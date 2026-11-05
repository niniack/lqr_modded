from __future__ import annotations

import torch

from .lqr import time_varying_lqr_noB, time_varying_lqr_noB_mem_efficient


class LQRSteering:
    """Apply LQR feedback to all decoder layers or only the final N layers.

    steer_last_n_layers=None selects all layers; 0 disables steering hooks.
    Jacobians and Riccati gains retain their full-model shapes.
    """

    def __init__(
        self,
        model,
        tokenizer,
        A: torch.Tensor,
        contrastive_vecs: torch.Tensor,
        q: float = 10.0,
        r: float = 10.0,
        qf: float = 1.0,
        preserve_mem: bool = True,
        *,
        steer_last_n_layers: int | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = model.model.layers
        self.T = len(self.layers)
        num_steered_layers = (
            self.T if steer_last_n_layers is None else steer_last_n_layers
        )
        if (
            not isinstance(num_steered_layers, int)
            or isinstance(num_steered_layers, bool)
            or not 0 <= num_steered_layers <= self.T
        ):
            raise ValueError(
                "steer_last_n_layers must be None or an integer between "
                f"0 and {self.T}."
            )
        self.steered_layer_indices = range(self.T - num_steered_layers, self.T)
        self.n = model.get_input_embeddings().weight.shape[1]
        self.input_device = model.get_input_embeddings().weight.device
        self.device = next(self.layers[0].parameters()).device

        if A.shape != (self.T, self.n, self.n):
            raise ValueError(
                f"A must have shape ({self.T}, {self.n}, {self.n}); got {tuple(A.shape)}."
            )
        if contrastive_vecs.shape != (self.T + 1, self.n):
            raise ValueError(
                "contrastive_vecs must have shape "
                f"({self.T + 1}, {self.n}); got {tuple(contrastive_vecs.shape)}."
            )

        self.E = contrastive_vecs.detach().to(
            device=self.device, dtype=torch.float32
        )
        identity = torch.eye(self.n, device=self.device, dtype=torch.float32)
        Q = (q * identity).expand(self.T, -1, -1)
        R = (r * identity).expand(self.T, -1, -1)
        Qf = qf * identity
        if preserve_mem:
            self.K = time_varying_lqr_noB_mem_efficient(
                A, Q, R, Qf, device=self.device
            )
        else:
            self.K = time_varying_lqr_noB(
                A.to(device=self.device, dtype=torch.float32), Q, R, Qf
            )

        self._layer_gains = [
            self.K[index].to(device=next(layer.parameters()).device)
            for index, layer in enumerate(self.layers)
        ]
        self.X = None
        self.U = torch.zeros(self.T, self.n, device=self.device, dtype=torch.float32)
        self.betas = None
        self.E_unit = None
        self._layer_directions = None
        self._layer_betas = None
        self.setpoint_type = "linear"
        self.ALL_TOKENS = False
        self.hooks = []

    def hook_setpoint_tracking(self, layer_idx, module, module_inputs, module_output):
        hidden_states = module_inputs[0]
        direction = self._layer_directions[layer_idx].to(hidden_states.device)
        gain = self._layer_gains[layer_idx].to(hidden_states.device)
        beta = self._layer_betas[layer_idx].to(hidden_states.device)
        hidden_states_fp32 = hidden_states.float()

        if self.ALL_TOKENS:
            x = hidden_states_fp32
            error_size = beta - torch.sum(x * direction, dim=-1, keepdim=True)
            error = error_size * direction
            control = error @ gain.T
            self.X[layer_idx] = x[-1, -1].to(self.device)
            self.U[layer_idx] = control[-1, -1].to(self.device)
        else:
            x = hidden_states_fp32[:, -1, :]
            error_size = beta - x @ direction
            error = error_size.unsqueeze(-1) * direction
            control = error @ gain.T
            self.X[layer_idx] = x[-1].to(self.device)
            self.U[layer_idx] = control[-1].to(self.device)

        if isinstance(module_output, (tuple, list)):
            output_hidden_states = module_output[0]
        else:
            output_hidden_states = module_output
        updated_hidden_states = output_hidden_states.clone()
        if self.ALL_TOKENS:
            updated_hidden_states = updated_hidden_states + control.to(
                dtype=updated_hidden_states.dtype
            )
        else:
            updated_hidden_states[:, -1, :] = (
                updated_hidden_states[:, -1, :]
                + control.to(dtype=updated_hidden_states.dtype)
            )

        if layer_idx == self.T - 1:
            self.X[self.T] = updated_hidden_states[-1, -1].float().to(self.device)

        if isinstance(module_output, tuple):
            return (updated_hidden_states, *module_output[1:])
        if isinstance(module_output, list):
            return [updated_hidden_states, *module_output[1:]]
        return updated_hidden_states

    def register_setpoint_tracking_hooks(self):
        if self.hooks:
            raise RuntimeError("Setpoint-tracking hooks are already registered.")
        try:
            for layer_idx in self.steered_layer_indices:
                layer = self.layers[layer_idx]
                def hook_wrapper(index):
                    def hook(module, module_inputs, module_output):
                        return self.hook_setpoint_tracking(
                            index, module, module_inputs, module_output
                        )

                    return hook

                self.hooks.append(
                    layer.register_forward_hook(hook_wrapper(layer_idx))
                )
        except Exception:
            self.remove_hooks()
            raise

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __enter__(self):
        self.register_setpoint_tracking_hooks()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.remove_hooks()

    def track_setpoint(
        self,
        prompt,
        max_new_tokens: int,
        lmbda: float = 1.0,
        do_sample: bool = False,
        temp: float = 1.0,
        all_tokens: bool = False,
        return_tokens: bool = False,
        *,
        top_p: float = 0.3,
    ):
        """Generate text while tracking each layer's contrastive setpoint."""
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative.")

        norms = torch.linalg.vector_norm(self.E, dim=-1)
        self.E_unit = torch.where(
            norms[:, None] > 0,
            self.E / norms.clamp_min(torch.finfo(torch.float32).tiny)[:, None],
            torch.zeros_like(self.E),
        )
        self.betas = lmbda * norms
        self._layer_directions = [
            self.E_unit[index].to(device=next(layer.parameters()).device)
            for index, layer in enumerate(self.layers)
        ]
        self._layer_betas = [
            self.betas[index].to(device=next(layer.parameters()).device)
            for index, layer in enumerate(self.layers)
        ]
        self.ALL_TOKENS = all_tokens
        self.X = torch.zeros(
            self.T + 1, self.n, device=self.device, dtype=torch.float32
        )

        original_padding_side = self.tokenizer.padding_side
        original_pad_token = self.tokenizer.pad_token
        try:
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token_id is None:
                if self.tokenizer.eos_token is None:
                    raise ValueError("Tokenizer needs a pad token or an EOS token.")
                self.tokenizer.pad_token = self.tokenizer.eos_token
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.input_device)
            generation_kwargs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "max_new_tokens": max_new_tokens,
                "return_dict_in_generate": True,
                "do_sample": do_sample,
                "use_cache": True,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            if do_sample:
                generation_kwargs["temperature"] = temp
                generation_kwargs["top_p"] = top_p
                generation_kwargs["repetition_penalty"] = 1.2

            with self:
                with torch.inference_mode():
                    output = self.model.generate(**generation_kwargs)

            if return_tokens:
                return output.sequences
            return self.tokenizer.batch_decode(
                output.sequences, skip_special_tokens=True
            )
        finally:
            self.ALL_TOKENS = False
            self.tokenizer.padding_side = original_padding_side
            if self.tokenizer.pad_token != original_pad_token:
                self.tokenizer.pad_token = original_pad_token
