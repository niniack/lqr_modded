"""Frozen native Gemma forward passes in raw transformer-block coordinates.

Only final nonpadding-token states are retained, except for the optional context
at the first selected boundary. Attention masks and rotary embeddings are owned
entirely by Transformers; extraction never calls the language-model head.
"""

import re

import torch
import transformers
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


class TeacherHead(nn.Module):
    """Frozen final norm and softcapped LM head, differentiable in raw inputs."""

    def __init__(self, norm: nn.Module, lm_head: nn.Module, softcap: float | None):
        super().__init__()
        self.norm = norm
        self.lm_head = lm_head
        self.softcap = softcap
        self.eval().requires_grad_(False)

    def forward(self, raw_state: torch.Tensor) -> torch.Tensor:
        # Do not disable autograd: the KL loss must reach student predictions.
        with torch.autocast(raw_state.device.type, enabled=False):
            logits = self.lm_head(self.norm(raw_state.float()))
            if self.softcap is not None:
                logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits


class _SuffixPrecision:
    """Disable prefix autocast inside native suffix blocks without replacing them."""

    def __init__(self, backbone, start, enabled):
        self._contexts = []
        self._handles = []
        if not enabled:
            return

        def enter_fp32(module, args):
            context = torch.autocast("cuda", enabled=False)
            context.__enter__()
            self._contexts.append((module, context))

        def leave_fp32(module, args, output):
            if self._contexts and self._contexts[-1][0] is module:
                self._contexts.pop()[1].__exit__(None, None, None)

        try:
            for layer in backbone.layers[start:]:
                self._handles.append(layer.register_forward_pre_hook(enter_fp32))
                self._handles.append(layer.register_forward_hook(leave_fp32, always_call=True))
        except BaseException:
            self.close()
            raise

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        while self._contexts:
            self._contexts.pop()[1].__exit__(None, None, None)


class GemmaTeacher:
    """One frozen FP32 Gemma backbone, tokenizer, head and clean target extractor.

    ``layers`` are contiguous 1-based raw block-output boundaries. The first is
    the student input; the rest are its trajectory targets. ``last_layers=N``
    selects the input and outputs of the final N blocks (default N=4).
    BF16 is optional and applies only before the selected input boundary, on
    CUDA devices with native BF16 support. Parameters and raw outputs stay FP32.
    """

    def __init__(self, model_name, device, *, revision=None, layers=None,
                 last_layers=None, max_length=512, prefix_dtype=torch.float32,
                 collect_context=False):
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Gemma teacher extraction requires CPU or CUDA")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA teacher extraction requires available CUDA")
        if prefix_dtype not in {torch.float32, torch.bfloat16}:
            raise ValueError("prefix_dtype must be torch.float32 or torch.bfloat16")
        if prefix_dtype == torch.bfloat16:
            if self.device.type != "cuda":
                raise ValueError("BF16 prefix requires native CUDA BF16 support; use float32 on CPU")
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported(including_emulation=False):
                    raise ValueError("BF16 prefix requires native CUDA BF16 support; use float32")
        if type(max_length) is not int or max_length < 1:
            raise ValueError("max_length must be a positive integer")
        if layers is not None and last_layers is not None:
            raise ValueError("Select layers or last_layers, not both")
        self.max_length = max_length
        self.collect_context = bool(collect_context)
        self.prefix_dtype = prefix_dtype
        self._closed = False

        config = AutoConfig.from_pretrained(model_name, revision=revision)
        text_config = config.get_text_config(decoder=True)
        if text_config is not config:
            text_config._commit_hash = getattr(config, "_commit_hash", None)
        if text_config.model_type not in {"gemma", "gemma2", "gemma3_text"}:
            raise ValueError(f"Expected a Gemma text backbone, got {text_config.model_type!r}")
        actual_revision = getattr(text_config, "_commit_hash", None)
        if (revision is not None and actual_revision is not None
                and re.fullmatch(r"[0-9a-fA-F]{40}", revision)
                and revision != actual_revision):
            raise ValueError("Resolved teacher revision disagrees with the requested checkpoint pin")
        resolved_revision = actual_revision or revision
        self.num_layers = text_config.num_hidden_layers
        self.hidden_width = text_config.hidden_size
        if layers is None:
            steps = 4 if last_layers is None else last_layers
            if type(steps) is not int or not 1 <= steps < self.num_layers:
                raise ValueError("last_layers must be between 1 and num_hidden_layers - 1")
            layers = range(self.num_layers - steps, self.num_layers + 1)
        self.layers = list(layers)
        if (len(self.layers) < 2
                or any(type(layer) is not int for layer in self.layers)
                or self.layers != list(range(self.layers[0], self.layers[-1] + 1))
                or not 1 <= self.layers[0] < self.layers[-1] <= self.num_layers):
            raise ValueError("layers must contain at least two contiguous 1-based raw block boundaries")

        # A multimodal Gemma3 checkpoint is loaded with its text config only.
        # Pin weights and tokenizer to the exact revision resolved by AutoConfig.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=resolved_revision, config=text_config, dtype=torch.float32,
        )
        self.model.to(device=self.device, dtype=torch.float32).eval().requires_grad_(False)
        backbone = self.model.model
        if (len(backbone.layers) != self.num_layers
                or self.model.config.hidden_size != self.hidden_width):
            raise ValueError("Loaded Gemma backbone dimensions disagree with its configuration")
        parameter_dtypes = sorted({str(parameter.dtype) for parameter in self.model.parameters()})
        if parameter_dtypes != [str(torch.float32)]:
            raise ValueError("Gemma teacher parameters must all be FP32")
        loaded_revision = getattr(self.model.config, "_commit_hash", None)
        if actual_revision is not None and loaded_revision not in {None, actual_revision}:
            raise ValueError("Loaded teacher revision disagrees with its resolved configuration")
        resolved_revision = loaded_revision or resolved_revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=resolved_revision)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is None:
                raise ValueError("Tokenizer needs a pad token or EOS token to batch documents")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.head = TeacherHead(
            backbone.norm, self.model.get_output_embeddings(),
            getattr(self.model.config, "final_logit_softcapping", None),
        )
        self._precision = _SuffixPrecision(
            backbone, self.layers[0], enabled=prefix_dtype == torch.bfloat16,
        )
        self.metadata = {
            "teacher_model": str(model_name),
            "teacher_model_revision": resolved_revision,
            "teacher_checkpoint_path": str(model_name),
            "teacher_checkpoint_revision": resolved_revision,
            "teacher_model_type": text_config.model_type,
            "teacher_parameter_dtypes": parameter_dtypes,
            "teacher_dtype": str(torch.float32),
            "teacher_attention_implementation": getattr(self.model.config, "_attn_implementation", None),
            "transformers_version": transformers.__version__,
            "teacher_extraction": "native_backbone_forward_hooks",
            "teacher_compute_dtype": str(prefix_dtype),
            "teacher_prefix_compute_dtype": str(prefix_dtype),
            "teacher_suffix_compute_dtype": str(torch.float32),
            "teacher_autocast_dtype": str(prefix_dtype) if prefix_dtype == torch.bfloat16 else None,
            "teacher_output_dtype": str(torch.float32),
            "teacher_head_parameter_dtypes": parameter_dtypes,
            "teacher_head_compute_dtype": str(torch.float32),
            "teacher_head_shared": True,
            "teacher_head_executed_during_extraction": False,
            "num_transformer_layers": self.num_layers,
            "hidden_width": self.hidden_width,
            "cached_layers": self.layers,
            "max_length": self.max_length,
            "layer_indexing": "1-based raw transformer-block outputs; final model norm is not applied",
            "final_token": "last non-padding token after truncation",
            "context_boundary": self.layers[0] if self.collect_context else None,
            "tokenizer_padding_side": self.tokenizer.padding_side,
            "tokenizer_pad_token_id": self.tokenizer.pad_token_id,
            "tokenizer_eos_token_id": self.tokenizer.eos_token_id,
            "tokenizer_add_bos_token": getattr(self.tokenizer, "add_bos_token", None),
            "tokenizer_add_eos_token": getattr(self.tokenizer, "add_eos_token", None),
            "tokenizer_truncation_side": self.tokenizer.truncation_side,
        }

    def tokenize(self, text):
        """Return unpadded, capped CPU input_ids/attention_mask of shape [1,T]."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Gemma teacher requires a nonempty text string")
        batch = self.tokenizer(
            text, return_tensors="pt", padding=False, truncation=True,
            max_length=self.max_length,
        )
        return {key: batch[key].to("cpu") for key in ("input_ids", "attention_mask")}

    def _batch_tokens(self, tokenized_batches):
        if not tokenized_batches:
            raise ValueError("Teacher extraction requires at least one tokenized document")
        lengths = []
        for tokens in tokenized_batches:
            ids, mask = tokens["input_ids"], tokens["attention_mask"]
            if (ids.device.type != "cpu" or mask.device.type != "cpu"
                    or ids.ndim != 2 or ids.shape[0] != 1 or mask.shape != ids.shape
                    or ids.dtype != torch.long or not 1 <= ids.shape[1] <= self.max_length
                    or not bool((mask == 1).all())):
                raise ValueError("Expected unpadded CPU input_ids/attention_mask [1,T] within max_length")
            lengths.append(ids.shape[1])
        input_ids = torch.full(
            (len(lengths), max(lengths)), self.tokenizer.pad_token_id, dtype=torch.long,
        )
        mask = torch.zeros_like(input_ids)
        for row, (tokens, length) in enumerate(zip(tokenized_batches, lengths)):
            input_ids[row, :length] = tokens["input_ids"][0]
            mask[row, :length] = 1
        return input_ids.to(self.device), mask.to(self.device), lengths

    def targets_batch(self, tokenized_batches):
        """Extract detached FP32 input [B,D], targets [B,S,D] and optional context."""
        if self._closed:
            raise RuntimeError("Gemma teacher is closed")
        input_ids, mask, lengths = self._batch_tokens(tokenized_batches)
        rows = torch.arange(len(lengths), device=self.device)
        positions = torch.tensor(lengths, device=self.device) - 1
        captured = {}
        handles = []
        context = None
        context_mask = None
        if self.collect_context:
            context_mask = (
                torch.arange(self.max_length - 1, device=self.device)[None]
                < positions[:, None]
            )

        def capture(boundary):
            def hook(module, args, output):
                nonlocal context
                hidden = output[0] if isinstance(output, tuple) else output
                captured[boundary] = hidden[rows, positions].detach()
                if self.collect_context and boundary == self.layers[0]:
                    context = hidden.new_zeros((len(lengths), self.max_length - 1, self.hidden_width))
                    available = min(hidden.shape[1], self.max_length - 1)
                    context[:, :available].copy_(hidden[:, :available].detach())
                    context.masked_fill_(~context_mask[:, :, None], 0)
            return hook

        try:
            for boundary in self.layers:
                handles.append(self.model.model.layers[boundary - 1].register_forward_hook(capture(boundary)))
            with torch.no_grad(), torch.autocast(
                self.device.type, dtype=self.prefix_dtype,
                enabled=self.prefix_dtype == torch.bfloat16,
            ):
                self.model.model(input_ids=input_ids, attention_mask=mask, use_cache=False)
            if set(captured) != set(self.layers):
                raise RuntimeError("Native teacher forward missed a requested raw block boundary")
            states = torch.stack([captured[boundary] for boundary in self.layers], dim=1)
            result = {"inputs": states[:, 0], "targets": states[:, 1:]}
            if self.collect_context:
                if context is None:
                    raise RuntimeError("Native teacher forward missed the input-boundary context")
                result.update(context=context, context_mask=context_mask)
            if any(value.dtype != torch.float32 or not bool(torch.isfinite(value).all())
                   for key, value in result.items() if key != "context_mask"):
                raise RuntimeError("Teacher raw states and context must be finite FP32 tensors")
            return result
        finally:
            for handle in handles:
                handle.remove()

    def contexts_batch(self, tokenized_batches):
        """Extract only the optional input-boundary context from exact CPU tokens."""
        if not self.collect_context:
            raise RuntimeError("Context extraction requires collect_context=True")
        batch = self.targets_batch(tokenized_batches)
        return {key: batch[key] for key in ("context", "context_mask")}

    def close(self):
        """Remove owned precision hooks; safe to call repeatedly."""
        if not self._closed:
            self._closed = True
            self._precision.close()
