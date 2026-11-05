from __future__ import annotations

import copy
import csv
import random
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .jacobian import linearize
from .wrappers import tf_block_wrapper, transformerBlockControl


class ContrastiveBuilder:
    """Focused port of the upstream activation-collection path.

    Collects each decoder block's final-prompt-token activation by running a
    one-token generation with forward hooks, matching the upstream
    ``collect_data_batch`` procedure without its pickle-file side effect.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.layers = model.model.layers
        self.num_layers = len(self.layers)
        self.hidden_size = model.get_input_embeddings().embedding_dim
        self._last_token_indices = None
        self._batch_activations = None
        self.hooks = []
        self._capture_full_sequence = False
        self._jacobian_metadata = None

    def _capture_activations(self, hidden_states):
        if hidden_states.ndim != 3:
            raise ValueError(
                "Decoder block activations must have shape (batch, sequence, hidden)."
            )
        if self._capture_full_sequence:
            if hidden_states.shape[0] != 1:
                raise ValueError("Jacobian collection requires a single prompt per pass.")
            return hidden_states[0].detach().to(device="cpu", dtype=torch.float32)
        batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        token_indices = self._last_token_indices.to(hidden_states.device)
        return hidden_states[
            batch_indices, token_indices
        ].detach().to(device="cpu", dtype=torch.float32)

    def hook_collector(self, layer_index, module, module_inputs, module_output):
        self._batch_activations[layer_index] = self._capture_activations(
            module_inputs[0]
        )
        if layer_index == self.num_layers - 1:
            final_hidden_states = (
                module_output[0]
                if isinstance(module_output, (tuple, list))
                else module_output
            )
            self._batch_activations[self.num_layers] = self._capture_activations(
                final_hidden_states
            )
        return module_output

    def register_hooks(self):
        try:
            for layer_index, layer in enumerate(self.layers):
                def hook_wrapper(index):
                    def hook(module, module_inputs, module_output):
                        return self.hook_collector(
                            index, module, module_inputs, module_output
                        )

                    return hook

                self.hooks.append(layer.register_forward_hook(hook_wrapper(layer_index)))
        except Exception:
            self.remove_hooks()
            raise

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.remove_hooks()

    def collect_data_batch(
        self,
        prompts: list[str],
        num_samples: int,
        batch_size: int = 50,
    ) -> torch.Tensor:
        """Return mean final-token activations for sampled prompts.

        The result has shape ``(num_layers + 1, hidden_size)``. Row zero is
        the embedding output; subsequent rows are decoder block outputs.
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self._capture_full_sequence = False
        samples = random.sample(prompts, num_samples)
        activation_sum = torch.zeros(self.num_layers + 1, self.hidden_size)
        original_padding_side = self.tokenizer.padding_side
        original_pad_token = self.tokenizer.pad_token

        try:
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token_id is None:
                if self.tokenizer.eos_token is None:
                    raise ValueError("Tokenizer needs a pad token or an EOS token.")
                self.tokenizer.pad_token = self.tokenizer.eos_token

            for start in range(0, len(samples), batch_size):
                batch = samples[start : start + batch_size]
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device)
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"].float()
                positions = torch.arange(
                    attention_mask.shape[1], device=attention_mask.device
                )
                self._last_token_indices = (attention_mask * positions).argmax(dim=1)
                self._batch_activations = [None] * (self.num_layers + 1)
                with self:
                    with torch.inference_mode():
                        self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=1,
                            return_dict_in_generate=True,
                            do_sample=False,
                            use_cache=False,
                            pad_token_id=self.tokenizer.eos_token_id,
                        )

                if any(activation is None for activation in self._batch_activations):
                    raise RuntimeError(
                        "Generation did not visit every decoder block; activations are incomplete."
                    )
                batch_activations = torch.stack(self._batch_activations)
                activation_sum += batch_activations.sum(dim=1)
        finally:
            self.tokenizer.padding_side = original_padding_side
            if self.tokenizer.pad_token != original_pad_token:
                self.tokenizer.pad_token = original_pad_token
            self._last_token_indices = None
            self._batch_activations = None
            self._capture_full_sequence = False
            self._jacobian_metadata = None

        return activation_sum / num_samples

    def collect_jacobians(
        self,
        prompts: list[str],
        num_samples: int,
        max_ctx: int = 32,
    ) -> torch.Tensor:
        """Average dense per-layer state Jacobians over sampled prompts."""
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if max_ctx <= 0:
            raise ValueError("max_ctx must be positive.")

        samples = random.sample(prompts, num_samples)
        jacobian_sum = torch.zeros(
            self.num_layers, self.hidden_size, self.hidden_size
        )
        progress = tqdm(
            total=len(samples) * self.num_layers,
            desc="Computing Jacobians",
            unit="Jacobian",
        )
        self._capture_full_sequence = True

        try:
            for prompt_index, prompt in enumerate(samples, start=1):
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_ctx,
                ).to(self.device)
                self._batch_activations = [None] * (self.num_layers + 1)
                self._jacobian_metadata = None

                def capture_metadata(module, module_inputs, module_kwargs):
                    if self._jacobian_metadata is None:
                        self._jacobian_metadata = {
                            name: module_kwargs.get(name)
                            for name in (
                                "attention_mask",
                                "position_ids",
                                "position_embeddings",
                            )
                        }

                metadata_hook = self.layers[0].register_forward_pre_hook(
                    capture_metadata,
                    with_kwargs=True,
                )
                try:
                    with self:
                        with torch.no_grad():
                            self.model.generate(
                                input_ids=inputs["input_ids"],
                                attention_mask=inputs["attention_mask"],
                                max_new_tokens=1,
                                return_dict_in_generate=True,
                                do_sample=False,
                                use_cache=False,
                                pad_token_id=self.tokenizer.eos_token_id,
                            )
                finally:
                    metadata_hook.remove()

                if any(activation is None for activation in self._batch_activations):
                    raise RuntimeError(
                        "Generation did not visit every decoder block; activations are incomplete."
                    )
                if self._jacobian_metadata is None:
                    raise RuntimeError("Could not capture decoder position metadata.")
                if (
                    self._jacobian_metadata["position_ids"] is None
                    or self._jacobian_metadata["position_embeddings"] is None
                ):
                    raise RuntimeError(
                        "Decoder did not provide position IDs and rotary embeddings."
                    )

                nominal_states = torch.stack(self._batch_activations).to(
                    dtype=torch.float32
                )
                attention_mask = self._jacobian_metadata["attention_mask"]
                position_ids = self._jacobian_metadata["position_ids"]
                position_embeddings = self._jacobian_metadata[
                    "position_embeddings"
                ]

                for layer_index, layer in enumerate(self.layers):
                    layer_device = next(layer.parameters()).device
                    x_nominal = nominal_states[layer_index : layer_index + 2].to(
                        device=layer_device, dtype=torch.float32
                    )
                    layer_attention_mask = (
                        attention_mask.to(
                            device=layer_device, dtype=torch.float32
                        )
                        if attention_mask is not None
                        else None
                    )
                    layer_position_ids = position_ids.to(device=layer_device)
                    layer_position_embeddings = tuple(
                        embedding.to(device=layer_device, dtype=torch.float32)
                        for embedding in position_embeddings
                    )
                    try:
                        float32_layer = copy.deepcopy(layer)
                    except Exception as error:
                        raise RuntimeError(
                            f"Could not copy decoder layer {layer_index} for "
                            "float32 Jacobian computation; use unquantized "
                            "decoder layers."
                        ) from error
                    float32_layer.to(device=layer_device, dtype=torch.float32)
                    float32_layer.eval()
                    for parameter in float32_layer.parameters():
                        parameter.requires_grad_(False)

                    transition = partial(
                        transformerBlockControl,
                        partial(
                            tf_block_wrapper,
                            float32_layer,
                            layer_attention_mask,
                            layer_position_ids,
                            layer_position_embeddings,
                        ),
                    )
                    layer_jacobian = linearize(
                        [transition],
                        1,
                        self.hidden_size,
                        x_nominal,
                    )
                    jacobian_sum[layer_index] += layer_jacobian[0].detach().to(
                        device="cpu", dtype=torch.float32
                    )
                    del transition, float32_layer, layer_jacobian, x_nominal
                    progress.set_postfix(
                        prompt=f"{prompt_index}/{len(samples)}",
                        layer=f"{layer_index + 1}/{self.num_layers}",
                        refresh=False,
                    )
                    progress.update(1)
        finally:
            progress.close()
            self._capture_full_sequence = False
            self._jacobian_metadata = None
            self._batch_activations = None
            self._last_token_indices = None
        return jacobian_sum / num_samples


SENTENCES_CSV = Path(__file__).resolve().parent.parent / "data" / "filtered_sentences.csv"


def load_concept_prompts(
    concept: str,
    sample_size: int = 200,
    csv_path: str | Path = SENTENCES_CSV,
) -> tuple[list[str], list[str]]:
    """Sample target and balanced non-target sentences for a concept."""
    target_sentences = []
    other_sentences_by_concept = defaultdict(list)

    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            label = row["concept"]
            sentence = row["sentence"]
            if label == f"{concept}.NOUN":
                target_sentences.append(sentence)
            else:
                other_sentences_by_concept[label].append(sentence)

    samples_per_other_concept = min(
        len(sentences) for sentences in other_sentences_by_concept.values()
    )
    background_sentences = [
        other_sentences_by_concept[other_concept][sample_index]
        for sample_index in range(samples_per_other_concept)
        for other_concept in other_sentences_by_concept
    ]

    return (
        random.sample(target_sentences, sample_size),
        random.sample(background_sentences, sample_size),
    )


def compute_concept_direction(
    concept: str,
    model,
    tokenizer,
    sample_size: int = 200,
    batch_size: int = 50,
    csv_path: str | Path = SENTENCES_CSV,
) -> torch.Tensor:
    """Return per-layer target-minus-background mean activations."""
    target_sentences, background_sentences = load_concept_prompts(
        concept,
        sample_size=sample_size,
        csv_path=csv_path,
    )
    builder = ContrastiveBuilder(model, tokenizer)
    target_mean = builder.collect_data_batch(
        target_sentences,
        num_samples=sample_size,
        batch_size=batch_size,
    )
    background_mean = builder.collect_data_batch(
        background_sentences,
        num_samples=sample_size,
        batch_size=batch_size,
    )
    return target_mean - background_mean


__all__ = [
    "ContrastiveBuilder",
    "SENTENCES_CSV",
    "load_concept_prompts",
    "compute_concept_direction",
]


