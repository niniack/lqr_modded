"""Cache raw transformer-layer activations and optional final-token JVPs."""

import argparse
import gc
import json
from pathlib import Path

import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


SHUFFLE_BUFFER_SIZE = 10_000
SHARD_SIZE = 256
TEACHER_DTYPE = torch.float32
CACHE_DTYPE = torch.float32


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face causal-LM ID or local model path")
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset ID or builder, e.g. json")
    parser.add_argument("--dataset-config")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--data-files", nargs="+", help="Optional files for dataset builders such as json")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--train-docs", type=int, required=True)
    parser.add_argument("--val-docs", type=int, required=True)
    parser.add_argument("--test-docs", type=int, required=True)
    layer_selection = parser.add_mutually_exclusive_group()
    layer_selection.add_argument(
        "--layers",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Inclusive 1-based transformer-layer outputs; default all layers",
    )
    layer_selection.add_argument(
        "--last-layers",
        type=int,
        metavar="STEPS",
        help="Replace the last STEPS transformer blocks; cache their input and outputs",
    )
    parser.add_argument(
        "--jvp-directions",
        type=int,
        default=0,
        help="Unit random directions per document for JVPs; 0 disables JVP collection",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", default="auto", help="Device such as cuda, cuda:0, or cpu")
    return parser


def _layer_hidden(output):
    return output[0] if isinstance(output, tuple) else output


def _capture_layer_call(module, args, kwargs, layer_index, captured_calls):
    extra_args = args[1:] if args else ()
    layer_kwargs = dict(kwargs)
    if not args:
        layer_kwargs.pop("hidden_states", None)
    captured_calls[layer_index] = extra_args, layer_kwargs


def _run_layer_slice(base_model, hidden_states, layer_calls, start_layer, end_layer, last_positions):
    batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    outputs = []
    for layer_index in range(start_layer, end_layer):
        extra_args, layer_kwargs = layer_calls[layer_index]
        layer_output = base_model.layers[layer_index](hidden_states, *extra_args, **layer_kwargs)
        hidden_states = _layer_hidden(layer_output)
        outputs.append(hidden_states[batch_indices, last_positions])
    return torch.stack(outputs, dim=1)


def _last_token_activations(
    model,
    tokenizer,
    texts,
    layer_indices,
    device,
    max_length,
    jvp_directions,
    direction_generator,
):
    batch = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    last_positions = attention_mask.sum(dim=1) - 1
    if torch.any(last_positions < 0):
        raise ValueError("A tokenized document contains no valid tokens")

    base_model = getattr(model, "model", None)
    if base_model is None or not hasattr(base_model, "layers"):
        raise ValueError("Activation caching requires a model.model decoder with transformer layers")
    if len(base_model.layers) != model.config.num_hidden_layers:
        raise ValueError("Decoder layer count does not match model.config.num_hidden_layers")

    captured_activations = {}
    captured_calls = {}
    captured_jvp_input = {}
    handles = []
    batch_indices = torch.arange(input_ids.shape[0], device=device)

    for output_layer in layer_indices:
        block_index = output_layer - 1
        layer = base_model.layers[block_index]

        def capture_output(module, args, output, output_layer=output_layer):
            hidden_states = _layer_hidden(output)
            captured_activations[output_layer] = hidden_states[batch_indices, last_positions].detach()
            if jvp_directions and output_layer == layer_indices[0]:
                captured_jvp_input["hidden_states"] = hidden_states.detach()

        handles.append(layer.register_forward_hook(capture_output))

    if jvp_directions:
        for layer_index in range(layer_indices[0], layer_indices[-1]):
            layer = base_model.layers[layer_index]
            hook = lambda module, args, kwargs, layer_index=layer_index: _capture_layer_call(
                module, args, kwargs, layer_index, captured_calls
            )
            handles.append(layer.register_forward_pre_hook(hook, with_kwargs=True))

    try:
        with torch.no_grad():
            base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    if set(captured_activations) != set(layer_indices):
        raise RuntimeError("Teacher forward did not execute every requested transformer layer")
    activations = torch.stack([captured_activations[layer] for layer in layer_indices], dim=1)
    token_lengths = attention_mask.sum(dim=1).tolist()
    unpadded_token_ids = [
        input_ids[index, :length].detach().to(device="cpu", dtype=torch.long)
        for index, length in enumerate(token_lengths)
    ]
    activations = activations.to(device="cpu", dtype=CACHE_DTYPE)
    if not jvp_directions:
        return activations, unpadded_token_ids, None

    start_layer, end_layer = layer_indices[0], layer_indices[-1]
    primal_inputs = captured_jvp_input.get("hidden_states")
    if primal_inputs is None:
        raise RuntimeError(f"Teacher forward did not capture the full sequence at layer {start_layer}")
    if set(captured_calls) != set(range(start_layer, end_layer)):
        raise RuntimeError("Teacher forward did not capture the arguments for every JVP slice layer")

    # These are the exact positional and keyword arguments used by the teacher's own block calls.
    slice_fn = lambda inputs: _run_layer_slice(
        base_model,
        inputs,
        captured_calls,
        start_layer,
        end_layer,
        last_positions,
    )
    expected_primals = torch.stack(
        [captured_activations[layer] for layer in range(start_layer + 1, end_layer + 1)],
        dim=1,
    )
    directions = []
    responses = []
    for _ in range(jvp_directions):
        direction = torch.randn(
            (input_ids.shape[0], primal_inputs.shape[-1]),
            generator=direction_generator,
            dtype=TEACHER_DTYPE,
        ).to(device=device)
        direction = torch.nn.functional.normalize(direction, dim=-1)
        tangent = torch.zeros_like(primal_inputs)
        tangent[batch_indices, last_positions] = direction
        try:
            with torch.no_grad():
                primal, response = torch.func.jvp(slice_fn, (primal_inputs,), (tangent,))
        except NotImplementedError as error:
            raise RuntimeError("A teacher layer operation does not support forward-mode AD") from error
        if not torch.allclose(primal, expected_primals, rtol=1e-5, atol=1e-6):
            max_error = (primal - expected_primals).abs().max().item()
            raise RuntimeError(f"JVP slice primal differs from cached teacher activations (max abs error {max_error})")
        directions.append(direction.to(device="cpu", dtype=CACHE_DTYPE))
        responses.append(response.to(device="cpu", dtype=CACHE_DTYPE))

    return activations, unpadded_token_ids, {
        "directions": torch.stack(directions),
        "responses": torch.stack(responses),
    }


def _cache_split(
    split_name,
    document_count,
    source,
    text_column,
    output_dir,
    model,
    tokenizer,
    layer_indices,
    device,
    batch_size,
    max_length,
    jvp_directions,
    direction_generator,
):
    split_dir = output_dir / split_name
    split_dir.mkdir()
    jvp_split_dir = None
    if jvp_directions:
        jvp_split_dir = output_dir / "jvp" / split_name
        jvp_split_dir.mkdir(parents=True)

    shard_names = []
    pending_activations = []
    pending_input_ids = []
    pending_directions = []
    pending_responses = []
    pending_count = 0
    documents_written = 0
    shard_index = 0
    saved_tensor_dtypes = None

    def save_shard():
        nonlocal pending_count, shard_index, saved_tensor_dtypes
        if not pending_count:
            return
        shard_name = f"{shard_index:06d}.pt"
        shard_activations = torch.cat(pending_activations)
        input_offsets = [0]
        for token_ids in pending_input_ids:
            input_offsets.append(input_offsets[-1] + token_ids.numel())
        shard_input_ids = torch.cat(pending_input_ids)
        shard_input_offsets = torch.tensor(input_offsets, dtype=torch.long)
        torch.save(
            {
                "activations": shard_activations,
                "input_ids": shard_input_ids,
                "input_offsets": shard_input_offsets,
            },
            split_dir / shard_name,
        )
        dtypes = {
            "activations": str(shard_activations.dtype),
            "input_ids": str(shard_input_ids.dtype),
            "input_offsets": str(shard_input_offsets.dtype),
        }
        if jvp_split_dir is not None:
            shard_directions = torch.cat(pending_directions, dim=1)
            shard_responses = torch.cat(pending_responses, dim=1)
            torch.save(
                {"directions": shard_directions, "responses": shard_responses},
                jvp_split_dir / shard_name,
            )
            dtypes["jvp_directions"] = str(shard_directions.dtype)
            dtypes["jvp_responses"] = str(shard_responses.dtype)
        if saved_tensor_dtypes is None:
            saved_tensor_dtypes = dtypes
        shard_names.append(shard_name)
        pending_activations.clear()
        pending_input_ids.clear()
        pending_directions.clear()
        pending_responses.clear()
        pending_count = 0
        shard_index += 1

    while documents_written < document_count:
        batch_count = min(batch_size, document_count - documents_written)
        texts = []
        for _ in range(batch_count):
            try:
                example = next(source)
            except StopIteration as error:
                raise ValueError(
                    f"Dataset split ended before collecting {document_count} {split_name} documents"
                ) from error
            text = example.get(text_column)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Dataset field {text_column!r} must contain non-empty text in {split_name}"
                )
            texts.append(text)

        activations, token_ids, jvp_data = _last_token_activations(
            model,
            tokenizer,
            texts,
            layer_indices,
            device,
            max_length,
            jvp_directions,
            direction_generator,
        )
        offset = 0
        while offset < batch_count:
            chunk_count = min(SHARD_SIZE - pending_count, batch_count - offset)
            chunk_end = offset + chunk_count
            pending_activations.append(activations[offset:chunk_end])
            pending_input_ids.extend(token_ids[offset:chunk_end])
            if jvp_data is not None:
                pending_directions.append(jvp_data["directions"][:, offset:chunk_end])
                pending_responses.append(jvp_data["responses"][:, offset:chunk_end])
            pending_count += chunk_count
            offset = chunk_end
            if pending_count == SHARD_SIZE:
                save_shard()
        documents_written += batch_count

    save_shard()
    split_info = {
        "documents": documents_written,
        "shards": shard_names,
        "tensor_dtypes": saved_tensor_dtypes,
    }
    if jvp_split_dir is not None:
        split_info["jvp_shards"] = shard_names
    return split_info


def main() -> None:
    args = _parser().parse_args()
    if args.train_docs <= 0 or args.val_docs < 0 or args.test_docs < 0:
        raise ValueError("train-docs must be positive; val-docs and test-docs cannot be negative")
    if args.batch_size < 1 or args.max_length < 1:
        raise ValueError("batch-size and max-length must be positive")
    if args.jvp_directions < 0:
        raise ValueError("jvp-directions cannot be negative")
    if args.layers is not None and args.layers[0] > args.layers[1]:
        raise ValueError("layers must be an inclusive contiguous range: START must be <= END")

    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer needs a pad token or EOS token to batch documents")
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"dtype": TEACHER_DTYPE}
    if args.jvp_directions:
        model_kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.to(device=device, dtype=TEACHER_DTYPE).eval()
    base_model = getattr(model, "model", None)
    if base_model is None or not hasattr(base_model, "layers"):
        raise ValueError("Activation caching requires a model.model decoder with transformer layers")
    num_layers = len(base_model.layers)
    if num_layers != model.config.num_hidden_layers:
        raise ValueError("Decoder layer count does not match model.config.num_hidden_layers")

    if args.last_layers is not None:
        if not 1 <= args.last_layers < num_layers:
            raise ValueError("--last-layers must be between 1 and num_hidden_layers - 1")
        start_layer, end_layer = num_layers - args.last_layers, num_layers
    elif args.layers is None:
        start_layer, end_layer = 1, num_layers
    else:
        start_layer, end_layer = args.layers
    if start_layer < 1 or end_layer > num_layers:
        raise ValueError(f"Layer range must be within [1, {num_layers}]")
    if args.jvp_directions and start_layer == end_layer:
        raise ValueError("JVP collection needs at least two cached layers to define a layer slice")
    layer_indices = list(range(start_layer, end_layer + 1))

    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.dataset_split,
        streaming=True,
        data_files=args.data_files,
    )
    document_count = args.train_docs + args.val_docs + args.test_docs
    source = iter(dataset.shuffle(seed=args.seed, buffer_size=min(SHUFFLE_BUFFER_SIZE, document_count)))
    direction_generator = torch.Generator(device="cpu").manual_seed(args.seed)

    splits = {}
    try:
        for split_name, count in (
            ("train", args.train_docs),
            ("val", args.val_docs),
            ("test", args.test_docs),
        ):
            splits[split_name] = _cache_split(
                split_name,
                count,
                source,
                args.text_column,
                output_dir,
                model,
                tokenizer,
                layer_indices,
                device,
                args.batch_size,
                args.max_length,
                args.jvp_directions,
                direction_generator,
            )
            print(f"Cached {count} {split_name} documents")
    finally:
        close_source = getattr(source, "close", None)
        if close_source is not None:
            close_source()
        del source, dataset
        gc.collect()

    manifest = {
        "model": args.model,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "transformers_version": transformers.__version__,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "text_column": args.text_column,
        "seed": args.seed,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "shard_documents": SHARD_SIZE,
        "num_transformer_layers": num_layers,
        "hidden_width": model.config.hidden_size,
        "cached_layers": layer_indices,
        "layer_indexing": "1-based raw transformer-block outputs; final model norm is not applied",
        "activation_shape": "[documents, cached_layers, hidden_width]",
        "token_ids_layout": "flat unpadded input_ids; input_offsets delimit each document",
        "teacher_parameter_dtypes": sorted({str(parameter.dtype) for parameter in model.parameters()}),
        "saved_tensor_dtypes": splits["train"]["tensor_dtypes"],
        "final_token": "last non-padding token after truncation",
        "output_head_loaded_but_not_run": hasattr(model, "lm_head"),
        "jvp": None
        if not args.jvp_directions
        else {
            "directions_per_document": args.jvp_directions,
            "input_layer": start_layer,
            "output_layers": list(range(start_layer + 1, end_layer + 1)),
            "direction": "unit-norm Gaussian; non-final token tangents are zero",
            "final_model_norm_applied": False,
            "directions_shape": "[directions, documents, hidden_width]",
            "responses_shape": "[directions, documents, output_layers, hidden_width]",
            "primal_check": {"rtol": 1e-5, "atol": 1e-6},
        },
        "splits": splits,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Cache written to {output_dir}")


if __name__ == "__main__":
    main()
