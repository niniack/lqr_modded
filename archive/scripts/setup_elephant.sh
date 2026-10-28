#!/usr/bin/env bash
# Official Elephant serving code, isolated from the KAE/Gemma dependency lock.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CACHE="$ROOT/.cache/elephant-1"
ENV="$CACHE/venv"
SRC="$CACHE/OneJev"
REV=81ce62f1597c91e46767d4d02d6ac2e18534fe94
MODEL_REV=b3f8a93462fb1051a9de031e94a7ff380e254072

if [[ -f "$CACHE/.ready" && -x "$ENV/bin/python" ]]; then
    printf 'Elephant runtime ready: %s\n' "$ENV/bin/python"
    exit 0
fi
mkdir -p "$CACHE"
if [[ ! -x "$ENV/bin/python" ]]; then
    uv venv --python 3.11 "$ENV"
fi
if [[ ! -d "$SRC/.git" ]]; then
    git clone --quiet https://github.com/OmniJev/OneJev.git "$SRC"
    git -C "$SRC" checkout --quiet --detach "$REV"
fi
PATCH_FILE="$(uv run --frozen --group baseline python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('baselabs/elephant-1', 'serve.patch', revision='$MODEL_REV'))")"
CONSTRAINTS="$(uv run --frozen --group baseline python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('baselabs/elephant-1', 'constraints.txt', revision='$MODEL_REV'))")"
if git -C "$SRC" apply --check "$PATCH_FILE"; then
    git -C "$SRC" apply "$PATCH_FILE"
else
    git -C "$SRC" apply --reverse --check "$PATCH_FILE"
fi
# Match the publisher's Torch 2.8/CUDA 12 runtime; use its prebuilt conv kernel,
# avoiding a local CUDA compiler requirement. This environment is Python 3.11/x86-64.
uv pip install --python "$ENV/bin/python" --constraint "$CONSTRAINTS" \
    'torch==2.8.0' 'torchvision==0.23.0' 'accelerate==1.15.0' \
    'transformers==5.19.0' 'huggingface_hub==2.2.0' 'hf-xet==1.7.0' \
    'flash-linear-attention==0.5.2' brotli --editable "$SRC" \
    'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl'
"$ENV/bin/python" -c 'import torch, transformers, causal_conv1d; from qev.engine import DecisionEngine; print("Elephant runtime:", torch.__version__, transformers.__version__)'
touch "$CACHE/.ready"
