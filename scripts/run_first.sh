#!/usr/bin/env bash
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
python -m maud_qwen verify
python -m pytest -q
python -m maud_qwen evaluate --limit 32
python -m maud_qwen smoke
python -m maud_qwen evaluate
python -m maud_qwen estimate
python -m maud_qwen train
