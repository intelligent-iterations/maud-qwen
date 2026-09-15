#!/usr/bin/env bash
set -euo pipefail
# Run only when validation-led iteration is complete. This fixes selection before test access.
python -m maud_qwen select
selection=manifests/final_selection.json
adapter=$(python -c 'import json; print(json.load(open("manifests/final_selection.json"))["adapter"])')
python -m maud_qwen majority --split test --final-record "$selection"
python -m maud_qwen evaluate --split test --final-record "$selection"
python -m maud_qwen evaluate --split test --adapter "$adapter" --final-record "$selection"
python -m maud_qwen compare --split test --final-record "$selection" \
  --left outputs/untuned-qwen/test.jsonl --right outputs/tuned-qwen/test.jsonl
