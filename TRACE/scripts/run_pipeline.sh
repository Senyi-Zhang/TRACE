#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/run_pipeline.sh CONFIG.yaml" >&2
  exit 2
fi

config_path="$1"

trace --config "$config_path" validate-data
trace --config "$config_path" build-trees --split train
trace --config "$config_path" build-trees --split test

setting="$(python - "$config_path" <<'PY'
import sys
import yaml
with open(sys.argv[1], encoding="utf-8") as stream:
    print((yaml.safe_load(stream) or {}).get("setting", "open"))
PY
)"

if [[ "$setting" == "open" ]]; then
  trace --config "$config_path" build-index
  trace --config "$config_path" train-reranker
fi

trace --config "$config_path" prepare-evidence --split train
trace --config "$config_path" prepare-evidence --split test
trace --config "$config_path" train-verifier
trace --config "$config_path" evaluate --split test

