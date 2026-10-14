#!/usr/bin/env bash
set -euo pipefail

if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "Activate the target Conda environment first (for example: conda activate mopd)." >&2
  exit 1
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX}/bin/python}"
export PYTHONNOUSERSITE=1
"${PYTHON_BIN}" - <<'PY'
import os
import sys

if os.path.realpath(sys.prefix) != os.path.realpath(os.environ["CONDA_PREFIX"]):
    raise SystemExit("PYTHON_BIN must belong to the active Conda environment.")
PY
PIP_BIN=("${PYTHON_BIN}" -m pip)

"${PIP_BIN[@]}" install swanlab math-verify matplotlib duckdb

"${PIP_BIN[@]}" install -e "${SCRIPT_DIR}/verl"

"${PIP_BIN[@]}" install ray==2.55.1
"${PIP_BIN[@]}" install transformers==4.57.6
"${PIP_BIN[@]}" install protobuf==6.33.5

if [ -f "${SCRIPT_DIR}/requirements-if-rl.txt" ]; then
  "${PIP_BIN[@]}" install -r "${SCRIPT_DIR}/requirements-if-rl.txt"
fi

VERIFIABLE_INSTRUCTIONS_PATH="${VERIFIABLE_INSTRUCTIONS_PATH:-}"
if [ -z "${VERIFIABLE_INSTRUCTIONS_PATH}" ] && [ -d "${SCRIPT_DIR}/third_party/verifiable-instructions" ]; then
  VERIFIABLE_INSTRUCTIONS_PATH="${SCRIPT_DIR}/third_party/verifiable-instructions"
fi
if [ -n "${VERIFIABLE_INSTRUCTIONS_PATH}" ] && [ -d "${VERIFIABLE_INSTRUCTIONS_PATH}" ]; then
  "${PIP_BIN[@]}" install -e "${VERIFIABLE_INSTRUCTIONS_PATH}"
else
  echo "[install_requirements] verifiable-instructions not installed; set VERIFIABLE_INSTRUCTIONS_PATH or add third_party/verifiable-instructions" >&2
fi

"${PYTHON_BIN}" - <<'PY' || true
import nltk
for resource in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)
PY
