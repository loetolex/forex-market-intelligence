#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="$ROOT_DIR/.venv"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "Creating isolated project environment at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

PY="$VENV_DIR/bin/python"

"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r requirements.txt

"$PY" -c 'import numpy, pandas, sklearn, pyarrow, ib_async, fastapi; print("isolated environment imports: OK"); print("numpy", numpy.__version__); print("pandas", pandas.__version__)'

"$PY" -m compileall -q app
"$PY" -m pytest -q

echo "LOCAL ENVIRONMENT: HEALTHY"
echo "Use $PY for project commands."
