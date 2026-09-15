#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${IBKR_BRIDGE_TOKEN:?Set IBKR_BRIDGE_TOKEN before starting the bridge}"

if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
fi

.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt

export IBKR_HOST="${IBKR_HOST:-127.0.0.1}"
export IBKR_PORT="${IBKR_PORT:-7497}"
export IBKR_CLIENT_ID="${IBKR_CLIENT_ID:-901}"
export IBKR_BRIDGE_PORT="${IBKR_BRIDGE_PORT:-8787}"

exec .venv/bin/python -m uvicorn app.ibkr_bridge:app --host 127.0.0.1 --port "$IBKR_BRIDGE_PORT"
