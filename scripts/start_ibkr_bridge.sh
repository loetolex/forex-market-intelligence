#!/bin/sh
set -eu

: "${IBKR_BRIDGE_TOKEN:?Set IBKR_BRIDGE_TOKEN before starting the bridge}"

exec uvicorn app.ibkr_bridge:app --host 127.0.0.1 --port "${IBKR_BRIDGE_PORT:-8787}"
