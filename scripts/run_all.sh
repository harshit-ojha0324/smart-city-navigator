#!/usr/bin/env bash
# Boot the 3 FastMCP servers, then the Flask gateway wired to them over
# streamable HTTP. Ctrl-C tears the whole thing down.
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
export NAVIGATOR_TRANSPORT=mcp
: "${NAVIGATOR_SIMULATE_FEED:=0}"   # set to 1 to force the offline feed simulation
export NAVIGATOR_SIMULATE_FEED

PY="${PYTHON:-python}"
pids=()
cleanup() { echo; echo "stopping…"; kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

for s in alerts routing geocode; do
  $PY -m navigator.mcp_servers.${s}_server &
  pids+=($!)
  echo "started MCP server: $s (pid $!)"
done

# Wait for all three ports before starting the gateway.
$PY - <<'PY'
import socket, time
for name, p in {"alerts":8071,"routing":8072,"geocode":8073}.items():
    end = time.time() + 20
    while time.time() < end:
        try:
            socket.create_connection(("127.0.0.1", p), 0.5).close(); break
        except OSError:
            time.sleep(0.2)
    else:
        raise SystemExit(f"{name} server never came up on :{p}")
print("all MCP servers ready")
PY

echo "starting gateway on http://localhost:${PORT:-8000} (transport=mcp)"
exec $PY -m navigator.gateway.app
