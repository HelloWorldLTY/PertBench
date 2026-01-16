#!/usr/bin/env sh
set -eu
cd GreenAgent
exec uv run python src/server.py --host 127.0.0.1 --port 19009
