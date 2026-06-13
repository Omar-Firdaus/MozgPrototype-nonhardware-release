#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "Setting up Mozg..."

case "$ROOT" in
  *"Mobile Documents"*|*"iCloud"*)
    echo "Heads up: iCloud paths are slow. ~/Projects/mozg is better."
    ;;
esac

if [[ ! -x ".venv/bin/python3" ]]; then
  echo "Creating .venv"
  python3 -m venv .venv
fi

PY=".venv/bin/python3"
PIP=".venv/bin/pip"

"$PIP" install --upgrade pip -q
"$PIP" install -r python/requirements.txt -q
if [[ -f python/social-analyzer/requirements.txt ]]; then
  "$PIP" install -r python/social-analyzer/requirements.txt -q
fi

npm install --silent

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env — add API keys if you want Assist / Calendar / Apollo."
fi

if [[ ! -f python/models/arcface.onnx ]]; then
  echo "Missing python/models/arcface.onnx — face labels won't work until you add it."
else
  echo "Face model ok."
fi

echo "Done. Run: npm start"
