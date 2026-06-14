#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "Bootstrapping Python + Node dependencies..."
bash scripts/bootstrap.sh

echo "Building macOS app..."
npm run build:mac

echo "Build complete. Artifacts are in dist/"
