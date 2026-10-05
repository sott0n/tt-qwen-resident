#!/bin/bash
# Fetch tt-metal at the pinned commit, apply this repo's ttnn patches, build it and its python env.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
git submodule update --init --recursive tt-metal
for p in patches/*.patch; do
  if git -C tt-metal apply --check "../$p" 2>/dev/null; then
    git -C tt-metal apply "../$p"
  else
    echo "skip $p (already applied or does not apply)"
  fi
done
cd tt-metal
./build_metal.sh
./create_venv.sh
