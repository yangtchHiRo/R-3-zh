#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-cu128}"
case "$mode" in
  cu128|cpu) ;;
  *) echo 'Usage: bash scripts/install.sh [cu128|cpu]' >&2; exit 2 ;;
esac
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url "https://download.pytorch.org/whl/$mode"
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
