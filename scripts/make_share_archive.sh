#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/dist}"
ARCHIVE_NAME="${1:-free-my-hands-share.tar.gz}"

mkdir -p "$OUT_DIR"
tar -C "$ROOT_DIR" \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache' \
  --exclude='.ruff_cache' \
  --exclude='.mypy_cache' \
  -czf "$OUT_DIR/$ARCHIVE_NAME" \
  README.md \
  pyproject.toml \
  config.example.toml \
  .gitignore \
  docs \
  scripts/start_fmh_tmux.sh \
  scripts/make_share_archive.sh \
  src \
  tests

echo "$OUT_DIR/$ARCHIVE_NAME"
