#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${FMH_TMUX_SESSION:-fmh}"
CONFIG="${FMH_CONFIG:-config.local.toml}"
CONFIG_Q="$(printf '%q' "$CONFIG")"

cd "$ROOT_DIR"

for env_name in HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy NO_PROXY no_proxy; do
  if [[ -v "$env_name" ]]; then
    tmux set-environment -g "$env_name" "${!env_name}"
  else
    tmux set-environment -gu "$env_name" 2>/dev/null || true
  fi
done

for legacy in fmh-poll fmh-review-auditor fmh-api; do
  if tmux has-session -t "$legacy" 2>/dev/null; then
    tmux kill-session -t "$legacy"
  fi
done

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" -n poll -c "$ROOT_DIR" \
  "python -m fmh.cli --config $CONFIG_Q poll"
tmux new-window -t "$SESSION:" -n review-auditor -c "$ROOT_DIR" \
  "python -m fmh.cli --config $CONFIG_Q review-auditor"
tmux new-window -t "$SESSION:" -n api -c "$ROOT_DIR" \
  "python -m fmh.cli --config $CONFIG_Q serve --host 0.0.0.0 --port 8787"
tmux select-window -t "$SESSION:poll"
