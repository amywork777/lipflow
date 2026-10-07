#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

inputs=$(git ls-files -z -- pyproject.toml uv.lock uv.toml .python-version .python-versions .config/uv/uv.toml | xargs -0 sha256sum | sha256sum | cut -d ' ' -f 1)
stamp=.venv/.capy-install.sha256

case "${1:-}" in
  initialize)
    sudo apt-get update -qq
    sudo apt-get install -y -qq libgl1 libglib2.0-0 libevdev2 libportaudio2 xclip xdotool xvfb xauth
    UV_FROZEN=1 ./setup-linux.sh --samples
    printf '%s\n' "$inputs" > "$stamp"
    ;;
  refresh)
    if [[ ! -x .venv/bin/python || ! -f "$stamp" || "$(cat "$stamp")" != "$inputs" ]]; then
      uv sync --frozen
      printf '%s\n' "$inputs" > "$stamp"
    fi
    ;;
  *)
    printf 'Usage: %s {initialize|refresh}\n' "$0" >&2
    exit 2
    ;;
esac
