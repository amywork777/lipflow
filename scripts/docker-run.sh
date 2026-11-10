#!/usr/bin/env bash
# Helper script to run Lipflow inside Docker on Linux with GUI, webcam and GPU pass-through.
set -euo pipefail

cd "$(dirname "$0")/.."

# Check if docker is installed
if ! command -v docker >/dev/null; then
    echo "Error: docker is not installed or not in PATH."
    exit 1
fi

# Clean up socks5 proxies if present to prevent Docker BuildKit lookup errors
if [[ "${HTTP_PROXY:-}" =~ socks5 ]] || [[ "${HTTPS_PROXY:-}" =~ socks5 ]]; then
    export HTTP_PROXY= HTTPS_PROXY= ALL_PROXY=
fi

# Determine current user UID and GID for non-root execution
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

# Build GPU flags if NVIDIA GPU is present
GPU_FLAGS=()
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_FLAGS=(--gpus all)
fi

# Determine available hardware devices
DEV_FLAGS=()
for dev in /dev/video* /dev/snd /dev/dri /dev/uinput; do
    if [ -e "$dev" ]; then
        DEV_FLAGS+=(--device="$dev:$dev")
    fi
done

# X11 Display configuration (no xhost +local:root needed)
X11_FLAGS=()
if [ -n "${DISPLAY:-}" ]; then
    X11_FLAGS+=(-e DISPLAY="${DISPLAY}" -v /tmp/.X11-unix:/tmp/.X11-unix:ro)
    if [ -n "${XAUTHORITY:-}" ] && [ -f "${XAUTHORITY}" ]; then
        X11_FLAGS+=(-e XAUTHORITY="${XAUTHORITY}" -v "${XAUTHORITY}:${XAUTHORITY}:ro")
    elif [ -f "${HOME}/.Xauthority" ]; then
        X11_FLAGS+=(-e XAUTHORITY="/home/lipflow/.Xauthority" -v "${HOME}/.Xauthority:/home/lipflow/.Xauthority:ro")
    fi
fi

# Wayland Display configuration: pass actual socket from $XDG_RUNTIME_DIR
WAYLAND_FLAGS=()
if [ -n "${WAYLAND_DISPLAY:-}" ] && [ -n "${XDG_RUNTIME_DIR:-}" ]; then
    WAYLAND_SOCK="${XDG_RUNTIME_DIR}/${WAYLAND_DISPLAY}"
    if [ -S "${WAYLAND_SOCK}" ]; then
        WAYLAND_FLAGS+=(
            -e WAYLAND_DISPLAY="${WAYLAND_DISPLAY}"
            -e XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR}"
            -v "${XDG_RUNTIME_DIR}:${XDG_RUNTIME_DIR}:ro"
        )
    fi
fi

# PulseAudio / PipeWire audio socket if present
AUDIO_FLAGS=()
if [ -n "${XDG_RUNTIME_DIR:-}" ] && [ -S "${XDG_RUNTIME_DIR}/pulse/native" ]; then
    AUDIO_FLAGS+=(-e PULSE_SERVER="unix:${XDG_RUNTIME_DIR}/pulse/native" -v "${XDG_RUNTIME_DIR}/pulse/native:${XDG_RUNTIME_DIR}/pulse/native:ro")
fi

# Ensure host data and models directories exist
mkdir -p "${PWD}/models" "${HOME}/.local/share/Lipflow"

# Build image if needed
if ! docker image inspect lipflow:latest >/dev/null 2>&1; then
    echo "Building lipflow:latest Docker image (UID=${HOST_UID}, GID=${HOST_GID})..."
    docker build --build-arg UID="${HOST_UID}" --build-arg GID="${HOST_GID}" -t lipflow:latest .
fi

echo "Starting Lipflow in Docker..."
exec docker run --rm -it \
    --network=host \
    --ipc=host \
    --user "${HOST_UID}:${HOST_GID}" \
    -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
    "${X11_FLAGS[@]}" \
    "${WAYLAND_FLAGS[@]}" \
    "${AUDIO_FLAGS[@]}" \
    -v "${PWD}/models:/app/models" \
    -v "${HOME}/.local/share/Lipflow:/home/lipflow/.local/share/Lipflow" \
    "${GPU_FLAGS[@]}" \
    "${DEV_FLAGS[@]}" \
    lipflow:latest "$@"
