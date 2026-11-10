FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    git \
    libgl1 \
    libglib2.0-0 \
    libgirepository-1.0-1 \
    gir1.2-ayatanaappindicator3-0.1 \
    libportaudio2 \
    python3-tk \
    xclip \
    xdotool \
    wl-clipboard \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Non-root user setup
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} lipflow 2>/dev/null || true \
    && useradd -u ${UID} -g ${GID} -m -s /bin/bash lipflow 2>/dev/null || true \
    && usermod -aG video,audio lipflow 2>/dev/null || true

ENV HOME=/home/lipflow \
    PATH="/home/lipflow/.local/bin:${PATH}"

WORKDIR /app

# Ensure directories exist and have proper ownership
RUN mkdir -p /app/models /home/lipflow/.local/share/Lipflow \
    && chown -R lipflow:lipflow /app /home/lipflow

USER lipflow

# Install standalone Python 3.12 managed by uv
RUN uv python install 3.12

# Copy dependency specifications first for layer caching
COPY --chown=lipflow:lipflow pyproject.toml uv.lock ./

# Install project dependencies
RUN uv sync --frozen --python 3.12

# Copy source code and helper scripts
COPY --chown=lipflow:lipflow lipflow/ ./lipflow/
COPY --chown=lipflow:lipflow espnet/ ./espnet/
COPY --chown=lipflow:lipflow scripts/ ./scripts/
COPY --chown=lipflow:lipflow README.md LICENSE NOTICE ./

# Install package in editable mode
RUN uv pip install -e . --no-deps

# Default entrypoint runs lipflow inside virtualenv
ENTRYPOINT ["uv", "run", "--no-sync", "lipflow"]
CMD ["doctor"]
