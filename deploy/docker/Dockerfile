# ============================================================================
# Genova – GPU-ready multi-stage Docker image
# Base: NVIDIA CUDA 12.1 + cuDNN 8 on Ubuntu 22.04
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1: builder – install Poetry, resolve deps, build wheel
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.7.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

# System deps required for building Python packages (pysam, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        python3-pip \
        build-essential \
        cmake \
        git \
        curl \
        zlib1g-dev \
        libbz2-dev \
        liblzma-dev \
        libcurl4-openssl-dev \
        libssl-dev \
        libffi-dev \
        libhts-dev \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

WORKDIR /build

# Copy only dependency manifests first (Docker layer caching)
COPY pyproject.toml poetry.lock* ./

# Install runtime deps (no dev group)
RUN poetry install --no-root --only main --no-interaction \
    || poetry lock --no-update && poetry install --no-root --only main --no-interaction

# Copy full source and install the project itself
COPY . .
RUN poetry install --only main --no-interaction

# ---------------------------------------------------------------------------
# Stage 2: runtime – slim image with only what we need
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Minimal runtime system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3-pip \
        libgomp1 \
        libcurl4 \
        zlib1g \
        libbz2-1.0 \
        liblzma5 \
        libhts3 \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/lib/python3/dist-packages /usr/lib/python3/dist-packages
COPY --from=builder /usr/lib/python3.10 /usr/lib/python3.10

# Copy the project
WORKDIR /app
COPY --from=builder /build /app

# Copy entry-point scripts
COPY --from=builder /usr/local/bin/genova /usr/local/bin/genova 2>/dev/null || true

# Create non-root user for production
RUN groupadd --gid 1000 genova \
    && useradd --uid 1000 --gid genova --shell /bin/bash --create-home genova \
    && chown -R genova:genova /app

# Shared volumes for data & checkpoints
RUN mkdir -p /data /checkpoints /logs \
    && chown -R genova:genova /data /checkpoints /logs

VOLUME ["/data", "/checkpoints", "/logs"]

USER genova

# Health-check for API mode
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: show help
ENTRYPOINT ["tini", "--"]
CMD ["python", "-m", "genova.cli.main", "--help"]
