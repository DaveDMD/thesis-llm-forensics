# Thesis Dockerfile — Membership Inference and Secret Extraction in LLMs
# Reproducible experimental infrastructure.

# ============================================================================
# Base image: PyTorch 2.4.0 + CUDA 12.4 + cuDNN 9, verified on an RTX 4070
# ============================================================================
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# Image metadata (code versioning)
LABEL description="Thesis experimental environment: MIA, extraction, RAG"
LABEL version="0.1.0"

# ============================================================================
# Build arguments (override via docker-compose or --build-arg)
# ============================================================================
# Pinned MIMIR commit (reproducibility). 'main' on the first build, then
# replaced with the actual hash to lock it down.
ARG MIMIR_COMMIT=main

# Non-root user UID/GID, aligned with the host to avoid permission issues
# on mounted volumes (HF cache, forensic logs)
ARG USER_UID=1000
ARG USER_GID=1000

# ============================================================================
# System dependencies
# ============================================================================
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        git-lfs \
        curl \
        wget \
        ca-certificates \
        build-essential \
        vim \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ============================================================================
# Non-root user (security)
# ============================================================================
RUN groupadd --gid ${USER_GID} researcher \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash researcher \
    && mkdir -p /workspace /opt/mimir \
    && chown -R researcher:researcher /workspace /opt/mimir

# ============================================================================
# Install MIMIR (Duan et al. 2024) at the specified commit
# ============================================================================
USER researcher
WORKDIR /opt

RUN git clone https://github.com/iamgroot42/mimir.git mimir \
    && cd mimir \
    && git checkout ${MIMIR_COMMIT} \
    && git rev-parse HEAD > /opt/mimir/.commit_hash

# ============================================================================
# Python dependencies (ours + MIMIR)
# ============================================================================
COPY --chown=researcher:researcher requirements.txt requirements.lock.txt /tmp/

# Upgrade pip and install our dependencies (they override MIMIR's where the
# versions are compatible)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /tmp/requirements.lock.txt \
    && pip install --no-cache-dir -e /opt/mimir \
    && rm /tmp/requirements.txt /tmp/requirements.lock.txt

# Build-time check: bitsandbytes imports and reports its version.
# (Loading the actual CUDA backend needs a GPU and happens at runtime;
# this catches at least a broken import before the first model run.)
RUN python -c "import bitsandbytes as bnb; print('bitsandbytes', bnb.__version__, 'import OK')"

# Download the NLTK data needed for the Neighbourhood attack (Mattern et al. 2023)
RUN python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True)"

# ============================================================================
# Runtime environment variables
# ============================================================================
ENV HF_HOME=/workspace/models/hf-cache \
    HF_DATASETS_CACHE=/workspace/models/hf-cache/datasets \
    MIMIR_CACHE_PATH=/workspace/data/mimir-cache \
    MIMIR_DATA_SOURCE=/workspace/data/mimir-data \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false

# ============================================================================
# Working directory: project code
# ============================================================================
WORKDIR /workspace

# Default: interactive bash
CMD ["/bin/bash"]

# ── forensic package (Tier-1 logging, indexer, verifier) ──────────────
# Installed editable: changes from the host (bind-mounted tools/) are reflected
# immediately in the container without a rebuild.
# NOTE: /workspace/tools/forensic must be bind-mounted by docker-compose.yml.
# The actual install happens the first time the container starts, via
# ENTRYPOINT or manually.
