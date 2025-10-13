# Dockerfile for UTH Prediction API
# Multi-stage build for optimized production container

# ================================
# Build Arguments
# ================================
ARG PYTHON_VERSION=3.9
ARG CUDA_VERSION=11.8
ARG UBUNTU_VERSION=22.04

# ================================
# Base Stage - CUDA-enabled Ubuntu
# ================================
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as base

# Build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Metadata
LABEL maintainer="saishashank3000@gmail.com, kvsm2k@gmail.com" \
      description="UTH Prediction API with UNet++ VAE and SAPHIR integration" \
      version=${VERSION} \
      build-date=${BUILD_DATE} \
      vcs-ref=${VCS_REF} \
      license="MIT" \
      vendor="BYU Hydroinformatics"

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_VISIBLE_DEVICES=0

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python and development tools
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    python3-pip \
    # System dependencies
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    # Scientific computing libraries
    libhdf5-dev \
    libnetcdf-dev \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    # Image processing
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    # System utilities
    htop \
    vim \
    nano \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create symbolic links for python
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# ================================
# Development Stage
# ================================
FROM base as development

# Install development dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Create application directory
WORKDIR /app

# Copy source code
COPY . /app/

# Set up permissions
RUN chmod +x /app/*.sh || true

# Development port
EXPOSE 8888

# Default command for development
CMD ["/bin/bash"]

# ================================
# Production Stage
# ================================
FROM base as production

# Create non-root user for security
RUN groupadd -r uth && useradd -r -g uth uth

# Create directories
RUN mkdir -p /app /data /output /checkpoints /logs && \
    chown -R uth:uth /app /data /output /checkpoints /logs

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt && \
    pip cache purge

# Copy application code
COPY --chown=uth:uth . /app/

# Set up scripts permissions
RUN chmod +x /app/*.sh || true && \
    chmod +x /app/*/<select file>.py || true && \
    chmod +x /app/*/inference.py || true

# Create volume mount points
VOLUME ["/data", "/output", "/checkpoints"]

# Switch to non-root user
USER uth

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('Health check passed')" || exit 1

# Default command
CMD ["python", "--version"]

# ================================
# Runtime Stage (Multi-arch support)
# ================================
FROM production as runtime

# Environment for runtime
ENV MODEL_PATH=/checkpoints \
    DATA_PATH=/data \
    OUTPUT_PATH=/output \
    LOG_LEVEL=INFO

# Copy entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN sudo chmod +x /usr/local/bin/docker-entrypoint.sh || chmod +x /usr/local/bin/docker-entrypoint.sh

# Use entrypoint
ENTRYPOINT ["docker-entrypoint.sh"]

# ================================
# Additional Build Stages
# ================================

# GPU-optimized stage
FROM runtime as gpu
ENV CUDA_VISIBLE_DEVICES=0
RUN python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"

# CPU-only stage  
FROM runtime as cpu
ENV CUDA_VISIBLE_DEVICES=""
RUN python -c "import torch; print(f'PyTorch CPU-only: {torch.__version__}')"

# ================================
# Documentation Stage
# ================================
FROM nginx:alpine as docs

# Copy documentation
COPY README.md /usr/share/nginx/html/
COPY DEVELOPMENT.md /usr/share/nginx/html/
COPY --from=production /app /usr/share/nginx/html/source/

# Expose documentation port
EXPOSE 80

# ================================
# Final Multi-stage Selection
# ================================
# Default to production stage
FROM production