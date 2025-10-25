FROM nvidia/cuda:13.0.1-base-ubuntu24.04

# Set working directory
WORKDIR /workspace

# Install Python dependencies
RUN apt update && \
        DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
        curl ca-certificates build-essential cuda-minimal-build-13-0 && \
        rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://astral.sh/uv/install.sh | sh
RUN uv python install 3.13

# Copy the project
COPY src src
COPY conf conf
COPY pyproject.toml pyproject.toml
ENV PATH="/root/.local/bin:$PATH"
RUN uv sync
RUN uv cache clean

# Set environment variables
ENV PYTHONPATH=/workspace
ENV CUDA_VISIBLE_DEVICES=0

# Default command (can be overridden)
CMD ["uv", "run", "src/alphaholdem/cli/train_rebel.py"]
