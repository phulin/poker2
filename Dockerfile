# AlphaHoldem Training Dockerfile for vast.ai
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project
COPY . .

# Install the package in development mode with all extras
RUN pip install -e .[all]

# Create directories for checkpoints and logs
RUN mkdir -p /workspace/checkpoints /workspace/logs

# Set environment variables
ENV PYTHONPATH=/workspace
ENV CUDA_VISIBLE_DEVICES=0

# Default command (can be overridden)
CMD ["python", "-m", "alphaholdem.cli.train_rebel", "--config-name", "config_rebel_cfr"]
