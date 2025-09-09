# AlphaHoldem Training Dockerfile for vast.ai
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

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

# Install additional dependencies for training
RUN pip install --no-cache-dir \
    wandb \
    tensorboard \
    matplotlib \
    seaborn \
    tqdm

# Copy the project
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create directories for checkpoints and logs
RUN mkdir -p /workspace/checkpoints /workspace/logs

# Set environment variables
ENV PYTHONPATH=/workspace
ENV CUDA_VISIBLE_DEVICES=0

# Default command (can be overridden)
CMD ["python", "alphaholdem/cli/train_kbest.py", "--device", "cuda", "--steps", "1000", "--use-tensor-env"]
