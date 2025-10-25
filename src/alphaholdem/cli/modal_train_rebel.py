#!/usr/bin/env python3
"""
Modal wrapper for ReBeL training script.

This script uploads the local source code to Modal and runs the ReBeL training
with the specified configuration. It handles all the necessary setup including
dependencies, file uploads, and environment configuration.
"""

import hydra
import modal
from omegaconf import DictConfig, OmegaConf

# Create Modal app
app = modal.App("rebel-training")

# Define the image with all necessary dependencies
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        [
            "torch>=2.9.0",
            "pyyaml>=6.0.2",
            "wandb>=0.15.0",
            "hydra-core>=1.3.0",
            "omegaconf>=2.3.0",
            "numpy>=1.24.0",
        ]
    )
    .pip_install("modal")  # Ensure modal is available
    .add_local_dir("conf", "/conf")
    .add_local_dir("src", "/src")
)

# Define the volume for persistent storage (checkpoints, etc.)
volume = modal.Volume.from_name("rebel-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/checkpoints": volume},
    timeout=86400,  # 1 day timeout
    secrets=[modal.Secret.from_name("wandb-secret")],  # For wandb API key
)
@hydra.main(version_base=None, config_path="/conf", config_name="config_rebel_cfr")
def train_rebel_modal(cfg: DictConfig):
    """
    Train ReBeL model in Modal.

    Args:
        config: Config object
    """
    import torch
    import sys
    import os

    # Add src to Python path
    sys.path.insert(0, "/src")

    # Import training function
    from alphaholdem.cli.train_rebel import train_rebel
    from alphaholdem.core.structured_config import (
        Config,
        EnvConfig,
        ModelConfig,
        SearchConfig,
        TrainingConfig,
    )

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    train_config = TrainingConfig(**cfg_dict.get("train", {}))
    model_config = ModelConfig(**cfg_dict.get("model", {}))
    env_config = EnvConfig(**cfg_dict.get("env", {}))
    search_config = SearchConfig(**cfg_dict.get("search", {}))

    config = Config(
        train=train_config,
        model=model_config,
        env=env_config,
        search=search_config,
        **{
            k: v
            for k, v in cfg_dict.items()
            if k not in ["train", "model", "env", "search"]
        },
    )

    print("Starting ReBeL training in Modal...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(
            f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    print(f"Configuration: {config}")

    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Start training
    train_rebel(config)
    print("Training completed successfully!")

    # Commit volume to persist checkpoints
    volume.commit()
    print("Checkpoints saved to volume")


@app.local_entrypoint()
def main():
    print("Starting ReBeL training in Modal...")

    # Run training
    train_rebel_modal.remote()


if __name__ == "__main__":
    main()
