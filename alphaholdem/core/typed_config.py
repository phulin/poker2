from dataclasses import dataclass
from typing import List, Optional
from omegaconf import DictConfig, OmegaConf


@dataclass
class TrainingConfig:
    learning_rate: float
    batch_size: int
    num_epochs: int
    gamma: float
    gae_lambda: float
    ppo_eps: float
    ppo_delta1: float
    value_coef: float
    entropy_coef: float
    grad_clip: float
    value_loss_type: str
    huber_delta: float


@dataclass
class ModelConfig:
    name: str
    kwargs: dict


@dataclass
class EnvConfig:
    stack: int
    sb: int
    bb: int
    bet_bins: List[float]
    card_encoder: dict
    action_encoder: dict


@dataclass
class Config:
    # Training parameters
    num_steps: int
    k_best_pool_size: int
    min_elo_diff: float
    checkpoint_interval: int
    eval_interval: int
    checkpoint_dir: str
    device: str
    use_tensor_env: bool
    num_envs: int
    use_wandb: bool
    wandb_project: str
    wandb_name: Optional[str]
    wandb_tags: List[str]
    wandb_run_id: Optional[str]
    resume_from: Optional[str]
    seed: int
    config: Optional[str]

    # Nested configs
    train: TrainingConfig
    model: ModelConfig
    env: EnvConfig


def load_typed_config(cfg: DictConfig) -> Config:
    """Convert Hydra DictConfig to typed dataclass."""
    return OmegaConf.to_object(cfg)
