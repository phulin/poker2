from dataclasses import dataclass
from typing import List, Optional
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    batch_size: int = 1024
    num_epochs: int = 4
    replay_buffer_batches: int = 4
    gamma: float = 0.999
    gae_lambda: float = 0.95
    ppo_eps: float = 0.2
    ppo_delta1: float = 3.0
    value_coef: float = 0.05
    entropy_coef: float = 0.01
    grad_clip: float = 1.0
    value_loss_type: str = "huber"
    huber_delta: float = 1.0


@dataclass
class ModelConfig:
    name: str = "siamese_convnet_v1"
    kwargs: Optional[dict] = None
    policy: Optional[dict] = None

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {
                "cards_channels": 6,
                "actions_channels": 24,
                "fusion_hidden": [1024, 1024],
                "num_actions": 8,
            }
        if self.policy is None:
            self.policy = {"name": "categorical_v1", "kwargs": {}}


@dataclass
class EnvConfig:
    stack: int = 1000
    sb: int = 5
    bb: int = 10
    bet_bins: Optional[List[float]] = None
    card_encoder: Optional[dict] = None
    action_encoder: Optional[dict] = None

    def __post_init__(self):
        if self.bet_bins is None:
            self.bet_bins = [0.5, 0.75, 1.0, 1.5, 2.0]
        if self.card_encoder is None:
            self.card_encoder = {"name": "cards_planes_v1", "kwargs": {}}
        if self.action_encoder is None:
            self.action_encoder = {
                "name": "actions_hu_v1",
                "kwargs": {"history_actions_per_round": 6},
            }


@dataclass
class Config:
    # Training parameters
    num_steps: int = 2000
    k_best_pool_size: int = 10
    min_elo_diff: float = 50.0
    k_factor: float = 32.0  # ELO K-factor for rating changes
    checkpoint_interval: int = 50
    eval_interval: int = 100
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda"
    use_tensor_env: bool = True
    num_envs: int = 512
    use_wandb: bool = True
    wandb_project: str = "poker-kbest"
    wandb_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    wandb_run_id: Optional[str] = None
    resume_from: Optional[str] = None
    seed: int = 42
    config: Optional[str] = None
    economize_checkpoints: bool = False

    # Nested configs
    train: TrainingConfig = MISSING
    model: ModelConfig = MISSING
    env: EnvConfig = MISSING

    def __post_init__(self):
        if self.wandb_tags is None:
            self.wandb_tags = ["kbest", "poker", "ppo"]


# Register with Hydra
cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="train", name="default", node=TrainingConfig)
cs.store(group="model", name="default", node=ModelConfig)
cs.store(group="env", name="default", node=EnvConfig)
