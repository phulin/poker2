from dataclasses import dataclass
from typing import List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    learning_rate_final: float = 1e-5
    lr_schedule: str = "cosine"  # Learning rate schedule type
    batch_size: int = 1024
    episodes_per_step: int = 4
    replay_buffer_batches: int = 4
    max_trajectory_length: int = 12  # Maximum steps per trajectory in replay buffer
    max_sequence_length: int = 50  # Maximum sequence length for transformer models
    gamma: float = 0.999
    gae_lambda: float = 0.95
    ppo_eps: float = 0.2
    ppo_delta1: float = 3.0
    value_coef: float = 0.05
    entropy_coef: float = 0.01
    entropy_coef_final: float = 0.002
    entropy_decay_portion: float = 0.6  # Portion of training for linear entropy decay
    grad_clip: float = 1.0
    value_loss_type: str = "huber"
    huber_delta: float = 1.0
    use_mixed_precision: bool = False  # Enable automatic mixed precision
    loss_scale: float = 128.0  # Initial loss scale for mixed precision

    # Transformer-specific training parameters
    auxiliary_loss_coef: float = (
        0.0  # Coefficient for auxiliary losses (e.g., hand range prediction)
    )
    warmup_steps: int = 0  # Learning rate warmup steps
    weight_decay: float = 0.0  # Weight decay for regularization

    use_kv_cache: bool = True

    # Distributional critic configuration
    quantile_huber_kappa: float = 1.0


@dataclass
class ModelConfig:
    name: str = "siamese_convnet_v1"
    kwargs: Optional[dict] = None
    policy: Optional[dict] = None
    # backwards compatibility
    use_gradient_checkpointing: Optional[bool] = None
    value_head_type: Optional[str] = None
    value_head_num_quantiles: Optional[int] = None

    def __post_init__(self):
        if self.kwargs is None and self.name == "siamese_convnet_v1":
            self.kwargs = {
                "cards_channels": 6,
                "actions_channels": 24,
                "cards_hidden": 256,
                "actions_hidden": 256,
                "fusion_hidden": [1024, 1024],
                "num_actions": 8,
            }
        if self.policy is None:
            self.policy = {"name": "categorical_v1", "kwargs": {}}
        if self.value_head_type is None:
            self.value_head_type = "scalar"
        if self.value_head_num_quantiles is None:
            self.value_head_num_quantiles = 51


@dataclass
class EnvConfig:
    stack: int = 1000
    sb: int = 5
    bb: int = 10
    bet_bins: Optional[List[float]] = None
    card_encoder: Optional[dict] = None
    action_encoder: Optional[dict] = None
    debug_step_table: bool = (
        False  # Print debug table during step_bins when batch_size <= 8
    )
    flop_showdown: bool = False  # Skip turn/river, go directly to showdown after flop

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
class StateEncoderConfig:
    name: str = "cnn"  # "cnn" or "transformer"
    kwargs: Optional[dict] = None

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


@dataclass
class ExploiterConfig:
    """Configuration for exploiter training."""

    enabled: bool = True
    training_interval: int = 1000  # Train every 1000 steps
    training_steps: int = 50  # Number of steps to train exploiter
    learning_rate: float = 3e-4  # Higher LR for faster adaptation
    batch_size: int = 512  # Smaller batch for faster training
    episodes_per_step: int = 2  # Fewer epochs per update
    entropy_coef: float = 0.005  # Lower entropy for more focused exploitation


@dataclass
class Config:
    # Training parameters
    num_steps: int = 2000
    opponent_pool_type: str = "k_best"  # "k_best" or "dred"
    k_best_pool_size: int = 10
    min_elo_diff: float = 50.0
    min_step_diff: int = (
        300  # Minimum step difference before considering for pool updates
    )
    k_factor: float = 32.0  # ELO K-factor for rating changes
    checkpoint_interval: int = 50
    eval_interval: int = 100
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda"
    use_tensor_env: bool = True
    num_envs: int = 512
    use_wandb: bool = False
    wandb_project: Optional[str] = "poker-kbest"
    wandb_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    wandb_run_id: Optional[str] = None
    resume_from: Optional[str] = None
    seed: int = 42
    config: Optional[str] = None
    economize_checkpoints: bool = False
    strict_model_loading: bool = False  # Use strict model loading (default: False)

    # Nested configs
    train: TrainingConfig = MISSING
    model: ModelConfig = MISSING
    env: EnvConfig = MISSING
    state_encoder: StateEncoderConfig = MISSING
    exploiter: ExploiterConfig = MISSING

    def __post_init__(self):
        if self.wandb_tags is None:
            self.wandb_tags = ["kbest", "poker", "ppo"]


# Register with Hydra
cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="train", name="default", node=TrainingConfig)
cs.store(group="model", name="default", node=ModelConfig)
cs.store(group="env", name="default", node=EnvConfig)
cs.store(group="state_encoder", name="default", node=StateEncoderConfig)
cs.store(group="exploiter", name="default", node=ExploiterConfig)
