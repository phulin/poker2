from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf


class ValueLossType(str, Enum):
    huber = "huber"
    mse = "mse"
    quantile = "quantile"


class LrSchedule(str, Enum):
    cosine = "cosine"
    linear = "linear"


class ValueHeadType(str, Enum):
    scalar = "scalar"
    quantile = "quantile"


class PPOClipping(str, Enum):
    none = "none"
    single = "single"
    dual = "dual"


class KLType(str, Enum):
    forward = "forward"
    reverse = "reverse"
    none = "none"


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    learning_rate_final: float = 1e-5
    lr_schedule: LrSchedule = LrSchedule.cosine
    batch_size: int = 1024
    episodes_per_step: int = 4
    replay_buffer_batches: int = 4
    max_trajectory_length: int = 12  # Maximum steps per trajectory in replay buffer
    max_sequence_length: int = 50  # Maximum sequence length for transformer models
    gamma: float = 0.999
    gae_lambda: float = 0.95
    ppo_eps: float = 0.2
    ppo_delta1: float = 3.0
    ppo_clipping: PPOClipping = PPOClipping.dual
    ppo_dual_clip: float = 3.0  # Dual clip parameter for negative advantages
    # KL divergence type: "forward", "reverse", or "none"
    kl_type: KLType = KLType.reverse
    value_coef: float = 0.05
    entropy_coef: float = 0.01
    entropy_coef_final: float = 0.002
    entropy_decay_portion: float = 0.6  # Portion of training for linear entropy decay
    grad_clip: float = 1.0
    value_loss_type: ValueLossType = ValueLossType.huber
    huber_delta: float = 1.0
    return_clipping: bool = True  # Enable return clipping in value loss
    use_mixed_precision: bool = False  # Enable automatic mixed precision
    loss_scale: float = 128.0  # Initial loss scale for mixed precision

    # KL beta controller bounds
    beta_min: float = 1e-4
    beta_max: float = 10.0

    # KL beta controller dynamics
    beta_increase_factor: float = 2.0
    # If None, defaults to 1.0 / beta_increase_factor
    beta_decrease_factor: float = 0.5

    # Transformer-specific training parameters
    auxiliary_loss_coef: float = (
        0.0  # Coefficient for auxiliary losses (e.g., hand range prediction)
    )
    warmup_steps: int = 0  # Learning rate warmup steps
    weight_decay: float = 0.0  # Weight decay for regularization

    use_kv_cache: bool = False

    # Distributional critic configuration
    quantile_huber_kappa: float = 1.0

    # Learning rate configuration for different model components
    value_head_learning_rate: float = 1e-4  # Learning rate for value head
    value_head_learning_rate_final: float = 1e-5  # Final learning rate for value head
    policy_trunk_learning_rate: float = 1e-4  # Learning rate for policy head and trunk
    policy_trunk_learning_rate_final: float = (
        1e-5  # Final learning rate for policy head and trunk
    )

    # KL divergence configuration
    target_kl: float = 0.02  # Target KL divergence for adaptive controllers

    # Learning rate scaling controller
    lr_scaling_init_value: float = 1.0  # Initial LR scaling factor
    lr_scaling_min_value: float = 3e-2  # Minimum LR scaling factor
    lr_scaling_max_value: float = 3e0  # Maximum LR scaling factor
    lr_scaling_increase_factor: float = 1.5  # Factor to increase LR scaling
    lr_scaling_decrease_factor: float = 1.0 / 1.5  # Factor to decrease LR scaling
    lr_scaling_upper_threshold: float = 1.5  # Upper threshold multiplier
    lr_scaling_lower_threshold: float = 0.67  # Lower threshold multiplier

    # ReBeL/DCFR self-play exploration
    cfr_action_epsilon: float = 0.25  # Epsilon for action sampling during self-play


@dataclass
class ModelConfig:
    name: str = "siamese_convnet_v1"
    value_head_type: ValueHeadType = ValueHeadType.scalar
    value_head_num_quantiles: int = -1
    use_gradient_checkpointing: bool = True
    detach_value_head: bool = False

    # CNN-specific parameters (with defaults)
    cards_channels: int = 6
    actions_channels: int = 24
    cards_hidden: int = 256
    actions_hidden: int = 256
    fusion_hidden: list[int] = field(default_factory=lambda: [1024, 1024])
    num_actions: int = -1

    # Transformer-specific parameters (with defaults)
    max_sequence_length: int = 47
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 2
    dropout: float = 0.0

    # ReBeL FFN parameters
    input_dim: int = 2661
    hidden_dim: int = 1536
    num_hidden_layers: int = 6


@dataclass
class EnvConfig:
    stack: int = 1000
    sb: int = 5
    bb: int = 10
    bet_bins: List[float] = field(default_factory=lambda: [0.5, 0.75, 1.0, 1.5, 2.0])
    debug_step_table: bool = (
        False  # Print debug table during step_bins when batch_size <= 8
    )
    flop_showdown: bool = False  # Skip turn/river, go directly to showdown after flop


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
class SearchConfig:
    enabled: bool = False
    depth: int = 2
    iterations: int = 100
    warm_start_iterations: int = 15
    branching: int = 4
    belief_samples: int = 16
    max_belief_enumeration: int = 0  # 0 -> use belief_samples uniformly
    dcfr_alpha: float = 1.5
    dcfr_beta: float = 0.0
    dcfr_gamma: float = 2.0
    include_average_policy: bool = True
    linear_cfr: bool = True
    cfr_avg: bool = True


@dataclass
class Config:
    # Training parameters
    num_steps: int = 2000
    opponent_pool_type: str = "k_best"  # "k_best" or "dred"
    k_best_pool_size: int = 10
    min_elo_diff: float = 50.0
    # Minimum step difference before considering for pool updates
    min_step_diff: int = 300
    k_factor: float = 1.0  # ELO K-factor for rating changes
    checkpoint_interval: int = 50
    eval_interval: int = 100
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda"
    use_tensor_env: bool = True
    num_envs: int = 512
    # Move opponent models back to CPU after inference to save GPU memory
    offload_opponent_models: bool = True
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
    train: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    exploiter: ExploiterConfig = field(default_factory=ExploiterConfig)
    search: SearchConfig = field(default_factory=SearchConfig)

    def __post_init__(self):
        if self.wandb_tags is None:
            self.wandb_tags = ["kbest", "poker", "ppo"]
        # Derive action space size directly from bet bins so search/env/model stay aligned.
        self.model.num_actions = len(self.env.bet_bins) + 3

    @classmethod
    def from_dict_config(cls, dict_config: DictConfig) -> "Config":
        container = OmegaConf.to_container(dict_config, resolve=True)
        container["train"] = TrainingConfig(**(container.get("train", {})))
        container["model"] = ModelConfig(**(container.get("model", {})))
        container["env"] = EnvConfig(**(container.get("env", {})))
        container["exploiter"] = ExploiterConfig(**(container.get("exploiter", {})))
        container["search"] = SearchConfig(**(container.get("search", {})))
        return cls(**container)


# Register with Hydra
cs = ConfigStore.instance()
cs.store(name="config", node=Config)
