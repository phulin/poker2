from typing import TypedDict, List, Optional, Dict, Any


class TrainingConfigDict(TypedDict):
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


class ModelConfigDict(TypedDict):
    name: str
    kwargs: Dict[str, Any]


class EnvConfigDict(TypedDict):
    stack: int
    sb: int
    bb: int
    bet_bins: List[float]
    card_encoder: Dict[str, Any]
    action_encoder: Dict[str, Any]


class ConfigDict(TypedDict):
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
    train: TrainingConfigDict
    model: ModelConfigDict
    env: EnvConfigDict
