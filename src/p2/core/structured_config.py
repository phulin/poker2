from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from p2.core.action_schedule import apply_action_schedule_to_config


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


class ModelType(str, Enum):
    siamese_convnet_v1 = "SiameseConvnetV1"
    poker_transformer_v1 = "PokerTransformerV1"
    rebel_ffn = "RebelFFN"
    better_ffn = "BetterFFN"
    better_trm = "BetterTRM"


class PPOClipping(str, Enum):
    none = "none"
    single = "single"
    dual = "dual"


class KLType(str, Enum):
    forward = "forward"
    reverse = "reverse"
    none = "none"


class PolicyNodeWeighting(str, Enum):
    uniform = "uniform"
    reach = "reach"
    sqrt_reach = "sqrt_reach"
    clipped_reach = "clipped_reach"


class PolicyLossType(str, Enum):
    cross_entropy = "cross_entropy"
    mse = "mse"


class CFRType(str, Enum):
    standard = "standard"
    linear = "linear"
    discounted = "discounted"


class WarmStartType(str, Enum):
    model = "model"
    model_br = "model_br"


class NonlinearityType(str, Enum):
    relu = "relu"
    leaky_relu = "leaky_relu"
    gelu = "gelu"
    silu = "silu"
    swiglu = "swiglu"


torch.serialization.add_safe_globals(ValueLossType)
torch.serialization.add_safe_globals(LrSchedule)
torch.serialization.add_safe_globals(ValueHeadType)
torch.serialization.add_safe_globals(ModelType)
torch.serialization.add_safe_globals(PPOClipping)
torch.serialization.add_safe_globals(PolicyNodeWeighting)
torch.serialization.add_safe_globals(PolicyLossType)
torch.serialization.add_safe_globals(KLType)
torch.serialization.add_safe_globals(CFRType)
torch.serialization.add_safe_globals(WarmStartType)
torch.serialization.add_safe_globals(NonlinearityType)
torch.serialization.add_safe_globals(PPOClipping)


@dataclass
class StratifyConfig:
    threshold: int
    probabilities: list[float]


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    learning_rate_final: float = 1e-5
    lr_schedule: LrSchedule = LrSchedule.cosine
    warmup_steps: int = 0
    batch_size: int = 1024
    episodes_per_step: int = 4

    # how many batches of data do we want to store in the replay buffer?
    # i.e. maximum age of a sample.
    replay_buffer_batches: int = 4
    replay_buffer_device: str = "cpu"

    # how many times do we want to reuse the same policy/value data?
    value_reuse_goal: float = 8.0
    policy_capacity_factor: int = 10
    policy_depth_stratify_decimate: bool = True
    policy_depth_stratify_sample: bool = True
    policy_depth_stratify_probs: list[float] | None = None
    policy_node_weighting: PolicyNodeWeighting = PolicyNodeWeighting.uniform
    policy_loss_type: PolicyLossType = PolicyLossType.cross_entropy
    policy_logit_l2_coef: float = 0.0
    policy_extra_updates_per_step: int = 0
    policy_extra_batch_size: int | None = None

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
    permutation_coef: float = 0.01
    grad_clip: float = 1.0
    # Separate clip for the policy net's gradients. Only takes effect when the
    # policy and value heads have disjoint parameters (separate nets, or a TRM
    # with shared_trunk=False); with a shared trunk grad_clip is used for all
    # parameters. Looser than grad_clip because policy CE gradients run hotter.
    policy_grad_clip: float = 10.0
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
    # Coefficient for auxiliary losses (e.g., hand range prediction)
    auxiliary_loss_coef: float = 0.0
    warmup_steps: int = 0  # Learning rate warmup steps
    weight_decay: float = 0.01  # Weight decay for regularization
    optimizer: str = "adamw"  # ReBeL trainer: "adamw", "muon", or "normuon"
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_eps: float = 1e-7
    muon_ns_steps: int = 5
    muon_adjust_lr_fn: str | None = None
    normuon_beta1: float = 0.95
    normuon_beta2: float = 0.95
    normuon_eps: float = 1e-8
    policy_head_muon_learning_rate: float = 0.05
    adamw_learning_rate: float | None = None

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

    # Stratify streets until the given step with the given probabilities.
    stratify_streets: list[StratifyConfig] = field(default_factory=list)

    # Exponential Moving Average (EMA) for model weights
    # If None, EMA is disabled. If float, it's the EMA decay rate (typically 0.999)
    model_ema: float | None = None


@dataclass
class ModelConfig:
    name: ModelType = ModelType.siamese_convnet_v1
    compile: str = "default"
    value_head_type: ValueHeadType = ValueHeadType.scalar
    value_head_num_quantiles: int = -1
    use_gradient_checkpointing: bool = True
    detach_value_head: bool = False
    nonlinearity: NonlinearityType = NonlinearityType.leaky_relu

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
    num_hidden_layers: int = 3
    num_policy_layers: int = 3
    num_value_layers: int = 3

    # Better FFN parameters
    range_hidden_dim: int = 128
    ffn_dim: int = 1024
    shared_trunk: bool = True
    enforce_zero_sum: bool = True
    board_interaction_dim: int = 0
    policy_rank: int = 64
    policy_hand_bias_rank: int = 32
    value_rank: int = 128

    # Better TRM parameters
    num_recursions: int = 6
    num_iterations: int = 3
    num_supervisions: int = 16


@dataclass
class EnvConfig:
    num_players: int = 2
    stack: int = 10000
    stack_mode: str = "fixed"
    min_stack_bb: int = 10
    mid_stack_bb: int = 200
    max_stack_bb: int = 400
    high_stack_mass_ratio: float = 1.0 / 3.0
    sb: int = 50
    bb: int = 100
    bet_bins: List[float] = field(default_factory=lambda: [0.5, 0.75, 1.0, 1.5, 2.0])
    randomize_stacks: bool = False
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
    iterations_final: int | None = (
        None  # If set, linearly interpolate from iterations to iterations_final
    )
    warm_start_iterations: int = 15
    warm_start_type: WarmStartType = WarmStartType.model_br
    warm_start_multiplier: float = 1.0
    branching: int = 4
    belief_samples: int = 16
    sample_epsilon: float = 0.25
    dcfr_alpha: float = 1.5
    dcfr_alpha_final: float | None = None
    dcfr_beta: float = 0.0
    dcfr_beta_final: float | None = None
    dcfr_gamma: float = 2.0
    dcfr_gamma_final: float | None = None
    dcfr_plus_delay: int = 50
    include_average_policy: bool = True
    cfr_type: CFRType = CFRType.linear
    cfr_plus: bool = True
    cfr_avg: bool = True
    sparse: bool = True
    sparse_fused: bool = False
    value_targets_from_final_policy: bool = False
    allin_call_terminal_abstraction: bool = True
    preflop_allin_table_path: str | None = None
    closing_leaf_checkpoint: str | None = None
    street_model_checkpoints: dict[str, str] = field(default_factory=dict)
    bet_bins_by_depth: list[list[float]] | None = None
    allin_by_depth: list[bool] | None = None
    continuation_value_target_sampling: bool = False
    continuation_value_target_streets: list[int] = field(default_factory=lambda: [0])
    continuation_value_target_min_depth: int = 0
    continuation_value_target_max_depth: int | None = None
    continuation_value_targets_replace_roots: bool = False


@dataclass
class TrueSkillConfig:
    enabled: bool = False
    # How often to snapshot and evaluate, as a fraction of total training run.
    snapshot_frac: float = 0.01
    # Target evaluation cost as a fraction of total training compute, expressed
    # in "actions". Total scheduled training actions are estimated as
    # num_steps * num_envs * actions_per_game (a game = ~16 actions).
    game_budget_frac: float = 0.03
    actions_per_game: int = 16
    # TrueSkill plays two independently searched models at each decision and
    # runs with less efficient batching than data generation. These factors
    # convert the nominal action budget into a more realistic game budget.
    eval_solves_per_action: float = 2.0
    eval_inefficiency_factor: float = 2.0
    # Recency weighting: weight of opponent i (1 = oldest) in the sampling
    # distribution is exp(-(N - i) / recency_tau_frac * N), where N = number
    # of stored snapshots. Larger -> more uniform; smaller -> more recent-heavy.
    recency_tau_frac: float = 0.25
    # Number of prior snapshots sampled for each eval. When fewer snapshots
    # exist, all available opponents are used.
    opponents_per_eval: int = 10
    # Minimum and maximum games to play against any sampled opponent per eval.
    min_games_per_opponent: int = 1
    max_games_per_opponent: int = 64
    # TrueSkill priors.
    initial_mu: float = 25.0
    initial_sigma: float = 25.0 / 3.0
    beta: float = 25.0 / 6.0
    tau: float = 25.0 / 300.0
    # Parallel Gaussian reward rating, in stack-normalized reward units.
    # Each opponent matchup observes mean_reward ~= mu_a - mu_b with
    # observation std gaussian_beta / sqrt(num_games).
    gaussian_initial_mu: float = 0.0
    gaussian_initial_sigma: float = 1.0
    gaussian_beta: float = 1.0
    gaussian_tau: float = 0.01
    # Snapshots are kept in CPU RAM as bfloat16 shadow weights.
    snapshot_dtype: str = "bfloat16"


@dataclass
class PregeneratedDatasetConfig:
    path: str
    value_weight: float = 1.0
    policy_weight: float = 1.0
    min_step: int = 0
    max_step: int | None = None


@dataclass
class PregeneratedDataConfig:
    value_batch_size: int | None = None
    policy_batch_size: int | None = None
    shuffle: bool = True
    direct_sample: bool = False
    pin_memory: bool = True
    async_shard_prefetch: bool = True
    validate_manifest: bool = True
    datasets: list[PregeneratedDatasetConfig] = field(default_factory=list)


@dataclass
class DataConfig:
    mode: str = "live"
    live_root_source: str = "self_play"
    include_pre_chance_value_batches: bool = True
    pregenerated: PregeneratedDataConfig = field(default_factory=PregeneratedDataConfig)


@dataclass
class CurriculumSubstepConfig:
    kind: str = "train"
    net: str = ""
    num_steps: int = 0
    from_net: str | None = None
    closing_net: str | None = None
    closing_checkpoint: str | None = None
    chance: str | None = None
    checkpoint: str | None = None
    output_dir: str | None = None
    data_overrides: dict[str, Any] = field(default_factory=dict)
    search_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class CurriculumConfig:
    stages: list[str] = field(default_factory=list)
    wandb_group: str | None = None
    promote_dir: str | None = None
    substeps: dict[str, CurriculumSubstepConfig] = field(default_factory=dict)


@dataclass
class RebelPregenerateConfig:
    output_dir: str = "outputs/rebel_postflop/river_v1"
    stage: str = "river"
    value_target_min: int = 100_000
    policy_target_min: int = 1_000_000
    generation_batch_size: int = 512
    max_generation_batches: int | None = None
    root_source: str | None = None
    storage_dtype: str | None = None


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
    trueskill: TrueSkillConfig = field(default_factory=TrueSkillConfig)
    data: DataConfig = field(default_factory=DataConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    rebel_pregenerate: RebelPregenerateConfig = field(
        default_factory=RebelPregenerateConfig
    )

    def __post_init__(self):
        if self.wandb_tags is None:
            self.wandb_tags = ["kbest", "poker", "ppo"]
        # Derive action space size directly from the search schedule so search,
        # env, and model stay aligned on one global action id space.
        apply_action_schedule_to_config(self)

    @classmethod
    def from_dict_config(cls, dict_config: DictConfig) -> "Config":
        container = OmegaConf.to_container(dict_config, resolve=True)
        return cls.from_dict(container)

    @classmethod
    def from_dict(cls, container: dict[str, Any]) -> "Config":
        container.pop("defaults", None)
        train_container = container.get("train", {})
        train_container["stratify_streets"] = [
            StratifyConfig(**config)
            for config in train_container.get("stratify_streets", [])
        ]
        container["train"] = TrainingConfig(**train_container)
        model_container = container.get("model", {})
        if "board_interaction_dim" not in model_container:
            rank_interaction_dim = model_container.pop(
                "rank_board_interaction_dim", None
            )
            suit_interaction_dim = model_container.pop(
                "suit_board_interaction_dim", None
            )
            if rank_interaction_dim is not None or suit_interaction_dim is not None:
                rank_interaction_dim = rank_interaction_dim or 0
                suit_interaction_dim = suit_interaction_dim or 0
                if rank_interaction_dim != suit_interaction_dim:
                    raise ValueError(
                        "rank_board_interaction_dim and suit_board_interaction_dim "
                        "must match; use board_interaction_dim"
                    )
                model_container["board_interaction_dim"] = rank_interaction_dim
        else:
            model_container.pop("rank_board_interaction_dim", None)
            model_container.pop("suit_board_interaction_dim", None)
        compile_setting = model_container.get("compile", "default")
        if isinstance(compile_setting, bool):
            model_container["compile"] = "default" if compile_setting else "off"
        model_container.pop("policy_factor_scale", None)
        container["model"] = ModelConfig(**model_container)
        container["env"] = EnvConfig(**(container.get("env", {})))
        container["exploiter"] = ExploiterConfig(**(container.get("exploiter", {})))
        container["search"] = SearchConfig(**(container.get("search", {})))
        container["trueskill"] = TrueSkillConfig(**(container.get("trueskill", {})))
        data_container = container.get("data", {})
        pregenerated_container = data_container.get("pregenerated", {})
        pregenerated_container["datasets"] = [
            PregeneratedDatasetConfig(**dataset)
            for dataset in pregenerated_container.get("datasets", [])
        ]
        data_container["pregenerated"] = PregeneratedDataConfig(
            **pregenerated_container
        )
        container["data"] = DataConfig(**data_container)
        curriculum_container = container.get("curriculum", {})
        known_curriculum_keys = {"stages", "wandb_group", "promote_dir", "substeps"}
        substep_containers = dict(curriculum_container.get("substeps", {}))
        for key, value in list(curriculum_container.items()):
            if key not in known_curriculum_keys and isinstance(value, dict):
                substep_containers[key] = value
        substeps = {}
        for name, substep_container in substep_containers.items():
            substep_container = dict(substep_container)
            if "from" in substep_container:
                substep_container["from_net"] = substep_container.pop("from")
            substeps[name] = CurriculumSubstepConfig(**substep_container)
        curriculum_clean = {
            key: value
            for key, value in curriculum_container.items()
            if key in known_curriculum_keys
        }
        curriculum_clean["substeps"] = substeps
        container["curriculum"] = CurriculumConfig(**curriculum_clean)
        container["rebel_pregenerate"] = RebelPregenerateConfig(
            **container.get("rebel_pregenerate", {})
        )
        return cls(**container)


# Register with Hydra
cs = ConfigStore.instance()
cs.store(name="config", node=Config)
