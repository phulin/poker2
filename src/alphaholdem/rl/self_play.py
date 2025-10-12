from __future__ import annotations

import copy
import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from alphaholdem.core.structured_config import Config, KLType, LrSchedule, ValueHeadType
from alphaholdem.encoding.action_mapping import bin_to_action, get_legal_mask
from alphaholdem.env.hunl_env import HUNLEnv
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.cnn.cnn_embedding_data import CNNEmbeddingData
from alphaholdem.models.cnn.siamese_convnet import SiameseConvNetV1
from alphaholdem.models.cnn.state_encoder import CNNStateEncoder
from alphaholdem.models.policy import CategoricalPolicyV1
from alphaholdem.models.transformer.kv_cache_manager import SelfPlayKVCacheManager
from alphaholdem.models.transformer.poker_transformer import PokerTransformerV1
from alphaholdem.models.transformer.token_sequence_builder import TokenSequenceBuilder
from alphaholdem.rl.agent_snapshot import AgentSnapshot
from alphaholdem.rl.exponential_controller import ExponentialController
from alphaholdem.rl.dred_pool import DREDPool
from alphaholdem.rl.k_best_pool import KBestOpponentPool
from alphaholdem.rl.losses import KLPolicyPPOLoss
from alphaholdem.rl.opponent_pool import OpponentPool
from alphaholdem.rl.popart_normalizer import PopArtNormalizer
from alphaholdem.rl.replay import Trajectory, Transition
from alphaholdem.rl.vectorized_replay import BatchSample, VectorizedReplayBuffer
from alphaholdem.utils.ema import EMA
from alphaholdem.utils.model_context import model_eval
from alphaholdem.utils.model_utils import (
    get_batch_log_probs,
    get_logits_log_probs_values,
)
from alphaholdem.utils.profiling import profile


class SelfDummySnapshot:
    def __init__(self, trainer: SelfPlayTrainer):
        self.trainer = trainer
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.games_played = 0
        self.total_rewards = 0.0
        self.data = None

    @property
    def model(self):
        return self.trainer.model

    @property
    def model_dtype(self):
        return self.trainer.model.dtype

    @property
    def step(self):
        return self.trainer.step

    @property
    def elo(self):
        return self.trainer.opponent_pool.current_elo

    @elo.setter
    def elo(self, value):
        pass


class ModelHistory:
    def __init__(self):
        self._models = {}

    def add_model(self, age: int, model: torch.nn.Module) -> None:
        self._models[age] = copy.deepcopy(model).eval().requires_grad_(False)

    def get_model(self, age: int) -> torch.nn.Module:
        return self._models[age]

    def clear_before(self, threshold: int) -> None:
        for age in list(self._models.keys()):
            if age < threshold:
                del self._models[age]


class SelfPlayTrainer:
    # Typed instance attributes (PEP 526 annotations)
    cfg: Config
    device: torch.device
    batch_size: int
    episodes_per_step: int
    bet_bins: list[float]
    gamma: float
    gae_lambda: float
    epsilon: float
    replay_buffer_batches: int
    max_trajectory_length: int
    delta1: float
    value_coef: float
    entropy_coef: float
    grad_clip: float
    learning_rate: float
    learning_rate_final: float
    lr_schedule: LrSchedule
    entropy_coef_start: float
    entropy_coef_final: float
    entropy_decay_portion: float
    k_best_pool_size: int
    min_elo_diff: float
    min_step_diff: int
    k_factor: float
    use_mixed_precision: bool
    loss_scale: float
    use_kv_cache: bool
    use_tensor_env: bool
    num_envs: int
    offload_opponent_models: bool
    use_wandb: bool
    wandb_project: str
    wandb_name: str
    wandb_tags: Optional[list[str]]
    wandb_run_id: Optional[str]

    float_dtype: torch.dtype
    rng: torch.Generator
    is_transformer: bool

    tensor_env: HUNLTensorEnv
    env: Optional[HUNLEnv]
    model: torch.nn.Module
    policy: Any
    state_encoder: Union[TokenSequenceBuilder, CNNStateEncoder]
    replay_buffer: VectorizedReplayBuffer
    opponent_pool_type: str
    opponent_pool: OpponentPool

    kl_ema: EMA

    def __init__(self, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.device = device

        # Hydra config - extract parameters from nested structure
        self.batch_size = cfg.train.batch_size
        self.episodes_per_step = cfg.train.episodes_per_step
        self.gamma = cfg.train.gamma
        self.gae_lambda = cfg.train.gae_lambda
        self.epsilon = cfg.train.ppo_eps
        self.replay_buffer_batches = cfg.train.replay_buffer_batches
        self.max_trajectory_length = cfg.train.max_trajectory_length
        self.delta1 = cfg.train.ppo_delta1
        self.value_coef = cfg.train.value_coef
        self.entropy_coef = cfg.train.entropy_coef
        self.grad_clip = cfg.train.grad_clip
        self.quantile_huber_kappa = cfg.train.quantile_huber_kappa
        self.lr_schedule = cfg.train.lr_schedule
        # Separate learning rates for value head vs policy/trunk
        self.value_head_learning_rate = cfg.train.value_head_learning_rate
        self.value_head_learning_rate_final = cfg.train.value_head_learning_rate_final
        self.policy_trunk_learning_rate = cfg.train.policy_trunk_learning_rate
        self.policy_trunk_learning_rate_final = (
            cfg.train.policy_trunk_learning_rate_final
        )

        # Learning rate scaling controller
        self.lr_scaling_init_value = cfg.train.lr_scaling_init_value
        self.lr_scaling_min_value = cfg.train.lr_scaling_min_value
        self.lr_scaling_max_value = cfg.train.lr_scaling_max_value
        self.lr_scaling_increase_factor = cfg.train.lr_scaling_increase_factor
        self.lr_scaling_decrease_factor = cfg.train.lr_scaling_decrease_factor
        self.lr_scaling_upper_threshold = cfg.train.lr_scaling_upper_threshold
        self.lr_scaling_lower_threshold = cfg.train.lr_scaling_lower_threshold
        self.entropy_coef_start = cfg.train.entropy_coef
        self.entropy_coef_final = cfg.train.entropy_coef_final
        self.entropy_decay_portion = cfg.train.entropy_decay_portion
        self.k_best_pool_size = cfg.k_best_pool_size
        self.min_elo_diff = cfg.min_elo_diff
        self.min_step_diff = cfg.min_step_diff
        self.k_factor = cfg.k_factor
        self.use_mixed_precision = cfg.train.use_mixed_precision
        self.loss_scale = cfg.train.loss_scale
        self.use_kv_cache = cfg.train.use_kv_cache
        self.use_tensor_env = cfg.use_tensor_env
        self.num_envs = cfg.num_envs
        self.offload_opponent_models = cfg.offload_opponent_models
        self.use_wandb = cfg.use_wandb
        self.wandb_project = cfg.wandb_project
        self.wandb_name = cfg.wandb_name
        self.wandb_tags = cfg.wandb_tags
        self.wandb_run_id = cfg.wandb_run_id

        self.float_dtype = torch.float32

        # Initialize RNG (manually-seeded for reproducibility)
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(self.cfg.seed)

        # Determine model type
        self.is_transformer = cfg.model.name.startswith("poker_transformer")

        # Initialize components
        # Always create tensor_env for state encoder, even if we don't use it for training
        self.tensor_env = HUNLTensorEnv(
            num_envs=self.num_envs,
            starting_stack=self.cfg.env.stack,
            sb=self.cfg.env.sb,
            bb=self.cfg.env.bb,
            device=self.device,
            rng=self.rng,
            float_dtype=self.float_dtype,
            debug_step_table=self.cfg.env.debug_step_table,
            flop_showdown=getattr(self.cfg.env, "flop_showdown", False),
        )

        if not self.use_tensor_env:
            # Also create regular env for non-tensorized training
            self.env = HUNLEnv(
                starting_stack=self.cfg.env.stack,
                sb=self.cfg.env.sb,
                bb=self.cfg.env.bb,
            )

        self.value_head_type = self.cfg.model.value_head_type
        self.value_head_num_quantiles = self.cfg.model.value_head_num_quantiles
        self.use_quantile_value_head = self.value_head_type == ValueHeadType.quantile

        self.bet_bins = self.cfg.env.bet_bins
        self.num_bet_bins = len(self.bet_bins) + 3

        torch.zeros((1,), device=self.device)
        if self.is_transformer:
            self.model = PokerTransformerV1(
                max_sequence_length=self.cfg.model.max_sequence_length,
                d_model=self.cfg.model.d_model,
                n_layers=self.cfg.model.n_layers,
                n_heads=self.cfg.model.n_heads,
                num_bet_bins=self.num_bet_bins,
                dropout=self.cfg.model.dropout,
                use_gradient_checkpointing=self.cfg.model.use_gradient_checkpointing,
                value_head_type=self.value_head_type,
                value_head_num_quantiles=self.value_head_num_quantiles,
                rng=self.rng,
            )
        else:
            self.model = SiameseConvNetV1(
                cards_channels=self.cfg.model.cards_channels,
                actions_channels=self.cfg.model.actions_channels,
                cards_hidden=self.cfg.model.cards_hidden,
                actions_hidden=self.cfg.model.actions_hidden,
                fusion_hidden=self.cfg.model.fusion_hidden,
                num_actions=self.num_bet_bins,
                use_gradient_checkpointing=self.cfg.model.use_gradient_checkpointing,
                value_head_type=self.value_head_type,
                value_head_num_quantiles=self.value_head_num_quantiles,
                rng=self.rng,
            )

        # Ensure bins align with model output size to avoid mask/logit mismatch
        if hasattr(self.model, "policy_head") and hasattr(
            self.model.policy_head, "out_features"
        ):
            assert self.num_bet_bins == int(self.model.policy_head.out_features)

        self.policy = CategoricalPolicyV1()
        self.model_age = 1
        self.model_history = ModelHistory()
        self.last_action_mix = {
            "fold": 0,
            "check_call": 0,
            "bet": 0,
            "all_in": 0,
        }

        # Create state encoder based on model type (TSB becomes the encoder for transformer)
        if self.is_transformer:
            # Use TSB directly as the state encoder
            # Choose a safe sequence length (matches tests/typical config)
            self.state_encoder = TokenSequenceBuilder(
                tensor_env=self.tensor_env,
                sequence_length=self.cfg.train.max_sequence_length,
                num_bet_bins=self.num_bet_bins,
                device=self.device,
                float_dtype=self.float_dtype,
            )
        else:
            # Use CNN state encoder
            self.state_encoder = CNNStateEncoder(
                self.tensor_env, self.device, self.num_bet_bins
            )

        self.model.to(self.device)  # Move model to device
        # Replay buffer capacity in steps is batch_size * replay_buffer_batches
        # Add an extra batch so we can reserve space for the next batch.
        buffer_capacity = self.batch_size * max(1, 1 + self.replay_buffer_batches)

        # Use vectorized replay buffer for efficient tensor operations
        sequence_length = (
            self.state_encoder.sequence_length if self.is_transformer else -1
        )
        self.replay_buffer = VectorizedReplayBuffer(
            capacity=buffer_capacity,  # Number of trajectories
            max_trajectory_length=self.max_trajectory_length,  # Maximum steps per trajectory
            num_bet_bins=self.num_bet_bins,  # Number of bet bins for actions tensor
            device=self.device,
            float_dtype=self.float_dtype,  # Use appropriate dtype for mixed precision
            is_transformer=self.is_transformer,  # Whether this buffer is for transformer models
            max_sequence_length=sequence_length,  # Sequence length for transformer models
        )

        # Initialize opponent pool based on configuration
        self.opponent_pool_type = self.cfg.opponent_pool_type
        if self.opponent_pool_type == "k_best":
            self.opponent_pool = KBestOpponentPool(
                k=self.k_best_pool_size,
                min_elo_diff=self.min_elo_diff,
                min_step_diff=self.min_step_diff,
                k_factor=self.k_factor,
                use_mixed_precision=self.use_mixed_precision,
            )
        elif self.opponent_pool_type == "dred":
            self.opponent_pool = DREDPool(
                max_size=self.k_best_pool_size * 10,  # DRED can handle larger pools
                k_factor=self.k_factor,
                use_mixed_precision=self.use_mixed_precision,
            )
        else:
            raise ValueError(f"Unknown opponent pool type: {self.opponent_pool_type}")

        # Initialize KV cache manager for transformer models
        if self.is_transformer and self.use_kv_cache:
            self.kv_cache_manager = SelfPlayKVCacheManager(self.model, self.device)
        else:
            self.kv_cache_manager = None

        # Separate optimizers for different components
        value_head_params = []
        policy_trunk_params = []
        cards_params = []

        for name, param in self.model.named_parameters():
            if "value_head" in name:
                value_head_params.append(param)
            elif "cards_trunk" in name:
                cards_params.append(param)
            else:
                policy_trunk_params.append(param)

        # Create separate optimizers
        if cards_params:  # CNN model
            self.policy_trunk_optimizer = torch.optim.AdamW(
                [
                    {
                        "params": cards_params,
                        "lr": self.policy_trunk_learning_rate * 0.1,
                    },  # 10x lower for CNN
                    {
                        "params": policy_trunk_params,
                        "lr": self.policy_trunk_learning_rate,
                    },
                ]
            )
        else:  # Transformer model (no cards_trunk)
            self.policy_trunk_optimizer = torch.optim.AdamW(
                [
                    {
                        "params": policy_trunk_params,
                        "lr": self.policy_trunk_learning_rate,
                    },
                ]
            )

        self.value_head_optimizer = torch.optim.AdamW(
            [
                {"params": value_head_params, "lr": self.value_head_learning_rate},
            ]
        )

        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler(
            self.device.type,
            init_scale=self.loss_scale,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000,
            enabled=False,  # self.use_mixed_precision,
        )

        # KL divergence exponential moving average tracking
        self.kl_ema = EMA(decay=0.99, initial_value=self.cfg.train.target_kl)

        # Initialize PopArt normalizer (disabled when using quantile value head)
        self.popart_normalizer = (
            None if self.use_quantile_value_head else PopArtNormalizer()
        )

        self.beta_controller = ExponentialController(
            target_value=self.cfg.train.target_kl,
            init_value=1.0,
            min_value=float(self.cfg.train.beta_min),
            max_value=float(self.cfg.train.beta_max),
            increase_factor=float(self.cfg.train.beta_increase_factor),
            decrease_factor=float(self.cfg.train.beta_decrease_factor),
        )

        # Learning rate scaling controller
        self.lr_scaling_controller = ExponentialController(
            target_value=self.cfg.train.target_kl,
            init_value=self.lr_scaling_init_value,
            min_value=self.lr_scaling_min_value,
            max_value=self.lr_scaling_max_value,
            increase_factor=self.lr_scaling_increase_factor,
            decrease_factor=self.lr_scaling_decrease_factor,
            upper_threshold=self.lr_scaling_upper_threshold,
            lower_threshold=self.lr_scaling_lower_threshold,
            direction="reverse",
        )

        # Initialize loss calculator
        loss_value_type = (
            "quantile"
            if self.use_quantile_value_head
            else self.cfg.train.value_loss_type
        )
        self.loss_calculator = KLPolicyPPOLoss(
            popart_normalizer=self.popart_normalizer,
            beta_controller=self.beta_controller,
            value_coef=self.value_coef,
            entropy_coef=self.entropy_coef,
            value_loss_type=loss_value_type,
            clipping=self.cfg.train.ppo_clipping,
            epsilon=self.cfg.train.ppo_eps,
            dual_clip=self.cfg.train.ppo_dual_clip,
            huber_delta=self.cfg.train.huber_delta,
            return_clipping=self.cfg.train.return_clipping,
            kl_type=self.cfg.train.kl_type,
            quantile_kappa=self.quantile_huber_kappa,
            num_quantiles=(
                self.value_head_num_quantiles if self.use_quantile_value_head else None
            ),
        )

        # Training stats
        self.step_trajectories_collected = 0
        self.total_step_reward = 0.0
        self.total_trajectories_collected = (
            0  # Keep track of total episodes across all steps
        )
        self.total_episodes = 0
        self.total_transitions_trained = 0
        self.opponent_pool.current_elo = 1200.0  # Starting ELO rating

        # Weights & Biases setup (managed by CLI). Keep counters only.
        self.wandb_step = 0

    def _autocast(self):
        """
        Autocast context manager shortcut.

        Disabled if not using mixed precision.
        """
        return torch.amp.autocast(
            self.device.type,
            dtype=torch.bfloat16,
            enabled=self.use_mixed_precision,
        )

    @profile
    def collect_trajectory(
        self, opponent_snapshot: Optional[AgentSnapshot] = None
    ) -> Trajectory:
        """
        Collect a single trajectory from self-play or against an opponent.

        Args:
            opponent_snapshot: Optional opponent snapshot to play against.
                             If None, plays against self (self-play).
        """
        state = self.env.reset()
        transitions = []
        our_chips = self.env.sb if state.to_act == 0 else self.env.bb
        opp_chips = self.env.bb if state.to_act == 0 else self.env.sb
        max_steps = self.cfg.train.max_trajectory_length

        step_count = 0
        final_reward_from_our_perspective = 0.0

        while not state.terminal and step_count < max_steps:
            step_count += 1

            # Determine which player's turn it is
            current_player = state.to_act

            # Encode state for current player
            cards, actions_tensor = self.state_encoder.encode_single_state(
                state, seat=current_player
            )

            # Get model prediction
            with torch.no_grad():
                # Create CNNEmbeddingData
                embedding_data = CNNEmbeddingData(
                    cards=cards.unsqueeze(0), actions=actions_tensor.unsqueeze(0)
                )

                if opponent_snapshot is not None and current_player != 0:
                    outputs = opponent_snapshot.model(
                        embedding_data.to(opponent_snapshot.model_dtype)
                    )
                else:
                    outputs = self.model(embedding_data)
                logits = outputs.policy_logits.float()
                value = outputs.value.float()

                legal_mask = get_legal_mask(
                    state, self.num_bet_bins, dtype=self.float_dtype, device=self.device
                )

                # Sample action
                action_idx, log_prob = self.policy.action(logits.squeeze(0), legal_mask)

                # Convert to concrete action
                action = bin_to_action(action_idx, state, self.num_bet_bins)

            # Take step
            next_state, reward, done, _ = self.env.step(action)

            # Record transition (only for our player's actions)
            if current_player == 0:  # Our player
                # Scale factor for reward/targets: 100 big blinds
                scale = float(self.env.bb) * 100.0

                transition = Transition(
                    observation=torch.cat(
                        [cards.flatten(), actions_tensor.flatten()]
                    ),  # Simplified obs
                    action=action_idx,
                    log_prob=log_prob,
                    value=float(value.item()),
                    reward=reward,
                    done=done,
                    legal_mask=legal_mask,
                    chips_placed=action.amount,
                    # Per-sample clipping bounds: δ2 = -opp_cum, δ3 = our_cum
                    delta2=float(-opp_chips) / scale,
                    delta3=float(our_chips + action.amount) / scale,
                )
                transitions.append(transition)
                our_chips += action.amount
            else:
                opp_chips += action.amount

                # If the opponent's action ended the hand, record the final reward for our last transition
                if done and transitions:
                    # The reward is from our perspective, so it's -reward if opp ended the hand
                    transitions[-1].reward = -reward
                    transitions[-1].done = True

            # Track final reward from our perspective if episode ends
            if done:
                final_reward_from_our_perspective = (
                    float(reward) if current_player == 0 else float(-reward)
                )

            state = next_state

        if step_count >= max_steps:
            print(
                f"Warning: Trajectory reached max steps ({max_steps}), forcing termination"
            )
            # Force termination by setting final reward
            if transitions:
                transitions[-1].reward = 0.0
                transitions[-1].done = True
            final_reward_from_our_perspective = 0.0

        return (
            Trajectory(
                transitions=transitions,
            ),
            final_reward_from_our_perspective,
        )

    def _update_elo_from_rewards(
        self,
        per_env_rewards: torch.Tensor,
        mask: torch.Tensor,
        opp_env_groups: Optional[list[torch.Tensor]],
        all_opponent_snapshots: list[AgentSnapshot],
    ) -> None:
        # Update ELO ratings for done environments (vectorized by opponent)
        # Use premade chunking to efficiently group done environments by opponent
        if opp_env_groups is None:
            return

        # Vectorized ELO update per opponent using existing chunking
        for opponent_idx, opp_env_indices in enumerate(opp_env_groups):
            if opp_env_indices.numel() == 0:
                continue

            opponent = all_opponent_snapshots[opponent_idx]

            masked_opp_env_indices_mask = mask[opp_env_indices]
            masked_opp_env_indices = opp_env_indices[masked_opp_env_indices_mask]

            if masked_opp_env_indices.numel() > 0:
                opponent_rewards = per_env_rewards[masked_opp_env_indices]

                # Update ELO for this opponent with all their games
                self.opponent_pool.update_elo_batch_vectorized(
                    opponent,
                    opponent_rewards,
                )

    def _model_with_kv_cache(
        self,
        model,
        states,
        opponent_idx: int = -1,  # -1 for self
    ):
        """
        Run model forward pass with KV cache handling.

        Args:
            model: The model to run forward pass on
            states: Input states for the model
            cache_type: Type of cache ("self" or "opponent")
            opponent_idx: Index of opponent (only used for opponent cache)
            move_to_device: Whether to move the model to the correct device first

        Returns:
            Model outputs
        """
        model.to(device=self.device)

        with self._autocast():
            if self.is_transformer and self.kv_cache_manager is not None:
                if opponent_idx == -1:
                    # Get our cache
                    cache = self.kv_cache_manager.get_self_cache()
                    outputs = model(states, kv_cache=cache)
                    # Update our cache
                    if outputs.kv_cache is not None:
                        self.kv_cache_manager.update_self_cache(outputs.kv_cache)
                else:  # opponent
                    # Get opponent cache
                    cache = self.kv_cache_manager.get_opponent_cache(opponent_idx)
                    outputs = model(states, kv_cache=cache)
                    # Update opponent cache
                    if outputs.kv_cache is not None:
                        self.kv_cache_manager.update_opponent_cache(
                            opponent_idx, outputs.kv_cache
                        )
            else:
                outputs = model(states)

        return outputs

    def _reset_tensor_env_and_encoder_kv(
        self, opponent_snapshots: Optional[list[AgentSnapshot]]
    ) -> None:
        """Reset the tensorized environment and token sequence builder."""
        self.tensor_env.reset()

        if self.kv_cache_manager is not None:
            self.kv_cache_manager.reset_for_new_game()
            # Initialize caches for self and opponent
            self.kv_cache_manager.initialize_self_cache(self.num_envs)
            if opponent_snapshots is not None:
                for i in range(len(opponent_snapshots)):
                    self.kv_cache_manager.initialize_opponent_cache(i, self.num_envs)
            else:
                self.kv_cache_manager.initialize_opponent_cache(0, self.num_envs)

        all_env_indices = torch.arange(self.num_envs, device=self.device)
        if self.is_transformer and isinstance(self.state_encoder, TokenSequenceBuilder):
            self.state_encoder.reset()

            # Game-invariant state
            self.state_encoder.add_game(all_env_indices)
            # Hole cards for P0
            self.state_encoder.add_card(
                all_env_indices,
                self.tensor_env.hole_indices[all_env_indices, 0, 0],
            )
            self.state_encoder.add_card(
                all_env_indices,
                self.tensor_env.hole_indices[all_env_indices, 0, 1],
            )
            # Preflop street marker
            self.state_encoder.add_street(
                all_env_indices, torch.zeros_like(all_env_indices)
            )

    def _compute_reference_distributions(
        self, batch: BatchSample, step_start_age: int
    ) -> torch.Tensor:
        """Compute the reference action distributions for a batch of transitions."""
        step_start_model = self.model_history.get_model(step_start_age)
        with torch.no_grad(), model_eval(step_start_model), self._autocast():
            # Get indices for non-current model (current model gets done below)
            other_model_indices = torch.where(batch.model_ages != step_start_age)[0]
            if len(other_model_indices) > 0:
                log_probs = get_batch_log_probs(
                    step_start_model, batch[other_model_indices]
                )
                batch.update_step_log_probs(other_model_indices, log_probs)

            for age, batch_indices in batch.group_by_model_age():
                frozen_model = self.model_history.get_model(age)
                with model_eval(frozen_model):
                    log_probs = get_batch_log_probs(frozen_model, batch[batch_indices])
                    batch.update_frozen_log_probs(batch_indices, log_probs)
                    if age == step_start_age:
                        # this is self.model.
                        batch.update_step_log_probs(batch_indices, log_probs)

    def _sample_legal_actions(
        self,
        active_mask: torch.Tensor,
        opp_env_groups: list[torch.Tensor],
        all_opponent_snapshots: list[AgentSnapshot],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample legal actions for active environments.

        Args:
            active_mask: Mask of active environments
            opp_env_groups: Groups of opponent environments
            all_opponent_snapshots: All opponent snapshots

        Returns:
            Tuple of (sampled action bins, raw logits, values, our states)
        """
        # Initialize logits and values tensors for active environments only
        env_logits = torch.zeros(
            self.num_envs,
            self.num_bet_bins,
            dtype=self.float_dtype,
            device=self.device,
        )
        env_values = torch.zeros(
            self.num_envs, dtype=self.float_dtype, device=self.device
        )

        legal_bins_mask = self.tensor_env.legal_bins_mask(self.bet_bins)
        active_indices = torch.where(active_mask)[0]
        we_act_mask = self.tensor_env.to_act == 0
        opp_acts_mask = self.tensor_env.to_act == 1
        env_active_we_act_mask = active_mask & we_act_mask
        env_active_we_act = torch.where(env_active_we_act_mask)[0]
        env_active_opp_acts_mask = active_mask & opp_acts_mask
        env_active_opp_acts = torch.where(env_active_opp_acts_mask)[0]

        # Encode states from our perspective (player=0)
        our_states = self.state_encoder.encode_tensor_states(
            player=0, idxs=env_active_we_act
        )

        # Get predictions from our model for our turns
        if env_active_we_act.numel() > 0:
            outputs = self._model_with_kv_cache(self.model, our_states, opponent_idx=-1)

            if torch.isnan(outputs.policy_logits).any():
                nan_envs = torch.where(torch.isnan(outputs.policy_logits).any(dim=-1))[
                    0
                ]
                print(
                    "[SelfPlayTrainer] NaN policy logits detected",
                    {
                        "env_indices": env_active_we_act[nan_envs].tolist(),
                        "logits": outputs.policy_logits[nan_envs]
                        .detach()
                        .cpu()
                        .tolist(),
                    },
                )
            if torch.isnan(outputs.value).any():
                nan_envs = torch.where(torch.isnan(outputs.value))[0]
                print(
                    "[SelfPlayTrainer] NaN values detected",
                    {
                        "env_indices": env_active_we_act[nan_envs].tolist(),
                        "values": outputs.value[nan_envs].detach().cpu().tolist(),
                    },
                )

            env_logits[env_active_we_act] = outputs.policy_logits.float()
            value_tensor = outputs.value.float()
            if self.popart_normalizer is not None:
                value_tensor = self.popart_normalizer.denormalize_value(value_tensor)
            env_values[env_active_we_act] = value_tensor

        # Get predictions from opponent models for opponent turns
        if opp_env_groups is not None and env_active_opp_acts.numel() > 0:
            for opponent_idx, opp_env_indices in enumerate(opp_env_groups):
                opponent = all_opponent_snapshots[opponent_idx]

                opp_working_env_indices = opp_env_indices[
                    env_active_opp_acts_mask[opp_env_indices]
                ]

                if opp_working_env_indices.numel() == 0:
                    continue

                opp_states = self.state_encoder.encode_tensor_states(
                    player=1, idxs=opp_working_env_indices
                )

                outputs = self._model_with_kv_cache(
                    opponent.model,
                    opp_states,
                    opponent_idx=opponent_idx,
                )
                env_logits[opp_working_env_indices] = outputs.policy_logits.float()
                value_tensor = outputs.value.float()
                if self.popart_normalizer is not None:
                    value_tensor = self.popart_normalizer.denormalize_value(
                        value_tensor
                    )
                env_values[opp_working_env_indices] = value_tensor

        active_env_logits = env_logits[active_indices]
        active_env_legal_mask = legal_bins_mask[active_indices]
        action_bins_active, log_probs_active_full = self.policy.action_batch(
            active_env_logits, active_env_legal_mask
        )
        if not active_env_legal_mask.gather(1, action_bins_active.unsqueeze(1)).all():
            raise ValueError(
                f"Action bins {action_bins_active} are not legal in active environments {active_indices}"
            )

        return action_bins_active, env_logits, env_values, our_states

    @profile
    @torch.no_grad()
    def collect_tensor_trajectories(
        self,
        min_steps: int = 0,
        min_trajectories: int = 0,
        all_opponent_snapshots: Optional[list[AgentSnapshot]] = None,
        add_to_replay_buffer: bool = True,
    ) -> torch.Tensor:
        """
        Collect trajectories from tensorized environments until we have at least min_steps.
        Only processes non-done environments in each iteration to avoid bias towards short episodes.
        Resets all environments only when every environment is done.
        Collects complete trajectories before adding them to the replay buffer.

        Args:
            min_steps: Minimum number of steps to collect across all trajectories
            opponent_snapshots: Optional list of opponent snapshots to play against.
                              If None, plays against self (self-play).
                              If provided, should have length <= num_envs.
            add_to_replay_buffer: If True, add collected trajectories to replay buffer.
                                If False, only collect and return statistics (for evaluation).

        Returns:
            Tuple of (total reward, episode count, tensor of individual episode rewards)
        """
        if not self.use_tensor_env:
            raise ValueError("collect_tensor_trajectories requires use_tensor_env=True")

        if min_steps == 0 and min_trajectories == 0:
            raise ValueError("Either min_steps or min_trajectories must be provided")

        if min_steps > 0 and min_trajectories > 0:
            raise ValueError(
                "Only one of min_steps or min_trajectories can be provided"
            )

        trajectory_count = 0
        steps_collected = 0
        batch_steps_collected = 0
        # Track individual episode rewards as list of tensors
        collected_trajectory_rewards = []

        # About to collect trajectories, so add current model to history
        self.model_history.add_model(self.model_age, self.model)

        # Initialize all environments
        self._reset_tensor_env_and_encoder_kv(all_opponent_snapshots)

        # Per-environment reward tracking and step counts
        per_env_rewards = torch.zeros(self.num_envs, device=self.device)
        max_steps = self.cfg.train.max_trajectory_length
        steps_since_reset = 0
        # Use the trainer's state_encoder as the TSB for transformer path

        # Chunk opponents by tensor env index
        env_indices = torch.arange(self.num_envs, device=self.device)
        if all_opponent_snapshots is not None and len(all_opponent_snapshots) > 0:
            # Use opponent models - assign opponents to environments
            num_opps = len(all_opponent_snapshots)
            # Split active_opp_acts into num_opps approximately equal groups
            opp_env_groups = torch.chunk(env_indices, num_opps)
        else:
            all_opponent_snapshots = [SelfDummySnapshot(self)]
            opp_env_groups = [env_indices]

        # Outer loop: collect complete trajectories until we have enough steps
        loop_count = 0
        max_loops = 1000  # Safety limit to prevent infinite loops

        adding_trajectories = False

        while (
            steps_collected < min_steps or trajectory_count < min_trajectories
        ) and loop_count < max_loops:
            if not adding_trajectories and add_to_replay_buffer:
                # Initialize trajectory collection; reserve space in buffer.
                self.replay_buffer.start_adding_trajectory_batches(
                    self.num_envs, self.model_age
                )
                adding_trajectories = True

            loop_count += 1

            # Only process non-done environments to avoid bias towards short episodes
            active_mask = ~self.tensor_env.done
            active_indices = torch.where(active_mask)[0]
            we_act_mask = self.tensor_env.to_act == 0
            opp_acts_mask = self.tensor_env.to_act == 1
            env_active_we_act = torch.where(active_mask & we_act_mask)[0]
            env_active_opp_acts = torch.where(active_mask & opp_acts_mask)[0]

            # Get legal action masks for active environments only
            legal_bins_amounts, legal_bins_mask = (
                self.tensor_env.legal_bins_amounts_and_mask(self.bet_bins)
            )

            # Get model predictions for active environments only
            # Add pre-action context token
            if self.is_transformer and isinstance(
                self.state_encoder, TokenSequenceBuilder
            ):
                self.state_encoder.add_context(active_indices)

            action_bins_active, env_logits, env_values, our_states = (
                self._sample_legal_actions(
                    active_mask, opp_env_groups, all_opponent_snapshots
                )
            )

            # Create full-size tensors for stepping (needed by tensor_env.step_bins)
            action_bins = torch.full(
                (self.num_envs,), -1, dtype=torch.long, device=self.device
            )
            action_bins[active_indices] = action_bins_active

            # Add street before step, as stepping changes values
            if self.is_transformer and isinstance(
                self.state_encoder, TokenSequenceBuilder
            ):
                self.state_encoder.add_action(
                    active_indices,
                    self.tensor_env.to_act[active_indices],
                    action_bins_active,
                    legal_bins_mask[active_indices],
                    legal_bins_amounts[active_indices, action_bins_active],
                    self.tensor_env.street[active_indices],
                )

            # Take steps in all environments (tensor_env expects full-size tensors)
            # NOTE: THIS CHANGES self.tensor_env.to_act!!
            rewards, dones, _, new_streets, dealt_cards = self.tensor_env.step_bins(
                action_bins, legal_bins_amounts, legal_bins_mask, self.bet_bins
            )
            newly_done_mask = dones & active_mask

            # After environment step, append any newly revealed non-action tokens
            if self.is_transformer and isinstance(
                self.state_encoder, TokenSequenceBuilder
            ):
                # Use environment return to detect street progress and dealt cards
                advanced_envs = torch.where(new_streets >= 0)[0]
                if advanced_envs.numel() > 0:
                    # Add street markers for environments that advanced
                    self.state_encoder.add_street(
                        advanced_envs, new_streets[advanced_envs]
                    )
                    for i in range(3):
                        valid_cards = torch.where(dealt_cards[:, i] >= 0)[0]
                        self.state_encoder.add_card(
                            valid_cards,
                            dealt_cards[valid_cards, i],
                        )

            # Update per-environment reward tracking
            per_env_rewards += rewards

            # Store transitions for our actions
            if env_active_we_act.numel() > 0:
                # Scale factor for reward/targets: 100 big blinds
                scale = float(self.tensor_env.bb) * 100.0

                # Compute delta bounds from our (p0/actor) perspective AFTER step
                # For our acting envs, actor is player 0
                chips = self.tensor_env.chips_placed.to(self.float_dtype)
                # Convert active_we_act to full environment indices
                our_delta2_tensor = (
                    -chips[env_active_we_act, 1] / scale
                )  # -opponent chips
                our_delta3_tensor = chips[env_active_we_act, 0] / scale  # our chips

                # Extract tensor values efficiently (no .tolist() calls)
                active_we_act = torch.where(we_act_mask[active_indices])[0]
                our_action_indices = action_bins_active[active_we_act]
                our_logits_tensor = env_logits[env_active_we_act]
                our_values_tensor = env_values[env_active_we_act]
                our_rewards_tensor = rewards[env_active_we_act]
                our_dones_tensor = dones[env_active_we_act]
                our_legal_masks_tensor = legal_bins_mask[env_active_we_act]

                # Track action type counts for metrics
                self.last_action_mix["fold"] += (our_action_indices == 0).sum().item()
                self.last_action_mix["check_call"] += (
                    (our_action_indices == 1).sum().item()
                )
                self.last_action_mix["bet"] += (
                    (
                        (our_action_indices >= 2)
                        & (our_action_indices < (self.num_bet_bins - 1))
                    )
                    .sum()
                    .item()
                )
                self.last_action_mix["all_in"] += (
                    ((our_action_indices == (self.num_bet_bins - 1))).sum().item()
                )

                # Add transitions immediately using vectorized operations
                if add_to_replay_buffer:
                    self.replay_buffer.add_transitions(
                        embedding_data=our_states,
                        action_indices=our_action_indices,
                        logits=our_logits_tensor,
                        rewards=our_rewards_tensor,
                        dones=our_dones_tensor,
                        legal_masks=our_legal_masks_tensor.bool(),
                        delta2=our_delta2_tensor,
                        delta3=our_delta3_tensor,
                        values=our_values_tensor,
                        trajectory_indices=env_active_we_act,
                    )

                batch_steps_collected += env_active_we_act.numel()

            if env_active_opp_acts.numel() > 0:
                # Check if any opponent actions ended the hand
                opp_ended_hands_mask = newly_done_mask & opp_acts_mask
                opp_ended_hands = torch.where(opp_ended_hands_mask)[0]
                # rewards are from p0's perspective
                opp_ended_hands_rewards = rewards[opp_ended_hands]

                # For environments where opponent's action ended the hand,
                # update p0's reward for the last transition
                if opp_ended_hands.numel() > 0 and add_to_replay_buffer:
                    self.replay_buffer.update_last_transition_rewards(
                        opp_ended_hands, opp_ended_hands_rewards
                    )

            # Handle environments that exceed max steps
            steps_since_reset += 1
            if steps_since_reset >= max_steps:
                bad_mask = torch.zeros(
                    active_indices.shape[0], dtype=torch.bool, device=self.device
                )
                if add_to_replay_buffer:
                    bad_mask |= (
                        self.replay_buffer.get_current_transition_counts(active_indices)
                        >= max_steps
                    )
                    if bad_mask.any():
                        print(
                            f"Warning: Environments {torch.where(bad_mask)[0].tolist()} reached max steps ({max_steps}), forcing termination"
                        )
                if self.is_transformer and isinstance(
                    self.state_encoder, TokenSequenceBuilder
                ):
                    sequence_length_mask = (
                        self.state_encoder.lengths[active_indices]
                        >= self.cfg.train.max_sequence_length
                    )
                    if sequence_length_mask.any():
                        print(
                            f"Warning: Environments {torch.where(sequence_length_mask)[0].tolist()} reached max sequence length ({self.cfg.train.max_sequence_length}), forcing termination"
                        )
                    bad_mask |= sequence_length_mask
                bad_indices = active_indices[bad_mask]
                # Mark them as done
                self.tensor_env.done[bad_indices] = True

            # If all environments are done, reset all of them and take credit for steps and trajectories collected
            if dones.all():
                if add_to_replay_buffer:
                    num_trajectories, _ = (
                        self.replay_buffer.finish_adding_trajectory_batches()
                    )
                    adding_trajectories = False
                    trajectory_count += num_trajectories
                else:
                    trajectory_count += self.tensor_env.N

                if self.cfg.env.debug_step_table:
                    print(f"=> Batch steps collected: {batch_steps_collected}")
                    print(f"=> Updating ELO from rewards", per_env_rewards)

                self._update_elo_from_rewards(
                    per_env_rewards, dones, opp_env_groups, all_opponent_snapshots
                )

                steps_collected += batch_steps_collected
                batch_steps_collected = 0

                # Collect individual episode rewards for accurate win counting
                collected_trajectory_rewards.append(per_env_rewards.clone())
                per_env_rewards[:] = 0.0

                steps_since_reset = 0

                # Reset environments if we're going through the loop again
                if steps_collected < min_steps or trajectory_count < min_trajectories:
                    self._reset_tensor_env_and_encoder_kv(all_opponent_snapshots)

        if loop_count >= max_loops:
            print(
                f"Warning: Reached maximum loop count ({max_loops}), stopping collection early"
            )
            print(f"Collected {steps_collected} steps out of {min_steps} requested")

        # Concatenate all trajectory rewards into a single tensor
        if collected_trajectory_rewards:
            collected_trajectory_rewards_tensor = torch.cat(
                collected_trajectory_rewards, dim=0
            )
        else:
            collected_trajectory_rewards_tensor = torch.tensor([], device=self.device)

        # Now that we've collected trajectories, and overwritten older ones, remove old models from history
        model_ages_nonzero = self.replay_buffer.model_ages.nonzero()
        self.model_history.clear_before(
            self.replay_buffer.model_ages[model_ages_nonzero].min().item()
        )

        return collected_trajectory_rewards_tensor

    def _compute_pool_kl_divergence(
        self, batch: BatchSample, log_probs: torch.Tensor
    ) -> float:
        """
        Compute KL divergence between current model and last admitted opponent.

        Args:
            batch: Current batch sample
            log_probs: Log probs from current model forward pass

        Returns:
            KL divergence value (0.0 if no last admitted opponent)
        """
        last_admitted_opponent = self.opponent_pool.get_last_admitted_snapshot()
        if last_admitted_opponent is None:
            return 0.0
        else:
            with (
                torch.no_grad(),
                model_eval(last_admitted_opponent.model),
                model_eval(self.model),
                self._autocast(),
            ):
                # Get last admitted opponent model logits
                last_admitted_opponent.model.to(self.device)
                opponent_log_probs = get_batch_log_probs(
                    last_admitted_opponent.model, batch
                )

                # KL(opponent || current model)
                return F.kl_div(
                    log_probs,
                    opponent_log_probs,
                    log_target=True,
                    reduction="batchmean",
                )

    def _compute_diagnostic_kl_divergence(
        self, step_start_age: int, batch: BatchSample, final_log_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute diagnostic KL divergence between current model and old model.

        Args:
            step_start_age: Age of the model at the start of the step

        Returns:
            KL divergence value computed from current and old model
        """
        # Compute current log probs on sample for diagnostic KL
        old_model = self.model_history.get_model(step_start_age)
        with (
            torch.no_grad(),
            model_eval(old_model),
            self._autocast(),
        ):
            kl_old_log_probs = get_batch_log_probs(old_model, batch)
            # KL(old || new)
            return F.kl_div(
                final_log_probs,
                kl_old_log_probs,
                log_target=True,
                reduction="batchmean",
            )

    @profile
    def update_model(self, step: int) -> dict:
        """Perform PPO update on collected trajectories."""
        # Require enough samples (transitions) in buffer
        if self.replay_buffer.num_steps() < self.batch_size:
            raise ValueError(
                f"Not enough samples in replay buffer: {self.replay_buffer.num_steps()} < {self.batch_size}"
            )

        # Compute GAE for all stored trajectories
        self.replay_buffer.compute_gae_returns(
            gamma=self.gamma, lambda_=self.gae_lambda
        )

        # Initialize tracking variables once before the epoch loop
        total_loss, total_policy_loss, total_value_loss = 0.0, 0.0, 0.0
        total_kl_loss = 0.0
        total_entropy, total_clipfrac = 0.0, 0.0
        total_ppo_clipfrac, total_return_clipfrac = 0.0, 0.0
        total_explained_var, total_pearson_r, total_epsilon = 0.0, 0.0, 0.0
        total_advantage_mean_raw, total_advantage_std_raw = 0.0, 0.0
        total_return_abs_mean, total_return_std = 0.0, 0.0
        total_small_adv_rate = 0.0
        total_grad_norm_unclipped, total_grad_norm_clipped = 0.0, 0.0
        minibatch_count = 0

        # Freeze value normalizer (PopArt) at the beginning of each epoch cycle
        if self.popart_normalizer is not None:
            self.popart_normalizer.freeze_stats()

        # Store the age at the beginning of the step for reference distributions
        step_start_age = self.model_age
        policy_episodes = 0
        value_only = False

        for episode in range(self.episodes_per_step):
            # Sample batch from vectorized buffer
            batch = self.replay_buffer.sample_batch(self.rng, self.batch_size)
            batch = batch.to(torch.float32)

            # Permute suits to augment data.
            batch.embedding_data.permute_suits(self.rng)

            # Update log-probs for the suit-permuted batch.
            self._compute_reference_distributions(batch, step_start_age)

            # Normalize advantages across the batch for stability (mean=0, std=1)
            adv = batch.advantages
            adv_mean_raw = adv.mean()
            adv_std_raw = adv.std().clamp_min(1e-8)
            batch.advantages = (adv - adv_mean_raw) / adv_std_raw
            total_advantage_mean_raw += adv_mean_raw.item()
            total_advantage_std_raw += adv_std_raw.item()

            # Calculate rate of small advantages (|A_raw| < 1e-3)
            small_adv_mask = adv.abs() < 1e-3
            small_adv_rate = small_adv_mask.float().mean().item()
            total_small_adv_rate += small_adv_rate

            total_return_abs_mean += batch.returns.abs().mean().item()
            total_return_std += batch.returns.std().item()

            with self._autocast():
                embedding_data = batch.embedding_data
                logits, log_probs, values, value_quantiles = (
                    get_logits_log_probs_values(
                        self.model, embedding_data, batch.legal_masks
                    )
                )

                loss_result = self.loss_calculator.compute_loss(
                    logits=logits,
                    values=values,
                    batch=batch,
                    value_quantiles=value_quantiles,
                )

            # Debugging metrics: explained variance
            with torch.no_grad():
                y = batch.returns
                if self.popart_normalizer is not None:
                    yhat = self.popart_normalizer.denormalize_value(values.detach())
                else:
                    yhat = values.detach()
                # Explained variance of value predictions against targets
                var_y = y.var(correction=0)
                var_err = (y - yhat).var(correction=0)
                explained_var = 1.0 - (var_err / (var_y + 1e-8))

                # 3) Also log correlation (often more stable/interpretible)
                # Use returns (y) and predicted values (yhat)
                pearson_r = 0.0
                if y.numel() >= 2:
                    pearson_r = torch.corrcoef(torch.stack([y, yhat]))[0, 1].item()

            total_loss += loss_result.total_loss.item()
            total_policy_loss += loss_result.policy_loss
            total_value_loss += loss_result.value_loss
            total_kl_loss += loss_result.penalty_kl * self.beta_controller.current_value
            total_entropy += loss_result.entropy
            total_clipfrac += loss_result.clipfrac
            total_ppo_clipfrac += loss_result.ppo_clipfrac
            total_return_clipfrac += loss_result.return_clipfrac
            total_explained_var += explained_var
            total_pearson_r += pearson_r
            total_epsilon += loss_result.epsilon
            minibatch_count += 1

            self.policy_trunk_optimizer.zero_grad()
            self.value_head_optimizer.zero_grad()

            # If we've gone too far from the starting policy, stop (skip this update).
            if (
                loss_result.forward_kl is not None
                and loss_result.forward_kl > 2 * self.cfg.train.target_kl
            ):
                value_only = True
                loss = loss_result.value_loss_tensor * self.value_coef
                # Don't break; instead just train value.
            else:
                policy_episodes += 1
                loss = loss_result.total_loss

            # Use scaler for mixed precision backward pass. Will have enabled=False if not using mixed precision.
            self.scaler.scale(loss).backward()

            # Always unscale both optimizers for gradient norm computation
            self.scaler.unscale_(self.value_head_optimizer)
            self.scaler.unscale_(self.policy_trunk_optimizer)

            grad_has_nan = False
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    grad_has_nan = True
                    print(
                        "[SelfPlayTrainer] NaN/Inf gradient detected",
                        {"param": name},
                    )
            # Apply stricter gradient clipping to CNN layers to prevent explosion
            for name, param in self.model.named_parameters():
                if param.grad is not None and "cards_trunk" in name:
                    torch.nn.utils.clip_grad_norm_(
                        [param], 0.5
                    )  # Stricter clipping for CNN
            grad_norm_unclipped = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            ).item()

            if grad_has_nan or not math.isfinite(grad_norm_unclipped):
                print(
                    "[SelfPlayTrainer] Gradient norm issue",
                    {"grad_norm": grad_norm_unclipped},
                )
            elif grad_norm_unclipped > 1e3:
                print(
                    "[SelfPlayTrainer] Large gradient norm",
                    {"grad_norm": grad_norm_unclipped},
                )

            # Accumulate gradient norm for averaging (always compute, but only step conditionally)
            if not value_only:
                total_grad_norm_unclipped += grad_norm_unclipped
                total_grad_norm_clipped += torch.nn.utils.get_total_norm(
                    p.grad for p in self.model.parameters()
                ).item()

            # Only step optimizers conditionally based on whether policy was trained
            self.scaler.step(self.value_head_optimizer)
            if not value_only:
                self.scaler.step(self.policy_trunk_optimizer)

            self.scaler.update()
            with torch.no_grad():
                param_nan = False
                for name, param in self.model.named_parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        param_nan = True
                        print(
                            "[SelfPlayTrainer] NaN/Inf parameter detected",
                            {"param": name},
                        )
                        break
                if param_nan:
                    print("[SelfPlayTrainer] Stopping due to invalid parameters")
                    raise RuntimeError("Detected NaN/Inf in model parameters")
            self.model_age += 1

            # Update value normalizer stats (if enabled)
            if self.popart_normalizer is not None:
                self.popart_normalizer.update_stats(batch.returns)

        # Get log probs for final diagnostic statistics.
        diagnostic_batch_size = min(self.batch_size, max(8, self.batch_size // 4))
        diagnostic_batch = batch[:diagnostic_batch_size]
        if value_only:
            # If we stopped early, skipped last update => log probs up to date.
            final_log_probs = log_probs[:diagnostic_batch_size].detach()
        else:
            # Else our log probs are one update behind, so we have to recompute.
            with torch.no_grad(), model_eval(self.model), self._autocast():
                final_log_probs = get_batch_log_probs(self.model, diagnostic_batch)

        # Check if we should add current model to opponent pool
        pool_kl_divergence = self._compute_pool_kl_divergence(
            diagnostic_batch, final_log_probs
        )
        if self.opponent_pool.should_add_snapshot(step, pool_kl_divergence):
            if isinstance(self.opponent_pool, DREDPool):
                self.opponent_pool.set_last_batch_data(batch.embedding_data)

            self.opponent_pool.add_snapshot(self.model, step)

        # Estimate diagnostic KL: divergence from last model batch.
        current_kl = self._compute_diagnostic_kl_divergence(
            step_start_age, diagnostic_batch, final_log_probs
        )

        # Update Beta Controller each epoch based on penalty KL
        if self.cfg.train.kl_type != KLType.none:
            self.beta_controller.update(current_kl)

        # Update LR scaling controller based on per-epoch diagnostic KL
        self.lr_scaling_controller.update(current_kl)

        # Update KL divergence exponential moving average
        self.kl_ema.update(current_kl)

        # Apply final PopArt rescaling after last epoch
        if (
            self.popart_normalizer is not None
            and self.popart_normalizer.mean_ema.initialized
        ):
            weight_scale, bias_adjustment = (
                self.popart_normalizer.compute_rescaling_adjustments()
            )
            if weight_scale is not None and bias_adjustment is not None:
                # Apply final rescaling to model (pass floats directly)
                self.model.adjust_scale(weight_scale, bias_adjustment)

        # Calculate average reward for this update step using captured values
        # (These values all come from trajectory collection, compute it here to include
        # in training stats)
        avg_reward = (
            self.total_step_reward / max(1, self.step_trajectories_collected)
            if self.step_trajectories_collected > 0
            else 0.0
        )
        if abs(avg_reward) > 1:
            print(f"Warning: Avg reward is {avg_reward}, which is outside [-1, 1].")
            print(f"Total step reward: {self.total_step_reward}")
            print(f"Step trajectories collected: {self.step_trajectories_collected}")

        avg_trajectory_length = self.replay_buffer.num_steps() / self.replay_buffer.size
        denom = max(1, minibatch_count)

        # Get all valid returns and advantages from replay buffer for histograms
        valid_returns = self.replay_buffer.get_valid_returns()
        valid_advantages = self.replay_buffer.get_valid_advantages()

        result = {
            "episodes": episode + 1,
            "policy_episodes": policy_episodes,
            "avg_reward": avg_reward,
            "num_samples": self.batch_size * self.episodes_per_step,
            "delta2_mean": batch.delta2.mean().item(),
            "delta3_mean": batch.delta3.mean().item(),
            "avg_trajectory_length": avg_trajectory_length,
            "avg_loss": total_loss / denom,
            "policy_loss": total_policy_loss / denom,
            "value_loss": total_value_loss / denom,
            "kl_loss": total_kl_loss / denom,
            "entropy": total_entropy / denom,
            "approx_kl": current_kl,
            "kl_ema": self.kl_ema.value,
            "clipfrac": total_clipfrac / denom,
            "ppo_clipfrac": total_ppo_clipfrac / denom,
            "return_clipfrac": total_return_clipfrac / denom,
            "explained_var": total_explained_var / denom,
            "pearson_r": total_pearson_r / denom,
            "epsilon": total_epsilon / denom,
            "advantage_mean_raw": total_advantage_mean_raw / denom,
            "advantage_std_raw": total_advantage_std_raw / denom,
            "advantage_histogram": wandb.Histogram(valid_advantages.cpu()),
            "small_adv": total_small_adv_rate / denom,
            "return_abs_mean": total_return_abs_mean / denom,
            "return_std": total_return_std / denom,
            "return_histogram": wandb.Histogram(valid_returns.cpu()),
            "grad_norm_unclipped": total_grad_norm_unclipped / max(1, policy_episodes),
            "grad_norm_clipped": total_grad_norm_clipped / max(1, policy_episodes),
            "beta": self.beta_controller.current_value,
        }

        # Get PopArt statistics
        if self.popart_normalizer is not None:
            popart_mu, popart_sigma = self.popart_normalizer.get_current_stats()
            result["popart_mu"] = popart_mu
            result["popart_sigma"] = popart_sigma

        return result

    @profile
    def train_step(self, step: int) -> dict:
        """
        Single training step: collect trajectories against K-Best opponents and update model.

        Args:
            step: Training step number for logging and opponent pool management.

        Returns:
            Dictionary with training statistics
        """

        # Apply LR/entropy schedules for this step
        self._apply_schedules(step)

        # This is updated while filling the replay buffer
        self.step_trajectories_collected = 0
        self.total_step_reward = 0.0
        self.last_action_mix = {
            "fold": 0,
            "check_call": 0,
            "bet": 0,
            "all_in": 0,
        }

        self.model.eval()
        target_steps = self.batch_size * max(self.cfg.train.replay_buffer_batches, 1)
        if self.replay_buffer.num_steps() == 0:
            # Warmup: fill replay buffer with minimum required samples
            print(f"Warmup: filling replay buffer to {target_steps} steps...")
            self._fill_replay_buffer(target_steps)
        else:
            # Before update, add one batch worth of fresh steps
            self._fill_replay_buffer(self.batch_size)

        # Update model
        self.model.train()
        update_stats = self.update_model(step)
        self.model.eval()

        # Prepare training stats for return and logging
        policy_learning_rate = self.policy_trunk_optimizer.param_groups[-1][
            "lr"
        ]  # Policy/trunk LR
        value_learning_rate = self.value_head_optimizer.param_groups[0][
            "lr"
        ]  # Value head LR
        self.total_transitions_trained += (
            update_stats["policy_episodes"] * self.batch_size
        )
        self.total_episodes += update_stats["policy_episodes"]
        total_actions = sum(self.last_action_mix.values())
        action_rates = {k: v / total_actions for k, v in self.last_action_mix.items()}
        training_stats = {
            "step": step,
            "trajectories_collected": self.step_trajectories_collected,
            "total_trajectories_collected": self.total_trajectories_collected,
            "total_transitions_trained": self.total_transitions_trained,
            "total_episodes": self.total_episodes,
            "current_elo": self.opponent_pool.current_elo,
            "pool_stats": self.opponent_pool.get_pool_stats(),
            "learning_rate_policy": policy_learning_rate,
            "learning_rate_value": value_learning_rate,
            "lr_scaling_factor": self.lr_scaling_controller.current_value,
            "beta": self.beta_controller.current_value,
            "grad_scale": self.scaler.get_scale(),
            "entropy_coef_current": self.entropy_coef,
            "action_rates": action_rates,
            **update_stats,
        }

        # Log to wandb if enabled
        if self.use_wandb:
            wandb.log(training_stats, step=step)

        return training_stats

    def _apply_schedules(self, step: int) -> None:
        """Apply cosine LR decay and linear entropy decay with floor."""
        total_steps = max(1, self.cfg.num_steps)
        t = min(1.0, max(0.0, step / float(total_steps)))

        # Learning rate schedules for different components
        # Policy/trunk learning rate schedule
        policy_lr_start = float(self.policy_trunk_learning_rate)
        policy_lr_final = float(self.policy_trunk_learning_rate_final)
        if self.lr_schedule == LrSchedule.cosine and policy_lr_final != policy_lr_start:
            policy_lr_now = policy_lr_final + 0.5 * (
                policy_lr_start - policy_lr_final
            ) * (1.0 + math.cos(math.pi * t))
        else:
            policy_lr_now = policy_lr_start

        # Value head learning rate schedule
        value_lr_start = float(self.value_head_learning_rate)
        value_lr_final = float(self.value_head_learning_rate_final)
        if self.lr_schedule == LrSchedule.cosine and value_lr_final != value_lr_start:
            value_lr_now = value_lr_final + 0.5 * (value_lr_start - value_lr_final) * (
                1.0 + math.cos(math.pi * t)
            )
        else:
            value_lr_now = value_lr_start

        # Apply LR scaling factor to both learning rates (but don't reduce value)
        lr_scale = self.lr_scaling_controller.current_value
        value_lr_scale = max(1.0, lr_scale)

        policy_lr_now *= lr_scale
        value_lr_now *= value_lr_scale

        # Update separate optimizers with their respective learning rates
        if (
            len(self.policy_trunk_optimizer.param_groups) > 1
        ):  # CNN model has cards_trunk
            self.policy_trunk_optimizer.param_groups[0]["lr"] = (
                policy_lr_now * 0.1
            )  # cards_trunk
            self.policy_trunk_optimizer.param_groups[1][
                "lr"
            ] = policy_lr_now  # policy/trunk
        else:  # Transformer model
            self.policy_trunk_optimizer.param_groups[0]["lr"] = policy_lr_now

        self.value_head_optimizer.param_groups[0]["lr"] = value_lr_now

        # Entropy coef linear decay with floor over first portion, then hold
        ent_start = self.entropy_coef_start
        ent_final = self.entropy_coef_final
        portion = self.entropy_decay_portion
        if portion > 0:
            if t <= portion:
                frac = t / portion
                self.entropy_coef = ent_start + (ent_final - ent_start) * frac
            else:
                self.entropy_coef = ent_final
        else:
            self.entropy_coef = ent_start

    @profile
    def _fill_replay_buffer(self, min_steps: int) -> None:
        """
        Fill replay buffer with at least min_steps samples using either tensor or scalar envs.

        Args:
            min_steps: Minimum number of steps to add to replay buffer
        """
        if self.use_tensor_env:
            # Sample 10 opponents for replay buffer filling
            sampled_opponents = self.opponent_pool.sample(k=10)

            # Move opponent models into GPU for inference
            for opponent in sampled_opponents:
                if opponent.model is not None:
                    opponent.model.to(device=self.device, non_blocking=True)

            trajectory_rewards = self.collect_tensor_trajectories(
                min_steps=min_steps,
                all_opponent_snapshots=sampled_opponents,
            )

            # Move opponent models back to CPU to save GPU memory (if enabled)
            if self.offload_opponent_models:
                last_admitted_opponent = self.opponent_pool.get_last_admitted_snapshot()
                for opponent in sampled_opponents:
                    if (
                        opponent.model is not None
                        and opponent != last_admitted_opponent
                    ):
                        opponent.model.to(device="cpu")

            self.total_step_reward += trajectory_rewards.sum().item()
            self.step_trajectories_collected += trajectory_rewards.numel()
            self.total_trajectories_collected += trajectory_rewards.numel()

        else:
            steps_added = 0
            while steps_added < min_steps:
                # Use scalar collection
                sampled_opponent = self.opponent_pool.sample(k=1)
                opponent = sampled_opponent[0] if sampled_opponent else None
                trajectory, reward = self.collect_trajectory(opponent)

                if len(trajectory.transitions) > 0:
                    self.replay_buffer.add_trajectory(trajectory)
                    self.step_trajectories_collected += 1
                    self.total_trajectories_collected += 1
                    steps_added += len(trajectory.transitions)
                    self.total_step_reward += reward

                    # Update opponent pool for scalar environment
                    if opponent is not None:
                        if reward > 0:
                            result = "win"
                        elif reward < 0:
                            result = "loss"
                        else:
                            result = "draw"
                        self.opponent_pool.update_elo_after_game(opponent, result)

    def save_checkpoint(
        self, path: str, step: int, save_optimizer: bool = True
    ) -> None:
        """Save model checkpoint and opponent pool."""
        # Serialize opponent pool inline
        pool_data = {
            "pool_type": self.opponent_pool_type,
            "current_elo": self.opponent_pool.current_elo,
            "snapshots": [],
        }

        # Add pool-specific attributes
        if hasattr(self.opponent_pool, "k"):
            pool_data["k"] = self.opponent_pool.k
        if hasattr(self.opponent_pool, "min_elo_diff"):
            pool_data["min_elo_diff"] = self.opponent_pool.min_elo_diff
        if hasattr(self.opponent_pool, "max_size"):
            pool_data["max_size"] = self.opponent_pool.max_size
        if hasattr(self.opponent_pool, "embedding_dim"):
            pool_data["embedding_dim"] = self.opponent_pool.embedding_dim
        for snapshot in self.opponent_pool.snapshots:
            snapshot_data = {
                "step": snapshot.step,
                "elo": snapshot.elo,
                "games_played": snapshot.games_played,
                "total_rewards": snapshot.total_rewards,
                "wins": snapshot.wins,
                "losses": snapshot.losses,
                "draws": snapshot.draws,
                "model_state_dict": snapshot.model.state_dict(),
                "model_dtype": snapshot.model_dtype,
            }

            # Add DRED-specific data if present
            if hasattr(snapshot, "data") and snapshot.data is not None:
                if hasattr(snapshot.data, "age"):
                    snapshot_data["dred_age"] = snapshot.data.age
                if hasattr(snapshot.data, "alpha"):
                    snapshot_data["dred_alpha"] = snapshot.data.alpha
                if hasattr(snapshot.data, "beta"):
                    snapshot_data["dred_beta"] = snapshot.data.beta

            pool_data["snapshots"].append(snapshot_data)

        checkpoint = {
            "step": step,  # Store the training step
            "total_trajectories_collected": self.total_trajectories_collected,
            "total_episodes": self.total_episodes,
            "total_transitions_trained": self.total_transitions_trained,
            "current_elo": self.opponent_pool.current_elo,
            "model_state_dict": self.model.state_dict(),
            # Store opponent pool inline for single-file checkpoints
            "opponent_pool": pool_data,
            # Store wandb run ID for resumption
            "wandb_run_id": wandb.run.id if self.use_wandb and wandb.run else None,
            # Store wandb step for consistent logging
            "wandb_step": self.wandb_step,
            # Store full config for complete restoration
            "full_config": self.cfg,
            "beta_controller": self.beta_controller.state_dict(),
            "lr_scaling_controller": self.lr_scaling_controller.state_dict(),
            "kl_ema": self.kl_ema.state_dict(),
            "popart_normalizer": (
                self.popart_normalizer.state_dict()
                if self.popart_normalizer is not None
                else None
            ),
            # Legacy config for backward compatibility
            "config": {
                "num_bet_bins": self.num_bet_bins,
                "batch_size": self.batch_size,
                "episodes_per_step": self.episodes_per_step,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "epsilon": self.epsilon,
                "delta1": self.delta1,
                "value_coef": self.value_coef,
                "entropy_coef": self.entropy_coef,
                "grad_clip": self.grad_clip,
            },
        }

        if save_optimizer:
            checkpoint["policy_trunk_optimizer_state_dict"] = (
                self.policy_trunk_optimizer.state_dict()
            )
            checkpoint["value_head_optimizer_state_dict"] = (
                self.value_head_optimizer.state_dict()
            )

        # Conditionally include replay buffer based on economize_checkpoints flag
        if not self.cfg.economize_checkpoints:
            checkpoint["replay_buffer"] = self.replay_buffer

        torch.save(checkpoint, path, _use_new_zipfile_serialization=True)

        if "checkpoint_step_" in path:
            # Create symlink for latest_model.pt pointing to current checkpoint
            self._create_latest_symlink(path)

            # Clean up old checkpoints if economize_checkpoints is enabled
            if self.cfg.economize_checkpoints:
                self._cleanup_old_checkpoints(path)

        compression_note = (
            " (compressed)"
            if not self.cfg.economize_checkpoints
            else " (compressed, replay buffer excluded)"
        )
        print(f"Checkpoint saved to {path}{compression_note}")

    def _create_latest_symlink(self, checkpoint_path: str) -> None:
        """Create or update symlink from latest_model.pt to the current checkpoint."""
        import os

        # Resolve symlinks to get the actual checkpoint file
        actual_checkpoint_path = os.path.realpath(checkpoint_path)
        checkpoint_dir = os.path.dirname(actual_checkpoint_path)
        latest_path = os.path.join(checkpoint_dir, "latest_model.pt")
        checkpoint_filename = os.path.basename(actual_checkpoint_path)

        # Remove existing symlink or file (force removal)
        if os.path.exists(latest_path):
            try:
                os.remove(latest_path)
            except OSError as e:
                print(f"Warning: Could not remove existing latest_model.pt: {e}")
                return

        # Create symlink using relative path to avoid circular references
        try:
            os.symlink(checkpoint_filename, latest_path)
        except OSError as e:
            print(f"Warning: Could not create symlink latest_model.pt: {e}")

    def _cleanup_old_checkpoints(self, current_path: str) -> None:
        """Clean up old checkpoints, keeping only best_model.pt and latest checkpoint."""
        import glob
        import os

        # Resolve symlinks to get the actual checkpoint file
        actual_current_path = os.path.realpath(current_path)
        checkpoint_dir = os.path.dirname(actual_current_path)

        if not os.path.exists(checkpoint_dir):
            return

        # Find all checkpoint files
        checkpoint_files = glob.glob(
            os.path.join(checkpoint_dir, "checkpoint_step_*.pt")
        )

        # Keep best_model.pt and the current checkpoint
        # Note: latest_model.pt is a symlink, so we don't need to keep it in files_to_keep
        files_to_keep = {
            os.path.join(checkpoint_dir, "best_model.pt"),
            actual_current_path,  # Keep the current checkpoint (resolved path)
        }

        # Remove old checkpoint files
        deleted_count = 0
        for file_path in checkpoint_files:
            if file_path not in files_to_keep:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except OSError as e:
                    print(f"Warning: Could not delete {file_path}: {e}")

        if deleted_count > 0:
            print(f"Cleaned up {deleted_count} old checkpoint(s)")

    def load_checkpoint(
        self, path: str, drop_prefixes: list[str] = None
    ) -> tuple[int, str | None]:
        """Load model checkpoint and opponent pool. Returns (step_number, wandb_run_id)."""
        # PyTorch 2.6 defaults to weights_only=True which blocks unpickling
        # custom classes like our ReplayBuffer. We trust our local checkpoints,
        # so explicitly allow full load.
        checkpoint = torch.load(path, weights_only=False, map_location=self.device)

        # Filter model state dict to drop parameters with specified prefixes
        model_state_dict = checkpoint["model_state_dict"]
        if drop_prefixes:
            dropped_keys = []

            for key in list(model_state_dict.keys()):
                should_drop = any(key.startswith(prefix) for prefix in drop_prefixes)
                if should_drop:
                    del model_state_dict[key]
                    dropped_keys.append(key)

            if dropped_keys:
                print(
                    f"Dropped {len(dropped_keys)} parameters with specified prefixes:"
                )
                for key in dropped_keys:
                    print(f"  - {key}")

        self.model.load_state_dict(
            model_state_dict, strict=self.cfg.strict_model_loading
        )
        if "policy_trunk_optimizer_state_dict" in checkpoint:
            self.policy_trunk_optimizer.load_state_dict(
                checkpoint["policy_trunk_optimizer_state_dict"]
            )
        if "value_head_optimizer_state_dict" in checkpoint:
            self.value_head_optimizer.load_state_dict(
                checkpoint["value_head_optimizer_state_dict"]
            )

        # Handle backward compatibility with old single optimizer checkpoints
        if (
            "optimizer_state_dict" in checkpoint
            and "policy_trunk_optimizer_state_dict" not in checkpoint
        ):
            print(
                "Warning: Loading old single optimizer checkpoint. This may not work correctly with separate optimizers."
            )
            # For now, we'll skip loading the old optimizer state
            # In a real migration, you'd need to split the old optimizer state

        # Handle both old and new checkpoint formats
        if "total_episodes_completed" in checkpoint:
            # Newer checkpoints (before rename) stored this field
            self.total_trajectories_collected = checkpoint["total_episodes_completed"]
        elif "total_trajectories_collected" in checkpoint:
            # Current checkpoints save this field directly
            self.total_trajectories_collected = checkpoint[
                "total_trajectories_collected"
            ]
        else:
            # Backwards compatibility with the oldest format
            self.total_trajectories_collected = checkpoint.get("episode_count", 0)

        # Per-step counters rebuilt each run
        self.step_trajectories_collected = 0
        self.total_step_reward = 0.0

        # Restore aggregate logging counters when present
        self.total_episodes = checkpoint.get("total_episodes", self.total_episodes)
        self.total_transitions_trained = checkpoint.get(
            "total_transitions_trained", self.total_transitions_trained
        )

        kl_state = checkpoint.get("kl_ema")
        if kl_state is not None:
            self.kl_ema.load_state_dict(kl_state)

        self.opponent_pool.current_elo = checkpoint.get("current_elo", 1200.0)

        beta_state = checkpoint.get("beta_controller")
        if beta_state is not None:
            self.beta_controller.load_state_dict(beta_state)

        lr_scaling_state = checkpoint.get("lr_scaling_controller")
        if lr_scaling_state is not None:
            self.lr_scaling_controller.load_state_dict(lr_scaling_state)

        popart_state = checkpoint.get("popart_normalizer")
        if popart_state is not None and self.popart_normalizer is not None:
            self.popart_normalizer.load_state_dict(popart_state)

        # Note: step counter is now managed externally

        # Restore wandb step for consistent logging
        # If wandb_step not in checkpoint (old checkpoint), set it to 0 since we no longer store step
        self.wandb_step = checkpoint.get("wandb_step", 0)

        # Restore replay buffer if present:
        if "replay_buffer" in checkpoint and checkpoint["replay_buffer"] is not None:
            self.replay_buffer = checkpoint["replay_buffer"]

        print(f"Checkpoint loaded from {path}")

        # Restore opponent pool from inline data if available; fallback to old separate file
        pool_data = checkpoint.get("opponent_pool")
        if pool_data is not None:
            # Restore pool-specific attributes
            if hasattr(self.opponent_pool, "k") and "k" in pool_data:
                self.opponent_pool.k = pool_data.get("k", self.opponent_pool.k)
            if (
                hasattr(self.opponent_pool, "min_elo_diff")
                and "min_elo_diff" in pool_data
            ):
                self.opponent_pool.min_elo_diff = pool_data.get(
                    "min_elo_diff", self.opponent_pool.min_elo_diff
                )
            if hasattr(self.opponent_pool, "max_size") and "max_size" in pool_data:
                self.opponent_pool.max_size = pool_data.get(
                    "max_size", self.opponent_pool.max_size
                )
            if (
                hasattr(self.opponent_pool, "embedding_dim")
                and "embedding_dim" in pool_data
            ):
                self.opponent_pool.embedding_dim = pool_data.get(
                    "embedding_dim", self.opponent_pool.embedding_dim
                )

            self.opponent_pool.current_elo = pool_data.get(
                "current_elo", self.opponent_pool.current_elo
            )
            self.opponent_pool.snapshots = []
            for snapshot_data in pool_data.get("snapshots", []):
                if self.is_transformer:
                    model = PokerTransformerV1(
                        max_sequence_length=self.cfg.model.max_sequence_length,
                        d_model=self.cfg.model.d_model,
                        n_layers=self.cfg.model.n_layers,
                        n_heads=self.cfg.model.n_heads,
                        num_bet_bins=self.num_bet_bins,
                        dropout=self.cfg.model.dropout,
                        use_gradient_checkpointing=self.cfg.model.use_gradient_checkpointing,
                        value_head_type=self.value_head_type,
                        value_head_num_quantiles=self.value_head_num_quantiles,
                    )
                else:
                    model = SiameseConvNetV1(
                        cards_channels=self.cfg.model.cards_channels,
                        actions_channels=self.cfg.model.actions_channels,
                        cards_hidden=self.cfg.model.cards_hidden,
                        actions_hidden=self.cfg.model.actions_hidden,
                        fusion_hidden=self.cfg.model.fusion_hidden,
                        num_actions=self.num_bet_bins,
                        use_gradient_checkpointing=self.cfg.model.use_gradient_checkpointing,
                        value_head_type=self.value_head_type,
                        value_head_num_quantiles=self.value_head_num_quantiles,
                    )
                model_state_dict = snapshot_data["model_state_dict"]
                if drop_prefixes:
                    dropped_keys = []

                    for key in list(model_state_dict.keys()):
                        should_drop = any(
                            key.startswith(prefix) for prefix in drop_prefixes
                        )
                        if should_drop:
                            del model_state_dict[key]
                            dropped_keys.append(key)

                model.load_state_dict(
                    model_state_dict,
                    strict=self.cfg.strict_model_loading,
                )

                # Handle backward compatibility for older snapshots
                model_dtype = snapshot_data.get("model_dtype", torch.float32)
                # Handle legacy use_mixed_precision field
                if (
                    "use_mixed_precision" in snapshot_data
                    and "model_dtype" not in snapshot_data
                ):
                    model_dtype = (
                        torch.bfloat16
                        if snapshot_data["use_mixed_precision"]
                        else torch.float32
                    )

                # Convert model to the stored dtype and device
                model = model.to(dtype=model_dtype, device=self.device)

                snapshot = AgentSnapshot(
                    model=model,
                    step=snapshot_data.get("step", 0),
                    elo=snapshot_data.get("elo", 1200.0),
                    model_dtype=model_dtype,
                )
                snapshot.games_played = snapshot_data.get("games_played", 0)
                snapshot.wins = snapshot_data.get("wins", 0)
                snapshot.losses = snapshot_data.get("losses", 0)
                snapshot.draws = snapshot_data.get("draws", 0)

                # Restore DRED-specific data if present
                if "dred_age" in snapshot_data:
                    from alphaholdem.rl.dred_pool import DREDSnapshotData

                    snapshot.data = DREDSnapshotData(
                        age=snapshot_data["dred_age"],
                        alpha=snapshot_data.get("dred_alpha", 1.0),
                        beta=snapshot_data.get("dred_beta", 1.0),
                    )

                self.opponent_pool.snapshots.append(snapshot)
            print("Opponent pool restored from checkpoint file")
        else:
            # Backward-compatibility path: load separate pool file if present
            pool_path = path.replace(".pt", "_pool.pt")
            try:
                # For backward compatibility, assume CNN model if separate pool file exists
                self.opponent_pool.load_pool(pool_path, SiameseConvNetV1)
                # Ensure snapshot models are on the correct device and dtype
                for snap in self.opponent_pool.snapshots:
                    snap.model.to(dtype=snap.model_dtype, device=self.device)
                print(f"Opponent pool loaded from {pool_path}")
            except FileNotFoundError:
                print(
                    f"No opponent pool found at {pool_path}; starting with empty pool"
                )

        # Extract wandb run ID for resumption
        wandb_run_id = checkpoint.get("wandb_run_id")

        # Extract step number from checkpoint (default to 0 for old checkpoints)
        step = checkpoint.get("step", 0)

        return step, wandb_run_id

    def evaluate_against_pool(self, min_games: int = 200) -> dict:
        """
        Evaluate current model against all opponents in the pool.

        Args:
            num_games: Number of games to play against each opponent

        Returns:
            Dictionary with evaluation results

        Note: Uses tensorized evaluation with individual episode reward tracking for accurate win counting.
        """
        if not self.opponent_pool.snapshots:
            return {"error": "No opponents in pool"}

        results = {}
        total_wins = 0
        total_games = 0

        # Use no_grad to prevent gradient computation during evaluation
        with torch.no_grad():
            opponents = list(self.opponent_pool.snapshots)

            # Snapshot opponent stats BEFORE evaluation
            before_stats = [(opp.losses, opp.games_played) for opp in opponents]

            if self.use_tensor_env:
                # Collect trajectories ONCE against all opponents; size to roughly hit min_games/opponent
                total_target_trajectories = max(1, min_games * len(opponents))
                for opp in opponents:
                    opp.model.to(device=self.device)
                _ = self.collect_tensor_trajectories(
                    min_trajectories=total_target_trajectories,
                    all_opponent_snapshots=opponents,
                    add_to_replay_buffer=False,
                )
                if self.offload_opponent_models:
                    for opp in opponents:
                        opp.model.to(device="cpu")
            else:
                # Scalar fallback: play min_games per opponent without adding to replay buffer
                for opp in opponents:
                    for _ in range(min_games):
                        self.collect_trajectory(opponent_snapshot=opp)

            # Compute per-opponent deltas AFTER evaluation
            for i, opp in enumerate(opponents):
                before_losses, before_games_played = before_stats[i]
                after_losses = opp.losses
                after_games = opp.games_played

                # From opponent's perspective, their losses are our wins
                our_wins = max(0, after_losses - before_losses)
                games = max(0, after_games - before_games_played)
                win_rate = (our_wins / games) if games > 0 else 0.0

                results[f"opponent_{i}_step_{opp.step}"] = {
                    "win_rate": win_rate,
                    "opponent_elo": opp.elo,
                    "wins": our_wins,
                    "total_games": games,
                }
                total_wins += our_wins
                total_games += games

        overall_win_rate = total_wins / total_games if total_games > 0 else 0.0

        return {
            "overall_win_rate": overall_win_rate,
            "total_games": total_games,
            "opponent_results": results,
            "pool_stats": self.opponent_pool.get_pool_stats(),
        }
