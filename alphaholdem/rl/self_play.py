from __future__ import annotations

from typing import Optional
import math

import torch
import torch.nn as nn

import wandb

from ..core.builders import build_components_from_config
from ..core.structured_config import Config
from ..encoding.action_mapping import bin_to_action, get_legal_mask
from ..env.hunl_env import HUNLEnv
from ..env.hunl_tensor_env import HUNLTensorEnv
from ..models.cnn_embedding_data import CNNEmbeddingData
from ..models.factory import ModelFactory
from ..models.state_encoder import CNNStateEncoder
from ..models.transformer.token_sequence_builder import TokenSequenceBuilder
from ..models.transformer.kv_cache_manager import SelfPlayKVCacheManager
from ..models.model_outputs import ModelOutput
from ..rl.agent_snapshot import AgentSnapshot
from ..rl.k_best_pool import KBestOpponentPool
from ..rl.dred_pool import DREDPool
from ..utils.kl_divergence import compute_kl_divergence_batch
from ..rl.losses import trinal_clip_ppo_loss
from ..rl.replay import Trajectory, Transition
from ..rl.vectorized_replay import VectorizedReplayBuffer
from ..utils.profiling import profile


class SelfPlayTrainer:
    def __init__(
        self,
        cfg: Config,
        device: torch.device,
    ):
        self.cfg = cfg
        self.device = device

        # Hydra config - extract parameters from nested structure
        self.batch_size = cfg.train.batch_size
        self.num_epochs = cfg.train.num_epochs
        self.gamma = cfg.train.gamma
        self.gae_lambda = cfg.train.gae_lambda
        self.epsilon = cfg.train.ppo_eps
        self.replay_buffer_batches = cfg.train.replay_buffer_batches
        self.max_trajectory_length = cfg.train.max_trajectory_length
        self.delta1 = cfg.train.ppo_delta1
        self.value_coef = cfg.train.value_coef
        self.entropy_coef = cfg.train.entropy_coef
        self.grad_clip = cfg.train.grad_clip
        self.learning_rate = cfg.train.learning_rate
        # LR and entropy schedules (assumed present in config)
        self.learning_rate_final = cfg.train.learning_rate_final
        self.lr_schedule = cfg.train.lr_schedule
        self.entropy_coef_start = cfg.train.entropy_coef
        self.entropy_coef_final = cfg.train.entropy_coef_final
        self.entropy_decay_portion = cfg.train.entropy_decay_portion
        self.k_best_pool_size = cfg.k_best_pool_size
        self.min_elo_diff = cfg.min_elo_diff
        self.min_step_diff = cfg.min_step_diff
        self.k_factor = cfg.k_factor
        self.use_mixed_precision = (
            cfg.train.use_mixed_precision and self.device.type in ["cuda", "mps"]
        )
        self.loss_scale = cfg.train.loss_scale
        self.use_kv_cache = cfg.train.use_kv_cache
        self.use_tensor_env = cfg.use_tensor_env
        self.num_envs = cfg.num_envs
        self.use_wandb = cfg.use_wandb
        self.wandb_project = cfg.wandb_project
        self.wandb_name = cfg.wandb_name
        self.wandb_tags = cfg.wandb_tags
        self.wandb_run_id = cfg.wandb_run_id

        self.float_dtype = torch.float32

        # Initialize RNG
        self.rng = torch.Generator(device=self.device)

        # Determine model type
        self.is_transformer = cfg.model.name.startswith("poker_transformer")

        # Initialize components
        # Always create tensor_env for state encoder, even if we don't use it for training
        self.tensor_env = HUNLTensorEnv(
            num_envs=self.num_envs,
            starting_stack=self.cfg.env.stack,
            sb=self.cfg.env.sb,
            bb=self.cfg.env.bb,
            bet_bins=self.cfg.env.bet_bins,
            store_action_history=not self.is_transformer,
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

        # Config-driven components
        _, _, model, policy = build_components_from_config(self.cfg)
        self.model = model
        self.policy = policy

        # Ensure bins align with model output size to avoid mask/logit mismatch
        if hasattr(self.model, "policy_head") and hasattr(
            self.model.policy_head, "out_features"
        ):
            self.num_bet_bins = int(self.model.policy_head.out_features)
        else:
            self.num_bet_bins = len(self.cfg.env.bet_bins) + 3

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
            self.state_encoder = CNNStateEncoder(self.tensor_env, self.device)

        self.model.to(self.device)  # Move model to device
        self._initialize_weights()  # Initialize with better weights
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

        # Optimizer with different learning rates for different components
        # CNN layers (cards_trunk) need lower learning rate to prevent gradient explosion
        lr = self.learning_rate

        cards_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if "cards_trunk" in name:
                cards_params.append(param)
            else:
                other_params.append(param)

        # Handle different model types
        if cards_params:  # CNN model
            self.optimizer = torch.optim.Adam(
                [
                    {"params": cards_params, "lr": lr * 0.1},  # 10x lower for CNN
                    {"params": other_params, "lr": lr},
                ]
            )
            self._optimizer_group_scales = [0.1, 1.0]
        else:  # Transformer model (no cards_trunk)
            self.optimizer = torch.optim.Adam([{"params": other_params, "lr": lr}])
            self._optimizer_group_scales = [1.0]

        # Mixed precision scaler
        self.scaler = (
            torch.amp.GradScaler(
                self.device.type,
                init_scale=self.loss_scale,
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=2000,
            )
            if self.use_mixed_precision
            else None
        )

        # Training stats
        self.step_trajectories_collected = 0
        self.total_step_reward = 0.0
        self.total_trajectories_collected = (
            0  # Keep track of total episodes across all steps
        )
        self.opponent_pool.current_elo = 1200.0  # Starting ELO rating

        # Weights & Biases setup
        self.use_wandb = self.use_wandb
        self.wandb_step = 0
        if self.use_wandb:
            # Determine if we're resuming an existing run
            wandb_init_kwargs = {
                "project": self.wandb_project,
                "name": self.wandb_name,
                "tags": self.wandb_tags or [],
                "config": {
                    "learning_rate": lr,
                    "batch_size": self.batch_size,
                    "num_epochs": self.num_epochs,
                    "gamma": self.gamma,
                    "gae_lambda": self.gae_lambda,
                    "epsilon": self.epsilon,
                    "delta1": self.delta1,
                    "value_coef": self.value_coef,
                    "entropy_coef": self.entropy_coef,
                    "grad_clip": self.grad_clip,
                    "k_best_pool_size": self.k_best_pool_size,
                    "min_elo_diff": self.min_elo_diff,
                    "use_tensor_env": self.use_tensor_env,
                    "num_envs": self.num_envs,
                    "device": str(self.device),
                },
            }
            try:
                if self.wandb_run_id:
                    wandb_init_kwargs["id"] = self.wandb_run_id
                    wandb_init_kwargs["resume"] = "must"
                    wandb.init(**wandb_init_kwargs)
                    print(
                        f"Wandb resumed run {self.wandb_run_id} for project: {self.wandb_project}"
                    )
                else:
                    wandb.init(**wandb_init_kwargs)
                print(f"Wandb initialized new run for project: {self.wandb_project}")
            except Exception as exc:  # pragma: no cover - offline fallback path
                print(
                    f"Wandb initialization failed ({exc}); continuing without wandb logging."
                )
                self.use_wandb = False

    def _initialize_weights(self):
        """Initialize model weights to prevent dead neurons."""
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

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
            self.state_encoder.reset_envs(
                torch.arange(self.num_envs, device=self.device)
            )

            # CLS
            self.state_encoder.add_cls(all_env_indices)
            # Preflop street marker
            self.state_encoder.add_street(
                all_env_indices, torch.zeros_like(all_env_indices)
            )
            # Hole cards for P0
            self.state_encoder.add_card(
                all_env_indices,
                self.tensor_env.hole_indices[all_env_indices, 0, 0],
            )
            self.state_encoder.add_card(
                all_env_indices,
                self.tensor_env.hole_indices[all_env_indices, 0, 1],
            )

    @profile
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

        # Initialize all environments
        self._reset_tensor_env_and_encoder_kv(all_opponent_snapshots)

        # Per-environment reward tracking and step counts
        per_env_rewards = torch.zeros(self.num_envs, device=self.device)
        max_steps = self.cfg.train.max_trajectory_length
        steps_since_reset = 0
        # Use the trainer's state_encoder as the TSB for transformer path

        # Chunk opponents by tensor env index
        opp_env_groups = None
        env_indices = torch.arange(self.num_envs, device=self.device)
        if all_opponent_snapshots is not None and len(all_opponent_snapshots) > 0:
            # Use opponent models - assign opponents to environments
            num_opps = len(all_opponent_snapshots)
            # Split active_opp_acts into num_opps approximately equal groups
            opp_env_groups = torch.chunk(env_indices, num_opps)

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

        # Outer loop: collect complete trajectories until we have enough steps
        loop_count = 0
        max_loops = 1000  # Safety limit to prevent infinite loops

        adding_trajectories = False

        while (
            steps_collected < min_steps or trajectory_count < min_trajectories
        ) and loop_count < max_loops:
            if not adding_trajectories and add_to_replay_buffer:
                # Initialize trajectory collection; reserve space in buffer.
                self.replay_buffer.start_adding_trajectory_batches(self.num_envs)
                adding_trajectories = True

            loop_count += 1

            # Only process non-done environments to avoid bias towards short episodes
            active_mask = ~self.tensor_env.done
            active_indices = torch.where(active_mask)[0]
            we_act_mask = self.tensor_env.to_act == 0
            opp_acts_mask = self.tensor_env.to_act == 1
            env_active_we_act_mask = active_mask & we_act_mask
            env_active_we_act = torch.where(env_active_we_act_mask)[0]
            env_active_opp_acts_mask = active_mask & opp_acts_mask
            env_active_opp_acts = torch.where(env_active_opp_acts_mask)[0]

            # Get legal action masks for active environments only
            legal_bins_amounts, legal_bins_mask = (
                self.tensor_env.legal_bins_amounts_and_mask()
            )

            # Get model predictions for active environments only
            with torch.no_grad():
                # Add pre-action context token
                if self.is_transformer and isinstance(
                    self.state_encoder, TokenSequenceBuilder
                ):
                    self.state_encoder.add_context(active_indices)

                # Encode states from our perspective (player=0)
                our_states = self.state_encoder.encode_tensor_states(
                    player=0, idxs=env_active_we_act
                )

                env_logits.zero_()
                env_values.zero_()

                # active_first_action = ~self.tensor_env.acted_since_reset[active_indices]

                # Get predictions from our model for our turns
                if env_active_we_act.numel() > 0:
                    with torch.amp.autocast(
                        self.device.type,
                        dtype=torch.bfloat16,
                        enabled=self.use_mixed_precision,
                    ):
                        # Use KV caching for transformer models
                        if self.is_transformer and self.kv_cache_manager is not None:
                            # Get our cache
                            our_cache = self.kv_cache_manager.get_self_cache()
                            outputs = self.model(our_states, kv_cache=our_cache)
                            # Update our cache
                            if outputs.kv_cache is not None:
                                self.kv_cache_manager.update_self_cache(
                                    outputs.kv_cache
                                )
                        else:
                            outputs = self.model(our_states)

                    env_logits[env_active_we_act] = outputs.policy_logits.float()
                    env_values[env_active_we_act] = outputs.value.float()

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

                        # Use indices directly into active arrays
                        with torch.amp.autocast(
                            self.device.type,
                            dtype=torch.bfloat16,
                            enabled=self.use_mixed_precision,
                        ):
                            # Use KV caching for transformer models
                            if (
                                self.is_transformer
                                and self.kv_cache_manager is not None
                            ):
                                # Get opponent cache
                                opp_cache = self.kv_cache_manager.get_opponent_cache(
                                    opponent_idx
                                )
                                outputs = opponent.model(
                                    opp_states.to(opponent.model_dtype),
                                    kv_cache=opp_cache,
                                )
                                # Update opponent cache
                                if outputs.kv_cache is not None:
                                    self.kv_cache_manager.update_opponent_cache(
                                        opponent_idx, outputs.kv_cache
                                    )
                            else:
                                outputs = opponent.model(
                                    opp_states.to(opponent.model_dtype)
                                )
                            env_logits[opp_working_env_indices] = (
                                outputs.policy_logits.float()
                            )
                            env_values[opp_working_env_indices] = outputs.value.float()

                elif env_active_opp_acts.numel() > 0:
                    opp_states = self.state_encoder.encode_tensor_states(
                        player=1, idxs=env_active_opp_acts
                    )
                    # Self-play: use our model for opponent turns too
                    with torch.amp.autocast(
                        self.device.type,
                        dtype=torch.bfloat16,
                        enabled=self.use_mixed_precision,
                    ):
                        # Use KV caching for transformer models
                        if self.is_transformer and self.kv_cache_manager is not None:
                            # For self-play, we can reuse our own cache for opponent turns
                            # or create a separate cache for the opponent player
                            opp_cache = self.kv_cache_manager.get_opponent_cache(
                                0
                            )  # Player 1
                            outputs = self.model(opp_states, kv_cache=opp_cache)
                            # Update opponent cache
                            if outputs.kv_cache is not None:
                                self.kv_cache_manager.update_opponent_cache(
                                    0, outputs.kv_cache
                                )
                        else:
                            outputs = self.model(opp_states)
                        env_logits[env_active_opp_acts] = outputs.policy_logits.float()
                        env_values[env_active_opp_acts] = outputs.value.float()

                # Sample actions for active environments only.
                action_bins_active, log_probs_active_full = self.policy.action_batch(
                    env_logits[active_indices], legal_bins_mask[active_indices]
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
                    self.tensor_env.street[active_indices],
                )

            # Take steps in all environments (tensor_env expects full-size tensors)
            # NOTE: THIS CHANGES self.tensor_env.to_act!!
            rewards, dones, _, new_streets, dealt_cards = self.tensor_env.step_bins(
                action_bins, legal_bins_amounts, legal_bins_mask
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
                our_log_probs_full = log_probs_active_full[active_we_act]
                our_values_tensor = env_values[env_active_we_act]
                our_rewards_tensor = rewards[env_active_we_act]
                our_dones_tensor = dones[env_active_we_act]
                our_legal_masks_tensor = legal_bins_mask[env_active_we_act]

                # Add transitions immediately using vectorized operations
                if add_to_replay_buffer:
                    self.replay_buffer.add_transitions(
                        embedding_data=our_states,
                        action_indices=our_action_indices,
                        log_probs=our_log_probs_full,
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

        # Concatenate all episode rewards into a single tensor
        if collected_trajectory_rewards:
            collected_trajectory_rewards_tensor = torch.cat(
                collected_trajectory_rewards, dim=0
            )
        else:
            collected_trajectory_rewards_tensor = torch.tensor([], device=self.device)

        return collected_trajectory_rewards_tensor

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
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clipfrac = 0.0
        total_explained_var = 0.0
        total_minibatches = 0

        for _ in range(self.num_epochs):
            # Sample batch from vectorized buffer
            batch = self.replay_buffer.sample_batch(self.rng, self.batch_size)

            # Normalize advantages across the batch for stability (mean=0, std=1)
            adv = batch["advantages"]
            adv_mean = adv.mean()
            adv_std = adv.std().clamp_min(1e-8)
            batch["advantages"] = (adv - adv_mean) / adv_std

            # Per-sample value clipping bounds computed in prepare_ppo_batch
            # batch['delta2'] and batch['delta3'] are length-N tensors aligned with samples
            delta2_vec = batch["delta2"]
            delta3_vec = batch["delta3"]

            # No minibatches: operate on the full batch at once
            is_transformer = self.cfg.model.name.startswith("poker_transformer")

            with torch.amp.autocast(
                self.device.type,
                dtype=torch.bfloat16,
                enabled=self.use_mixed_precision,
            ):
                outputs = self.model(batch["embedding_data"])
                logits = outputs.policy_logits.float()
                values = outputs.value.float()

                if is_transformer:
                    # Transformer-specific loss with auxiliary hand range loss
                    loss_dict = self._compute_transformer_loss(
                        outputs, batch, delta2_vec, delta3_vec
                    )
                else:
                    # CNN loss
                    loss_dict = trinal_clip_ppo_loss(
                        logits=logits,
                        values=values,
                        actions=batch["action_indices"],
                        log_probs_old=batch["log_probs_old"].float(),
                        advantages=batch["advantages"].float(),
                        returns=batch["returns"].float(),
                        legal_masks=batch["legal_masks"],
                        epsilon=self.epsilon,
                        delta1=self.delta1,
                        delta2=delta2_vec.float(),
                        delta3=delta3_vec.float(),
                        value_coef=self.value_coef,
                        entropy_coef=self.entropy_coef,
                        value_loss_type=self.cfg.train.value_loss_type,
                        huber_delta=self.cfg.train.huber_delta,
                    )

            # Debugging metrics: exact KL (from old full dist), clipfrac, explained variance
            with torch.no_grad():
                legal_mb = batch["legal_masks"].bool()
                masked_logits_new = torch.where(legal_mb, logits, -1e9)
                log_probs_new_full = torch.log_softmax(masked_logits_new, dim=-1)
                log_probs_old_full = batch.get("log_probs_old_full")
                # Compute exact KL: E_{a~p_old}[log p_old - log p_new]
                p_old = torch.exp(log_probs_old_full)
                exact_kl_vec = (p_old * (log_probs_old_full - log_probs_new_full)).sum(
                    dim=-1
                )
                approx_kl = exact_kl_vec.mean()
                # Compute ratio/clipfrac for monitoring
                a_mb = batch["action_indices"]
                logp_new = log_probs_new_full.gather(1, a_mb.unsqueeze(1)).squeeze(1)
                logp_old = batch["log_probs_old"]
                ratio = torch.exp(logp_new - logp_old)
                clip_low, clip_high = 1.0 - self.epsilon, 1.0 + self.epsilon
                clipped = torch.clamp(ratio, clip_low, clip_high)
                clipfrac = (torch.abs(clipped - ratio) > 1e-8).float().mean()

                # Explained variance calculation using clipped returns (same as loss)
                ret_mb = batch["returns"].float()
                d2_mb = delta2_vec.float()
                d3_mb = delta3_vec.float()
                # Ensure per-sample bounds are ordered to avoid clamp warnings
                ret_clipped_mb = torch.clamp(ret_mb, min=d2_mb, max=d3_mb)
                # Explained variance of value predictions against clipped targets
                var_y = torch.var(ret_clipped_mb)
                var_err = torch.var(ret_clipped_mb - values.detach())
                explained_var = 1.0 - (var_err / (var_y + 1e-8))

            self.optimizer.zero_grad()

            # Use scaler for mixed precision backward pass
            if self.scaler is not None:
                self.scaler.scale(loss_dict["total_loss"]).backward()

                # Apply stricter gradient clipping to CNN layers to prevent explosion
                self.scaler.unscale_(self.optimizer)
                for name, param in self.model.named_parameters():
                    if param.grad is not None and "cards_trunk" in name:
                        torch.nn.utils.clip_grad_norm_(
                            [param], 0.5
                        )  # Stricter clipping for CNN

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict["total_loss"].backward()

                # Apply stricter gradient clipping to CNN layers to prevent explosion
                for name, param in self.model.named_parameters():
                    if param.grad is not None and "cards_trunk" in name:
                        torch.nn.utils.clip_grad_norm_(
                            [param], 0.5
                        )  # Stricter clipping for CNN

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()

            total_loss += loss_dict["total_loss"].item()
            total_policy_loss += float(loss_dict["policy_loss"].item())
            total_value_loss += float(loss_dict["value_loss"].item())
            total_entropy += float(loss_dict["entropy"].item())
            total_approx_kl += float(approx_kl.item())
            total_clipfrac += float(clipfrac.item())
            total_explained_var += float(explained_var.item())
            total_minibatches += 1

            # Check if we should add current model to opponent pool
            # Compute KL divergence between current model and last admitted opponent
            kl_divergence = 0.0
            last_admitted_opponent = self.opponent_pool.get_last_admitted_snapshot()

            if last_admitted_opponent is not None:
                # Take a 64-element sample from the current batch without replacement
                batch_size = batch["returns"].shape[0]
                sample_size = min(64, batch_size)
                sample_indices = torch.randperm(
                    batch_size, device=batch["returns"].device
                )[:sample_size]

                # Get states for KL computation
                kl_states = batch["embedding_data"][sample_indices]

                # Get current model logits
                with (
                    torch.no_grad(),
                    torch.amp.autocast(
                        self.device.type,
                        dtype=torch.bfloat16,
                        enabled=self.use_mixed_precision,
                    ),
                ):
                    # Get last admitted opponent model logits
                    opponent_outputs = last_admitted_opponent.model(
                        kl_states.to(last_admitted_opponent.model_dtype)
                    )
                    opponent_logits = opponent_outputs.policy_logits.float()

                # Compute KL divergence
                current_logits = outputs.policy_logits[sample_indices].float()
                kl_divergence = compute_kl_divergence_batch(
                    current_logits, opponent_logits
                )

            if self.opponent_pool.should_add_snapshot(step, kl_divergence):
                sample_batch_size = min(1024, len(batch["embedding_data"]))
                sample_batch = batch["embedding_data"][:sample_batch_size]
                self.opponent_pool.add_snapshot(self.model, step, sample_batch)

        # Calculate average reward for this update step using captured values
        avg_reward = (
            self.total_step_reward / max(1, self.step_trajectories_collected)
            if self.step_trajectories_collected > 0
            else 0.0
        )
        if abs(avg_reward) > 1:
            print(f"Warning: Avg reward is {avg_reward}, which is outside [-1, 1].")
            print(f"Total step reward: {self.total_step_reward}")
            print(f"Step trajectories collected: {self.step_trajectories_collected}")

        denom = max(1, total_minibatches)
        return {
            "avg_reward": avg_reward,
            "num_samples": self.batch_size * self.num_epochs,
            "delta2_mean": float(delta2_vec.mean().item()),
            "delta3_mean": float(delta3_vec.mean().item()),
            "avg_loss": total_loss / denom,
            "policy_loss": total_policy_loss / denom,
            "value_loss": total_value_loss / denom,
            "entropy": total_entropy / denom,
            "approx_kl": total_approx_kl / denom,
            "clipfrac": total_clipfrac / denom,
            "explained_var": total_explained_var / denom,
        }

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

        # These are updated while filling the replay buffer
        self.total_step_reward = 0.0
        self.step_trajectories_collected = 0

        target_steps = self.batch_size * max(self.cfg.train.replay_buffer_batches, 1)
        if self.replay_buffer.num_steps() == 0:
            # Warmup: fill replay buffer with minimum required samples
            print(f"Warmup: filling replay buffer to {target_steps} steps...")
            self._fill_replay_buffer(target_steps)
        else:
            # Before update, add one batch worth of fresh steps
            self._fill_replay_buffer(self.batch_size)

        # Update model
        update_stats = self.update_model(step)

        # Prepare training stats for return and logging
        training_stats = {
            "step": step,
            "trajectories_collected": self.step_trajectories_collected,
            "total_trajectories_collected": self.total_trajectories_collected,
            "current_elo": self.opponent_pool.current_elo,
            "pool_stats": self.opponent_pool.get_pool_stats(),
            **update_stats,
        }

        # Reset step counters for next training step
        self.total_step_reward = 0.0
        self.step_trajectories_collected = 0

        # Log to wandb if enabled
        if self.use_wandb:
            wandb.log(
                {
                    "step": step,  # Match CLI display (1-indexed)
                    "trajectories_collected": training_stats["trajectories_collected"],
                    "avg_reward": training_stats["avg_reward"],
                    "current_elo": training_stats["current_elo"],
                    "policy_loss": training_stats["policy_loss"],
                    "value_loss": training_stats["value_loss"],
                    "entropy": training_stats["entropy"],
                    "approx_kl": training_stats["approx_kl"],
                    "clipfrac": training_stats["clipfrac"],
                    "explained_var": training_stats["explained_var"],
                    "avg_loss": training_stats["avg_loss"],
                    "num_samples": training_stats["num_samples"],
                    "lr": self.optimizer.param_groups[-1]["lr"],
                    "entropy_coef_current": self.entropy_coef,
                },
                step=step,
            )

        return training_stats

    def _apply_schedules(self, step: int) -> None:
        """Apply cosine LR decay and linear entropy decay with floor."""
        total_steps = max(1, self.cfg.num_steps)
        t = min(1.0, max(0.0, step / float(total_steps)))

        # Learning rate schedule
        lr_start = float(self.learning_rate)
        lr_final = float(self.learning_rate_final)
        if self.lr_schedule == "cosine" and lr_final != lr_start:
            lr_now = lr_final + 0.5 * (lr_start - lr_final) * (
                1.0 + math.cos(math.pi * t)
            )
        else:
            lr_now = lr_start

        # Update optimizer groups preserving relative scales
        for scale, group in zip(
            self._optimizer_group_scales, self.optimizer.param_groups
        ):
            group["lr"] = lr_now * scale

        # Entropy coef linear decay with floor over first portion, then hold
        ent_start = float(self.entropy_coef_start)
        ent_final = float(self.entropy_coef_final)
        portion = float(self.entropy_decay_portion)
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
            trajectory_rewards = self.collect_tensor_trajectories(
                min_steps=min_steps,
                all_opponent_snapshots=sampled_opponents,
            )

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

    def save_checkpoint(self, path: str, step: int) -> None:
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
            "current_elo": self.opponent_pool.current_elo,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            # Store opponent pool inline for single-file checkpoints
            "opponent_pool": pool_data,
            # Store wandb run ID for resumption
            "wandb_run_id": wandb.run.id if self.use_wandb and wandb.run else None,
            # Store wandb step for consistent logging
            "wandb_step": self.wandb_step,
            # Store full config for complete restoration
            "full_config": self.cfg,
            # Legacy config for backward compatibility
            "config": {
                "num_bet_bins": self.num_bet_bins,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "epsilon": self.epsilon,
                "delta1": self.delta1,
                "value_coef": self.value_coef,
                "entropy_coef": self.entropy_coef,
                "grad_clip": self.grad_clip,
            },
        }

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

    def load_checkpoint(self, path: str) -> tuple[int, str | None]:
        """Load model checkpoint and opponent pool. Returns (step_number, wandb_run_id)."""
        # PyTorch 2.6 defaults to weights_only=True which blocks unpickling
        # custom classes like our ReplayBuffer. We trust our local checkpoints,
        # so explicitly allow full load.
        checkpoint = torch.load(path, weights_only=False, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Handle both old and new checkpoint formats
        if "total_episodes_completed" in checkpoint:
            # New format
            self.total_trajectories_collected = checkpoint["total_episodes_completed"]
            self.step_trajectories_collected = 0  # Reset step counter
            self.total_step_reward = 0.0  # Reset step reward
        else:
            # Old format - migrate to new format
            self.total_trajectories_collected = checkpoint.get("episode_count", 0)
            self.step_trajectories_collected = 0  # Reset step counter
            self.total_step_reward = 0.0  # Reset step reward

        self.opponent_pool.current_elo = checkpoint.get("current_elo", 1200.0)

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
                # Create model using ModelFactory to support both CNN and transformer
                model_kwargs = (
                    self.cfg.model.kwargs.copy() if self.cfg.model.kwargs else {}
                )
                model_kwargs["use_gradient_checkpointing"] = (
                    self.cfg.model.use_gradient_checkpointing
                )
                model = ModelFactory.create_model(
                    self.cfg.model.name, model_kwargs, self.device
                )
                model.load_state_dict(snapshot_data["model_state_dict"])
                model.to(self.device)
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
                    from .dred_pool import DREDSnapshotData

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
                from ..models.cnn import SiameseConvNetV1

                self.opponent_pool.load_pool(pool_path, SiameseConvNetV1)
                # Ensure snapshot models are on the correct device
                for snap in self.opponent_pool.snapshots:
                    snap.model.to(self.device)
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

    def evaluate_against_pool(self, min_games: int = 100) -> dict:
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
            before_stats = [
                (opp.wins, opp.losses, opp.games_played) for opp in opponents
            ]

            if self.use_tensor_env:
                # Collect trajectories ONCE against all opponents; size to roughly hit min_games/opponent
                total_target_trajectories = max(1, min_games * len(opponents))
                _ = self.collect_tensor_trajectories(
                    min_trajectories=total_target_trajectories,
                    all_opponent_snapshots=opponents,
                    add_to_replay_buffer=False,
                )
            else:
                # Scalar fallback: play min_games per opponent without adding to replay buffer
                for opp in opponents:
                    for _ in range(min_games):
                        self.collect_trajectory(opponent_snapshot=opp)

            # Compute per-opponent deltas AFTER evaluation
            for i, opp in enumerate(opponents):
                _, before_losses, before_games_played = before_stats[i]
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

    def _compute_transformer_loss(
        self,
        outputs: ModelOutput,
        batch: dict,
        delta2_vec: torch.Tensor,
        delta3_vec: torch.Tensor,
    ) -> dict:
        """Compute transformer-specific loss with auxiliary hand range loss."""

        logits = outputs.policy_logits.float()
        values = outputs.value.float()

        # Standard PPO policy loss
        return trinal_clip_ppo_loss(
            logits=logits,
            values=values,
            actions=batch["action_indices"],
            log_probs_old=batch["log_probs_old"].float(),
            advantages=batch["advantages"].float(),
            returns=batch["returns"].float(),
            legal_masks=batch["legal_masks"],
            epsilon=self.epsilon,
            delta1=self.delta1,
            delta2=delta2_vec.float(),
            delta3=delta3_vec.float(),
            value_coef=self.value_coef,
            entropy_coef=self.entropy_coef,
            value_loss_type=self.cfg.train.value_loss_type,
            huber_delta=self.cfg.train.huber_delta,
        )
