from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn

from ..utils.profiling import profile


import wandb

from ..core.structured_config import Config
from ..env.hunl_env import HUNLEnv
from ..env.hunl_tensor_env import HUNLTensorEnv
from ..models.cnn import CardsPlanesV1, ActionsHUEncoderV1
from ..models.transformer.embedding_data import StructuredEmbeddingData
from ..models.cnn_embedding_data import CNNEmbeddingData
from ..encoding.action_mapping import bin_to_action, get_legal_mask
from ..models.factory import ModelFactory
from ..models.state_encoder import CNNStateEncoder
from ..models.cnn import SiameseConvNetV1
from ..rl.replay import (
    Transition,
    Trajectory,
)
from ..rl.vectorized_replay import VectorizedReplayBuffer
from ..rl.losses import trinal_clip_ppo_loss
from ..rl.k_best_pool import KBestOpponentPool, AgentSnapshot
from ..core.builders import build_components_from_config


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
        self.k_best_pool_size = cfg.k_best_pool_size
        self.min_elo_diff = cfg.min_elo_diff
        self.min_step_diff = cfg.min_step_diff
        self.k_factor = cfg.k_factor
        self.use_mixed_precision = (
            cfg.train.use_mixed_precision and self.device.type in ["cuda", "mps"]
        )
        self.loss_scale = cfg.train.loss_scale
        self.use_tensor_env = cfg.use_tensor_env
        self.num_envs = cfg.num_envs
        self.use_wandb = cfg.use_wandb
        self.wandb_project = cfg.wandb_project
        self.wandb_name = cfg.wandb_name
        self.wandb_tags = cfg.wandb_tags
        self.wandb_run_id = cfg.wandb_run_id

        self.float_dtype = torch.bfloat16 if self.use_mixed_precision else torch.float32

        # Initialize RNG
        self.rng = torch.Generator(device=self.device)

        # Initialize components
        # Always create tensor_env for state encoder, even if we don't use it for training
        self.tensor_env = HUNLTensorEnv(
            num_envs=self.num_envs,
            starting_stack=self.cfg.env.stack,
            sb=self.cfg.env.sb,
            bb=self.cfg.env.bb,
            bet_bins=self.cfg.env.bet_bins,
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
        ce, ae, model, policy = build_components_from_config(self.cfg)
        self.cards_encoder = ce
        self.actions_encoder = ae
        self.model = model
        self.policy = policy

        # Create state encoder based on model type
        is_transformer = cfg.model.name.startswith("poker_transformer")
        if is_transformer:
            # Use transformer state encoder
            self.state_encoder = ModelFactory.create_state_encoder(
                "transformer", self.device, tensor_env=self.tensor_env
            )
            # Always use structured embeddings for transformer models
            self.use_structured_embeddings = True
        else:
            # Use CNN state encoder
            self.state_encoder = CNNStateEncoder(self.tensor_env, self.device)
            self.use_structured_embeddings = False

        # Ensure bins align with model output size to avoid mask/logit mismatch
        if hasattr(self.model, "policy_head") and hasattr(
            self.model.policy_head, "out_features"
        ):
            self.num_bet_bins = int(self.model.policy_head.out_features)
        else:
            self.num_bet_bins = len(self.cfg.env.bet_bins) + 3

        self.model.to(self.device)  # Move model to device
        self._initialize_weights()  # Initialize with better weights
        # Replay buffer capacity in steps is batch_size * replay_buffer_batches
        # Add an extra batch so we can reserve space for the next batch.
        buffer_capacity = self.batch_size * max(1, 1 + self.replay_buffer_batches)

        # Use vectorized replay buffer for efficient tensor operations
        sequence_length = (
            self.state_encoder.get_sequence_length() if is_transformer else 50
        )
        self.replay_buffer = VectorizedReplayBuffer(
            capacity=buffer_capacity,  # Number of trajectories
            max_trajectory_length=self.max_trajectory_length,  # Maximum steps per trajectory
            num_bet_bins=self.num_bet_bins,  # Number of bet bins for actions tensor
            device=self.device,
            float_dtype=self.float_dtype,  # Use appropriate dtype for mixed precision
            is_transformer=is_transformer,  # Whether this buffer is for transformer models
            sequence_length=sequence_length,  # Sequence length for transformer models
        )

        # K-Best opponent pool
        self.opponent_pool = KBestOpponentPool(
            k=self.k_best_pool_size,
            min_elo_diff=self.min_elo_diff,
            min_step_diff=self.min_step_diff,
            k_factor=self.k_factor,
        )

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
        else:  # Transformer model (no cards_trunk)
            self.optimizer = torch.optim.Adam([{"params": other_params, "lr": lr}])

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
        max_steps = 50  # Safety limit to prevent infinite loops

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
                    outputs = opponent_snapshot.model(embedding_data)
                else:
                    outputs = self.model(embedding_data)
                logits = outputs["policy_logits"]
                value = outputs["value"]

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

    @profile
    def collect_tensor_trajectories(
        self,
        min_steps: int,
        all_opponent_snapshots: Optional[list[AgentSnapshot]] = None,
        add_to_replay_buffer: bool = True,
    ) -> tuple[float, int, torch.Tensor]:
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

        total_reward = 0.0
        trajectory_count = 0
        steps_collected = 0
        # Track individual episode rewards as list of tensors
        collected_trajectory_rewards = []

        # Initialize all environments
        self.tensor_env.reset()

        # Per-environment reward tracking and step counts
        per_env_rewards = torch.zeros(self.num_envs, device=self.device)
        per_traj_step_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        max_steps = 50  # Safety limit per trajectory

        # Outer loop: collect complete trajectories until we have enough steps
        loop_count = 0
        max_loops = 1000  # Safety limit to prevent infinite loops
        adding_trajectories = False

        while steps_collected < min_steps and loop_count < max_loops:
            if not adding_trajectories and add_to_replay_buffer:
                # Initialize trajectory collection; reserve space in buffer.
                self.replay_buffer.start_adding_trajectory_batches(self.num_envs)
                adding_trajectories = True

            loop_count += 1
            # print("loop", steps_collected, min_steps, f"(loop {loop_count})")
            # Only process non-done environments to avoid bias towards short episodes
            active_mask = ~self.tensor_env.done
            active_indices = torch.where(active_mask)[0]
            env_active_we_act = torch.where(
                active_mask & (self.tensor_env.to_act == 0)
            )[0]
            env_active_opp_acts = torch.where(
                active_mask & (self.tensor_env.to_act == 1)
            )[0]

            # Get legal action masks for active environments only
            legal_bins_amounts, legal_bins_mask = (
                self.tensor_env.legal_bins_amounts_and_mask()
            )
            active_legal_masks = legal_bins_mask[active_indices]  # [N_active, B]

            # Get model predictions for active environments only
            with torch.no_grad():
                # Determine which active environments need our model vs opponent models
                # Save actor mask BEFORE stepping; these are the envs where we act now
                original_to_act = self.tensor_env.to_act.clone()
                active_who_acts = original_to_act[active_indices]
                # these two are indices INTO THE ACTIVE ENV ARRAY only
                active_we_act = torch.where(active_who_acts == 0)[0]
                active_opp_acts = torch.where(active_who_acts == 1)[0]

                # Encode states from our perspective (player=0)
                our_states = self._encode_tensor_states(
                    player=0, idxs=env_active_we_act
                )

                active_first_action = self.tensor_env.acted_since_reset[active_indices]

                # Initialize logits and values tensors for active environments only
                active_logits = torch.zeros(
                    active_indices.numel(),
                    self.num_bet_bins,
                    dtype=self.float_dtype,
                    device=self.device,
                )
                active_values = torch.zeros(
                    active_indices.numel(), dtype=self.float_dtype, device=self.device
                )

                # Get predictions from our model for our turns
                if active_we_act.numel() > 0:
                    with torch.amp.autocast(
                        self.device.type,
                        dtype=torch.bfloat16,
                        enabled=self.use_mixed_precision,
                    ):
                        outputs = self.model(our_states)

                    active_logits[active_we_act] = outputs["policy_logits"]
                    active_values[active_we_act] = outputs["value"].squeeze(-1)

                # SPECIAL CASE: If first action of hand, opponent folding illegal
                # (because it would leave an empty trajectory)
                # This doesn't bias our sampling because we would never see the empty
                # trajectory if it were sampled anyway, since there is no decision of the
                # actor to train. We could in the future train only the value function.
                # NB that this will make our average reward look worse.
                if add_to_replay_buffer:
                    active_legal_masks[
                        (active_who_acts == 1) & active_first_action, 0
                    ] = False

                # Get predictions from opponent models for opponent turns
                opp_env_groups: Tuple[torch.Tensor, ...] | None = None
                if (
                    all_opponent_snapshots is not None
                    and len(all_opponent_snapshots) > 0
                    and env_active_opp_acts.numel() > 0
                ):
                    # Use opponent models - assign opponents to environments
                    num_opps = len(all_opponent_snapshots)
                    # Split active_opp_acts into num_opps approximately equal groups
                    opp_env_groups = torch.chunk(env_active_opp_acts, num_opps)
                    processed = 0
                    for opponent_idx, opp_env_indices in enumerate(opp_env_groups):
                        if opp_env_indices.numel() == 0:
                            continue
                        opponent = all_opponent_snapshots[opponent_idx]

                        opp_states = self._encode_tensor_states(
                            player=1, idxs=opp_env_indices
                        )

                        # Use indices directly into active arrays
                        with torch.amp.autocast(
                            self.device.type,
                            dtype=torch.bfloat16,
                            enabled=self.use_mixed_precision,
                        ):
                            outputs = opponent.model(opp_states)
                            opp_logits = outputs["policy_logits"]
                            opp_values = outputs["value"]

                        just_processed = opp_env_indices.numel()
                        active_logits[
                            active_opp_acts[processed : processed + just_processed]
                        ] = opp_logits
                        active_values[
                            active_opp_acts[processed : processed + just_processed]
                        ] = opp_values.squeeze(-1)
                        processed += just_processed
                elif active_opp_acts.numel() > 0:
                    opp_states = self._encode_tensor_states(
                        player=1, idxs=env_active_opp_acts
                    )
                    # Self-play: use our model for opponent turns too
                    with torch.amp.autocast(
                        self.device.type,
                        dtype=torch.bfloat16,
                        enabled=self.use_mixed_precision,
                    ):
                        outputs = self.model(opp_states)
                        opp_logits = outputs["policy_logits"]
                        opp_values = outputs["value"]

                    active_logits[active_opp_acts] = opp_logits
                    active_values[active_opp_acts] = opp_values.squeeze(-1)

                # Sample actions for active environments only
                action_values_active, log_probs_active = self.policy.action_batch(
                    active_logits, active_legal_masks
                )

            # Create full-size tensors for stepping (needed by tensor_env.step_bins)
            action_values = torch.full(
                (self.num_envs,), -1, dtype=torch.long, device=self.device
            )
            action_values[active_indices] = action_values_active

            # Take steps in all environments (tensor_env expects full-size tensors)
            # NOTE: THIS CHANGES self.tensor_env.to_act!!
            rewards, dones, _ = self.tensor_env.step_bins(
                action_values, legal_bins_amounts, legal_bins_mask
            )

            active_rewards = rewards[active_indices]
            active_dones = dones[active_indices]

            # Store transitions for our actions (don't add to replay buffer yet)
            if active_we_act.numel() > 0:
                # Scale factor for reward/targets: 100 big blinds
                scale = float(self.tensor_env.bb) * 100.0

                # Compute delta bounds from actor perspective AFTER step
                # For our acting envs, actor is player 0
                chips = self.tensor_env.chips_placed.to(self.float_dtype)
                # Convert active_we_act to full environment indices
                our_env_indices = active_indices[active_we_act]
                our_delta2_tensor = (
                    -chips[our_env_indices, 1] / scale
                )  # -opponent chips
                our_delta3_tensor = chips[our_env_indices, 0] / scale  # our chips

                # Extract tensor values efficiently (no .tolist() calls)
                our_action_indices = action_values_active[active_we_act]
                our_log_probs_tensor = log_probs_active[active_we_act]
                our_values_tensor = active_values[active_we_act]
                our_rewards_tensor = active_rewards[active_we_act]
                our_dones_tensor = active_dones[active_we_act]
                our_legal_masks_tensor = active_legal_masks[active_we_act]

                # Add transitions immediately using vectorized operations
                if add_to_replay_buffer:
                    self.replay_buffer.add_batch(
                        embedding_data=our_states,
                        action_indices=our_action_indices,
                        log_probs=our_log_probs_tensor,
                        rewards=our_rewards_tensor,
                        dones=our_dones_tensor,
                        legal_masks=our_legal_masks_tensor.bool(),
                        delta2=our_delta2_tensor,
                        delta3=our_delta3_tensor,
                        values=our_values_tensor,
                        trajectory_indices=our_env_indices,  # Use full environment indices
                    )

                # Update per-environment reward tracking
                per_env_rewards[our_env_indices] += our_rewards_tensor

            if active_opp_acts.numel() > 0:
                # Check if any opponent actions ended the hand
                opp_ended_hands_mask = active_mask & dones & (original_to_act == 1)
                opp_ended_hands = torch.where(opp_ended_hands_mask)[0]
                opp_rewards = rewards[opp_ended_hands]

                # For environments where opponent's action ended the hand,
                # update the last transition's reward using vectorized method
                if opp_ended_hands.numel() > 0 and add_to_replay_buffer:
                    # Use vectorized method to update rewards
                    self.replay_buffer.update_opponent_rewards(
                        opp_ended_hands, opp_rewards
                    )

                # Update per-environment reward tracking
                per_env_rewards[active_indices[active_opp_acts]] += active_rewards[
                    active_opp_acts
                ]

            # Update step counts for active environments
            per_traj_step_count[active_indices] += 1

            # Handle environments that exceed max steps
            over_limit = (per_traj_step_count >= max_steps) & (~self.tensor_env.done)
            if over_limit.any():
                env_indices = torch.where(over_limit)[0].tolist()
                print(
                    f"Warning: Environments {env_indices} reached max steps ({max_steps}), forcing termination"
                )
                # Mark them as done
                self.tensor_env.done[env_indices] = True

            # Create trajectories for done environments
            newly_done_indices = torch.where(dones & active_mask)[0]

            if newly_done_indices.numel() > 0:
                trajectory_count += newly_done_indices.numel()

            # Trajectories are now added immediately via vectorized operations
            # No need for separate trajectory completion logic

            # Update ELO ratings for done environments (vectorized by opponent)
            # Use premade chunking to efficiently group done environments by opponent
            if opp_env_groups is not None:
                # Vectorized ELO update per opponent using existing chunking
                for opponent_idx, opp_env_indices in enumerate(opp_env_groups):
                    if opp_env_indices.numel() == 0:
                        continue

                    opponent = all_opponent_snapshots[opponent_idx]

                    # opp_env_indices are already full environment indices from env_active_opp_acts
                    # Find which environments in this opponent's group are done
                    done_mask = dones[opp_env_indices]
                    done_env_idxs_for_opponent = opp_env_indices[done_mask]

                    if done_env_idxs_for_opponent.numel() > 0:
                        # Slice rewards tensor directly - no .item() calls needed
                        opponent_rewards = per_env_rewards[done_env_idxs_for_opponent]

                        # Update ELO for this opponent with all their games
                        self.opponent_pool.update_elo_batch_vectorized(
                            opponent, opponent_rewards
                        )
            else:
                # Fallback for self-play case (no opponents, so no ELO tracking)
                pass

            # If all environments are done, reset all of them
            if dones.all():
                if add_to_replay_buffer:
                    _, steps_added = (
                        self.replay_buffer.finish_adding_trajectory_batches()
                    )
                    steps_collected += steps_added
                else:
                    # For evaluation, just count episodes without adding to buffer
                    steps_collected += per_traj_step_count.sum().item()

                # Collect individual episode rewards for accurate win counting
                collected_trajectory_rewards.append(per_env_rewards.clone())
                per_env_rewards[:] = 0.0
                per_traj_step_count[:] = 0
                adding_trajectories = False

                # Reset environments if we're going through the loop again
                if steps_collected < min_steps:
                    self.tensor_env.reset()

        if loop_count >= max_loops:
            print(
                f"Warning: Reached maximum loop count ({max_loops}), stopping collection early"
            )
            print(f"Collected {steps_collected} steps out of {min_steps} requested")

        if self.tensor_env.done.any().item():
            collected_trajectory_rewards.append(per_env_rewards[self.tensor_env.done])

        # Concatenate all episode rewards into a single tensor
        if collected_trajectory_rewards:
            collected_trajectory_rewards_tensor = torch.cat(
                collected_trajectory_rewards, dim=0
            )
        else:
            collected_trajectory_rewards_tensor = torch.tensor([], device=self.device)
        total_reward = collected_trajectory_rewards_tensor.sum().item()

        return total_reward, trajectory_count, collected_trajectory_rewards_tensor

    def _encode_tensor_states(self, player: int, idxs: torch.Tensor) -> dict:
        """
        Encode states for all tensor_environments in the tensorized environment.

        Returns:
            Dictionary with 'cards' and 'actions' tensors for CNN models,
            or structured embedding components for transformer models
        """
        return self.state_encoder.encode_tensor_states(player=player, idxs=idxs)

    @profile
    def update_model(self) -> dict:
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

            if is_transformer:
                # Transformer model with structured embeddings
                with torch.amp.autocast(
                    self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.use_mixed_precision,
                ):
                    # Create structured data from batch
                    batch_structured_data = StructuredEmbeddingData.from_dict(batch)
                    outputs = self.model(batch_structured_data)
                    logits = outputs["policy_logits"]
                    values = outputs["value"]

                # Transformer-specific loss with auxiliary hand range loss
                loss_dict = self._compute_transformer_loss(
                    outputs, batch, delta2_vec, delta3_vec
                )
            else:
                # CNN model
                cards_features = batch[
                    "cards_features"
                ]  # [batch_size, 6, 4, 13] - bool
                actions_features = batch[
                    "actions_features"
                ]  # [batch_size, 24, 4, num_bet_bins] - bool

                # Convert bool tensors to appropriate dtype for model forward pass
                cards_float = cards_features.to(torch.float32)
                actions_float = actions_features.to(torch.float32)

                with torch.amp.autocast(
                    self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.use_mixed_precision,
                ):
                    # Create CNNEmbeddingData
                    embedding_data = CNNEmbeddingData(
                        cards=cards_float, actions=actions_float
                    )
                    outputs = self.model(embedding_data)
                    logits = outputs["policy_logits"]
                    values = outputs["value"]

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

            # Debugging metrics: approx KL, clipfrac, explained variance
            with torch.no_grad():
                legal_mb = batch["legal_masks"]
                masked_logits = torch.where(legal_mb.bool(), logits, -1e9)
                log_probs_new = torch.log_softmax(masked_logits, dim=-1)
                a_mb = batch["action_indices"]
                logp_new = log_probs_new.gather(1, a_mb.unsqueeze(1)).squeeze(1)
                logp_old = batch["log_probs_old"]
                ratio = torch.exp(logp_new - logp_old)
                approx_kl = (logp_old - logp_new).mean()
                clip_low, clip_high = 1.0 - self.epsilon, 1.0 + self.epsilon
                clipped = torch.clamp(ratio, clip_low, clip_high)
                clipfrac = (torch.abs(clipped - ratio) > 1e-8).float().mean()

                # Explained variance calculation using clipped returns (same as loss)
                ret_mb = batch["returns"].float()
                d2_mb = delta2_vec.float()
                d3_mb = delta3_vec.float()
                # Ensure per-sample bounds are ordered to avoid clamp warnings
                min_b = torch.minimum(d2_mb, d3_mb)
                max_b = torch.maximum(d2_mb, d3_mb)
                ret_clipped_mb = torch.clamp(ret_mb, min=min_b, max=max_b)
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
            "avg_loss": total_loss / self.num_epochs,
            "avg_reward": avg_reward,
            "num_samples": batch["action_indices"].shape[0],
            "delta2_mean": float(delta2_vec.mean().item()),
            "delta3_mean": float(delta3_vec.mean().item()),
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

        # Trim buffer back to target_steps
        self.replay_buffer.trim_to_steps(target_steps)

        # Update model
        update_stats = self.update_model()

        # Check if we should add current model to opponent pool
        if self.opponent_pool.should_add_snapshot(step):
            self.opponent_pool.add_snapshot(self.model, step)

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
                },
                step=step,
            )

        return training_stats

    @profile
    def _fill_replay_buffer(self, min_steps: int) -> None:
        """
        Fill replay buffer with at least min_steps samples using either tensor or scalar envs.

        Args:
            min_steps: Minimum number of steps to add to replay buffer
        """
        if self.use_tensor_env:
            total_reward, trajectory_count, _ = self.collect_tensor_trajectories(
                min_steps,
                all_opponent_snapshots=self.opponent_pool.snapshots,
            )

            self.total_step_reward += total_reward
            self.step_trajectories_collected += trajectory_count
            self.total_trajectories_collected += trajectory_count
        else:
            steps_added = 0
            while steps_added < min_steps:
                # Use scalar collection
                sampled_opponent = self.opponent_pool.sample(k=1)
                opponent = sampled_opponent[0] if sampled_opponent else None
                trajectory, reward = self.collect_trajectory(opponent)

                if len(trajectory.transitions) > 0:
                    self.replay_buffer.add_trajectory_legacy(trajectory)
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
            "k": self.opponent_pool.k,
            "min_elo_diff": self.opponent_pool.min_elo_diff,
            "current_elo": self.opponent_pool.current_elo,
            "snapshots": [],
        }
        for snapshot in self.opponent_pool.snapshots:
            snapshot_data = {
                "step": snapshot.step,
                "elo": snapshot.elo,
                "games_played": snapshot.games_played,
                "wins": snapshot.wins,
                "losses": snapshot.losses,
                "draws": snapshot.draws,
                "model_state_dict": snapshot.model.state_dict(),
            }
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
        import os
        import glob

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
            self.opponent_pool.k = pool_data.get("k", self.opponent_pool.k)
            self.opponent_pool.min_elo_diff = pool_data.get(
                "min_elo_diff", self.opponent_pool.min_elo_diff
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
                snapshot = AgentSnapshot(
                    model=model,
                    step=snapshot_data.get("step", 0),
                    elo=snapshot_data.get("elo", 1200.0),
                )
                snapshot.games_played = snapshot_data.get("games_played", 0)
                snapshot.wins = snapshot_data.get("wins", 0)
                snapshot.losses = snapshot_data.get("losses", 0)
                snapshot.draws = snapshot_data.get("draws", 0)
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
            for i, opponent in enumerate(self.opponent_pool.snapshots):
                wins = 0

                if self.use_tensor_env:
                    # Use tensorized evaluation with individual episode reward tracking
                    _, _, individual_rewards = self.collect_tensor_trajectories(
                        min_steps=min_games,
                        all_opponent_snapshots=[opponent],
                        add_to_replay_buffer=False,  # Don't pollute training data
                    )

                    # Count wins based on individual episode rewards (tensor operations)
                    wins = (individual_rewards > 0).sum().item()
                    games = individual_rewards.numel()
                else:
                    # Use scalar evaluation
                    for _ in range(min_games):
                        _, final_reward = self.collect_trajectory(
                            opponent_snapshot=opponent
                        )
                        if final_reward > 0:
                            wins += 1
                    games = min_games

                win_rate = wins / games
                results[f"opponent_{i}_step_{opponent.step}"] = {
                    "win_rate": win_rate,
                    "opponent_elo": opponent.elo,
                    "wins": wins,
                    "total_games": min_games,
                }
                total_wins += wins
                total_games += games

        overall_win_rate = total_wins / total_games if total_games > 0 else 0.0

        return {
            "overall_win_rate": overall_win_rate,
            "total_games": total_games,
            "opponent_results": results,
            "pool_stats": self.opponent_pool.get_pool_stats(),
        }

    def get_preflop_range_grid(self, seat: int = 0, metric: str = "allin") -> str:
        """Get preflop range as a grid showing selected action probabilities for button play.

        metric: "allin" or "fold"
        """
        # Create a 13x13 grid representing all possible hole card combinations
        # Rows/cols: A, K, Q, J, T, 9, 8, 7, 6, 5, 4, 3, 2
        ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]

        # Initialize grid with action probabilities
        grid = []
        header = "    " + " ".join(f"{rank:>2}" for rank in ranks)
        grid.append(header)
        grid.append("   " + "-" * 39)  # Separator line

        for i, rank1 in enumerate(ranks):
            row = [f"{rank1:>2} |"]
            for j, rank2 in enumerate(ranks):
                if i == j:
                    # Same rank (pairs) - always suited
                    card1, card2 = f"{rank1}s", f"{rank1}h"  # Suited pair
                elif i < j:
                    # Top-right triangle: suited hands (e.g., AKs, AQs)
                    card1, card2 = f"{rank1}s", f"{rank2}h"  # Suited
                else:
                    # Bottom-left triangle: off-suit hands (e.g., KAs, QAs)
                    card1, card2 = f"{rank2}s", f"{rank1}d"  # Off-suit

                # Get action probability for this hand
                prob = self._get_preflop_action_probability(card1, card2, seat, metric)
                row.append(f"{prob:>2}")

            grid.append(" ".join(row))

        return "\n".join(grid)

    def get_preflop_value_grid(self, seat: int = 0) -> str:
        """Get preflop value estimates as a grid showing value estimates for button play."""
        # Create a 13x13 grid representing all possible hole card combinations
        # Rows/cols: A, K, Q, J, T, 9, 8, 7, 6, 5, 4, 3, 2
        ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]

        # Initialize grid with value estimates
        grid = []
        header = "    " + " ".join(f"{rank:>2}" for rank in ranks)
        grid.append(header)
        grid.append("   " + "-" * 39)  # Separator line

        for i, rank1 in enumerate(ranks):
            row = [f"{rank1:>2} |"]
            for j, rank2 in enumerate(ranks):
                if i == j:
                    # Same rank (pairs) - always suited
                    card1, card2 = f"{rank1}s", f"{rank1}h"  # Suited pair
                elif i < j:
                    # Top-right triangle: suited hands (e.g., AKs, AQs)
                    card1, card2 = f"{rank1}s", f"{rank2}h"  # Suited
                else:
                    # Bottom-left triangle: off-suit hands (e.g., KAs, QAs)
                    card1, card2 = f"{rank2}s", f"{rank1}d"  # Off-suit

                # Get value estimate for this hand
                _, value = self._get_preflop_action_probability_and_value(
                    card1, card2, seat, "call"
                )

                # Format value estimate (multiply by 100 for readability, round to 1 decimal)
                formatted_value = f"{value:5.2f}"
                row.append(formatted_value)

            grid.append(" ".join(row))

        return "\n".join(grid)

    def _get_preflop_action_probability(
        self, card1: str, card2: str, seat: int, metric: str
    ) -> str:
        """Get action probability for a specific hole card combination using tensor environment.

        metric: "allin" or "fold"
        """
        prob, _ = self._get_preflop_action_probability_and_value(
            card1, card2, seat, metric
        )
        return prob

    def _get_preflop_action_probability_and_value(
        self, card1: str, card2: str, seat: int, metric: str
    ) -> tuple[str, float]:
        """Get action probability and value estimate for a specific hole card combination using tensor environment.

        Returns:
            tuple: (formatted_probability_string, value_estimate)
        """
        # Create a temporary tensor environment for this analysis
        temp_env = HUNLTensorEnv(
            num_envs=1,
            starting_stack=self.cfg.env.stack,
            sb=self.cfg.env.sb,
            bb=self.cfg.env.bb,
            bet_bins=self.cfg.env.bet_bins,
            device=self.device,
            rng=self.rng,
            flop_showdown=getattr(self.cfg.env, "flop_showdown", False),
        )

        # Reset and set up the specific hand
        temp_env.reset()

        # Set the button to the seat we're analyzing
        temp_env.button[0] = seat
        temp_env.to_act[0] = seat

        # Set hole cards for the seat we're analyzing
        card1_int = self._card_str_to_int(card1)
        card2_int = self._card_str_to_int(card2)

        # Convert cards to one-hot representation
        suit1, rank1 = card1_int // 13, card1_int % 13
        suit2, rank2 = card2_int // 13, card2_int % 13

        # Set hole cards in the tensor environment
        temp_env.hole_onehot[0, 0, 0, suit1, rank1] = True
        temp_env.hole_onehot[0, 0, 1, suit2, rank2] = True

        # Set hole card indices
        temp_env.hole_indices[0, 0, 0] = card1_int
        temp_env.hole_indices[0, 0, 1] = card2_int

        # Get model prediction using tensor environment
        with torch.no_grad():
            if self.use_structured_embeddings:
                # For transformer models, use the transformer state encoder
                # Create a temporary transformer state encoder for this analysis
                from ..models.transformer.state_encoder import TransformerStateEncoder

                temp_state_encoder = TransformerStateEncoder(temp_env, self.device)

                # Encode the state using the transformer state encoder
                structured_data = temp_state_encoder.encode_tensor_states(
                    seat, torch.tensor([0], device=self.device)
                )
                outputs = self.model(structured_data)
            else:
                # For CNN models, use the existing state encoder
                embedding_data = self.state_encoder.encode_tensor_states(
                    seat, torch.tensor([0], device=self.device)
                )
                outputs = self.model(embedding_data)

            logits = outputs["policy_logits"]
            value = outputs["value"].item()  # Get value estimate

            # Get legal actions from tensor environment
            legal_mask = temp_env.legal_bins_mask()[0]  # [num_bet_bins]

            # Apply legal mask
            masked_logits = torch.where(
                legal_mask == 0, torch.full_like(logits, -1e9), logits
            )

            # Get probabilities
            probs = torch.softmax(masked_logits, dim=-1).squeeze(0)

            # Select metric index
            if metric == "allin":
                idx = self.num_bet_bins - 1  # Last bin is all-in
            elif metric == "call":
                idx = 1  # Second bin is call
            elif metric == "fold":
                idx = 0  # First bin is fold
            else:
                idx = self.num_bet_bins - 1

            selected_prob = probs[idx].item()

            # Convert to percentage and format
            percentage = round(selected_prob * 100)
            if percentage >= 100:
                prob_str = "██"  # Two filled boxes to keep 2-char width
            else:
                prob_str = f"{percentage:2d}"

            return prob_str, value

    def _card_str_to_int(self, card_str: str) -> int:
        """Convert card string (e.g., 'As', 'Kh') to integer representation."""
        rank_map = {
            "A": 12,
            "K": 11,
            "Q": 10,
            "J": 9,
            "T": 8,
            "9": 7,
            "8": 6,
            "7": 5,
            "6": 4,
            "5": 3,
            "4": 2,
            "3": 1,
            "2": 0,
        }
        suit_map = {"s": 0, "h": 1, "d": 2, "c": 3}

        rank = card_str[0]
        suit = card_str[1]

        return rank_map[rank] * 4 + suit_map[suit]

    def _compute_transformer_loss(
        self,
        outputs: dict,
        batch: dict,
        delta2_vec: torch.Tensor,
        delta3_vec: torch.Tensor,
    ) -> dict:
        """Compute transformer-specific loss with auxiliary hand range loss."""
        import torch.nn.functional as F

        logits = outputs["policy_logits"]
        values = outputs["value"]

        # Standard PPO policy loss
        policy_loss = trinal_clip_ppo_loss(
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

        # Add auxiliary hand range loss if available
        aux_loss = 0.0
        if "hand_range_logits" in outputs and hasattr(
            self.cfg.train, "auxiliary_loss_coef"
        ):
            # For now, we don't have hand range targets, so skip auxiliary loss
            # In the future, this could be implemented with opponent hand range prediction
            aux_loss = 0.0

        total_loss = policy_loss["total_loss"] + aux_loss

        return {
            "policy_loss": policy_loss["policy_loss"],
            "value_loss": policy_loss["value_loss"],
            "entropy": policy_loss["entropy"],
            "aux_loss": aux_loss,
            "total_loss": total_loss,
        }
