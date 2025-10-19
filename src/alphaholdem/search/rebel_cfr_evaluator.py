from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Optional, ContextManager
import os
import time

import torch
import torch.nn.functional as F

from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.env.card_utils import (
    combo_blocking_tensor,
    combo_to_onehot_tensor,
    hand_combos_tensor,
)
from alphaholdem.env.rebel_feature_encoder import RebelFeatureEncoder
from alphaholdem.env.rules import rank_hands
from alphaholdem.models.mlp.rebel_ffn import NUM_HANDS, RebelFFN
from alphaholdem.rl.rebel_replay import RebelBatch
from alphaholdem.utils.model_utils import compute_masked_logits
from alphaholdem.utils.profiling import profile

T_WARM = 15


@dataclass
class PublicBeliefState:
    """Public belief state for both players."""

    env: HUNLTensorEnv
    beliefs: torch.Tensor  # [batch_size, num_players, NUM_HANDS]

    @classmethod
    def from_proto(
        cls,
        env_proto: HUNLTensorEnv,
        beliefs: torch.Tensor,
        num_envs: Optional[int] = None,
    ) -> PublicBeliefState:
        """Create a new belief state with an environment cloned from `env_proto`.

        Args:
            env_proto: Template environment whose configuration should be reused.
            beliefs: Initial belief tensor shaped `[batch, players, NUM_HANDS]`.
            num_envs: Optional override for the number of vectorised environments.
        """
        return PublicBeliefState(
            env=HUNLTensorEnv.from_proto(env_proto, num_envs=num_envs),
            beliefs=beliefs,
        )

    def __post_init__(self):
        assert self.beliefs.shape[0] == self.env.N


@dataclass
class HandRankData:
    sorted_indices: torch.Tensor
    inv_sorted: torch.Tensor
    H: torch.Tensor
    card_ok: torch.Tensor
    hand_ok_mask: torch.Tensor
    hand_ok_mask_sorted: torch.Tensor
    hands_c1c2_sorted: torch.Tensor
    L_idx: torch.Tensor
    R_idx: torch.Tensor


class RebelCFREvaluator:
    """ReBeL CFR Evaluator implementing the precise SELFPLAY algorithm."""

    search_batch_size: int
    model: RebelFFN
    max_depth: int
    bet_bins: list[float]
    cfr_iterations: int
    warm_start_iterations: int
    sample_epsilon: float
    device: torch.device
    float_dtype: torch.dtype
    generator: Optional[torch.Generator]
    num_players: int
    num_actions: int
    all_hands: torch.Tensor
    depth_offsets: list[int]
    total_nodes: int
    env: HUNLTensorEnv
    valid_mask: torch.Tensor
    leaf_mask: torch.Tensor
    policy_probs: torch.Tensor
    policy_probs_avg: torch.Tensor
    cumulative_regrets: torch.Tensor
    values: torch.Tensor
    beliefs: torch.Tensor
    legal_mask: Optional[torch.Tensor]
    feature_encoder: RebelFeatureEncoder
    hand_rank_data: Optional[HandRankData]

    def __init__(
        self,
        search_batch_size: int,
        env_proto: HUNLTensorEnv,
        model: RebelFFN,
        bet_bins: list[float],
        max_depth: int,
        cfr_iterations: int,
        device: torch.device,
        float_dtype: torch.dtype,
        generator: Optional[torch.Generator] = None,
        warm_start_iterations: int = T_WARM,
        sample_epsilon: float = 0.25,
    ):
        assert cfr_iterations > warm_start_iterations

        self.search_batch_size = search_batch_size
        self.model = model
        self.max_depth = max_depth
        self.bet_bins = bet_bins
        self.cfr_iterations = cfr_iterations
        self.warm_start_iterations = warm_start_iterations
        self.sample_epsilon = sample_epsilon
        self.device = device
        self.float_dtype = float_dtype
        self.generator = generator

        self.num_players = 2
        self.num_actions = len(bet_bins) + 3
        self.all_hands = torch.arange(NUM_HANDS, device=self.device)

        # Compute depth offsets: slice i holds nodes at depth i
        self.depth_offsets: list[int] = [0]
        nodes_at_depth = self.search_batch_size
        for _ in range(self.max_depth + 1):
            self.depth_offsets.append(self.depth_offsets[-1] + nodes_at_depth)
            nodes_at_depth *= self.num_actions
        self.total_nodes = self.depth_offsets[-1]

        # Subgame environment
        self.env = HUNLTensorEnv.from_proto(
            env_proto,
            num_envs=self.total_nodes,
        )
        self.valid_mask = torch.zeros(
            self.total_nodes, dtype=torch.bool, device=self.device
        )
        # If finished, don't continue searching below this node.
        # Different from env.done, which is set when the node is terminal.
        # Should be a subset of valid_mask.
        self.leaf_mask = torch.zeros(
            self.total_nodes, dtype=torch.bool, device=self.device
        )

        # Notionally, at its parent, what is the probability of acting to get to this node?
        self.policy_probs = torch.zeros(
            self.total_nodes,
            NUM_HANDS,
            device=self.device,
            dtype=self.float_dtype,
        )
        self.policy_probs_avg = torch.zeros_like(self.policy_probs)
        # Cumulative regret of taking this node vs the best at the parent node.
        self.cumulative_regrets = torch.zeros_like(self.policy_probs)

        # One value per node per player per hand
        self.values = torch.zeros(
            self.total_nodes,
            self.num_players,
            NUM_HANDS,
            device=self.device,
            dtype=self.float_dtype,
        )
        # One belief per node per player per hand
        # NB: beliefs[k, 0] is the belief ABOUT the acting player.
        # Not the belief OF the acting player.
        self.beliefs = torch.zeros(
            self.total_nodes,
            self.num_players,
            NUM_HANDS,
            device=self.device,
            dtype=self.float_dtype,
        )

        self.legal_mask = None

        self.combo_onehot_float = combo_to_onehot_tensor(device=self.device).float()

        # Feature encoder for belief computation
        self.feature_encoder = RebelFeatureEncoder(
            env=self.env,
            device=device,
            dtype=float_dtype,
        )

        self.hand_rank_data = None

        # PyTorch profiler setup
        self.profiler_enabled = False
        self.profiler = None

    def enable_profiler(self, output_dir: str = "profiler_logs") -> None:
        """Enable PyTorch profiler with stack traces."""
        self.profiler_enabled = True
        self.profiler_output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Create profiler with stack traces and TensorBoard support
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        self.profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,  # Enable stack traces
            with_flops=True,
            with_modules=True,
        )
        self.profiler.start()

    def disable_profiler(self) -> None:
        """Disable PyTorch profiler."""
        self.profiler_enabled = False
        if self.profiler is not None:
            self.profiler.stop()
            self.profiler = None

    def profiler_step(self) -> None:
        """Step the profiler if enabled."""
        if self.profiler_enabled and self.profiler is not None:
            self.profiler.step()

    @profile
    def initialize_search(
        self,
        src_env: HUNLTensorEnv,
        src_indices: torch.Tensor,
        initial_beliefs: torch.Tensor | None = None,
    ) -> None:
        """Copy root states into the search tree and reset per-node buffers.

        Args:
            src_env: Batched environment that holds the source root public states.
            src_indices: Row indices inside `src_env` to copy into the tree roots.
            initial_beliefs: Optional belief tensor aligned with `src_indices`.
        """
        assert src_indices.shape[0] == self.search_batch_size
        assert src_indices.min().item() >= 0
        if initial_beliefs is None:
            initial_beliefs = torch.full(
                (self.search_batch_size, self.num_players, NUM_HANDS),
                1.0 / NUM_HANDS,
                dtype=self.float_dtype,
                device=self.device,
            )
        else:
            assert initial_beliefs.shape[0] == src_indices.shape[0]
            initial_beliefs = initial_beliefs.to(
                device=self.device, dtype=self.float_dtype
            )

        N = self.search_batch_size

        dest_indices = torch.arange(N, device=self.device)
        self.env.reset()
        self.env.copy_state_from(src_env, src_indices, dest_indices, copy_deck=True)
        self.valid_mask.zero_()
        self.valid_mask[:N] = True
        self.leaf_mask.zero_()
        self.leaf_mask[:N] = self.env.done[:N]
        self.policy_probs.zero_()
        self.policy_probs_avg.zero_()
        self.cumulative_regrets.zero_()
        self.values.zero_()
        self.beliefs.zero_()
        self.beliefs[:N] = initial_beliefs
        self.legal_mask = None
        self.hand_rank_data = None

    @profile
    def construct_subgame(self) -> None:
        """Expand the tree by cloning legal successor states at each depth."""
        N, M, B = self.search_batch_size, self.total_nodes, self.num_actions

        for depth in range(self.max_depth):
            bin_amounts, legal_masks = self.env.legal_bins_amounts_and_mask()
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]

            action_bins = torch.full((M,), -1, dtype=torch.long, device=self.device)
            # don't currently have a way to get a subset of the masks
            for action in range(self.num_actions):
                current_legal_mask = (
                    legal_masks[offset:offset_next, action]
                    & self.valid_mask[offset:offset_next]
                    & ~self.env.done[offset:offset_next]
                )
                current_legal_indices = torch.where(current_legal_mask)[0] + offset
                if current_legal_indices.numel() == 0:
                    continue

                new_legal_indices = (
                    offset_next + (current_legal_indices - offset) * B + action
                )
                assert new_legal_indices.max().item() < M

                self.env.copy_state_from(
                    self.env, current_legal_indices, new_legal_indices
                )
                self.valid_mask[new_legal_indices] = True

                action_bins[new_legal_indices] = action

            # TODO: To stop on street, capture new_streets here.
            rewards, _, _ = self.env.step_bins(
                action_bins, bin_amounts=bin_amounts, legal_masks=legal_masks
            )

            # Showdown values get set in set_leaf_values.
            finished_folded = self.valid_mask & (action_bins == 0) & self.env.done
            self.values[finished_folded, 0] = rewards[finished_folded].view(-1, 1)
            self.values[finished_folded, 1] = -rewards[finished_folded].view(-1, 1)

        leaf_start = self.depth_offsets[self.max_depth]
        leaf_end = self.depth_offsets[self.max_depth + 1]
        self.leaf_mask[leaf_start:leaf_end] = self.valid_mask[leaf_start:leaf_end]
        self.leaf_mask |= self.valid_mask & self.env.done

        self.legal_mask = self.env.legal_bins_mask()
        valid_legal_masks = self.legal_mask[self.valid_mask & ~self.leaf_mask]
        has_legal = valid_legal_masks.any(dim=-1)
        assert has_legal.all(), "Every valid node must have at least one legal action."

        assert self.values[self.valid_mask].abs().max() <= 5

    @torch.no_grad()
    @profile
    def _get_model_policy_probs(self, indices: torch.Tensor) -> torch.Tensor:
        features = self.encode_current_states()
        model_output = self.model(features[indices])
        logits = model_output.policy_logits
        legal_masks = self.legal_mask[indices]
        masked_logits = compute_masked_logits(logits, legal_masks[:, None, :])
        return F.softmax(masked_logits, dim=-1)

    @profile
    def _propagate_all_beliefs(self) -> None:
        """Propagate beliefs from all valid nodes to all valid nodes."""

        bottom = self.depth_offsets[1]
        top = self.depth_offsets[-2]

        # beliefs: [M, 2, 1326]. expand by a factor of B
        beliefs_expanded = self.beliefs[:top, None].expand(-1, self.num_actions, -1, -1)
        beliefs_copied = beliefs_expanded.reshape(-1, self.num_players, NUM_HANDS)

        self.beliefs[bottom:] = beliefs_copied
        former_actor = 1 - self.env.to_act[bottom:]
        src_indices = former_actor[:, None, None].expand(-1, -1, NUM_HANDS)
        src = self.beliefs[bottom:].gather(1, src_indices)
        src *= self.policy_probs[bottom:, None]
        self.beliefs[bottom:].scatter_(1, src_indices, src)

    def _propagate_beliefs(
        self,
        current_legal_indices: torch.Tensor,
        next_legal_indices: torch.Tensor,
    ) -> None:
        """Propagate beliefs from current legal indices to next legal indices."""
        legal_actor = self.env.to_act[current_legal_indices]
        self.beliefs[next_legal_indices, legal_actor] = (
            self.beliefs[current_legal_indices, legal_actor]
            * self.policy_probs[next_legal_indices]
        )
        self.beliefs[next_legal_indices, 1 - legal_actor] = self.beliefs[
            current_legal_indices, 1 - legal_actor
        ]

    @profile
    def _block_beliefs(self) -> None:
        """Block beliefs based on the board."""
        combo_onehot = combo_to_onehot_tensor(device=self.device).float()
        board_onehot = self.env.board_onehot.any(dim=1).view(-1, 52).float()
        # [N, 52] @ [52, 1326]
        blocked = (board_onehot @ combo_onehot.T).clamp(0, 1)
        self.beliefs.masked_fill_(
            ~self.valid_mask[:, None, None] | (blocked[:, None, :] > 0.5), 0.0
        )

    @profile
    def _normalize_beliefs(self) -> None:
        """Normalize beliefs across hands in-place for valid nodes."""
        denom = self.beliefs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        # If the action probability of getting to a node is 0, our
        # bayesian update will make the beliefs in that state all 0.
        # So we set them to uniform.
        self.beliefs = torch.where(denom > 1e-9, self.beliefs / denom, 1.0 / NUM_HANDS)
        self.beliefs.masked_fill_(~self.valid_mask[:, None, None], 0.0)

    @profile
    def _block_and_normalize_beliefs(self) -> None:
        # A little inefficient, but normalize twice to handle the case where
        # the action probability of getting to a node is 0 (restore uniform beliefs
        # in the first normalize and then block/normalize again).
        self._normalize_beliefs()
        self._block_beliefs()
        self._normalize_beliefs()

    @torch.no_grad()
    @profile
    def initialize_policy_and_beliefs(self) -> None:
        """Push public beliefs down the tree using the freshly initialised policy."""
        self.policy_probs.zero_()
        self.policy_probs_avg.zero_()
        self.beliefs[self.search_batch_size :].zero_()

        top = self.depth_offsets[-2]
        bottom = self.depth_offsets[1]
        current_indices = torch.arange(top, device=self.device)
        probs = self._get_model_policy_probs(current_indices)
        probs_by_dest = self._push_down(probs.permute(0, 2, 1))
        self.policy_probs[bottom:] = probs_by_dest
        self.policy_probs.masked_fill_(~self.valid_mask[:, None], 0.0)
        self.policy_probs_avg[:] = self.policy_probs

        self._propagate_all_beliefs()
        self._block_and_normalize_beliefs()

    @profile
    def warm_start(self) -> None:
        N, B = self.search_batch_size, self.num_actions
        min_value = torch.finfo(self.float_dtype).min

        # [M, ]
        temp_values = torch.where(self.leaf_mask[:, None, None], self.values, 0.0)

        all_actions = torch.arange(B, device=self.device)[None, :]
        assert (self.values[~self.valid_mask] == 0.0).all()
        for depth, current_indices in self._valid_nodes(bottom_up=True):
            if depth == self.max_depth:
                continue

            offset_next = self.depth_offsets[depth + 1]
            offset = self.depth_offsets[depth]
            next_indices = (
                offset_next + (current_indices[:, None] - offset) * B + all_actions
            )
            # [n]
            actor = self.env.to_act[current_indices]
            opp = 1 - actor
            # [n, B, 1326]
            next_actor_values = temp_values[next_indices, actor[:, None]]
            next_opp_values = temp_values[next_indices, opp[:, None]]

            # [n, B] - dot product over ranges of hand values
            actor_action_values = (
                self.policy_probs[next_indices] * next_actor_values
            ).sum(dim=-1)
            opp_action_values_all = self.policy_probs[next_indices] * next_opp_values
            opp_action_values = opp_action_values_all.sum(dim=-1)

            # take the player-to-act's best action, and the other player's average action
            actor_action_values.masked_fill_(
                ~self.legal_mask[current_indices], min_value
            )
            opp_action_values.masked_fill_(~self.legal_mask[current_indices], 0)
            all_indices = torch.arange(current_indices.numel(), device=self.device)
            temp_values[current_indices, actor] = next_actor_values[
                all_indices, actor_action_values.argmax(dim=-1)
            ]
            temp_values[current_indices, opp] = opp_action_values_all.sum(dim=1)

        assert temp_values[self.valid_mask].isfinite().all()

        # heuristic: scale regrets by the number of warm start iterations
        regrets = self.compute_instantaneous_regrets(temp_values)
        self.cumulative_regrets += self.warm_start_iterations * regrets
        self.update_policy()
        self.policy_probs_avg[:] = self.policy_probs

    @torch.no_grad()
    @profile
    def set_leaf_values(self) -> None:
        """Populate cached per-hand payoffs for nodes marked as leaves."""

        # Set estimated leaf value from model for non-terminal nodes.
        model_mask = self.leaf_mask & ~self.env.done

        features = self.encode_current_states()
        model_output = self.model(features[model_mask])
        self.values[model_mask] = model_output.hand_values

        # Fold values were set in construct_subgame and don't need updating.
        # Showdown values need to be updated based on beliefs.
        # The env has hands it uses for showdown, but those are fake.
        showdown = torch.where(self.env.street == 4)[0]
        # assert self.env.done[showdown].all()
        # assert self.leaf_mask[showdown].all()
        showdown_values = self._showdown_value(showdown)
        self.values[showdown, 0] = showdown_values
        self.values[showdown, 1] = -showdown_values

    @profile
    def compute_expected_values(self) -> torch.Tensor:
        """Back up leaf hand values to their ancestors under the current policy."""

        new_values = self.values.clone()
        new_values.masked_fill_(~self.leaf_mask[:, None, None], 0.0)
        # First iteration: leaf values already populated; back propagate expectations
        for depth in range(self.max_depth - 1, -1, -1):
            self.profiler_step()  # Profile each depth iteration

            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]
            offset_next_next = self.depth_offsets[depth + 2]

            # pull back values to the source nodes
            values_weighted = (
                new_values[offset_next:offset_next_next]
                * self.policy_probs[offset_next:offset_next_next, None, :]
            )
            values_src = values_weighted.reshape(
                -1, self.num_actions, self.num_players, NUM_HANDS
            )
            torch.where(
                self.leaf_mask[offset:offset_next, None, None],
                new_values[offset:offset_next],
                values_src.sum(dim=1),
                out=new_values[offset:offset_next],
            )
            new_values[offset:offset_next].masked_fill_(
                ~self.valid_mask[offset:offset_next, None, None], 0.0
            )

        return new_values

    @profile
    def compute_instantaneous_regrets(self, values: torch.Tensor) -> torch.Tensor:
        """Compute regrets for every valid non-leaf information set."""

        regrets = torch.zeros_like(self.policy_probs)

        bottom = self.depth_offsets[1]

        actor_indices = self.env.to_act[:, None, None].expand(-1, -1, NUM_HANDS)
        actor_beliefs = (
            self.beliefs[bottom:].gather(1, actor_indices[bottom:]).squeeze(1)
        )
        # Flip actor as the child actor is reversed.
        actor_values = values.gather(1, 1 - actor_indices).squeeze(1)
        # opp_values = values.gather(1, actor_indices).squeeze(1)

        actor_values_weighted = actor_values * self.policy_probs
        actor_values_weighted_src = self._pull_back(actor_values_weighted)
        actor_values_expected = actor_values_weighted_src.sum(dim=1)
        actor_values_expected_dest = self._fan_out(actor_values_expected)

        advantages = actor_values[bottom:] - actor_values_expected_dest

        # combo_compat = combo_onehot_float @ combo_onehot_float.T - torch.eye(1326)
        # combo_compat = ~combo_blocking_tensor(device=self.device)
        combo_onehot = combo_to_onehot_tensor(device=self.device).float()
        weights = (
            actor_beliefs.sum(dim=-1, keepdim=True)
            - actor_beliefs @ combo_onehot @ combo_onehot.T
            + actor_beliefs
        )
        regrets[bottom:] = weights * advantages

        regrets[: self.search_batch_size] = 0.0
        regrets.masked_fill_(~self.valid_mask[:, None], 0.0)
        return regrets

    @profile
    def update_policy(self) -> None:
        """Apply a regret-matching update to every valid non-leaf information set."""

        top = self.depth_offsets[-2]
        bottom = self.depth_offsets[1]

        positive_regrets = self.cumulative_regrets.clamp(min=0)
        positive_src = self._pull_back(positive_regrets)
        regret_sum = positive_src.sum(dim=1)
        regret_sum_src = self._fan_out(regret_sum)

        updated = torch.where(
            regret_sum_src > 1e-8,
            positive_regrets[bottom:],
            1.0,
        )
        updated.masked_fill_(~self.valid_mask[bottom:, None], 0.0)

        updated_src = self._pull_back(updated, sliced=True)
        updated_src /= updated_src.sum(dim=1, keepdim=True).clamp(min=1e-8)
        updated_dest = self._push_down(updated_src)
        torch.where(
            self.valid_mask[bottom:, None],
            updated_dest,
            self.policy_probs[bottom:],
            out=self.policy_probs[bottom:],
        )

        self._propagate_all_beliefs()
        self._block_and_normalize_beliefs()

    @profile
    def sample_leaf(
        self,
        root_indices: torch.Tensor,
        pbs: PublicBeliefState,
        pbs_start_idx: int,
        training_mode: bool,
    ) -> None:
        """Roll out from `root_indices` and copy the reached leaves into `pbs`.

        Args:
            root_indices: Indices of root nodes to sample trajectories from.
            pbs: Destination public belief state that will hold sampled leaves.
            pbs_start_idx: Offset inside `pbs` to start writing sampled rows.
            training_mode: Whether epsilon-greedy exploration is enabled.
        """
        N, M, B = self.search_batch_size, self.total_nodes, self.num_actions
        valid_leaf_mask = self.leaf_mask & ~self.env.done
        count = root_indices.numel()
        assert count > 0
        assert count <= valid_leaf_mask.sum().item()

        players = torch.randint(
            0, 2, (count,), generator=self.generator, device=self.device
        )
        sample_epsilon = self.sample_epsilon if training_mode else 0.0
        legal_masks = self.env.legal_bins_mask()
        legal_counts = legal_masks.float().sum(dim=-1, keepdim=True)
        uniform = torch.where(legal_masks, 1 / legal_counts, 0)

        # select a hand for every valid node
        selected_hands = torch.zeros(M, dtype=torch.long, device=self.device)
        valid_indices = torch.where(self.valid_mask)[0]
        selected_hands[valid_indices] = torch.multinomial(
            self.beliefs[valid_indices, self.env.to_act[valid_indices]], 1
        ).squeeze(1)

        # start with root nodes and descend to leaves
        sampled_nodes = root_indices.clone()

        depth = 0
        active_mask = ~self.leaf_mask[sampled_nodes]
        policy_probs_by_src = self._pull_back(self.policy_probs)
        while active_mask.any():
            assert depth <= self.max_depth
            active_nodes = sampled_nodes[active_mask]
            active_count = active_nodes.numel()
            assert self.valid_mask[active_nodes].all()

            to_act = self.env.to_act[active_nodes]
            player_mask = players[active_mask]
            sample_uniformly = (
                torch.rand(active_count, generator=self.generator, device=self.device)
                < sample_epsilon
            )
            sample_uniformly &= to_act == player_mask

            policy_probs_active = policy_probs_by_src[
                active_nodes, :, selected_hands[active_nodes]
            ]
            actions = torch.multinomial(
                torch.where(
                    sample_uniformly.view(-1, 1),
                    uniform[active_nodes],
                    policy_probs_active,
                ),
                num_samples=1,
                generator=self.generator,
            ).squeeze(1)

            offset_next = self.depth_offsets[depth + 1]
            offset = self.depth_offsets[depth]
            sampled_nodes[active_mask] = (
                offset_next + (active_nodes - offset) * B + actions
            )
            active_mask = ~self.leaf_mask[sampled_nodes]
            depth += 1

        dest_indices = torch.arange(
            pbs_start_idx, pbs_start_idx + count, device=self.device
        )
        pbs.env.copy_state_from(self.env, sampled_nodes, dest_indices)
        pbs.beliefs[pbs_start_idx : pbs_start_idx + count] = self.beliefs[sampled_nodes]

    @profile
    def self_play_iteration(
        self, training_mode: bool = True
    ) -> Optional[PublicBeliefState]:
        """Run one iteration through the CFR loop and produce leaf samples for replay."""
        self.profiler_step()  # Profile start of iteration

        self.construct_subgame()
        self.profiler_step()  # Profile after subgame construction

        self.initialize_policy_and_beliefs()
        self.profiler_step()  # Profile after policy initialization

        if self.warm_start_iterations > 0:
            self.set_leaf_values()
            self.profiler_step()  # Profile after leaf values

            self.warm_start()
            self.profiler_step()  # Profile after warm start

        self.set_leaf_values()
        self.profiler_step()  # Profile after leaf values

        self.values = self.compute_expected_values()
        self.profiler_step()  # Profile after expected values computation

        leaf_indices = torch.where(self.leaf_mask & ~self.env.done)[0]
        sample_cap = max(1, self.search_batch_size // 2)
        sample_count = min(leaf_indices.numel(), sample_cap)
        next_pbs = None
        next_pbs_idx = 0
        if sample_count > 0:
            next_pbs = PublicBeliefState.from_proto(
                env_proto=self.env,
                beliefs=torch.zeros(
                    sample_count, self.num_players, NUM_HANDS, device=self.device
                ),
                num_envs=sample_count,
            )
            sample_low = min(self.warm_start_iterations, self.cfr_iterations - 1)
            sample_low = max(sample_low, 0)
            sample_high = max(self.cfr_iterations, sample_low + 1)
            t_sample = torch.randint(
                sample_low,
                sample_high,
                (sample_count,),
                device=self.device,
            )

        loop_start = min(self.warm_start_iterations, self.cfr_iterations - 1)
        loop_start = max(loop_start, 0)

        for t in range(loop_start, self.cfr_iterations):
            self.profiler_step()  # Profile start of CFR iteration

            if sample_count > 0:
                # If t == t_sample, sample leaf PBS
                sample_now = torch.where(t_sample == t)[0]
                if sample_now.numel() > 0:
                    self.sample_leaf(
                        sample_now,
                        next_pbs,
                        next_pbs_idx,
                        training_mode=training_mode,
                    )
                    next_pbs_idx += sample_now.numel()
                    self.profiler_step()  # Profile after leaf sampling

            regrets = self.compute_instantaneous_regrets(self.values)
            self.profiler_step()  # Profile after regret computation

            self.cumulative_regrets += regrets
            self.update_policy()
            self.profiler_step()  # Profile after policy update

            # Update average policy.
            self.policy_probs_avg *= t
            self.policy_probs_avg += self.policy_probs
            self.policy_probs_avg /= t + 1
            self.profiler_step()  # Profile after average policy update

            self.set_leaf_values()
            self.profiler_step()  # Profile after leaf values update

            new_expected_values = self.compute_expected_values()
            self.profiler_step()  # Profile after expected values computation

            # Clip accumulated values to prevent extreme outliers
            # Typical poker hand values should be bounded by stack size
            max_hand_value = (
                self.env.starting_stack / self.env.scale * 4
            )  # 2x starting stack in scaled units

            self.values = (t * self.values + new_expected_values) / (t + 1)

        # Debug logging for extreme values
        extreme_values = self.valid_mask[:, None, None] & (
            torch.abs(new_expected_values) > max_hand_value
        )
        if torch.any(extreme_values):  # Warn at 80% of max
            extreme_count = extreme_values.sum().item()
            max_val = new_expected_values[self.valid_mask].max().item()
            min_val = new_expected_values[self.valid_mask].min().item()
            print(f"WARNING: Large hand values detected")
            print(f"  Extreme values count: {extreme_count}")
            print(f"  Value range: [{min_val:.2f}, {max_val:.2f}]")
            print(f"  Max allowed: {max_hand_value:.2f}")

        return next_pbs

    def sample_data(self) -> RebelBatch:
        """Aggregate model targets from the current root batch for supervised learning."""
        indices = torch.where(self.valid_mask[: self.search_batch_size])[0]
        level_1_end = self.depth_offsets[2]
        policy_targets = self._pull_back(self.policy_probs_avg[:level_1_end])
        return RebelBatch(
            features=self.encode_current_states()[indices],
            policy_targets=policy_targets.permute(0, 2, 1),
            value_targets=self.values[indices],
            legal_masks=self.env.legal_bins_mask()[indices],
            acting_players=self.env.to_act[indices],
        )

    @profile
    def _showdown_value(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Exact river showdown EV using rank-CDF + blocker correction.
        Returns EV and per-hand EV (unsorted/original hand order) per env.
        """

        M = indices.numel()
        device = self.device
        dtype = torch.float32  # or match belief dtype

        if M == 0:
            return torch.zeros(0, NUM_HANDS, device=device, dtype=dtype)

        # --- Beliefs & boards ---
        beliefs = self.beliefs[indices]  # (M,2,1326)
        b_self = beliefs[:, 0, :].to(dtype)  # (M,1326)
        b_opp = beliefs[:, 1, :].to(dtype)  # (M,1326)
        board = self.env.board_indices[indices].int()  # (M,5)

        # Sorted position k (0..1325) replicated across batch
        k = torch.arange(NUM_HANDS, device=device).expand(M, -1)  # (M,1326)

        if self.hand_rank_data is None:
            # --- Ranks & sorted order per env (river deterministic strength) ---
            # hand_ranks: (M,1326) any integer/monotone rank key s.t. equal => tie
            # sorted_indices: argsort by (rank, tiebreak) ascending (weaker -> stronger)
            hand_ranks, sorted_indices = rank_hands(board)  # both (M,1326)

            # Ranks in sorted order
            ranks_sorted = torch.gather(hand_ranks, 1, sorted_indices)  # (M,1326)
            assert torch.all(
                ranks_sorted[:, 1:] >= ranks_sorted[:, :-1]
            ), "rank_hands order is descending; flip or fix rank_hands"

            # --- Tie groups: start flags, group ids, [L,R] spans per sorted position ---
            is_start = torch.ones_like(ranks_sorted, dtype=torch.bool)  # (M,1326)
            is_start[:, 1:] = ranks_sorted[:, 1:] != ranks_sorted[:, :-1]
            group_id = is_start.cumsum(dim=1, dtype=torch.int) - 1  # (M,1326), 0..G-1

            # For each group id, store first/last index in sorted order
            starts = torch.full(
                (M, NUM_HANDS), NUM_HANDS, dtype=torch.int, device=device
            )
            ends = torch.full((M, NUM_HANDS), -1, dtype=torch.int, device=device)
            starts.scatter_reduce_(
                1, group_id, k.int(), reduce="amin", include_self=True
            )
            ends.scatter_reduce_(1, group_id, k.int(), reduce="amax", include_self=True)

            # L,R per sorted position
            L = torch.gather(starts, 1, group_id)  # (M,1326)
            R = torch.gather(ends, 1, group_id)  # (M,1326)
            L_idx = L
            R_idx = (R + 1).clamp(max=NUM_HANDS)
            assert (L <= R).all(), "L must be <= R"
            assert torch.all(
                torch.gather(ranks_sorted, 1, L) == torch.gather(ranks_sorted, 1, R)
            ), "L/R must have same rank"

            # Inverse permutation (sorted->original) for mapping EV back
            inv_sorted = torch.argsort(sorted_indices, dim=1)  # (M,1326)

            # --- Hand/card incidence & board masking ---
            combo_to_onehot = combo_to_onehot_tensor(device=device)  # (1326,52)
            hands_c1c2 = hand_combos_tensor(device=device)  # (1326,2)

            # Per-env mask for cards not on the board: True = usable card
            card_ok = torch.ones((M, 52), dtype=torch.bool, device=device)
            card_ok.scatter_(1, board, False)  # False for board cards

            # Hand usable mask (unsorted): hand must use only ok cards
            H = combo_to_onehot.unsqueeze(0).expand(M, -1, -1)  # (M,1326,52)
            conflicts = H & (~card_ok.unsqueeze(1))  # mark hands using any blocked card
            hand_ok_mask = ~conflicts.any(
                dim=2
            )  # True only if both cards are available
            hand_ok_mask_sorted = torch.gather(hand_ok_mask, 1, sorted_indices)

            # Cards (c1,c2) of each *sorted* hand per env
            hands_c1c2_sorted = torch.gather(
                hands_c1c2.unsqueeze(0).expand(M, -1, -1),  # (M,1326,2)
                1,
                sorted_indices.unsqueeze(-1).expand(-1, -1, 2),
            )  # (M,1326,2)

            self.hand_rank_data = HandRankData(
                sorted_indices=sorted_indices,
                inv_sorted=inv_sorted,
                H=H,
                card_ok=card_ok,
                hand_ok_mask=hand_ok_mask,
                hand_ok_mask_sorted=hand_ok_mask_sorted,
                hands_c1c2_sorted=hands_c1c2_sorted,
                L_idx=L_idx,
                R_idx=R_idx,
            )
        else:
            sorted_indices = self.hand_rank_data.sorted_indices
            inv_sorted = self.hand_rank_data.inv_sorted
            H = self.hand_rank_data.H
            card_ok = self.hand_rank_data.card_ok
            hand_ok_mask = self.hand_rank_data.hand_ok_mask
            hand_ok_mask_sorted = self.hand_rank_data.hand_ok_mask_sorted
            hands_c1c2_sorted = self.hand_rank_data.hands_c1c2_sorted
            L_idx = self.hand_rank_data.L_idx
            R_idx = self.hand_rank_data.R_idx

        assert not (b_self * ~hand_ok_mask).any()
        assert not (b_opp * ~hand_ok_mask).any()
        assert torch.allclose(b_self.sum(dim=1), torch.ones_like(b_self.sum(dim=1)))
        assert torch.allclose(b_opp.sum(dim=1), torch.ones_like(b_opp.sum(dim=1)))

        c1 = hands_c1c2_sorted[..., 0]  # (M,1326)
        c2 = hands_c1c2_sorted[..., 1]  # (M,1326)

        # Sort opponent marginal by strength order
        b_opp_sorted = torch.gather(b_opp, 1, sorted_indices)  # (M,1326)

        # Hand->card incidence in sorted order with board columns zeroed
        H_sorted = torch.gather(H, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, 52))
        H_sorted = H_sorted & card_ok.unsqueeze(1)  # (M,1326,52)

        # --- Prefix sums over opponent mass (global and per-card), left-padded ---
        P = torch.cumsum(b_opp_sorted, dim=1)  # (M,1326)
        P = torch.cat(
            [torch.zeros(M, 1, device=device, dtype=dtype), P], dim=1
        )  # (M,1327)

        per_card_mass = H_sorted.to(dtype) * b_opp_sorted.unsqueeze(-1)  # (M,1326,52)
        Pcards = torch.cumsum(per_card_mass, dim=1)  # (M,1326,52)
        # -- Prefix sums over opponent mass, per card --
        Pcards = torch.cat(
            [torch.zeros(M, 1, 52, device=device, dtype=dtype), Pcards], dim=1
        )  # (M,1327,52)

        # --- Win/tie masses for each sorted position ---

        # Gather needed prefixes
        P_before = torch.gather(P, 1, L_idx)  # (M,1326)
        Pcards_before = torch.gather(
            Pcards, 1, L_idx.unsqueeze(-1).expand(-1, -1, 52)
        )  # (M,1326,52)

        # Win mass: all strictly weaker, excluding blockers
        Pcards_k_c1 = Pcards_before.gather(2, c1.unsqueeze(-1)).squeeze(-1)
        Pcards_k_c2 = Pcards_before.gather(2, c2.unsqueeze(-1)).squeeze(-1)
        win_mass = P_before - Pcards_k_c1 - Pcards_k_c2

        # Tie mass over [L,R] inclusive, excluding blockers
        P_R = torch.gather(P, 1, R_idx)
        P_L = torch.gather(P, 1, L_idx)
        seg_sum = P_R - P_L  # (M,1326)

        gL = L_idx.unsqueeze(-1).expand(-1, -1, 52)
        gR = R_idx.unsqueeze(-1).expand(-1, -1, 52)
        Pcards_R = torch.gather(Pcards, 1, gR)  # (M,1326,52)
        Pcards_L = torch.gather(Pcards, 1, gL)  # (M,1326,52)
        seg_c1 = (Pcards_R - Pcards_L).gather(2, c1.unsqueeze(-1)).squeeze(-1)
        seg_c2 = (Pcards_R - Pcards_L).gather(2, c2.unsqueeze(-1)).squeeze(-1)
        # Re-add hero combo mass (present in both seg_c1 and seg_c2)
        tie_mass = seg_sum - seg_c1 - seg_c2 + b_opp_sorted

        # --- Denominator: compatible opp mass for each hero hand (unsorted belief) ---
        Pc_last = Pcards[:, -1, :]  # (M, 52) totals per card
        denom = (
            1.0 - Pc_last.gather(1, c1) - Pc_last.gather(1, c2) + b_opp_sorted
        ).clamp(min=1e-12)
        assert ((denom > 1e-12) | ((win_mass < 1e-5) & (tie_mass < 1e-5))).all()

        # Probabilities & EV (in sorted order)
        win_prob = torch.where(denom > 1e-12, win_mass / denom, 0.0)
        tie_prob = torch.where(denom > 1e-12, tie_mass / denom, 0.0)

        EV_hand_sorted = win_prob + 0.5 * tie_prob

        # Map per-hand EV back to original hand order
        EV_hand = torch.gather(EV_hand_sorted, 1, inv_sorted)  # (M,1326)
        EV_hand = EV_hand * hand_ok_mask.to(dtype)  # zero impossible hands

        # Range EV for the player
        potential = (
            self.env.stacks[indices, 0]
            + self.env.pot[indices]
            - self.env.starting_stack
        )

        return EV_hand * potential[:, None] / self.env.scale

    def _valid_nodes(
        self, bottom_up: bool = False
    ) -> Generator[tuple[int, torch.Tensor]]:
        """Yield `(depth, indices)` pairs for nodes marked as valid."""
        for depth in (
            range(self.max_depth + 1)
            if not bottom_up
            else range(self.max_depth, -1, -1)
        ):
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]

            mask = self.valid_mask[offset:offset_next]
            yield depth, offset + torch.where(mask)[0]

    def _valid_actions(
        self, bottom_up: bool = False
    ) -> Generator[tuple[int, int, torch.Tensor, torch.Tensor]]:
        """Iterate over legal transitions, optionally in bottom-up order."""
        N, M, B = self.search_batch_size, self.total_nodes, self.num_actions

        for depth in (
            range(self.max_depth)
            if not bottom_up
            else range(self.max_depth - 1, -1, -1)
        ):
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]

            for action in range(B):
                current_legal_mask = (
                    self.legal_mask[offset:offset_next, action]
                    & self.valid_mask[offset:offset_next]
                    & ~self.leaf_mask[offset:offset_next]
                )
                if not current_legal_mask.any():
                    continue

                current_legal_indices = torch.where(current_legal_mask)[0] + offset
                next_legal_indices = (
                    offset_next + (current_legal_indices - offset) * B + action
                )
                assert next_legal_indices.max().item() < M
                assert self.valid_mask[next_legal_indices].all()

                yield depth, action, current_legal_indices, next_legal_indices

    def _pull_back(self, data: torch.Tensor, sliced=False) -> torch.Tensor:
        """Pull back data to all parent nodes.
        Args:
            data: Data to pull back, organized by destination node, shape [M, ...].
            sliced: Whether this is already sliced to the bottom of the tree ([M - N, ...]).
        Returns:
            Data by source node/action, shape [M - N * B ** D, B, *data.shape[1:]].
        """
        bottom = self.depth_offsets[1] if not sliced else 0
        return data[bottom:].view(-1, self.num_actions, *data.shape[1:])

    def _push_down(self, data: torch.Tensor) -> torch.Tensor:
        """Push down data to all child nodes.
        Args:
            data: Data to push down, shape [M, B, ...].
        Returns:
            Data by child node, shape [M - N, ...].
        """
        assert data.shape[1] == self.num_actions
        top = self.depth_offsets[-2]
        return data[:top, None].reshape(-1, *data.shape[2:])

    def _fan_out(self, data: torch.Tensor) -> torch.Tensor:
        """Fanout data to all children nodes.

        Args:
            data: Data to fanout.
        Returns:
            Fanout data, shape [M - N, *data.shape[1:]].
        """
        top = self.depth_offsets[-2]
        return (
            data[:top, None]
            .expand(-1, self.num_actions, *data.shape[1:])
            .reshape(-1, *data.shape[1:])
        )

    @profile
    def encode_current_states(self) -> torch.Tensor:
        """Encode environment states for policy network input."""
        return self.feature_encoder.encode(
            self.env.to_act,
            beliefs=self.beliefs,
        )

    def _leaf_node_indices(self) -> torch.Tensor:
        """Return flattened indices for valid nodes marked as leaves."""
        return torch.where(self.leaf_mask)[0]
