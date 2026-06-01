"""Base CFR evaluator class with shared methods."""

from __future__ import annotations

import os
import warnings
from abc import ABC
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from p2.core.structured_config import CFRType, WarmStartType
from p2.env.card_utils import (
    NUM_HANDS,
    calculate_unblocked_mass,
    combo_to_onehot_tensor,
    hand_combos_tensor,
)
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.env.pbs_env import PBSEnv
from p2.env.rules import rank_hands
from p2.models.base_mlp_model import BaseMLPModel
from p2.models.mlp.better_feature_encoder import BetterFeatureEncoder
from p2.models.mlp.better_trm import BetterTRM
from p2.models.mlp.mlp_features import MLPFeatures
from p2.models.mlp.rebel_feature_encoder import RebelFeatureEncoder
from p2.models.model_output import TRMLatent
from p2.search.allin_payoff import (
    FLOP_I8_SCALE,
    I16_SCALE,
    AllInPayoffResolver,
    _flop_combination_index_tensor,
    allin_values_from_payoff_batch,
    compute_postflop_payoff_quantized_triton_batched,
)
from p2.rl.rebel_batch import RebelBatch
from p2.rl.target_provenance import (
    TARGET_SOURCE_CFR_BACKUP,
    TARGET_SOURCE_CHANCE_EXPECTATION,
)
from p2.search.chance_node_helper import ChanceNodeHelper
from p2.utils.model_utils import compute_masked_logits
from p2.utils.profiling import profile

STREETS = ["preflop", "flop", "turn", "river"]


@dataclass
class ExploitabilityStats:
    local_exploitability: torch.Tensor
    local_best_response_values: torch.Tensor


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


@dataclass
class PublicBeliefState:
    """Public belief state for a vectorized poker environment.

    Attributes:
        env: Vectorised poker environment standing at a public state.
        beliefs: Beliefs representing the range at this node (post-chance for
            regular states, pre-chance for street-end nodes).
    """

    env: HUNLTensorEnv | PBSEnv
    beliefs: torch.Tensor  # [batch_size, num_players, NUM_HANDS]

    @classmethod
    def from_proto(
        cls,
        env_proto: HUNLTensorEnv | PBSEnv,
        beliefs: torch.Tensor,
        num_envs: int | None = None,
    ) -> PublicBeliefState:
        """Create a new belief state with an environment cloned from `env_proto`.

        Args:
            env_proto: Template environment whose configuration should be reused.
            beliefs: Belief tensor shaped `[batch, players, NUM_HANDS]`.
            num_envs: Optional override for the number of vectorised environments.
        """
        env_cls = type(env_proto)
        return PublicBeliefState(
            env=env_cls.from_proto(env_proto, num_envs=num_envs),
            beliefs=beliefs,
        )

    def __post_init__(self) -> None:
        assert self.beliefs.shape[0] == self.env.N


def padded_indices(mask: torch.Tensor, alignment: int) -> torch.Tensor:
    """Compute indices from mask, padded to a multiple of alignment by repeating the last item."""
    indices = torch.where(mask)[0]
    current_len = indices.numel()
    if current_len > 0:
        remainder = current_len % alignment
        if remainder != 0:
            padding_size = alignment - remainder
            last_item = indices[-1:]
            padding = last_item.repeat(padding_size)
            indices = torch.cat([indices, padding])
    return indices


class CFREvaluator(ABC):
    """Base class for CFR evaluators with shared methods."""

    # Per-step invariant checks (`isfinite().all()`, `min()/max() < bound`,
    # ordering asserts). Each one materializes a Python bool → host sync; the
    # CFR hot path triggers many per training step. Off by default; flip on
    # for debugging numerical issues.
    CHECK_INVARIANTS: bool = False

    model: BaseMLPModel
    device: torch.device
    env: HUNLTensorEnv | PBSEnv
    feature_encoder: RebelFeatureEncoder | BetterFeatureEncoder
    cfr_type: CFRType
    num_supervisions: int
    root_nodes: int
    total_nodes: int
    beliefs: torch.Tensor
    beliefs_avg: torch.Tensor
    legal_mask: torch.Tensor
    # Common fields shared by both evaluators
    float_dtype: torch.dtype
    num_players: int
    num_actions: int
    max_depth: int
    tree_depth: int
    cfr_iterations: int
    warm_start_iterations: int
    warm_start_type: WarmStartType
    warm_start_multiplier: float
    cfr_avg: bool
    cfr_plus: bool
    dcfr_alpha: float
    dcfr_beta: float
    dcfr_gamma: float
    dcfr_alpha_initial: float
    dcfr_beta_initial: float
    dcfr_gamma_initial: float
    dcfr_alpha_final: float | None
    dcfr_beta_final: float | None
    dcfr_gamma_final: float | None
    dcfr_delay: int
    sample_epsilon: float
    use_final_policy_values: bool
    generator: torch.Generator | None
    valid_mask: torch.Tensor
    leaf_mask: torch.Tensor
    child_mask: torch.Tensor
    child_count: torch.Tensor
    new_street_mask: torch.Tensor
    model_indices: torch.Tensor
    allowed_hands: torch.Tensor
    allowed_hands_prob: torch.Tensor
    policy_probs: torch.Tensor
    policy_probs_avg: torch.Tensor
    average_policy_numerator: torch.Tensor
    average_policy_denominator: torch.Tensor
    average_policy_initialized: bool
    policy_probs_sample: torch.Tensor
    beliefs_sample: torch.Tensor
    uniform_policy: torch.Tensor
    cumulative_regrets: torch.Tensor
    latest_values: torch.Tensor
    values_avg: torch.Tensor
    self_reach: torch.Tensor
    self_reach_avg: torch.Tensor
    root_pre_chance_beliefs: torch.Tensor
    latent: TRMLatent | None
    last_model_values: torch.Tensor | None
    showdown_indices: torch.Tensor
    showdown_actors: torch.Tensor
    showdown_potential: torch.Tensor
    allin_call_indices: torch.Tensor
    allin_call_parent_indices: torch.Tensor
    allin_call_mask: torch.Tensor
    allin_payoff_resolver: AllInPayoffResolver | None
    prev_actor: torch.Tensor
    combo_onehot_float: torch.Tensor
    chance_helper: ChanceNodeHelper
    stats: dict[str, float]
    depth_offsets: list[int]
    # Profiler fields (optional, initialized by subclasses if needed)
    profiler_enabled: bool
    profiler: any
    profiler_output_dir: str | None

    # ============================================================================
    # Abstract Methods (must be implemented by subclasses)
    # ============================================================================

    def _fan_out_deep(self, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement _fan_out_deep.")

    def _construct_subgame(
        self,
        src_env: HUNLTensorEnv | PBSEnv,
        src_indices: torch.Tensor,
    ) -> None:
        """Construct the subgame tree structure (subclass-specific implementation).

        This method should:
        - Copy root states from src_env to self.env
        - Expand the tree by creating child nodes
        - Set up depth_offsets, valid_mask, leaf_mask, etc.
        - Initialize environment states for all nodes

        Args:
            src_env: Batched environment that holds the source root public states.
            src_indices: Row indices inside `src_env` to copy into the tree roots.
        """
        raise NotImplementedError("Subclasses must implement _construct_subgame.")

    def sample_leaves(self, training_mode: bool) -> PublicBeliefState:
        """Sample leaves from `self.policy_probs_sample`.

        Returns:
            PublicBeliefState containing the sampled leaves.
        """
        raise NotImplementedError("Subclasses must implement sample_leaves.")

    def _sample_root_hands_by_player(self) -> torch.Tensor:
        """Sample one private hand per root and player from root beliefs."""
        N = self.root_nodes
        return torch.multinomial(
            self.beliefs[:N].reshape(N * self.num_players, NUM_HANDS),
            1,
            generator=self.generator,
        ).view(N, self.num_players)

    def _fan_out(self, data: torch.Tensor, level: int | None = None) -> torch.Tensor:
        """Fanout data to all children nodes."""
        raise NotImplementedError("Subclasses must implement _fan_out.")

    def _pull_back(self, data: torch.Tensor, level: int | None = None) -> torch.Tensor:
        """Pull back data to all parent nodes."""
        raise NotImplementedError("Subclasses must implement _pull_back.")

    def _pull_back_sum(
        self, tensor: torch.Tensor, out: torch.Tensor, level: int | None = None
    ) -> None:
        """Pull back tensor and sum into output tensor."""
        raise NotImplementedError("Subclasses must implement _pull_back_sum.")

    def _push_down(self, data: torch.Tensor, level: int | None = None) -> torch.Tensor:
        """Push down data to all child nodes.

        Args:
            data: Data to push down, shape [M, B, ...].
            level: Depth level to push down from, or None for all levels.
        Returns:
            Data by child node, shape [M - N, ...].
        """
        raise NotImplementedError("Subclasses must implement _push_down.")

    def _mask_invalid(self, tensor: torch.Tensor) -> None:
        """Mask invalid nodes in the tensor. Noop for sparse evaluator."""
        raise NotImplementedError("Subclasses must implement _mask_invalid.")

    def _propagate_level_beliefs(self, depth: int) -> None:
        """Propagate beliefs from all nodes at a given level to all nodes at the next level."""
        raise NotImplementedError("Subclasses must implement _propagate_level_beliefs.")

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def _block_beliefs(self, target: torch.Tensor | None = None) -> None:
        """Block beliefs based on the board."""
        if target is None:
            target = self.beliefs
        target.masked_fill_((~self.allowed_hands)[:, None, :], 0.0)

    def _normalize_beliefs(self, target: torch.Tensor | None = None) -> None:
        """Normalize beliefs across hands in-place for valid nodes.

        Note: allowed_hands_prob should be 0 on invalid nodes, so invalid nodes
        will automatically get 0 beliefs when denom is 0.
        """
        if target is None:
            target = self.beliefs

        denom = target.sum(dim=-1, keepdim=True)
        # If the action probability of getting to a node is 0, our
        # bayesian update will make the beliefs in that state all 0.
        # So we set them to uniform (allowed_hands_prob).
        # For invalid nodes, allowed_hands_prob is 0, so they get 0 beliefs.
        torch.where(
            denom > 1e-5,
            target / denom,
            self.allowed_hands_prob[:, None, :],
            out=target,
        )

    def _compute_model_indices(self) -> torch.Tensor:
        """Compute model indices from leaf mask, padded to a multiple of num_envs.

        Returns:
            Tensor of indices where model evaluation is needed, padded to a multiple
            of num_envs by repeating the last item.
        """
        model_mask = self.leaf_mask & ~self.env.done
        allin_mask = getattr(self, "allin_call_mask", None)
        if allin_mask is not None and allin_mask.shape == model_mask.shape:
            model_mask = model_mask & ~allin_mask
        return padded_indices(model_mask, self.root_nodes)

    def _uses_street_cutoff_schedule(self) -> bool:
        schedule = getattr(self, "action_schedule", None)
        return bool(
            schedule is not None and getattr(schedule, "bet_bins_by_depth", None) is not None
        )

    def _validate_model_leaf_phases(self) -> None:
        if not self._uses_street_cutoff_schedule() or self.model_indices.numel() == 0:
            return
        model_leaf_mask = self.leaf_mask & ~self.env.done
        allin_mask = getattr(self, "allin_call_mask", None)
        if allin_mask is not None and allin_mask.shape == model_leaf_mask.shape:
            model_leaf_mask &= ~allin_mask
        same_street = model_leaf_mask & ~self.new_street_mask
        if same_street.any():
            raise RuntimeError(
                "Street-cutoff search produced same-street non-terminal model "
                "leaves. Increase search.bet_bins_by_depth coverage or restrict "
                "the final depth to actions that close the street."
            )

    def _allin_abstraction_enabled(self) -> bool:
        cfg = getattr(self, "cfg", None)
        search_cfg = getattr(cfg, "search", None)
        return bool(
            getattr(
                search_cfg,
                "allin_call_terminal_abstraction",
                getattr(self, "allin_call_terminal_abstraction", False),
            )
        )

    def _ensure_allin_payoff_resolver(self) -> AllInPayoffResolver:
        resolver = getattr(self, "allin_payoff_resolver", None)
        if resolver is None:
            search_cfg = getattr(getattr(self, "cfg", None), "search", None)
            resolver = AllInPayoffResolver(
                device=self.device,
                preflop_table_path=getattr(
                    search_cfg,
                    "preflop_allin_table_path",
                    getattr(self, "preflop_allin_table_path", None),
                ),
            )
            self.allin_payoff_resolver = resolver
        return resolver

    def _mark_allin_call_leaves(self) -> None:
        """Mark children reached by calling an all-in bet as terminal leaves.

        The child environment has already stepped and may have dealt a future
        street. The all-in payoff is evaluated from the parent public board and
        the child's post-call pot/stacks.
        """
        empty = torch.empty(0, dtype=torch.long, device=self.device)
        self.allin_call_indices = empty
        self.allin_call_parent_indices = empty
        self.allin_call_indices_by_street = (empty, empty, empty)
        self.allin_call_parent_indices_by_street = (empty, empty, empty)
        self.allin_call_boards_by_street = (
            torch.empty(0, 5, dtype=torch.long, device=self.device),
            torch.empty(0, 5, dtype=torch.long, device=self.device),
            torch.empty(0, 5, dtype=torch.long, device=self.device),
        )
        self.allin_turn_tables_i16 = torch.empty(
            0, NUM_HANDS, NUM_HANDS, dtype=torch.int16, device=self.device
        )
        self.allin_turn_table_ids = empty
        self.allin_turn_stats_buffer = torch.empty(
            0, 2, 53, dtype=torch.float32, device=self.device
        )
        self.allin_flop_tables_i8 = torch.empty(
            0, NUM_HANDS, NUM_HANDS, dtype=torch.int8, device=self.device
        )
        self.allin_flop_table_ids = empty
        self.allin_flop_stats_buffer = torch.empty(
            0, 2, 53, dtype=torch.float32, device=self.device
        )
        self.allin_preflop_stats_buffer = torch.empty(
            0, 2, 53, dtype=torch.float32, device=self.device
        )
        self.allin_call_mask = torch.zeros(
            self.total_nodes, dtype=torch.bool, device=self.device
        )
        if not self._allin_abstraction_enabled() or self.total_nodes <= self.root_nodes:
            return

        child_indices = torch.arange(self.root_nodes, self.total_nodes, device=self.device)
        parent, action = self._parent_action_for_nodes(child_indices)
        actor = self.env.to_act[parent]
        opp = 1 - actor
        parent_to_call = (
            self.env.committed[parent, opp] - self.env.committed[parent, actor]
        )
        parent_street = self.env.street[parent]
        search_cfg = getattr(getattr(self, "cfg", None), "search", None)
        has_preflop_table = (
            getattr(
                search_cfg,
                "preflop_allin_table_path",
                getattr(self, "preflop_allin_table_path", None),
            )
            is not None
        )

        mask = (
            (action == 1)
            & self.env.is_allin[parent, opp]
            & (parent_to_call > 0)
            & (parent_street < 3)
        )
        if not has_preflop_table:
            mask &= parent_street > 0
        indices = child_indices[mask]
        if indices.numel() == 0:
            return
        self.allin_call_indices = indices.contiguous()
        self.allin_call_parent_indices = parent[mask].contiguous()
        self.allin_call_mask[self.allin_call_indices] = True
        self.leaf_mask[self.allin_call_indices] = True
        self.new_street_mask[self.allin_call_indices] = False
        self._cache_allin_call_street_partitions(parent_street[mask].contiguous())
        self._prune_allin_call_descendants()

    def _cache_allin_call_street_partitions(self, parent_streets: torch.Tensor) -> None:
        if self.allin_call_indices.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=self.device)
            empty_boards = torch.empty(0, 5, dtype=torch.long, device=self.device)
            self.allin_call_indices_by_street = (empty, empty, empty)
            self.allin_call_parent_indices_by_street = (empty, empty, empty)
            self.allin_call_boards_by_street = (empty_boards, empty_boards, empty_boards)
            self.allin_turn_tables_i16 = torch.empty(
                0, NUM_HANDS, NUM_HANDS, dtype=torch.int16, device=self.device
            )
            self.allin_turn_table_ids = empty
            self.allin_turn_stats_buffer = torch.empty(
                0, 2, 53, dtype=torch.float32, device=self.device
            )
            self.allin_flop_tables_i8 = torch.empty(
                0, NUM_HANDS, NUM_HANDS, dtype=torch.int8, device=self.device
            )
            self.allin_flop_table_ids = empty
            self.allin_flop_stats_buffer = torch.empty(
                0, 2, 53, dtype=torch.float32, device=self.device
            )
            self.allin_preflop_stats_buffer = torch.empty(
                0, 2, 53, dtype=torch.float32, device=self.device
            )
            return

        boards = self.env.board_indices[self.allin_call_parent_indices].long()
        street0 = parent_streets == 0
        street1 = parent_streets == 1
        street2 = parent_streets == 2
        self.allin_call_indices_by_street = (
            self.allin_call_indices[street0].contiguous(),
            self.allin_call_indices[street1].contiguous(),
            self.allin_call_indices[street2].contiguous(),
        )
        self.allin_call_parent_indices_by_street = (
            self.allin_call_parent_indices[street0].contiguous(),
            self.allin_call_parent_indices[street1].contiguous(),
            self.allin_call_parent_indices[street2].contiguous(),
        )
        self.allin_call_boards_by_street = (
            boards[street0].contiguous(),
            boards[street1].contiguous(),
            boards[street2].contiguous(),
        )
        self.allin_preflop_stats_buffer = torch.empty(
            self.allin_call_indices_by_street[0].shape[0],
            2,
            53,
            dtype=torch.float32,
            device=self.device,
        )
        self._cache_allin_flop_tables()
        self._cache_allin_turn_tables()

    def _cache_allin_flop_tables(self) -> None:
        empty = torch.empty(0, dtype=torch.long, device=self.device)
        flop_boards = (
            self.allin_call_boards_by_street[1][:, :3].long().sort(dim=1).values
        )
        if flop_boards.numel() == 0 or self.device.type != "cuda":
            self.allin_flop_tables_i8 = torch.empty(
                0, NUM_HANDS, NUM_HANDS, dtype=torch.int8, device=self.device
            )
            self.allin_flop_table_ids = empty
            self.allin_flop_stats_buffer = torch.empty(
                0, 2, 53, dtype=torch.float32, device=self.device
            )
            return

        resolver = self._ensure_allin_payoff_resolver()
        if resolver._flop_i8 is None:
            self.allin_flop_tables_i8 = torch.empty(
                0, NUM_HANDS, NUM_HANDS, dtype=torch.int8, device=self.device
            )
            self.allin_flop_table_ids = empty
            self.allin_flop_stats_buffer = torch.empty(
                0, 2, 53, dtype=torch.float32, device=self.device
            )
            return

        actual_to_canon, actual_perm, combo_perms = resolver.flop_lookup_tensors()
        actual_idx = _flop_combination_index_tensor(flop_boards)
        unique_actual_idx, inverse = torch.unique(
            actual_idx,
            sorted=True,
            return_inverse=True,
        )
        canon_ids = actual_to_canon.index_select(0, unique_actual_idx)
        perm_ids = actual_perm.index_select(0, unique_actual_idx)
        tables = resolver._flop_i8.index_select(0, canon_ids)
        perms = combo_perms.index_select(0, perm_ids)
        self.allin_flop_tables_i8 = tables.gather(
            1,
            perms[:, :, None].expand(-1, -1, NUM_HANDS),
        ).gather(
            2,
            perms[:, None, :].expand(-1, NUM_HANDS, -1),
        ).contiguous()
        self.allin_flop_table_ids = inverse.contiguous()
        self.allin_flop_stats_buffer = torch.empty(
            flop_boards.shape[0],
            2,
            53,
            dtype=torch.float32,
            device=self.device,
        )

    def _cache_allin_turn_tables(self) -> None:
        empty = torch.empty(0, dtype=torch.long, device=self.device)
        turn_boards = self.allin_call_boards_by_street[2][:, :4].long().contiguous()
        if turn_boards.numel() == 0 or self.device.type != "cuda":
            self.allin_turn_tables_i16 = torch.empty(
                0, NUM_HANDS, NUM_HANDS, dtype=torch.int16, device=self.device
            )
            self.allin_turn_table_ids = empty
            self.allin_turn_stats_buffer = torch.empty(
                0, 2, 53, dtype=torch.float32, device=self.device
            )
            return

        unique_boards, inverse = torch.unique(
            turn_boards,
            dim=0,
            sorted=True,
            return_inverse=True,
        )
        self.allin_turn_tables_i16 = compute_postflop_payoff_quantized_triton_batched(
            unique_boards,
            dtype=torch.int16,
        ).contiguous()
        self.allin_turn_table_ids = inverse.contiguous()
        self.allin_turn_stats_buffer = torch.empty(
            turn_boards.shape[0],
            2,
            53,
            dtype=torch.float32,
            device=self.device,
        )

    def _allin_call_child_mask(
        self,
        parent_env: HUNLTensorEnv | PBSEnv,
        parent_local_indices: torch.Tensor,
        action_bins: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.num_players != 2
            or not self._allin_abstraction_enabled()
            or action_bins.numel() == 0
        ):
            return torch.zeros_like(action_bins, dtype=torch.bool)

        actor = parent_env.to_act[parent_local_indices]
        opp = 1 - actor
        parent_to_call = (
            parent_env.committed[parent_local_indices, opp]
            - parent_env.committed[parent_local_indices, actor]
        )
        parent_street = parent_env.street[parent_local_indices]
        search_cfg = getattr(getattr(self, "cfg", None), "search", None)
        has_preflop_table = (
            getattr(
                search_cfg,
                "preflop_allin_table_path",
                getattr(self, "preflop_allin_table_path", None),
            )
            is not None
        )
        mask = (
            (action_bins == 1)
            & parent_env.is_allin[parent_local_indices, opp]
            & (parent_to_call > 0)
            & (parent_street < 3)
        )
        if not has_preflop_table:
            mask &= parent_street > 0
        return mask

    def _prune_allin_call_descendants(self) -> None:
        if not hasattr(self, "valid_mask") or self.allin_call_indices.numel() == 0:
            return
        pruned = self.allin_call_mask.clone()
        for depth in range(1, self.tree_depth + 1):
            start = self.depth_offsets[depth]
            end = self.depth_offsets[depth + 1]
            if start >= end:
                continue
            nodes = torch.arange(start, end, device=self.device)
            parent, _ = self._parent_action_for_nodes(nodes)
            invalid = pruned[parent]
            if invalid.any():
                self.valid_mask[nodes[invalid]] = False
                self.leaf_mask[nodes[invalid]] = False
                self.new_street_mask[nodes[invalid]] = False
                pruned[nodes[invalid]] = True

    def _parent_action_for_nodes(
        self, node_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parent_index = getattr(self, "parent_index", None)
        action_from_parent = getattr(self, "action_from_parent", None)
        if parent_index is not None and action_from_parent is not None:
            return parent_index[node_indices], action_from_parent[node_indices]

        offsets = torch.tensor(self.depth_offsets, dtype=torch.long, device=self.device)
        parent_depth = torch.bucketize(node_indices, offsets[1:], right=False)
        parent_start = offsets[parent_depth]
        child_start = offsets[parent_depth + 1]
        local = node_indices - child_start
        parent = parent_start + torch.div(local, self.num_actions, rounding_mode="floor")
        action = local.remainder(self.num_actions)
        return parent, action

    def _set_allin_call_values(self, beliefs: torch.Tensor) -> None:
        indices = getattr(self, "allin_call_indices", None)
        if indices is None or indices.numel() == 0:
            return
        resolver = self._ensure_allin_payoff_resolver()
        indices_by_street = getattr(self, "allin_call_indices_by_street", None)
        boards_by_street = getattr(self, "allin_call_boards_by_street", None)
        if indices_by_street is None or boards_by_street is None:
            self._cache_allin_call_street_partitions(
                self.env.street[self.allin_call_parent_indices]
            )
            indices_by_street = self.allin_call_indices_by_street
            boards_by_street = self.allin_call_boards_by_street

        node_idx = indices_by_street[0]
        if node_idx.numel() > 0:
            values = resolver.values_for_boards(
                street=0,
                boards=boards_by_street[0],
                beliefs=beliefs[node_idx],
            )
            self._write_scaled_allin_values(node_idx, values)

        node_idx = indices_by_street[1]
        if node_idx.numel() > 0:
            flop_tables = getattr(self, "allin_flop_tables_i8", None)
            flop_ids = getattr(self, "allin_flop_table_ids", None)
            if (
                flop_tables is not None
                and flop_ids is not None
                and flop_tables.numel() > 0
            ):
                values = allin_values_from_payoff_batch(
                    flop_tables.index_select(0, flop_ids),
                    beliefs[node_idx],
                    scale=FLOP_I8_SCALE,
                )
            else:
                values = resolver.values_for_boards(
                    street=1,
                    boards=boards_by_street[1],
                    beliefs=beliefs[node_idx],
                )
            self._write_scaled_allin_values(node_idx, values)

        node_idx = indices_by_street[2]
        if node_idx.numel() > 0:
            turn_tables = getattr(self, "allin_turn_tables_i16", None)
            turn_ids = getattr(self, "allin_turn_table_ids", None)
            if (
                turn_tables is not None
                and turn_ids is not None
                and turn_tables.numel() > 0
            ):
                values = allin_values_from_payoff_batch(
                    turn_tables.index_select(0, turn_ids),
                    beliefs[node_idx],
                    scale=I16_SCALE,
                )
            else:
                values = resolver.values_for_boards(
                    street=2,
                    boards=boards_by_street[2],
                    beliefs=beliefs[node_idx],
                )
            self._write_scaled_allin_values(node_idx, values)

    def _write_scaled_allin_values(self, node_idx: torch.Tensor, values: torch.Tensor) -> None:
        potential = (
            self.env.stacks[node_idx].to(values.dtype)
            + self.env.pot[node_idx, None].to(values.dtype)
            - self.env.starting_stacks[node_idx].to(values.dtype)
        )
        env_scale = self.env.scale[node_idx].to(values.dtype).clamp_min(1e-8)
        values *= potential[:, :, None] / env_scale[:, None, None]
        self.latest_values[node_idx] = values

    def _get_mixing_weights(self, t: int) -> tuple[float, float]:
        """Get the mixing weights for the current iteration (0-indexed).

        For iteration t (0-indexed), returns (old, new) where:
        - old: weight for the previous average policy
        - new: weight for the current iteration's policy
        """
        if self.cfr_type == CFRType.standard:
            return t, 1
        elif self.cfr_type == CFRType.linear:
            return t, 2
        elif self.cfr_type == CFRType.discounted:
            new = self._get_average_policy_weight(t)
            if new == 0:
                return 0.0, 0.0
            old = sum(
                self._get_average_policy_weight(k)
                for k in range(self.warm_start_iterations, t)
            )
            return float(old), float(new)
        raise ValueError(f"Unsupported CFR type: {self.cfr_type}")

    def _get_average_policy_weight(self, t: int) -> float:
        """Return the current-iteration weight for CFR average strategy."""
        if self.cfr_type == CFRType.discounted:
            if self._average_accumulation_delayed(t):
                return 0.0
            progress = max(0.0, float(t - self.dcfr_delay)) / float(
                self._average_accumulation_window()
            )
            return float(progress ** self._get_dcfr_gamma_for_iteration(t))
        _, new = self._get_mixing_weights(t)
        return float(new)

    def _average_accumulation_delayed(self, t: int) -> bool:
        return self.cfr_type == CFRType.discounted and t <= self.dcfr_delay

    def _average_accumulation_window(self) -> int:
        return max(1, self.cfr_iterations - self.dcfr_delay)

    def _get_dcfr_gamma_for_iteration(self, t: int) -> float:
        """Return the scheduled gamma for one iteration without mutating state."""
        if self.dcfr_gamma_final is None:
            return float(self.dcfr_gamma)
        total_iterations = max(1, self.cfr_iterations - self.warm_start_iterations)
        iteration_progress = max(0, t - self.warm_start_iterations)
        t_normalized = min(1.0, max(0.0, iteration_progress / float(total_iterations)))
        return float(
            self.dcfr_gamma_initial
            + (self.dcfr_gamma_final - self.dcfr_gamma_initial) * t_normalized
        )

    def _get_average_policy_weight_tensor(
        self, iterations: torch.Tensor
    ) -> torch.Tensor:
        """Return current-iteration average-policy weights for many iterations."""
        t = iterations.to(device=self.device, dtype=torch.float32)
        if self.cfr_type == CFRType.standard:
            return torch.ones_like(t)
        if self.cfr_type == CFRType.linear:
            return torch.full_like(t, 2.0)
        if self.cfr_type == CFRType.discounted:
            gamma = self._get_dcfr_gamma_tensor(iterations)
            progress = (t - float(self.dcfr_delay)).clamp(min=0.0) / float(
                self._average_accumulation_window()
            )
            weights = progress.pow(gamma)
            return torch.where(
                t > float(self.dcfr_delay),
                weights,
                torch.zeros_like(t),
            )
        raise ValueError(f"Unsupported CFR type: {self.cfr_type}")

    def _get_dcfr_gamma_tensor(self, iterations: torch.Tensor) -> torch.Tensor:
        """Return the gamma value that apply_schedules uses at each iteration."""
        t = iterations.to(device=self.device, dtype=torch.float32)
        if self.dcfr_gamma_final is None:
            return torch.full_like(t, float(self.dcfr_gamma))

        total_iterations = max(1, self.cfr_iterations - self.warm_start_iterations)
        progress = (t - float(self.warm_start_iterations)).clamp(min=0.0)
        t_normalized = (progress / float(total_iterations)).clamp(min=0.0, max=1.0)
        return float(self.dcfr_gamma_initial) + (
            float(self.dcfr_gamma_final) - float(self.dcfr_gamma_initial)
        ) * t_normalized

    def _reset_average_policy_accumulators(self) -> None:
        """Clear true CFR average-strategy accumulators for a fresh subgame."""
        if (
            hasattr(self, "average_policy_numerator")
            and self.average_policy_numerator.shape == self.policy_probs_avg.shape
        ):
            self.average_policy_numerator.zero_()
            self.average_policy_denominator.zero_()
        else:
            self.average_policy_numerator = torch.zeros_like(self.policy_probs_avg)
            self.average_policy_denominator = torch.zeros_like(self.policy_probs_avg)
        self.average_policy_initialized = False

    def _ensure_average_policy_accumulators(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            not hasattr(self, "average_policy_numerator")
            or self.average_policy_numerator.shape != self.policy_probs_avg.shape
        ):
            self.average_policy_numerator = torch.zeros_like(self.policy_probs_avg)
            self.average_policy_denominator = torch.zeros_like(self.policy_probs_avg)
            self.average_policy_initialized = False
        return self.average_policy_numerator, self.average_policy_denominator

    @torch.no_grad()
    def _get_model_policy_probs(self, indices: torch.Tensor) -> torch.Tensor:
        """Get policy probabilities from model for given indices."""
        policy_encoder = getattr(self, "policy_feature_encoder", self.feature_encoder)
        policy_model = getattr(self, "policy_model", self.model)
        features = policy_encoder.encode(self.beliefs, indices=indices)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if isinstance(policy_model, BetterTRM):
                latent = None
                for supervision in range(self.num_supervisions):
                    model_output = policy_model(
                        features,
                        include_policy=supervision == self.num_supervisions - 1,
                        include_value=False,
                        latent=latent,
                    )
                    latent = model_output.latent
            else:
                model_output = policy_model(features, include_policy=True)

        logits = model_output.policy_logits.float()
        legal_masks = self.legal_mask[indices]
        masked_logits = compute_masked_logits(logits, legal_masks[:, None, :])
        probs = F.softmax(masked_logits, dim=-1)
        probs.masked_fill_(
            (self.child_count[indices] == 0)[:, None, None],
            0.0,
        )
        return probs

    def _calculate_reach_weights(
        self, target: torch.Tensor, policy: torch.Tensor
    ) -> None:
        """Calculate self reach weights for each node.

        Note: Root nodes should already be initialized to 1.0 in initialize_subgame
        and are never updated by this method.
        """
        for depth in range(self.tree_depth):
            offset_next = self.depth_offsets[depth + 1]
            offset_next_next = self.depth_offsets[depth + 2]

            target_dest = target[offset_next:offset_next_next]
            target_dest[:] = self._fan_out(target, level=depth)

            prev_actor_dest = self.prev_actor[offset_next:offset_next_next]
            prev_actor_indices = prev_actor_dest[:, None, None].expand(
                -1, -1, NUM_HANDS
            )
            policy_dest = policy[offset_next:offset_next_next]
            target_dest.scatter_reduce_(
                dim=1,
                index=prev_actor_indices,
                src=policy_dest[:, None],
                reduce="prod",
                include_self=True,
            )

        self._mask_invalid(target)
        self._block_beliefs(target)

    def _propagate_all_beliefs(
        self,
        target: torch.Tensor | None = None,
        reach_weights: torch.Tensor | None = None,
    ) -> None:
        """Propagate beliefs from all valid nodes to all valid nodes."""
        N = self.root_nodes

        if target is None:
            target = self.beliefs
        if reach_weights is None:
            reach_weights = self.self_reach

        target[:] = self._fan_out_deep(target[:N]) * reach_weights

        # Precondition: reach_weights should be board-blocked, so the multiplication
        # will block target as well. All that's left is normalizing.
        self._normalize_beliefs(target)

    def _get_sampling_schedule(self) -> torch.Tensor:
        N = self.root_nodes
        if self.cfr_type == CFRType.discounted:
            sample_low = max(self.warm_start_iterations, self.dcfr_delay) + 1
        else:
            sample_low = self.warm_start_iterations + 1
        sample_low = min(sample_low, self.cfr_iterations)
        sample_high = max(self.cfr_iterations, sample_low + 1)
        iterations = torch.arange(
            sample_low, sample_high, dtype=torch.long, device=self.device
        )
        distribution = self._get_average_policy_weight_tensor(iterations)
        distribution_sum = distribution.sum()
        if distribution_sum <= 0:
            distribution.fill_(1.0)
            distribution_sum = distribution.sum()
        distribution /= distribution_sum
        t_sample = torch.multinomial(
            distribution, N, replacement=True, generator=self.generator
        )
        t_sample += sample_low

        return self._fan_out_deep(t_sample)

    def _init_hand_rank_data(self) -> None:
        device = self.device
        indices = self.showdown_indices
        M = indices.numel()
        board = self.env.board_indices[indices].int()  # (M,5)

        # Sorted position k (0..1325) replicated across batch
        k = torch.arange(NUM_HANDS, device=device).expand(
            indices.numel(), -1
        )  # (M,1326)

        # --- Ranks & sorted order per env (river deterministic strength) ---
        # hand_ranks: (M,1326) any integer/monotone rank key s.t. equal => tie
        # sorted_indices: argsort by (rank, tiebreak) ascending (weaker -> stronger)
        hand_ranks, sorted_indices = rank_hands(board)  # both (M,1326)

        # Ranks in sorted order
        ranks_sorted = torch.gather(hand_ranks, 1, sorted_indices)  # (M,1326)
        assert torch.all(ranks_sorted[:, 1:] >= ranks_sorted[:, :-1]), (
            "rank_hands order is descending; flip or fix rank_hands"
        )

        # --- Tie groups: start flags, group ids, [L,R] spans per sorted position ---
        is_start = torch.ones_like(ranks_sorted, dtype=torch.bool)  # (M,1326)
        is_start[:, 1:] = ranks_sorted[:, 1:] != ranks_sorted[:, :-1]
        group_id = is_start.cumsum(dim=1, dtype=torch.int) - 1  # (M,1326), 0..G-1

        # For each group id, store first/last index in sorted order
        starts = torch.full((M, NUM_HANDS), NUM_HANDS, dtype=torch.int, device=device)
        ends = torch.full((M, NUM_HANDS), -1, dtype=torch.int, device=device)
        starts.scatter_reduce_(1, group_id, k.int(), reduce="amin", include_self=True)
        ends.scatter_reduce_(1, group_id, k.int(), reduce="amax", include_self=True)

        # L,R per sorted position
        L = torch.gather(starts, 1, group_id)  # (M,1326)
        R = torch.gather(ends, 1, group_id)  # (M,1326)
        L_idx = L
        R_idx = (R + 1).clamp(max=NUM_HANDS)
        if self.CHECK_INVARIANTS:
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
        hand_ok_mask = self.allowed_hands[indices]
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

    @torch.compile(dynamic=True)
    def _showdown_value_both(self, beliefs: torch.Tensor) -> torch.Tensor:
        """Compute showdown values for both players."""
        result = torch.empty_like(beliefs)
        result[:, 0] = self._showdown_value(beliefs, 0)
        result[:, 1] = self._showdown_value(beliefs, 1)
        return result

    def _showdown_value(self, beliefs: torch.Tensor, hero: int) -> torch.Tensor:
        """
        Exact river showdown EV using rank-CDF + blocker correction.
        Returns per-hand EV [N, 1326] (unsorted/original hand order) per env.
        Result is from hero perspective.

        Args:
            hero: Index of hero player (0 or 1).
            indices: Indices of nodes to compute showdown values for.

        Returns:
            Per-hand EV [N, 1326] (unsorted/original hand order) per env.
        """
        indices = self.showdown_indices
        M = indices.numel()
        device = self.device
        dtype = torch.float32  # or match belief dtype
        villain = 1 - hero

        if M == 0:
            return torch.zeros(0, NUM_HANDS, device=device, dtype=dtype)

        # --- Beliefs & boards ---
        # Showdown value always uses the normal beliefs, not the average beliefs.
        # We store it in latest_values which always corresponds to non-average beliefs.
        b_opp = beliefs[:, villain, :].to(dtype)  # (M,1326)

        sorted_indices = self.hand_rank_data.sorted_indices
        inv_sorted = self.hand_rank_data.inv_sorted
        H = self.hand_rank_data.H
        card_ok = self.hand_rank_data.card_ok
        hand_ok_mask = self.hand_rank_data.hand_ok_mask
        hands_c1c2_sorted = self.hand_rank_data.hands_c1c2_sorted
        L_idx = self.hand_rank_data.L_idx
        R_idx = self.hand_rank_data.R_idx

        c1 = hands_c1c2_sorted[..., 0]  # (M,1326)
        c2 = hands_c1c2_sorted[..., 1]  # (M,1326)

        # Sort opponent marginal by strength order
        b_opp_sorted = b_opp.gather(1, sorted_indices)  # (M,1326)

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
        ).clamp(min=1e-8)
        valid_denom = denom > 1e-8
        if self.CHECK_INVARIANTS:
            assert ((valid_denom) | ((win_mass < 1e-5) & (tie_mass < 1e-5))).all()

        # Probabilities & EV (in sorted order)
        win_prob = torch.where(valid_denom, win_mass / denom, 0.0)
        tie_prob = torch.where(valid_denom, tie_mass / denom, 0.0)
        loss_prob = torch.where(valid_denom, 1.0 - win_prob - tie_prob, 0.0)

        EV_hand_sorted = win_prob - loss_prob

        # Map per-hand EV back to original hand order
        EV_hand = torch.gather(EV_hand_sorted, 1, inv_sorted)  # (M,1326)
        EV_hand = EV_hand * hand_ok_mask.to(dtype)  # zero impossible hands

        # Range EV for the player
        potential = self.showdown_potential[:, hero]
        scale = self.env.scale[indices]

        return EV_hand * potential[:, None] / scale[:, None]

    def _best_response_values(
        self,
        policy: torch.Tensor,
        beliefs: torch.Tensor,
        base_values: torch.Tensor,
        deviating_player: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute best response values."""
        N, B = self.root_nodes, self.num_actions
        top = self.depth_offsets[-2]
        if deviating_player is None:
            deviating_player = self._fan_out_deep(self.env.to_act[:N])

        values_br = torch.where(self.leaf_mask[:, None, None], base_values, 0.0)

        min_value = torch.finfo(base_values.dtype).min

        policy_src_all = self._pull_back(policy)

        actor_indices = self.env.to_act[:, None, None].expand(-1, -1, NUM_HANDS)
        actor_beliefs = beliefs.gather(1, actor_indices).squeeze(1)[:top]

        marginal_policy = policy_src_all * actor_beliefs[:, None, :]
        policy_blocked = calculate_unblocked_mass(marginal_policy)
        matchup_mass = calculate_unblocked_mass(actor_beliefs)
        opponent_conditioned_policy = torch.where(
            matchup_mass[:, None, :] > 1e-5,
            policy_blocked / matchup_mass[:, None, :],
            0.0,
        )

        for depth in range(self.tree_depth - 1, -1, -1):
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]

            indices = torch.arange(offset_next - offset, device=self.device)
            actor = self.env.to_act[offset:offset_next]
            deviator = deviating_player[offset:offset_next]
            invalid_children = ~self.child_mask[offset:offset_next]

            values_src = self._pull_back(values_br, level=depth)  # [K, B, 2, 1326]
            policy_src = policy_src_all[offset:offset_next]
            opponent_policy = opponent_conditioned_policy[offset:offset_next]

            actor_indices = actor[:, None, None, None].expand(-1, B, 1, NUM_HANDS)
            opp_indices = (1 - actor)[:, None, None, None].expand(-1, B, 1, NUM_HANDS)
            # Both [K, B, 1326]
            actor_values_src = values_src.gather(2, actor_indices).squeeze(2)
            opp_values_src = values_src.gather(2, opp_indices).squeeze(2)

            actor_values_for_best = actor_values_src.masked_fill(
                invalid_children[:, :, None], min_value
            )
            best_action = actor_values_for_best.argmax(dim=1)  # [K, 1326]
            # [K, 1326]
            best_actor_values = actor_values_src.gather(
                1, best_action[:, None, :]
            ).squeeze(1)

            # Public belief over deviator hands at s (not action-dependent)
            deviator_beliefs = actor_beliefs[offset:offset_next]

            # 1) Histogram the deviator belief by the BR-chosen action a*(h_i)
            #    mass_by_action[a, h_i] = b_i(h_i|s) if a*(h_i)==a else 0
            mass_by_action = torch.zeros(
                deviator_beliefs.size(0),
                B,
                deviator_beliefs.size(1),
                dtype=deviator_beliefs.dtype,
                device=self.device,
            )  # [n_dev, A, H_dev]
            # Partition belief by best action.
            mass_by_action.scatter_add_(
                1, best_action[:, None, :], deviator_beliefs[:, None, :]
            )

            # 2) Blocker-project that mass to opponent hands and normalize per h_-i
            mass_blocked = calculate_unblocked_mass(mass_by_action)  # [M, B, 1326]
            dev_match = matchup_mass[offset:offset_next][:, None, :]  # [M, 1, 1326]
            P_dev = torch.where(
                dev_match > 1e-5,
                mass_blocked / dev_match,  # P_dev(a | s, h_-i)
                0.0,
            )  # [M, B, 1326]

            # 3) Expectation of opponent continuation values under P_dev
            v_opp_exp = (P_dev * opp_values_src).sum(dim=1)  # [M, 1326]

            # Actor: deviating player gets best value, otherwise average value.
            actor_values = torch.where(
                (deviator == actor)[:, None],
                best_actor_values,  # case 1
                (actor_values_src * policy_src).sum(dim=1),  # case 3
            )
            # Non-actor: deviating player gets average value.
            # Non-deviating player gets value assuming deviating player plays best action.
            opp_values = torch.where(
                (deviator == actor)[:, None],
                v_opp_exp,  # case 2
                (opp_values_src * opponent_policy).sum(dim=1),  # case 4
            )

            values_br[indices + offset, actor] = actor_values
            values_br[indices + offset, 1 - actor] = opp_values

            # Re-add leaf values (which were just overwritten).
            torch.where(
                self.leaf_mask[offset:offset_next, None, None],
                base_values[offset:offset_next],
                values_br[offset:offset_next],
                out=values_br[offset:offset_next],
            )

        return values_br

    def _compute_exploitability(self) -> ExploitabilityStats:
        N = self.root_nodes
        if N == 0:
            empty = torch.empty(0, device=self.device, dtype=self.float_dtype)
            empty2 = torch.empty(0, 2, device=self.device, dtype=self.float_dtype)
            return ExploitabilityStats(
                local_exploitability=empty,
                local_best_response_values=empty2,
            )

        policy = self.policy_probs_avg
        beliefs = self.beliefs_avg
        leaf_values = self.values_avg.clamp(-1.0, 1.0)

        base_values = torch.zeros_like(leaf_values)
        self.compute_expected_values(
            policy=policy, beliefs=beliefs, leaf_values=leaf_values, values=base_values
        )

        improvements_by_player = []
        br_values_by_player = []
        for player in range(self.num_players):
            deviator = torch.full(
                (N,), player, device=self.device, dtype=self.env.to_act.dtype
            )
            br_values = self._best_response_values(
                policy,
                beliefs,
                leaf_values,
                deviating_player=self._fan_out_deep(deviator),
            )

            base_root = base_values[:N, player]  # (N, NUM_HANDS)
            br_root = br_values[:N, player]  # (N, NUM_HANDS)
            root_beliefs = beliefs[:N, player]  # (N, NUM_HANDS)

            improvements_by_player.append(
                ((br_root - base_root) * root_beliefs).sum(dim=-1)
            )
            br_values_by_player.append((br_root * root_beliefs).sum(dim=-1))

        improvements = improvements_by_player[0] + improvements_by_player[1]
        br_values_agg = torch.stack(br_values_by_player, dim=-1)  # (N, 2)

        return ExploitabilityStats(
            local_exploitability=improvements, local_best_response_values=br_values_agg
        )

    def _local_exploitability_mbbg(
        self, local_exploitability: torch.Tensor
    ) -> torch.Tensor:
        """Convert stack-normalized exploitability to milli-big-blinds/game."""
        N = local_exploitability.shape[0]
        scale = self.env.scale[:N].to(dtype=local_exploitability.dtype)
        bb = max(float(self.env.bb), 1.0)
        return local_exploitability * scale * (1000.0 / bb)

    def _compute_policy_node_reach(self, top: int) -> torch.Tensor:
        """Approximate public-node reach under the average policy.

        This is the compatible private-hand-pair average of both players'
        average self-reach probabilities at each public node. The normalization
        keeps root nodes near 1.0 even when public board cards block combos.
        """
        allowed = self.allowed_hands[:top].to(dtype=self.float_dtype)
        reach = self.self_reach_avg[:top].to(dtype=self.float_dtype)
        opp_unblocked = calculate_unblocked_mass(reach[:, 1])
        numer = (reach[:, 0] * opp_unblocked * allowed).sum(dim=-1)

        allowed_unblocked = calculate_unblocked_mass(allowed)
        denom = (allowed * allowed_unblocked).sum(dim=-1).clamp(min=1e-12)
        return (numer / denom).clamp(min=0.0, max=1.0)

    def _should_record_policy_node_reach(self) -> bool:
        cfg = getattr(self, "cfg", None)
        train_cfg = getattr(cfg, "train", None)
        mode = getattr(train_cfg, "policy_node_weighting", None)
        if mode is None:
            return True
        return getattr(mode, "value", mode) != "uniform"

    # ============================================================================
    # Core Logic Methods (in order called by cfr_iteration and evaluate_cfr)
    # ============================================================================

    def initialize_subgame(
        self,
        src_env: HUNLTensorEnv | PBSEnv,
        src_indices: torch.Tensor,
        initial_beliefs: torch.Tensor | None = None,
    ) -> None:
        """Copy root states into the search tree, reset per-node buffers, and expand.

        Args:
            src_env: Batched environment that holds the source root public states.
            src_indices: Row indices inside `src_env` to copy into the tree roots.
            initial_beliefs: Optional belief tensor aligned with `src_indices`.
        """
        # Construct the subgame tree first (subclass-specific, allocates tensors)
        self._construct_subgame(src_env, src_indices)
        N = self.root_nodes

        # Handle initial beliefs
        if initial_beliefs is None:
            initial_beliefs = torch.full(
                (N, self.num_players, NUM_HANDS),
                1.0 / NUM_HANDS,
                dtype=self.float_dtype,
                device=self.device,
            )
        else:
            initial_beliefs = initial_beliefs.to(
                device=self.device, dtype=self.float_dtype
            )

        # Preserve the incoming root beliefs for pre-chance value targets. At
        # street transitions these may still contain hands blocked by the newly
        # dealt board, so the search-facing beliefs are normalized separately
        # after the root board mask is available below.
        self.root_pre_chance_beliefs[:N] = initial_beliefs
        self.self_reach[:N] = 1.0
        self.self_reach_avg[:N] = 1.0

        # latent always have shape [model_indices.numel(), model.hidden_dim]
        self.model_indices = self._compute_model_indices()
        self._validate_model_leaf_phases()
        self.latent = None

        # Compute allowed hands from root board
        board_mask_root = self.env.board_onehot[:N].any(dim=1).reshape(N, -1).float()
        root_allowed = (self.combo_onehot_float @ board_mask_root.T).T < 0.5
        root_allowed_prob = root_allowed.to(dtype=self.float_dtype)
        root_allowed_prob /= root_allowed_prob.sum(dim=-1, keepdim=True).clamp(min=1.0)

        # Fan out allowed hands to all nodes
        self.allowed_hands = self._fan_out_deep(root_allowed)
        self.allowed_hands_prob = self._fan_out_deep(root_allowed_prob)

        # Search/policy evaluation operates in the current public state. At street
        # transitions, initial_beliefs may still be pre-chance ranges from the
        # previous board, but initialize_policy_and_beliefs() asks the model for a
        # root policy before the usual per-level block/normalize pass runs. Block
        # and renormalize the search-facing roots here so model warm-start never
        # conditions on hands made impossible by the newly dealt board. Keep the
        # original pre-chance roots for transition-target supervision.
        root_beliefs = self.beliefs[:N]
        root_beliefs[:] = initial_beliefs
        root_beliefs.masked_fill_((~root_allowed)[:, None, :], 0.0)
        denom = root_beliefs.sum(dim=-1, keepdim=True)
        torch.where(
            denom > 1e-5,
            root_beliefs / denom,
            root_allowed_prob[:, None, :],
            out=root_beliefs,
        )
        self.beliefs_avg[:N] = root_beliefs

        # Initialize hand rank data
        self._init_hand_rank_data()

        # Record statistics
        self.stats["evaluator_street"] = self.env.street[:N].float().mean().item()
        self.stats["evaluator_total_nodes"] = float(self.total_nodes)
        self.stats["evaluator_root_nodes"] = float(self.root_nodes)
        self.stats["evaluator_tree_depth"] = float(self.tree_depth)

    @torch.no_grad()
    @profile
    def initialize_policy_and_beliefs(self) -> None:
        """Push public beliefs down the tree using the freshly initialised policy."""
        self.policy_probs.zero_()
        self.model.eval()

        # Use defensive loop bounds: len(depth_offsets) - 2 ensures we don't go out of bounds
        if self.tree_depth == 0:
            # No depth to process, just block and normalize beliefs
            self._block_beliefs()
            self._normalize_beliefs()
            self._mask_invalid(self.policy_probs)
            self._calculate_reach_weights(self.self_reach, self.policy_probs)
            self.policy_probs_avg[:] = self.policy_probs
            self.self_reach_avg[:] = self.self_reach
            self.beliefs_avg[:] = self.beliefs
            self.beliefs_sample[:] = self.beliefs
            self._reset_average_policy_accumulators()
            return

        # Pre-allocate policy_probs_src for efficiency (used by sparse, but harmless for dense)
        top = self.depth_offsets[-2] if len(self.depth_offsets) > 1 else self.root_nodes
        policy_probs_src = torch.empty(
            top, self.num_actions, NUM_HANDS, device=self.device, dtype=self.float_dtype
        )

        for depth in range(self.tree_depth):
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]
            offset_next_next = self.depth_offsets[depth + 2]

            # Get policy probabilities from model for nodes at current depth
            indices = torch.arange(offset, offset_next, device=self.device)
            model_policy = self._get_model_policy_probs(indices)  # [K, B, NUM_HANDS]
            policy_probs_src[offset:offset_next] = model_policy.permute(0, 2, 1)

            # Push down policy to children using _push_down (works for both dense and sparse)
            self.policy_probs[offset_next:offset_next_next] = self._push_down(
                policy_probs_src, level=depth
            )

            # Propagate beliefs from current level to next level
            self._propagate_level_beliefs(depth)

            # Block and normalize beliefs after each level
            self._block_beliefs()
            self._normalize_beliefs()

        # Mask invalid policy probs (noop for sparse, masks for dense)
        self._mask_invalid(self.policy_probs)

        # Calculate reach weights
        self._calculate_reach_weights(self.self_reach, self.policy_probs)

        # Initialize averages
        self.policy_probs_avg[:] = self.policy_probs
        self.self_reach_avg[:] = self.self_reach
        self.beliefs_avg[:] = self.beliefs
        self.beliefs_sample[:] = self.beliefs
        self._reset_average_policy_accumulators()

    def warm_start(self) -> None:
        """Simple warm start: use model values and do a best-response pass."""
        self.set_leaf_values(0)
        if self.CHECK_INVARIANTS and not self.latest_values.isfinite().all():
            num_nonfinite = (~self.latest_values.isfinite()).sum().item()
            raise ValueError(
                f"Non-finite values in latest_values after set_leaf_values: "
                f"{num_nonfinite} non-finite elements out of {self.latest_values.numel()}"
            )

        self.compute_expected_values(
            policy=self.policy_probs,
            beliefs=self.beliefs,
            leaf_values=self.latest_values,
            values=self.latest_values,
        )
        if self.CHECK_INVARIANTS and not self.latest_values.isfinite().all():
            num_nonfinite = (~self.latest_values.isfinite()).sum().item()
            raise ValueError(
                f"Non-finite values in latest_values after compute_expected_values: "
                f"{num_nonfinite} non-finite elements out of {self.latest_values.numel()}"
            )

        self._record_initial_exploitability()

        # If configured, seed regrets so regret matching replays the model policy.
        if self.warm_start_type == WarmStartType.model:
            bottom = self.depth_offsets[1]
            weight = float(self.warm_start_iterations) * float(
                self.warm_start_multiplier
            )
            regrets = self.compute_instantaneous_regrets(self.latest_values)
            weights = weight * regrets[bottom:].clamp(min=0.0).mean(dim=-1)
            self.cumulative_regrets[bottom:] = (
                self.policy_probs[bottom:] * weights[:, None]
            )
            self.update_policy(self.warm_start_iterations)
            return

        # [M, ]
        values_br_p0 = self._best_response_values(
            self.policy_probs,
            self.beliefs,
            self.latest_values,
            torch.zeros_like(self.env.to_act),
        )
        values_br_p1 = self._best_response_values(
            self.policy_probs,
            self.beliefs,
            self.latest_values,
            torch.ones_like(self.env.to_act),
        )
        # NB: Invalid on root nodes, but we don't use them for regret/policy calculation.
        values_br = torch.where(
            self.prev_actor[:, None, None] == 0, values_br_p0, values_br_p1
        )

        if self.CHECK_INVARIANTS:
            assert values_br.isfinite().all()

        # heuristic: scale regrets by the number of warm start iterations
        regrets = self.compute_instantaneous_regrets(
            values_achieved=values_br, values_expected=self.latest_values
        )
        scale = float(self.warm_start_iterations) * float(self.warm_start_multiplier)
        self.cumulative_regrets += scale * regrets
        self.update_policy(self.warm_start_iterations)

    def _maybe_enforce_zero_sum(
        self,
        hand_values: torch.Tensor,
        player_beliefs: torch.Tensor,
        ignore_mask: torch.Tensor | None = None,
    ) -> None:
        """
        Enforce zero-sum constraint on hand values by subtracting the weighted average.

        Args:
            hand_values: Tensor of shape (batch, num_players, NUM_HANDS)
            player_beliefs: Tensor of shape (batch, num_players, NUM_HANDS)
        """
        if self.model.enforce_zero_sum and self.num_players == 2:
            hand_value_sums = (
                (hand_values * player_beliefs)
                .sum(dim=2, keepdim=True)
                .mean(dim=1, keepdim=True)
            )
            if ignore_mask is not None:
                hand_value_sums.masked_fill_(ignore_mask[:, None, None], 0.0)
            return hand_values - hand_value_sums
        else:
            return hand_values

    def _eval_value_model(
        self, value_model, features: MLPFeatures, *, use_pre_head: bool
    ) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if isinstance(value_model, BetterTRM):
                # Note self.latent gets reinitialized for each subgame.
                model_output = value_model(
                    features,
                    include_policy=False,
                    latent=self.latent,
                )
                self.latent = model_output.latent
            else:
                if use_pre_head and hasattr(value_model, "forward_pre"):
                    hand_values = value_model.forward_pre(features).to(self.float_dtype)
                else:
                    model_output = value_model(features, include_policy=False)
                    hand_values = model_output.hand_values.to(self.float_dtype)
        return hand_values

    def _model_leaf_values(self, features: MLPFeatures) -> torch.Tensor:
        value_model = getattr(self, "value_model", self.model)
        closing_value_model = getattr(self, "closing_leaf_value_model", None)
        if closing_value_model is None:
            return self._eval_value_model(
                value_model,
                features,
                use_pre_head=(
                    self._uses_street_cutoff_schedule()
                    and hasattr(value_model, "forward_pre")
                ),
            )

        model_indices = self.model_indices
        closing_mask = self.new_street_mask[model_indices]
        hand_values = torch.empty(
            len(features),
            self.num_players,
            NUM_HANDS,
            device=features.context.device,
            dtype=self.float_dtype,
        )
        if (~closing_mask).any():
            same_street_values = self._eval_value_model(
                value_model,
                features[~closing_mask],
                use_pre_head=(
                    self._uses_street_cutoff_schedule()
                    and hasattr(value_model, "forward_pre")
                ),
            )
            hand_values[~closing_mask] = same_street_values
        if closing_mask.any():
            closing_values = self._eval_value_model(
                closing_value_model,
                features[closing_mask],
                use_pre_head=True,
            )
            hand_values[closing_mask] = closing_values
        return hand_values

    def _set_model_values_impl(
        self, t: int, beliefs: torch.Tensor, features: MLPFeatures
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Set model values for non-terminal leaves
        hand_values = self._model_leaf_values(features)

        if (
            not self.cfr_avg
            or t <= 1
            or self.last_model_values is None
            or self._average_accumulation_delayed(t)
        ):
            new_values = torch.index_copy(
                self.latest_values,
                0,
                self.model_indices,
                hand_values,
            )
        else:
            # Mix with previous values (CFR-AVG style)
            old, new = self._get_mixing_weights(t)
            unmixed = (old + new) * hand_values - old * self.last_model_values
            unmixed /= new
            unmixed = self._maybe_enforce_zero_sum(unmixed, beliefs)
            new_values = torch.index_copy(
                self.latest_values,
                0,
                self.model_indices,
                unmixed,
            )
        return new_values, hand_values

    def _set_model_values(
        self, t: int, beliefs: torch.Tensor, features: MLPFeatures
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._set_model_values_impl(t, beliefs, features)

    @torch.no_grad()
    def set_leaf_values(self, t: int, beliefs: torch.Tensor | None = None) -> None:
        """Set leaf values from model or terminal states."""
        if beliefs is None:
            beliefs = self.beliefs_avg if self.cfr_avg else self.beliefs

        if self.model_indices.numel() > 0:
            value_encoder = getattr(self, "value_feature_encoder", self.feature_encoder)
            features = value_encoder.encode(
                beliefs, pre_chance_node=self.new_street_mask
            )

            # Pass the same beliefs used for feature encoding to _set_model_values
            # so that zero-sum enforcement is consistent with the model input
            new_values, last_model_values = self._set_model_values(
                t, beliefs[self.model_indices], features[self.model_indices]
            )
            # this is necessary because of torch.compile.
            self.latest_values = new_values.clone()
            self.last_model_values = last_model_values.clone()
        else:
            self.last_model_values = self.latest_values.new_empty(
                (0, self.num_players, NUM_HANDS)
            )

        # Set showdown values. Heads-up keeps the exact per-hand river resolver;
        # multiway PBS currently falls back to the side-pot-aware marginal EV
        # helper and broadcasts the seat EV over private hands.
        if self.showdown_indices.numel() > 0:
            showdown_beliefs = beliefs[self.showdown_indices]
            if self.num_players == 2:
                showdown_values = self._showdown_value_both(showdown_beliefs)
                self.latest_values[self.showdown_indices] = showdown_values
            elif isinstance(self.env, PBSEnv):
                showdown_rewards = self.env.expected_showdown_rewards(
                    showdown_beliefs,
                    env_indices=self.showdown_indices,
                )
                self.latest_values[self.showdown_indices] = showdown_rewards[:, :, None]
            else:
                raise NotImplementedError(
                    "Multiway showdown values require a PBSEnv-backed public state."
                )
        set_allin = getattr(self, "_set_allin_call_values", None)
        if set_allin is not None:
            set_allin(beliefs)

    def compute_expected_values(
        self,
        policy: torch.Tensor | None = None,
        beliefs: torch.Tensor | None = None,
        leaf_values: torch.Tensor | None = None,
        values: torch.Tensor | None = None,
    ) -> None:
        """Back up values from leaves to root under the provided policy."""
        if policy is None:
            policy = self.policy_probs
        if beliefs is None:
            beliefs = self.beliefs
        if leaf_values is None:
            leaf_values = self.latest_values
        if values is None:
            values = leaf_values

        if leaf_values is values:
            values.masked_fill_((~self.leaf_mask)[:, None, None], 0.0)
        else:
            torch.where(
                self.leaf_mask[:, None, None],
                leaf_values,
                torch.zeros_like(values),
                out=values,
            )

        bottom, top = self.depth_offsets[1], self.depth_offsets[-2]
        actor_indices = self.env.to_act[:top]
        actor_indices_expanded = actor_indices[:top, None, None].expand(
            -1, -1, NUM_HANDS
        )
        actor_beliefs = beliefs[:top].gather(1, actor_indices_expanded).squeeze(1)
        beliefs_dest = self._fan_out(actor_beliefs)
        marginal_policy = beliefs_dest * policy[bottom:]

        policy_blocked = calculate_unblocked_mass(marginal_policy)
        matchup_values = calculate_unblocked_mass(beliefs_dest)
        opponent_conditioned_policy = torch.zeros_like(policy)
        torch.where(
            matchup_values > 1e-5,
            policy_blocked / matchup_values,
            torch.zeros_like(policy_blocked),
            out=opponent_conditioned_policy[bottom:],
        )

        for depth in range(self.tree_depth - 1, -1, -1):
            offset_next = self.depth_offsets[depth + 1]
            offset_next_next = self.depth_offsets[depth + 2]

            prev_actor_indices = self.prev_actor[offset_next:offset_next_next]
            weighted_child_values = values[offset_next:offset_next_next].clone()
            player_ids = torch.arange(self.num_players, device=self.device)
            is_actor = player_ids[None, :, None] == prev_actor_indices[:, None, None]
            action_weights = torch.where(
                is_actor,
                policy[offset_next:offset_next_next, None, :],
                opponent_conditioned_policy[offset_next:offset_next_next, None, :],
            )
            weighted_child_values *= action_weights

            self._pull_back_sum(weighted_child_values, values, level=depth)

    def compute_instantaneous_regrets(
        self, values_achieved: torch.Tensor, values_expected: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute instantaneous regrets for each action at each node.

        Args:
            values_achieved: [M, 2, 1326] tensor of values for each node.
            values_expected: [M, 2, 1326] tensor of expected values for each node, or none to use values_achieved.

        Returns:
            regrets: [M, 1326] tensor of regrets for taking the action to get to the node.
        """
        if values_expected is None:
            values_expected = values_achieved

        bottom = self.depth_offsets[1]
        beliefs = self.beliefs_avg if self.cfr_avg else self.beliefs

        regrets = torch.zeros_like(self.policy_probs)

        src_actor_indices = self.env.to_act[:, None, None].expand(-1, -1, NUM_HANDS)
        prev_actor_indices = self.prev_actor[bottom:, None, None].expand(
            -1, -1, NUM_HANDS
        )

        # This represents other players' reach mass at the source node, projected
        # into the acting player's hand space by blocker compatibility.
        unblocked_reach = calculate_unblocked_mass(beliefs)
        player_ids = torch.arange(self.num_players, device=self.device)
        other_live = player_ids[None, :, None] != self.env.to_act[:, None, None]
        if hasattr(self.env, "has_folded"):
            other_live &= ~self.env.has_folded[:, :, None]
        src_weights = torch.where(
            other_live,
            unblocked_reach.clamp_min(1e-12),
            torch.ones_like(unblocked_reach),
        ).prod(dim=1)
        src_weights *= self.allowed_hands.to(dtype=src_weights.dtype)

        # Weight advantages by our mass unblocked by the opponent hands.
        weights = self._fan_out(src_weights)

        # The value at a node is already the EV over all actions.
        actor_values = values_expected.gather(1, src_actor_indices).squeeze(1)  # bottom
        actor_values_expected = self._fan_out(actor_values)
        actor_values_achieved = (
            values_achieved[bottom:].gather(1, prev_actor_indices).squeeze(1)
        )

        advantages = actor_values_achieved - actor_values_expected

        regrets[bottom:] = weights * advantages

        # Mask invalid nodes (noop for sparse, masks invalid nodes for dense)
        self._mask_invalid(regrets)

        return regrets

    @profile
    def update_policy(self, t: int) -> None:
        """Update policy using regret matching."""
        bottom = self.depth_offsets[1]
        positive_regrets = self.cumulative_regrets.clamp(min=0.0)
        regret_sum = torch.zeros_like(self.policy_probs)

        self._pull_back_sum(positive_regrets, regret_sum)
        denom = self._fan_out(regret_sum)

        # Get uniform policy fallback (1.0 / num_actions per node)
        uniform_fallback = self.uniform_policy[bottom:]

        torch.where(
            denom > 1e-8,
            positive_regrets[bottom:] / denom.clamp(min=1e-8),
            uniform_fallback,
            out=self.policy_probs[bottom:],
        )
        self._mask_invalid(self.policy_probs)

        self._calculate_reach_weights(self.self_reach, self.policy_probs)
        self._propagate_all_beliefs(self.beliefs, self.self_reach)

        self.update_average_policy(t)
        self._calculate_reach_weights(self.self_reach_avg, self.policy_probs_avg)
        self._propagate_all_beliefs(self.beliefs_avg, self.self_reach_avg)

    def update_average_policy(self, t: int) -> None:
        """Update the average policy using true CFR reach-weighted sums."""
        if (
            self.cfr_type == CFRType.discounted and self._average_accumulation_delayed(t)
        ):
            self.policy_probs_avg[:] = self.policy_probs
            self.average_policy_initialized = False
            return

        N = self.root_nodes
        weight = self._get_average_policy_weight(t)
        numerator, denominator = self._ensure_average_policy_accumulators()

        # Get actor indices at source nodes (nodes that have children)
        # _fan_out expects tensors aligned with source nodes (0 to depth_offsets[-2])
        top = self.depth_offsets[-2]
        actor_indices = self.env.to_act[:top, None, None].expand(-1, -1, NUM_HANDS)
        reach_actor = self.self_reach[:top].gather(1, actor_indices).squeeze(1)

        # Fan out actor reach to get per-action CFR average weights.
        reach_actor_dest = self._fan_out(reach_actor)
        contribution_weight = reach_actor_dest * weight
        numerator[N:] += contribution_weight * self.policy_probs[N:]
        denominator[N:] += contribution_weight
        self.average_policy_initialized = True

        torch.where(
            denominator[N:] > 1e-5,
            numerator[N:] / denominator[N:].clamp(min=1e-8),
            self.policy_probs[N:],
            out=self.policy_probs_avg[N:],
        )

        policy_sum = torch.zeros(
            self.depth_offsets[-2],
            NUM_HANDS,
            device=self.device,
            dtype=self.float_dtype,
        )
        self._pull_back_sum(self.policy_probs_avg, policy_sum)
        policy_denom = self._fan_out(policy_sum)
        torch.where(
            policy_denom > 1e-5,
            self.policy_probs_avg[N:] / policy_denom,
            self.policy_probs_avg[N:],
            out=self.policy_probs_avg[N:],
        )

        # Root nodes don't have policies (they're decision nodes, not action nodes)
        self.policy_probs_avg[:N] = 0.0

    def update_average_values(self, t: int) -> None:
        """
        Update average values with weighted average and enforce zero-sum constraint.

        Args:
            t: Current iteration number
        """

        old, new = self._get_mixing_weights(t)
        total = old + new
        if total == 0:
            return
        self.values_avg *= old
        self.values_avg += new * self.latest_values
        self.values_avg /= total
        self.values_avg[:] = self._maybe_enforce_zero_sum(
            self.values_avg, self.beliefs_avg, ignore_mask=self.env.done
        )

    def update_average_values_final(self) -> None:
        """
        Update average values with final policy values.
        """
        # Seed latest_values with the leaf values under beliefs_avg
        self.set_leaf_values(0, beliefs=self.beliefs_avg)
        # Using latest_values as leaf values, compute EVs into values_avg
        self.compute_expected_values(
            self.policy_probs_avg,
            self.beliefs_avg,
            self.latest_values,
            self.values_avg,
        )
        # Possibly redundant: enforce zero-sum on values_avg
        self.values_avg[:] = self._maybe_enforce_zero_sum(
            self.values_avg, self.beliefs_avg, ignore_mask=self.env.done
        )

    def apply_schedules(self, t: int) -> None:
        """Apply DCFR parameter schedules based on iteration count.

        Args:
            t: Current iteration number (0-indexed)
        """
        # Calculate progress through CFR iterations (excluding warm start)
        total_iterations = max(1, self.cfr_iterations - self.warm_start_iterations)
        iteration_progress = max(0, t - self.warm_start_iterations)
        t_normalized = min(1.0, max(0.0, iteration_progress / float(total_iterations)))

        # DCFR parameter schedules (linear interpolation)
        if self.dcfr_alpha_final is not None:
            self.dcfr_alpha = (
                self.dcfr_alpha_initial
                + (self.dcfr_alpha_final - self.dcfr_alpha_initial) * t_normalized
            )
        else:
            self.dcfr_alpha = self.dcfr_alpha_initial

        if self.dcfr_beta_final is not None:
            self.dcfr_beta = (
                self.dcfr_beta_initial
                + (self.dcfr_beta_final - self.dcfr_beta_initial) * t_normalized
            )
        else:
            self.dcfr_beta = self.dcfr_beta_initial

        if self.dcfr_gamma_final is not None:
            self.dcfr_gamma = (
                self.dcfr_gamma_initial
                + (self.dcfr_gamma_final - self.dcfr_gamma_initial) * t_normalized
            )
        else:
            self.dcfr_gamma = self.dcfr_gamma_initial

    @profile
    def cfr_iteration(self, t: int) -> None:
        """Run one CFR iteration."""
        # Apply schedules at the beginning of each iteration
        self.apply_schedules(t)

        sample_mask = self.t_sample == t
        torch.where(
            sample_mask[:, None],
            self.policy_probs,
            self.policy_probs_sample,
            out=self.policy_probs_sample,
        )
        torch.where(
            sample_mask[:, None, None],
            self.beliefs,
            self.beliefs_sample,
            out=self.beliefs_sample,
        )

        # Compute regrets
        regrets = self.compute_instantaneous_regrets(self.latest_values)

        if self.cfr_type == CFRType.linear:  # Alternate updates.
            regrets.masked_fill_(self.prev_actor[:, None] == t % self.num_players, 0.0)
        elif self.cfr_type == CFRType.discounted:
            numerator = torch.where(
                self.cumulative_regrets > 0, t**self.dcfr_alpha, t**self.dcfr_beta
            )
            denominator = torch.where(
                self.cumulative_regrets > 0,
                t**self.dcfr_alpha + 1,
                t**self.dcfr_beta + 1,
            )
            self.cumulative_regrets *= numerator
            self.cumulative_regrets /= denominator
        # Update cumulative regrets
        self.cumulative_regrets += regrets

        # CFR+ trick: clamp regrets to non-negative
        if self.cfr_plus:
            self.cumulative_regrets.clamp_(min=0)

        # Update policy. Only clone old_policy_probs on the iterations where
        # _record_stats actually inspects it (5 percentile iters per CFR run).
        if t in self._record_stats_percentile_ts():
            old_policy_probs = self.policy_probs.clone()
            self.update_policy(t)
            self._record_stats(t, old_policy_probs)
        else:
            self.update_policy(t)

        # Set leaf values and back up
        self.set_leaf_values(t)
        self.compute_expected_values()

        # Update average values
        if not self.use_final_policy_values:
            self.update_average_values(t)

    @profile
    def training_data(
        self, exclude_start: bool = True
    ) -> tuple[RebelBatch, RebelBatch, RebelBatch]:
        """Return training data from CFR evaluation."""
        N = self.root_nodes
        top = self.depth_offsets[-2] if len(self.depth_offsets) > 1 else N

        policy_targets = self._pull_back(self.policy_probs_avg)
        policy_targets = policy_targets[:top].permute(0, 2, 1)

        source_values = (
            self.latest_values if self.use_final_policy_values else self.values_avg
        )
        value_targets = source_values[:N].clamp(-1.0, 1.0)

        policy_encoder = getattr(self, "policy_feature_encoder", self.feature_encoder)
        value_encoder = getattr(self, "value_feature_encoder", self.feature_encoder)
        policy_features = policy_encoder.encode(
            self.beliefs_avg, pre_chance_node=False
        )[:top]
        value_features = value_encoder.encode(
            self.beliefs_avg, pre_chance_node=False
        )[:top]
        bin_amounts, _ = self.env.legal_bins_amounts_and_mask()
        legal_masks = self.legal_mask
        node_depth = torch.zeros(top, dtype=torch.long, device=self.device)
        for depth in range(self.tree_depth):
            node_depth[self.depth_offsets[depth] : self.depth_offsets[depth + 1]] = (
                depth
            )

        statistics = {
            "to_act": self.env.to_act,
            "street": self.env.street,
            "stage": 2 * self.env.street,
            "board": self.env.board_indices,
            "pot": self.env.pot,
            "bet_amounts": bin_amounts,
            "node_depth": node_depth,
        }

        exploit_stats = self._compute_exploitability()
        exploit_mbbg = self._local_exploitability_mbbg(
            exploit_stats.local_exploitability
        )

        value_statistics = {key: statistics[key][:N] for key in statistics}
        value_statistics["target_source"] = torch.full(
            (N,),
            TARGET_SOURCE_CFR_BACKUP,
            dtype=torch.long,
            device=self.device,
        )
        value_statistics["local_exploitability"] = exploit_stats.local_exploitability
        value_statistics["local_exploitability_mbbg"] = exploit_mbbg
        value_statistics["local_best_response_values"] = (
            exploit_stats.local_best_response_values
        )

        # Policy batch gets all valid, non-leaf states.
        # Use valid_mask directly (works for both: sparse has all-ones, dense has computed mask)
        valid_top = self.valid_mask[:top] & ~self.leaf_mask[:top]

        policy_statistics = {
            key: statistics[key][:top][valid_top] for key in statistics
        }
        policy_statistics["policy_node_reach"] = self._compute_policy_node_reach(top)[
            valid_top
        ]

        value_batch = RebelBatch(
            features=value_features[:N],
            value_targets=value_targets,
            legal_masks=legal_masks[:N],
            statistics=value_statistics,
        )

        street_root = self.env.street[:N]
        actions_root = self.env.actions_this_round[:N]
        mid_street_value_roots = (street_root < 4) & (actions_root > 0)
        if mid_street_value_roots.any():
            count = int(mid_street_value_roots.sum().item())
            self.stats["mid_street_value_root_count"] = float(count)
            warnings.warn(
                "training_data() received mid-street roots for value targets; "
                "value supervision is intended for street-boundary roots only. "
                f"count={count}",
                RuntimeWarning,
                stacklevel=2,
            )
        root_nodes = (street_root == 0) & (actions_root == 0)
        if exclude_start:
            value_batch = value_batch[~root_nodes]

        policy_batch = RebelBatch(
            features=policy_features[valid_top],
            policy_targets=policy_targets[valid_top],
            legal_masks=legal_masks[:top][valid_top],
            statistics=policy_statistics,
        )

        pre_features_all = value_encoder.encode(
            self.beliefs, pre_chance_node=True
        )
        pre_features_root = pre_features_all[:N].clone()
        pre_beliefs = self.root_pre_chance_beliefs[:N].reshape(N, -1)
        pre_features_root.beliefs = pre_beliefs

        value_targets_pre = value_targets.clone()
        value_statistics_pre = {
            key: value_statistics[key].clone() for key in value_statistics
        }
        value_statistics_pre["board"] = self.env.last_board_indices[:N].clone()
        prev_street = torch.where(
            (street_root > 0) & (street_root < 4) & (actions_root == 0),
            street_root - 1,
            street_root,
        )
        value_statistics_pre["street"] = prev_street
        value_statistics_pre["stage"] = 2 * prev_street + 1
        value_statistics_pre["target_source"] = torch.full(
            (N,),
            TARGET_SOURCE_CHANCE_EXPECTATION,
            dtype=torch.long,
            device=self.device,
        )

        start_mask = actions_root == 0

        turn_river_mask = start_mask & ((street_root == 2) | (street_root == 3))
        if turn_river_mask.any():
            expected_turn_river = self.chance_helper.single_card_chance_values(
                torch.where(turn_river_mask)[0],
                value_features[:N],
                self.root_pre_chance_beliefs,
                self.env.last_board_indices,
            )
            value_targets_pre[turn_river_mask] = expected_turn_river

        flop_mask = start_mask & (street_root == 1)
        if flop_mask.any():
            expected_flop = self.chance_helper.flop_chance_values(
                torch.where(flop_mask)[0],
                value_features[:N],
                self.root_pre_chance_beliefs,
            )
            value_targets_pre[flop_mask] = expected_flop

        transition_mask = turn_river_mask | flop_mask
        pre_value_batch = RebelBatch(
            features=pre_features_root,
            value_targets=value_targets_pre,
            legal_masks=legal_masks[:N],
            statistics=value_statistics_pre,
        )[transition_mask]

        return value_batch, pre_value_batch, policy_batch

    def evaluate_cfr(self, training_mode: bool = True) -> PublicBeliefState:
        """Run CFR iterations to evaluate the subgame.

        Returns:
            PublicBeliefState containing the sampled leaves.
        """
        self.model.eval()

        self.initialize_policy_and_beliefs()

        if self.warm_start_iterations > 0:
            self.warm_start()

        # Use t=0 here so set_leaf_values doesn't do the CFR-AVG de-averaging.
        self.set_leaf_values(0)
        self.compute_expected_values()
        self.values_avg[:] = self.latest_values

        self.t_sample = self._get_sampling_schedule()
        for t in range(self.warm_start_iterations, self.cfr_iterations):
            self.profiler_step()  # Profile start of CFR iteration
            self.cfr_iteration(t)

        if self.use_final_policy_values:
            self.update_average_values_final()

        # Record statistics
        self._record_action_mix()
        self._record_cfr_entropy()
        self._record_cumulative_regret()

        return self.sample_leaves(training_mode)

    # ============================================================================
    # Profiler Methods
    # ============================================================================

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

    # ============================================================================
    # Statistics Methods
    # ============================================================================

    def _record_stats_percentile_ts(self) -> set[int]:
        """The 5 CFR iterations at which `_record_stats` actually consumes
        `old_policy_probs`. Used by `cfr_iteration` to skip the full-tensor
        clone on the other iterations."""
        return {
            int(x)
            for x in torch.linspace(
                self.warm_start_iterations, self.cfr_iterations - 1, 5
            )
            .round()
            .int()
            .tolist()
        }

    def _record_stats(self, t: int, old_policy_probs: torch.Tensor) -> None:
        """Record statistics about the policy update."""

        # Compute the 5 percentile points (0, 25, 50, 75, 100)
        percentile_ts = (
            torch.linspace(self.warm_start_iterations, self.cfr_iterations - 1, 5)
            .round()
            .int()
            .tolist()
        )
        percentiles = [0, 25, 50, 75, 100]

        if t in percentile_ts:
            # Find which percentile this t corresponds to
            percentile_idx = percentile_ts.index(t)
            percentile = percentiles[percentile_idx]

            # Can either player get to this node with a given hand?
            reachable = (self.self_reach > 0).any(dim=1)[: self.depth_offsets[-2]]
            reachable_hand_count = reachable.sum(dim=-1)
            reachable_nodes = reachable_hand_count > 0

            child_start = self.depth_offsets[1]
            child_end = self.total_nodes
            parent_count = self.depth_offsets[-2]
            child_delta = (
                self.policy_probs[child_start:child_end]
                - old_policy_probs[child_start:child_end]
            ).abs()
            delta_by_parent_hand = torch.zeros(
                (parent_count, child_delta.shape[-1]),
                dtype=child_delta.dtype,
                device=child_delta.device,
            )
            parent_indices = self.parent_index[child_start:child_end]
            delta_by_parent_hand.scatter_add_(
                0,
                parent_indices[:, None].expand(-1, child_delta.shape[-1]),
                child_delta,
            )
            delta_by_parent_hand.masked_fill_(~reachable, 0.0)

            # Sum over action probabilities and hands, then divide by reachable hand count.
            diff_sum_nodes = delta_by_parent_hand.sum(dim=1)
            node_delta = torch.where(
                reachable_nodes, diff_sum_nodes / reachable_hand_count, 0.0
            )
            node_delta_mean = node_delta.sum() / reachable_nodes.sum()
            self.stats[f"cfr_delta.{percentile}"] = node_delta_mean.item()

    def _record_cfr_entropy(self) -> None:
        """Record the entropy of the policy."""
        if self.max_depth == 0:
            return
        N = self.root_nodes
        actions = self._pull_back(self.policy_probs_avg)[:N]
        mask = self.valid_mask[:N] & ~self.leaf_mask[:N]
        probs = actions[mask]
        entropy = torch.where(probs > 1e-5, -(probs * probs.log()), 0.0)
        self.stats["cfr_entropy"] = entropy.sum(dim=1).mean().item()

    def _record_initial_exploitability(self) -> None:
        """Record the initial exploitability."""
        N = self.root_nodes
        root_streets = self.env.street[:N]
        exploit_stats = self._compute_exploitability()
        exploit_mbbg = self._local_exploitability_mbbg(
            exploit_stats.local_exploitability
        )
        self.stats["local_exploitability_init"] = (
            exploit_stats.local_exploitability.mean().item()
        )
        self.stats["local_exploitability_init_mbbg"] = exploit_mbbg.mean().item()
        self.stats["local_exploitability_init_street"] = {
            street_name: (
                exploit_stats.local_exploitability[root_streets == i].mean().item()
            )
            for i, street_name in enumerate(STREETS)
            if (root_streets == i).any()
        }
        self.stats["local_exploitability_init_mbbg_street"] = {
            street_name: exploit_mbbg[root_streets == i].mean().item()
            for i, street_name in enumerate(STREETS)
            if (root_streets == i).any()
        }

    def _record_cumulative_regret(self) -> None:
        self.stats["mean_positive_regret"] = (
            self.cumulative_regrets.clamp(min=0).mean().item()
        )

        # Compute and record exploitability as a generation-time statistic
        exploit_stats = self._compute_exploitability()
        exploit_mbbg = self._local_exploitability_mbbg(
            exploit_stats.local_exploitability
        )
        self.stats["local_exploitability"] = (
            exploit_stats.local_exploitability.mean().item()
        )
        self.stats["local_exploitability_mbbg"] = exploit_mbbg.mean().item()

        # Record exploitability by street
        N = self.root_nodes
        root_streets = self.env.street[:N]  # (N,)
        self.stats["local_exploitability_street"] = {
            street_name: exploit_stats.local_exploitability[root_streets == i]
            .mean()
            .item()
            for i, street_name in enumerate(STREETS)
            if (root_streets == i).any()
        }
        self.stats["local_exploitability_mbbg_street"] = {
            street_name: exploit_mbbg[root_streets == i].mean().item()
            for i, street_name in enumerate(STREETS)
            if (root_streets == i).any()
        }
        self.stats["local_exploitability_max"] = (
            exploit_stats.local_exploitability.max().item()
        )
        self.stats["local_exploitability_min"] = (
            exploit_stats.local_exploitability.min().item()
        )
        self._save_high_exploitability_roots(exploit_stats.local_exploitability)

    def _save_high_exploitability_roots(
        self, local_exploitability: torch.Tensor, threshold: float = 10.0
    ) -> None:
        """Persist small debug bundles for roots with unusually high exploitability."""
        high_roots = torch.where(local_exploitability > threshold)[0]
        if high_roots.numel() == 0:
            return

        for root_idx_tensor in high_roots.cpu():
            root_idx = int(root_idx_tensor.item())
            tree_mask = torch.zeros(
                self.total_nodes, dtype=torch.bool, device=self.device
            )
            tree_mask[root_idx] = True
            for depth in range(self.tree_depth):
                parent_start = self.depth_offsets[depth]
                child_start = self.depth_offsets[depth + 1]
                child_end = self.depth_offsets[depth + 2]
                parent_mask = tree_mask[parent_start:child_start]
                if not parent_mask.any():
                    continue
                parent_actions = parent_mask[:, None].expand(-1, self.num_actions)
                child_mask = self._push_down(parent_actions, level=depth)
                tree_mask[child_start:child_end] |= child_mask

            tree_indices = torch.where(tree_mask & self.valid_mask)[0]
            if tree_indices.numel() == 0:
                continue

            env_state = type(self.env).from_proto(self.env, num_envs=tree_indices.numel())
            env_state.copy_state_from(
                self.env,
                tree_indices.to(self.device),
                torch.arange(tree_indices.numel(), device=self.device),
                copy_deck=True,
            )
            payload = {
                "env_state": env_state,
                "tree_indices": tree_indices.cpu(),
                "root_idx": root_idx,
                "exploitability": float(local_exploitability[root_idx].item()),
                "model_state_dict": self.model.state_dict(),
            }
            torch.save(
                payload,
                f"high_exploitability_root_{root_idx}_{self.cfr_iterations}.pt",
            )

    def _record_action_mix(self) -> None:
        """Record the action mix of the policy."""
        actions = self._pull_back(self.policy_probs_avg)
        mask = self.valid_mask & ~self.leaf_mask
        mask = mask[: actions.shape[0]]
        allowed_hands = self.allowed_hands[: actions.shape[0]][mask]
        # self.policy_probs_avg is already masked by allowed hands.
        action_mix_by_node = actions[mask].sum(dim=2) / allowed_hands.sum(
            dim=1, keepdim=True
        )
        self.stats["action_mix"] = self._summarize_action_mix(action_mix_by_node)

        N = self.root_nodes
        root_mask = self.valid_mask[:N] & ~self.leaf_mask[:N]
        root_allowed_hands = self.allowed_hands[:N][root_mask]
        root_action_mix_by_node = actions[:N][root_mask].sum(
            dim=2
        ) / root_allowed_hands.sum(dim=1, keepdim=True)
        self.stats["root_action_mix"] = self._summarize_action_mix(
            root_action_mix_by_node
        )

    def _summarize_action_mix(self, action_mix_by_node: torch.Tensor) -> dict[str, float]:
        return {
            "fold": action_mix_by_node[:, 0].mean().item(),
            "call": action_mix_by_node[:, 1].mean().item(),
            "bet": action_mix_by_node[:, 2:-1].sum(dim=1).mean().item(),
            "allin": action_mix_by_node[:, -1].mean().item(),
        }
