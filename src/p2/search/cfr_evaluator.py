"""Base CFR evaluator class with shared methods."""

from __future__ import annotations

import math
import os
import warnings
from abc import ABC
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from p2.core.structured_config import CFRType, WarmStartType
from p2.env.card_utils import (
    NUM_HANDS,
    PREFLOP_HANDS,
    calculate_unblocked_mass,
    combo_to_onehot_tensor,
    collapse_1326_to_169,
    expand_169_to_1326,
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
    TARGET_SOURCE_CLOSING_NET,
    TARGET_SOURCE_EXACT_TERMINAL,
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
    hand_dim: int = NUM_HANDS
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
    new_street_indices: torch.Tensor
    cutoff_indices: torch.Tensor
    new_street_model_positions: torch.Tensor
    cutoff_model_positions: torch.Tensor
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
    _warm_start_policy_prior: torch.Tensor | None
    _warm_start_prior_tau: torch.Tensor | None
    _warm_start_prior_start_t: int
    _warm_start_prior_horizon: int
    _warm_start_regrets: torch.Tensor | None
    _warm_start_regret_decay: str
    _warm_start_regret_decay_horizon: int
    _warm_start_regret_decay_floor: float
    _warm_start_regret_start_t: int
    _warm_start_ftrl_enabled: bool
    _warm_start_ftrl_mode: str
    _warm_start_ftrl_tau_scale: float
    _warm_start_ftrl_horizon: int
    _warm_start_ftrl_floor: float
    _exploitability_cache_key: object | None = None
    _exploitability_cache: ExploitabilityStats | None = None

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
        hand_dim = self.hand_dim
        return torch.multinomial(
            self.beliefs[:N].reshape(N * self.num_players, hand_dim),
            1,
            generator=self.generator,
        ).view(N, self.num_players)

    def _fan_out(self, data: torch.Tensor, level: int | None = None) -> torch.Tensor:
        """Fanout data to all children nodes."""
        raise NotImplementedError("Subclasses must implement _fan_out.")

    def _pull_back(self, data: torch.Tensor, level: int | None = None) -> torch.Tensor:
        """Pull back data to all parent nodes."""
        raise NotImplementedError("Subclasses must implement _pull_back.")

    def _policy_targets_for_nodes(
        self, node_indices: torch.Tensor, top: int
    ) -> torch.Tensor:
        """Return per-hand action targets for selected policy nodes."""
        policy_targets = self._pull_back(self.policy_probs_avg)
        return policy_targets[:top].permute(0, 2, 1)[node_indices]

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

    def _update_model_index_partitions(self) -> None:
        """Partition model leaves into street-closing and same-street cutoffs.

        ``model_indices`` stays as the union batch used by existing feature and
        writeback kernels. The partition tensors expose both node indices and
        positions within that union, so mixed model dispatch can evaluate each
        model on only the rows it owns while still writing back through the
        existing aligned union path.
        """
        device = self.model_indices.device
        if self.model_indices.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            self.new_street_indices = empty
            self.cutoff_indices = empty
            self.new_street_model_positions = empty
            self.cutoff_model_positions = empty
            self.model_baseline_positions = empty
            self.model_hu_positions = empty
            self.new_street_baseline_model_positions = empty
            self.new_street_hu_model_positions = empty
            return

        new_street_at_model = self.new_street_mask[self.model_indices]
        positions = torch.arange(
            self.model_indices.numel(), dtype=torch.long, device=device
        )
        self.new_street_model_positions = positions[new_street_at_model].contiguous()
        self.cutoff_model_positions = positions[~new_street_at_model].contiguous()
        self.new_street_indices = self.model_indices[
            self.new_street_model_positions
        ].contiguous()
        self.cutoff_indices = self.model_indices[
            self.cutoff_model_positions
        ].contiguous()
        empty = torch.empty(0, dtype=torch.long, device=device)
        self.model_baseline_positions = empty
        self.model_hu_positions = positions
        self.new_street_baseline_model_positions = empty
        self.new_street_hu_model_positions = self.new_street_model_positions
        if (
            getattr(self, "closing_leaf_value_model", None) is not None
            and self._can_project_heads_up_closing_model()
        ):
            live_counts = self._live_counts_for_nodes(self.model_indices)
            hu_at_model = live_counts >= 2
            self.model_baseline_positions = positions[~hu_at_model].contiguous()
            self.model_hu_positions = positions[hu_at_model].contiguous()
            self.new_street_baseline_model_positions = positions[
                new_street_at_model & ~hu_at_model
            ].contiguous()
            self.new_street_hu_model_positions = positions[
                new_street_at_model & hu_at_model
            ].contiguous()

    def _refresh_model_indices(self) -> None:
        self.model_indices = self._compute_model_indices()
        self._update_model_index_partitions()

    def _ensure_model_index_partitions(self) -> None:
        if (
            not hasattr(self, "new_street_indices")
            or not hasattr(self, "cutoff_indices")
            or not hasattr(self, "new_street_model_positions")
            or not hasattr(self, "cutoff_model_positions")
            or not hasattr(self, "model_baseline_positions")
            or not hasattr(self, "model_hu_positions")
            or not hasattr(self, "new_street_baseline_model_positions")
            or not hasattr(self, "new_street_hu_model_positions")
            or self.new_street_model_positions.numel()
            + self.cutoff_model_positions.numel()
            != self.model_indices.numel()
        ):
            self._update_model_index_partitions()

    def _features_for_model_positions(
        self,
        features: MLPFeatures,
        positions: torch.Tensor,
        encoder: RebelFeatureEncoder | BetterFeatureEncoder | None = None,
    ) -> MLPFeatures:
        if encoder is None:
            return features[positions]
        node_indices = self.model_indices[positions]
        encoded = encoder.encode(
            self.beliefs,
            pre_chance_node=self.new_street_mask,
            indices=node_indices,
        )
        encoded.beliefs = features.beliefs[positions]
        return encoded

    def _closing_model_num_players(self) -> int:
        return self.closing_leaf_value_model.num_players

    def _closing_model_hand_dim(self) -> int:
        return self.closing_leaf_value_model.hand_dim

    def _can_project_heads_up_closing_model(self) -> bool:
        return (
            self._closing_model_num_players() == 2
            and self.num_players > 2
            and isinstance(getattr(self, "env", None), PBSEnv)
        )

    def _hand_dim_convert(
        self,
        tensor: torch.Tensor,
        *,
        source_hand_dim: int,
        target_hand_dim: int,
        is_belief: bool,
    ) -> torch.Tensor:
        if source_hand_dim == target_hand_dim:
            return tensor
        if source_hand_dim == PREFLOP_HANDS and target_hand_dim == NUM_HANDS:
            return expand_169_to_1326(
                tensor,
                divide_by_multiplicity=is_belief,
            )
        if source_hand_dim == NUM_HANDS and target_hand_dim == PREFLOP_HANDS:
            return collapse_1326_to_169(
                tensor,
                reduction="sum" if is_belief else "mean",
            )
        raise ValueError(
            "unsupported closing model hand-dimension conversion: "
            f"{source_hand_dim} -> {target_hand_dim}"
        )

    def _heads_up_live_players_for_nodes(
        self, node_indices: torch.Tensor
    ) -> torch.Tensor:
        if not isinstance(self.env, PBSEnv):
            raise TypeError("heads-up closing projection requires PBSEnv nodes")
        live = ~self.env.has_folded[node_indices]
        live_count = live.sum(dim=1)
        if not (live_count >= 2).all():
            raise RuntimeError(
                "2-player closing model projection requires at least two live players"
            )
        players = torch.arange(
            self.num_players, dtype=torch.long, device=node_indices.device
        )
        tie_break = self.num_players - players
        score = self.env.chips_placed[node_indices] * (self.num_players + 1) + tie_break
        score = torch.where(live, score, torch.full_like(score, -1))
        return score.topk(k=2, dim=1).indices.contiguous()

    def _live_counts_for_nodes(self, node_indices: torch.Tensor) -> torch.Tensor:
        if not isinstance(self.env, PBSEnv):
            raise TypeError("live-count query requires PBSEnv nodes")
        return (~self.env.has_folded[node_indices]).sum(dim=1)

    def _project_heads_up_pbs_env(
        self,
        node_indices: torch.Tensor,
        live_players: torch.Tensor,
    ) -> PBSEnv:
        if not isinstance(self.env, PBSEnv):
            raise TypeError("heads-up closing projection requires PBSEnv nodes")
        env = self.env
        projected = PBSEnv(
            num_envs=node_indices.numel(),
            num_players=2,
            mean_stack=env.mean_stack,
            sb=env.sb,
            bb=env.bb,
            default_bet_bins=env.default_bet_bins,
            device=env.device,
            rng=env.rng,
            float_dtype=env.float_dtype,
            stack_mode=env.stack_mode,
            min_stack_bb=env.min_stack_bb,
            mid_stack_bb=env.mid_stack_bb,
            max_stack_bb=env.max_stack_bb,
            high_stack_mass_ratio=env.high_stack_mass_ratio,
            force_heads_up_preflop_flop=env.force_heads_up_preflop_flop,
        )
        for field in (
            "street",
            "last_to_act",
            "pot",
            "min_raise",
            "last_aggressive_amount",
            "actions_this_round",
            "actions_last_round",
            "scale",
            "done",
            "winner",
            "board_indices",
            "last_board_indices",
            "board_onehot",
            "deck",
            "deck_pos",
        ):
            getattr(projected, field)[:] = getattr(env, field)[node_indices]

        for field in (
            "stacks",
            "starting_stacks",
            "committed",
            "chips_placed",
            "has_folded",
            "is_allin",
            "acted_this_round",
            "winners",
        ):
            src = getattr(env, field)[node_indices]
            getattr(projected, field)[:] = src.gather(
                1,
                live_players,
            )

        to_act_orig = env.to_act[node_indices]
        to_act_proj = (
            (live_players == to_act_orig[:, None]).to(torch.long).argmax(dim=1)
        )
        projected.to_act[:] = to_act_proj
        projected.button[:] = 1 - to_act_proj
        return projected

    def _heads_up_projected_closing_features(
        self,
        features: MLPFeatures,
        positions: torch.Tensor,
        encoder: RebelFeatureEncoder | BetterFeatureEncoder | None,
    ) -> tuple[MLPFeatures, torch.Tensor]:
        node_indices = self.model_indices[positions]
        live_players = self._heads_up_live_players_for_nodes(node_indices)
        source_hand_dim = features.hand_dim
        target_hand_dim = self._closing_model_hand_dim()
        selected_beliefs = features.beliefs[positions].view(
            -1, self.num_players, source_hand_dim
        )
        selected_beliefs = selected_beliefs.gather(
            1,
            live_players[:, :, None].expand(-1, 2, source_hand_dim),
        )
        selected_beliefs = self._hand_dim_convert(
            selected_beliefs,
            source_hand_dim=source_hand_dim,
            target_hand_dim=target_hand_dim,
            is_belief=True,
        )
        source_encoder = encoder
        if source_encoder is None:
            source_encoder = getattr(
                self, "value_feature_encoder", getattr(self, "feature_encoder", None)
            )
        if source_encoder is None:
            base = features[positions]
            return (
                MLPFeatures(
                    context=base.context,
                    street=base.street,
                    to_act=base.to_act,
                    board=base.board,
                    beliefs=selected_beliefs.reshape(
                        positions.numel(), 2 * target_hand_dim
                    ),
                    hand_dim=target_hand_dim,
                ),
                live_players,
            )

        projected_env = self._project_heads_up_pbs_env(node_indices, live_players)
        encoder_kwargs = {
            "env": projected_env,
            "device": self.device,
            "dtype": source_encoder.dtype,
        }
        projected_encoder = type(source_encoder)(**encoder_kwargs)
        projected_features = projected_encoder.encode(
            selected_beliefs,
            pre_chance_node=torch.ones(
                positions.numel(), dtype=torch.bool, device=self.device
            ),
        )
        return projected_features, live_players

    def _scatter_heads_up_closing_values(
        self,
        values: torch.Tensor,
        live_players: torch.Tensor,
        *,
        target_hand_dim: int,
        node_indices: torch.Tensor,
    ) -> torch.Tensor:
        values = self._hand_dim_convert(
            values,
            source_hand_dim=values.shape[-1],
            target_hand_dim=target_hand_dim,
            is_belief=False,
        )
        out = self._stack_value_baseline(node_indices, target_hand_dim)
        out.scatter_(
            1,
            live_players[:, :, None].expand(-1, 2, target_hand_dim),
            values.to(dtype=out.dtype),
        )
        return out

    def _stack_value_baseline(
        self, node_indices: torch.Tensor, hand_dim: int
    ) -> torch.Tensor:
        if not isinstance(self.env, PBSEnv):
            return self.latest_values.new_zeros(
                node_indices.numel(), self.num_players, hand_dim
            )
        denom = self.env.scale[node_indices].to(torch.float32).clamp_min(1.0)
        stack_value = (
            self.env.stacks[node_indices].to(torch.float32)
            - self.env.starting_stacks[node_indices].to(torch.float32)
        ) / denom[:, None]
        return (
            stack_value.to(dtype=self.latest_values.dtype)[:, :, None]
            .expand(-1, -1, hand_dim)
            .clone()
        )

    def _uses_street_cutoff_schedule(self) -> bool:
        schedule = getattr(self, "action_schedule", None)
        return bool(
            schedule is not None
            and getattr(schedule, "bet_bins_by_depth", None) is not None
        )

    def _model_scope(self) -> str:
        return self.cfg.search.model_scope.value

    def _validate_model_leaf_phases(self) -> None:
        if self.model_indices.numel() == 0:
            return
        scope = self._model_scope()
        if scope in ("mixed_street", "single_street"):
            return
        if scope != "end_of_street":
            raise ValueError(f"Unknown search.model_scope: {scope!r}")
        model_leaf_mask = self.leaf_mask & ~self.env.done
        allin_mask = getattr(self, "allin_call_mask", None)
        if allin_mask is not None and allin_mask.shape == model_leaf_mask.shape:
            model_leaf_mask &= ~allin_mask
        invalid = model_leaf_mask & ~self.new_street_mask
        if invalid.any():
            raise RuntimeError(
                f"search.model_scope={scope!r} requires all neural model leaves "
                "to be end-of-street, but the search produced same-street leaves."
            )

    def _allin_abstraction_enabled(self) -> bool:
        return bool(self.cfg.search.allin_call_terminal_abstraction)

    def _continuation_value_target_sampling_enabled(self) -> bool:
        return bool(self.cfg.search.continuation_value_target_sampling)

    def _continuation_value_target_streets(self) -> tuple[int, ...]:
        streets = self.cfg.search.continuation_value_target_streets
        if streets is None:
            return ()
        return tuple(int(street) for street in streets)

    def _continuation_value_target_depth_bounds(self) -> tuple[int, int] | None:
        min_depth = int(self.cfg.search.continuation_value_target_min_depth)
        max_depth_cfg = self.cfg.search.continuation_value_target_max_depth
        max_depth = self.tree_depth if max_depth_cfg is None else int(max_depth_cfg)
        min_depth = max(0, min_depth)
        max_depth = min(self.tree_depth, max_depth)
        if max_depth < min_depth:
            return None
        return min_depth, max_depth

    def _mid_street_value_roots_are_expected(self) -> bool:
        return (
            self._model_scope() == "mixed_street"
            and self._continuation_value_target_sampling_enabled()
        )

    def _ensure_allin_payoff_resolver(self) -> AllInPayoffResolver:
        resolver = getattr(self, "allin_payoff_resolver", None)
        if resolver is None:
            resolver = AllInPayoffResolver(
                device=self.device,
                preflop_table_path=self.cfg.search.preflop_allin_table_path,
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

        child_indices = torch.arange(
            self.root_nodes, self.total_nodes, device=self.device
        )
        parent, action = self._parent_action_for_nodes(child_indices)
        actor = self.env.to_act[parent]
        opp = 1 - actor
        parent_to_call = (
            self.env.committed[parent, opp] - self.env.committed[parent, actor]
        )
        parent_street = self.env.street[parent]
        mask = (
            (action == 1)
            & self.env.is_allin[parent, opp]
            & (parent_to_call > 0)
            & (parent_street > 0)
            & (parent_street < 3)
        )
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
            self.allin_call_boards_by_street = (
                empty_boards,
                empty_boards,
                empty_boards,
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
        self.allin_flop_tables_i8 = (
            tables.gather(
                1,
                perms[:, :, None].expand(-1, -1, NUM_HANDS),
            )
            .gather(
                2,
                perms[:, None, :].expand(-1, NUM_HANDS, -1),
            )
            .contiguous()
        )
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
        mask = (
            (action_bins == 1)
            & parent_env.is_allin[parent_local_indices, opp]
            & (parent_to_call > 0)
            & (parent_street > 0)
            & (parent_street < 3)
        )
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
        parent = parent_start + torch.div(
            local, self.num_actions, rounding_mode="floor"
        )
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

    def _write_scaled_allin_values(
        self, node_idx: torch.Tensor, values: torch.Tensor
    ) -> None:
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
        elif self.cfr_type == CFRType.linear or (
            self.cfr_type in (CFRType.pcfr, CFRType.sapcfr)
            and not self._predictive_cfr_uses_dcfr()
        ):
            return t, 2
        elif self.cfr_type == CFRType.discounted or self._predictive_cfr_uses_dcfr():
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
        if self.cfr_type == CFRType.discounted or self._predictive_cfr_uses_dcfr():
            if self._average_accumulation_delayed(t):
                return 0.0
            progress = max(0.0, float(t - self.dcfr_delay)) / float(
                self._average_accumulation_window()
            )
            return float(progress ** self._get_dcfr_gamma_for_iteration(t))
        _, new = self._get_mixing_weights(t)
        return float(new)

    def _average_accumulation_delayed(self, t: int) -> bool:
        return (
            self.cfr_type == CFRType.discounted or self._predictive_cfr_uses_dcfr()
        ) and t <= self.dcfr_delay

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
        if self.cfr_type == CFRType.linear or (
            self.cfr_type in (CFRType.pcfr, CFRType.sapcfr)
            and not self._predictive_cfr_uses_dcfr()
        ):
            return torch.full_like(t, 2.0)
        if self.cfr_type == CFRType.discounted or self._predictive_cfr_uses_dcfr():
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
        return (
            float(self.dcfr_gamma_initial)
            + (float(self.dcfr_gamma_final) - float(self.dcfr_gamma_initial))
            * t_normalized
        )

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
        policy_encoder = self.policy_feature_encoder
        policy_model = self.policy_model
        features = policy_encoder.encode(self.beliefs, indices=indices)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if type(policy_model) is BetterTRM:
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
            hand_dim = self.hand_dim

            target_dest = target[offset_next:offset_next_next]
            target_dest[:] = self._fan_out(target, level=depth)

            prev_actor_dest = self.prev_actor[offset_next:offset_next_next]
            prev_actor_indices = prev_actor_dest[:, None, None].expand(-1, -1, hand_dim)
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
        if self.cfr_type == CFRType.discounted or self._predictive_cfr_uses_dcfr():
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

    def _legal_bin_amounts_for(self, indices: torch.Tensor) -> torch.Tensor:
        """Return concrete action-bin amounts for selected evaluator nodes."""
        amount_fn = getattr(self.env, "legal_bins_amounts_for", None)
        if callable(amount_fn):
            return amount_fn(indices)
        amounts, _ = self.env.legal_bins_amounts_and_mask()
        return amounts[indices]

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
        return self.cfg.train.policy_node_weighting.value != "uniform"

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
        hand_dim = self.hand_dim
        if initial_beliefs is None:
            initial_beliefs = torch.full(
                (N, self.num_players, hand_dim),
                1.0 / hand_dim,
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
        self._refresh_model_indices()
        self._validate_model_leaf_phases()
        self.latent = None

        # Compute allowed hands from root board. Compact preflop rank classes have
        # no public-board blockers inside the evaluator; exact class blocker
        # weighting is handled by dedicated preflop code paths when enabled.
        if hand_dim == NUM_HANDS:
            board_mask_root = (
                self.env.board_onehot[:N].any(dim=1).reshape(N, -1).float()
            )
            root_allowed = (self.combo_onehot_float @ board_mask_root.T).T < 0.5
        else:
            root_allowed = torch.ones(N, hand_dim, dtype=torch.bool, device=self.device)
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
        hand_dim = self.hand_dim
        policy_probs_src = torch.empty(
            top, self.num_actions, hand_dim, device=self.device, dtype=self.float_dtype
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
            self._store_warm_start_policy_prior()
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
        avg_scale = float(self.warm_start_iterations) * float(
            self.warm_start_multiplier
        )
        regret_multiplier = getattr(
            self, "warm_start_regret_multiplier", self.warm_start_multiplier
        )
        regret_scale = float(self.warm_start_iterations) * float(regret_multiplier)
        if self._warm_start_regret_decay == "none":
            self.cumulative_regrets += regret_scale * regrets
            self._warm_start_regrets = None
        else:
            self._warm_start_regrets = (regret_scale * regrets).detach().clone()
            self._warm_start_regret_start_t = int(self.warm_start_iterations)

        # Seed the average strategy with the model policy as if it had been
        # played for `avg_scale` iterations (paper App. "CFR Warm Start Algorithm":
        # "the average policy effectively assumes that the warm start policy was
        # played for the first [warm_start] iterations of CFR"). policy_probs and
        # self_reach still hold the model policy here, so accumulate it into the
        # average before regret matching overwrites the current strategy.
        per_iter_weight = self._get_average_policy_weight(self.warm_start_iterations)
        self.update_average_policy(
            self.warm_start_iterations, weight_override=avg_scale * per_iter_weight
        )
        self._calculate_reach_weights(self.self_reach_avg, self.policy_probs_avg)
        self._propagate_all_beliefs(self.beliefs_avg, self.self_reach_avg)

        # Keep the model policy as a prior for the current-policy extractor.
        # The model_br regrets remain in cumulative_regrets, but the policy
        # update is KL-regularized toward the model policy while regrets are
        # still only a warm-start prior.
        self._store_warm_start_policy_prior()
        self._regret_match_current_policy(self.warm_start_iterations)

    def _store_warm_start_policy_prior(self) -> None:
        """Keep the warm-start policy as a KL/FTRL prior, when enabled."""
        self._warm_start_policy_prior = None
        self._warm_start_prior_tau = None
        self._warm_start_prior_start_t = int(self.warm_start_iterations)
        self._warm_start_prior_horizon = self._resolve_warm_start_horizon(
            self._warm_start_ftrl_horizon
        )
        if (
            not self._warm_start_ftrl_enabled
            or self._warm_start_ftrl_mode == "none"
            or self.warm_start_type != WarmStartType.model_br
            or self.warm_start_iterations <= 0
            or self.total_nodes <= self.depth_offsets[1]
        ):
            return
        bottom = self.depth_offsets[1]
        top = self.depth_offsets[-2]
        self._warm_start_policy_prior = self.policy_probs[bottom:].detach().clone()

        parent_regrets = self._pull_back(
            self._effective_cumulative_regrets(self.warm_start_iterations)
        )
        action_ids = torch.arange(self.num_actions, device=self.device)
        valid_actions = action_ids[None, :] < self.child_count[:top, None]
        valid_actions = valid_actions[:, :, None]
        high = parent_regrets.masked_fill(~valid_actions, -torch.inf).amax(dim=1)
        low = parent_regrets.masked_fill(~valid_actions, torch.inf).amin(dim=1)
        spread = high - low
        spread = torch.where(torch.isfinite(spread), spread, torch.ones_like(spread))
        self._warm_start_prior_tau = (
            spread.clamp_min(1.0) * float(self._warm_start_ftrl_tau_scale)
        ).detach()[:, None, :]

    def _resolve_warm_start_horizon(self, configured: int) -> int:
        if configured > 0:
            return int(configured)
        return max(
            int(self.warm_start_iterations),
            int(self.dcfr_delay) - int(self.warm_start_iterations),
        )

    def _warm_start_decay_factor(
        self, t: int | None, *, mode: str, start_t: int, horizon: int, floor: float
    ) -> float:
        if t is None or mode == "none":
            return 0.0
        elapsed = int(t) - int(start_t)
        if elapsed < 0:
            return 1.0
        horizon = max(1, int(horizon))
        if mode == "constant":
            return 1.0
        if mode == "linear":
            if elapsed >= horizon:
                return float(floor)
            progress = float(elapsed) / float(horizon)
            return float(floor) + (1.0 - float(floor)) * (1.0 - progress)
        if mode == "exp":
            progress = float(elapsed) / float(horizon)
            return float(floor) + (1.0 - float(floor)) * math.exp(-progress)
        raise ValueError(f"Unsupported warm-start decay mode: {mode}")

    def _warm_start_regret_factor(self, t: int | None) -> float:
        return self._warm_start_decay_factor(
            t,
            mode=self._warm_start_regret_decay,
            start_t=self._warm_start_regret_start_t,
            horizon=self._resolve_warm_start_horizon(
                self._warm_start_regret_decay_horizon
            ),
            floor=self._warm_start_regret_decay_floor,
        )

    def _effective_cumulative_regrets(self, t: int | None) -> torch.Tensor:
        warm_regrets = self._warm_start_regrets
        if warm_regrets is None:
            return self.cumulative_regrets
        factor = self._warm_start_regret_factor(t)
        if factor <= 0.0:
            return self.cumulative_regrets
        return self.cumulative_regrets + factor * warm_regrets

    def _predictive_cfr_prediction_scale(self) -> float:
        if self.cfr_type == CFRType.pcfr:
            return 1.0
        if self.cfr_type == CFRType.sapcfr:
            alpha = float(getattr(self, "sapcfr_alpha", 2.0))
            return 1.0 / (1.0 + max(0.0, alpha))
        return 0.0

    def _predictive_cfr_uses_dcfr(self) -> bool:
        return bool(
            getattr(self, "_predictive_cfr_dcfr_hybrid", False)
            and self.cfr_type in (CFRType.pcfr, CFRType.sapcfr)
        )

    def _predictive_cfr_delay_threshold(self) -> int:
        delay = int(getattr(self, "predictive_cfr_delay", -1))
        if delay < 0:
            return int(self.dcfr_delay)
        return delay

    def _predictive_cfr_active(self, t: int | None) -> bool:
        return (
            t is not None
            and getattr(self, "_predictive_cfr_enabled", False)
            and int(t) > self._predictive_cfr_delay_threshold()
        )

    def _current_policy_regrets(self, t: int | None) -> torch.Tensor:
        regrets = self._effective_cumulative_regrets(t)
        if not self._predictive_cfr_active(t):
            return regrets

        last_regrets = getattr(self, "_last_instantaneous_regrets", None)
        if last_regrets is None or last_regrets.shape != regrets.shape:
            return regrets

        scale = self._predictive_cfr_prediction_scale()
        if scale <= 0.0:
            return regrets
        return regrets + scale * last_regrets

    def _update_predictive_cfr_observation(self, regrets: torch.Tensor, t: int) -> None:
        if not getattr(self, "_predictive_cfr_enabled", False):
            return

        last_regrets = getattr(self, "_last_instantaneous_regrets", None)
        if last_regrets is None or last_regrets.shape != regrets.shape:
            self._last_instantaneous_regrets = torch.zeros_like(regrets)
            last_regrets = self._last_instantaneous_regrets

        # Match the existing alternating-update convention used by linear CFR:
        # only the player whose regrets are updated this iteration gets a fresh
        # prediction for the next policy extraction.
        observed = self.prev_actor[:, None] != (t % self.num_players)
        torch.where(observed, regrets, last_regrets, out=last_regrets)

    def _try_apply_warm_start_ftrl_policy(self, t: int | None) -> bool:
        """Extract policy with a KL prior: pi ∝ pi_prior * exp(R / tau)."""
        prior = self._warm_start_policy_prior
        tau = self._warm_start_prior_tau
        if prior is None or tau is None or t is None:
            return False

        tau_factor = self._warm_start_decay_factor(
            t,
            mode=self._warm_start_ftrl_mode,
            start_t=self._warm_start_prior_start_t,
            horizon=self._warm_start_prior_horizon,
            floor=self._warm_start_ftrl_floor,
        )
        if tau_factor <= 0.0:
            self._warm_start_policy_prior = None
            self._warm_start_prior_tau = None
            return False

        bottom = self.depth_offsets[1]
        top = self.depth_offsets[-2]
        if prior.shape != self.policy_probs[bottom:].shape:
            self._warm_start_policy_prior = None
            self._warm_start_prior_tau = None
            return False

        prior_full = torch.zeros_like(self.policy_probs)
        prior_full[bottom:] = prior
        parent_prior = self._pull_back(prior_full)
        parent_regrets = self._pull_back(self._effective_cumulative_regrets(t))

        action_ids = torch.arange(self.num_actions, device=self.device)
        valid_actions = action_ids[None, :] < self.child_count[:top, None]
        logits = parent_prior.clamp_min(1e-8).log() + parent_regrets / (
            tau * tau_factor
        ).clamp_min(1e-6)
        logits = logits.masked_fill(~valid_actions[:, :, None], -1e9)
        parent_policy = torch.softmax(logits, dim=1)
        self.policy_probs[bottom:] = self._push_down(parent_policy)
        self._mask_invalid(self.policy_probs)
        return True

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
            if type(value_model) is BetterTRM:
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
        value_model = self.value_model
        closing_value_model = self.closing_leaf_value_model
        scope = self._model_scope()
        if scope == "mixed_street" and closing_value_model is not None:
            self._ensure_model_index_partitions()
            hand_values = self.latest_values.new_empty(
                (
                    len(features),
                    self.num_players,
                    self.hand_dim,
                )
            )
            if self.cutoff_model_positions.numel() > 0:
                cutoff_values = self._eval_value_model(
                    value_model,
                    self._features_for_model_positions(
                        features, self.cutoff_model_positions
                    ),
                    use_pre_head=False,
                )
                hand_values.index_copy_(0, self.cutoff_model_positions, cutoff_values)
            if self.new_street_model_positions.numel() > 0:
                closing_encoder = self.closing_leaf_value_encoder
                if self._can_project_heads_up_closing_model():
                    node_indices = self.model_indices[self.new_street_model_positions]
                    live_counts = self._live_counts_for_nodes(node_indices)
                    baseline_local = torch.where(live_counts < 2)[0]
                    if baseline_local.numel() > 0:
                        baseline_positions = self.new_street_model_positions[
                            baseline_local
                        ]
                        baseline_values = self._stack_value_baseline(
                            node_indices[baseline_local],
                            self.hand_dim,
                        )
                        hand_values.index_copy_(0, baseline_positions, baseline_values)
                    hu_local = torch.where(live_counts >= 2)[0]
                    if hu_local.numel() == 0:
                        return hand_values
                    hu_positions = self.new_street_model_positions[hu_local]
                    closing_features, live_players = (
                        self._heads_up_projected_closing_features(
                            features,
                            hu_positions,
                            closing_encoder,
                        )
                    )
                    closing_values = self._eval_value_model(
                        closing_value_model,
                        closing_features,
                        use_pre_head=False,
                    )
                    closing_values = self._scatter_heads_up_closing_values(
                        closing_values,
                        live_players,
                        target_hand_dim=self.hand_dim,
                        node_indices=self.model_indices[hu_positions],
                    )
                    hand_values.index_copy_(0, hu_positions, closing_values)
                    return hand_values
                closing_values = self._eval_value_model(
                    closing_value_model,
                    self._features_for_model_positions(
                        features,
                        self.new_street_model_positions,
                        closing_encoder,
                    ),
                    use_pre_head=False,
                )
                hand_values.index_copy_(
                    0, self.new_street_model_positions, closing_values
                )
            return hand_values
        if scope in ("mixed_street", "single_street"):
            return self._eval_value_model(
                value_model,
                features,
                use_pre_head=False,
            )
        if scope != "end_of_street":
            raise ValueError(f"Unknown search.model_scope: {scope!r}")
        if closing_value_model is not None:
            positions = torch.arange(
                len(features), dtype=torch.long, device=features.context.device
            )
            closing_encoder = self.closing_leaf_value_encoder
            if self._can_project_heads_up_closing_model():
                node_indices = self.model_indices[positions]
                live_counts = self._live_counts_for_nodes(node_indices)
                projected_values = self.latest_values.new_empty(
                    (
                        len(features),
                        self.num_players,
                        self.hand_dim,
                    )
                )
                baseline_local = torch.where(live_counts < 2)[0]
                if baseline_local.numel() > 0:
                    projected_values.index_copy_(
                        0,
                        baseline_local,
                        self._stack_value_baseline(
                            node_indices[baseline_local],
                            self.hand_dim,
                        ),
                    )
                hu_local = torch.where(live_counts >= 2)[0]
                if hu_local.numel() == 0:
                    return projected_values
                hu_positions = positions[hu_local]
                closing_features, live_players = (
                    self._heads_up_projected_closing_features(
                        features,
                        hu_positions,
                        closing_encoder,
                    )
                )
                closing_values = self._eval_value_model(
                    closing_value_model,
                    closing_features,
                    use_pre_head=False,
                )
                closing_values = self._scatter_heads_up_closing_values(
                    closing_values,
                    live_players,
                    target_hand_dim=self.hand_dim,
                    node_indices=self.model_indices[hu_positions],
                )
                projected_values.index_copy_(0, hu_local, closing_values)
                return projected_values
            features = self._features_for_model_positions(
                features,
                positions,
                closing_encoder,
            )
        return self._eval_value_model(
            closing_value_model or value_model,
            features,
            use_pre_head=False,
        )

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
            value_encoder = self.value_feature_encoder
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
            hand_dim = self.hand_dim
            self.last_model_values = self.latest_values.new_empty(
                (0, self.num_players, hand_dim)
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
        hand_dim = self.hand_dim
        actor_indices = self.env.to_act[:top]
        actor_indices_expanded = actor_indices[:top, None, None].expand(
            -1, -1, hand_dim
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

        hand_dim = self.hand_dim
        src_actor_indices = self.env.to_act[:, None, None].expand(-1, -1, hand_dim)
        prev_actor_indices = self.prev_actor[bottom:, None, None].expand(
            -1, -1, hand_dim
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
        self._regret_match_current_policy(t)

        self.update_average_policy(t)
        self._calculate_reach_weights(self.self_reach_avg, self.policy_probs_avg)
        self._propagate_all_beliefs(self.beliefs_avg, self.self_reach_avg)

    def _regret_match_current_policy(self, t: int | None = None) -> None:
        """Set the current strategy from cumulative regrets via regret matching
        and refresh the current-strategy reach weights and beliefs.

        This is the per-iteration strategy update only; it does not touch the
        average-strategy accumulators, so callers can update the average
        separately (e.g. to seed it with a warm-start policy).
        """
        if self._try_apply_warm_start_ftrl_policy(t):
            self._calculate_reach_weights(self.self_reach, self.policy_probs)
            self._propagate_all_beliefs(self.beliefs, self.self_reach)
            return

        bottom = self.depth_offsets[1]
        positive_regrets = self._current_policy_regrets(t).clamp(min=0.0)
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

    def update_average_policy(
        self, t: int, weight_override: float | None = None
    ) -> None:
        """Update the average policy using true CFR reach-weighted sums.

        ``weight_override`` replaces the per-iteration accumulation weight. It is
        used to seed the average with the warm-start policy as if it had been
        played for several CFR iterations.
        """
        if (
            self.cfr_type == CFRType.discounted or self._predictive_cfr_uses_dcfr()
        ) and self._average_accumulation_delayed(t):
            self.policy_probs_avg[:] = self.policy_probs
            self.average_policy_initialized = False
            return

        N = self.root_nodes
        weight = (
            self._get_average_policy_weight(t)
            if weight_override is None
            else weight_override
        )
        numerator, denominator = self._ensure_average_policy_accumulators()

        # Get actor indices at source nodes (nodes that have children)
        # _fan_out expects tensors aligned with source nodes (0 to depth_offsets[-2])
        top = self.depth_offsets[-2]
        hand_dim = self.hand_dim
        actor_indices = self.env.to_act[:top, None, None].expand(-1, -1, hand_dim)
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
            hand_dim,
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

        if self.cfr_type == CFRType.linear or (
            self.cfr_type in (CFRType.pcfr, CFRType.sapcfr)
            and not self._predictive_cfr_uses_dcfr()
        ):  # Alternate updates.
            regrets.masked_fill_(self.prev_actor[:, None] == t % self.num_players, 0.0)
        elif self.cfr_type == CFRType.discounted or self._predictive_cfr_uses_dcfr():
            t_discount = max(1, t)
            numerator = torch.where(
                self.cumulative_regrets > 0,
                t_discount**self.dcfr_alpha,
                t_discount**self.dcfr_beta,
            )
            denominator = torch.where(
                self.cumulative_regrets > 0,
                t_discount**self.dcfr_alpha + 1,
                t_discount**self.dcfr_beta + 1,
            )
            self.cumulative_regrets *= numerator
            self.cumulative_regrets /= denominator
        # Update cumulative regrets
        self.cumulative_regrets += regrets

        # CFR+ trick: clamp regrets to non-negative
        if self.cfr_plus:
            self.cumulative_regrets.clamp_(min=0)

        self._update_predictive_cfr_observation(regrets, t)

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

    def _backup_consistency_enabled(self) -> bool:
        return float(self.cfg.train.backup_consistency_coef or 0.0) > 0.0

    def _backup_feature_storage_dtype(self, dtype: torch.dtype) -> torch.dtype:
        if self.device.type == "cuda" and dtype in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        }:
            return torch.bfloat16
        return dtype

    def _empty_backup_consistency_statistics(
        self,
        value_node_indices: torch.Tensor,
        value_features_all: MLPFeatures,
    ) -> dict[str, torch.Tensor]:
        count = int(value_node_indices.numel())
        action_count = int(self.num_actions)
        context_dtype = self._backup_feature_storage_dtype(
            value_features_all.context.dtype
        )
        belief_dtype = self._backup_feature_storage_dtype(
            value_features_all.beliefs.dtype
        )
        actor = self.env.to_act[value_node_indices].long()
        actor = actor.clamp(min=0, max=max(self.num_players - 1, 0))
        return {
            "backup_child_context": torch.zeros(
                count,
                action_count,
                value_features_all.context.shape[1],
                dtype=context_dtype,
                device=self.device,
            ),
            "backup_child_street": torch.zeros(
                count,
                action_count,
                dtype=torch.long,
                device=self.device,
            ),
            "backup_child_to_act": torch.zeros(
                count,
                action_count,
                dtype=torch.long,
                device=self.device,
            ),
            "backup_child_board": torch.full(
                (count, action_count, 5),
                -1,
                dtype=torch.long,
                device=self.device,
            ),
            "backup_child_beliefs": torch.zeros(
                count,
                action_count,
                value_features_all.beliefs.shape[1],
                dtype=belief_dtype,
                device=self.device,
            ),
            "backup_child_valid": torch.zeros(
                count,
                action_count,
                dtype=torch.bool,
                device=self.device,
            ),
            "backup_actor": actor,
        }

    def _backup_consistency_child_statistics(
        self,
        value_node_indices: torch.Tensor,
        value_features_all: MLPFeatures,
        top: int,
    ) -> dict[str, torch.Tensor]:
        stats = self._empty_backup_consistency_statistics(
            value_node_indices, value_features_all
        )
        if value_node_indices.numel() == 0 or len(self.depth_offsets) < 3 or top <= 0:
            return stats

        child_start = self.depth_offsets[1]
        child_end = min(self.depth_offsets[2], self.total_nodes)
        if child_start >= child_end:
            return stats

        child_nodes = torch.arange(child_start, child_end, device=self.device)
        parent, action = self._parent_action_for_nodes(child_nodes)
        in_range = (
            (parent >= 0)
            & (parent < self.root_nodes)
            & (action >= 0)
            & (action < self.num_actions)
        )
        child_by_parent_action = torch.full(
            (self.root_nodes, self.num_actions),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        child_by_parent_action[parent[in_range], action[in_range]] = child_nodes[
            in_range
        ]

        child_indices = child_by_parent_action[value_node_indices]
        has_child = child_indices >= 0
        safe_child_indices = child_indices.clamp(min=0, max=top - 1)
        parent_street = self.env.street[value_node_indices]
        child_to_act = self.env.to_act[safe_child_indices]
        child_valid = (
            has_child
            & (child_indices < top)
            & self.valid_mask[safe_child_indices]
            & ~self.env.done[safe_child_indices]
            & (self.env.street[safe_child_indices] == parent_street[:, None])
            & (child_to_act >= 0)
            & (child_to_act < self.num_players)
            & self.legal_mask[safe_child_indices].any(dim=-1)
        )
        allin_call_mask = getattr(self, "allin_call_mask", None)
        if allin_call_mask is not None and allin_call_mask.shape[0] >= self.total_nodes:
            child_valid &= ~allin_call_mask[safe_child_indices]

        stats["backup_child_context"] = value_features_all.context[
            safe_child_indices
        ].to(dtype=stats["backup_child_context"].dtype)
        stats["backup_child_street"] = value_features_all.street[safe_child_indices]
        stats["backup_child_to_act"] = value_features_all.to_act[safe_child_indices]
        stats["backup_child_board"] = value_features_all.board[safe_child_indices]
        stats["backup_child_beliefs"] = value_features_all.beliefs[
            safe_child_indices
        ].to(dtype=stats["backup_child_beliefs"].dtype)
        stats["backup_child_valid"] = child_valid
        return stats

    @profile
    def training_data(
        self,
        exclude_start: bool = True,
        *,
        include_pre_chance_value_batch: bool = True,
        include_policy_batch: bool = True,
    ) -> tuple[RebelBatch, RebelBatch | None, RebelBatch | None]:
        """Return training data from CFR evaluation."""
        N = self.root_nodes
        top = self.depth_offsets[-2] if len(self.depth_offsets) > 1 else N

        source_values = (
            self.latest_values if self.use_final_policy_values else self.values_avg
        )
        root_value_targets = source_values[:N].clamp(-1.0, 1.0)
        root_indices = torch.arange(N, dtype=torch.long, device=self.device)
        value_root_mask = torch.ones(N, dtype=torch.bool, device=self.device)
        if exclude_start:
            value_root_mask &= (
                self.valid_mask[:N]
                & ~self.env.done[:N]
                & (self.env.actions_this_round[:N] >= int(self.max_depth))
            )
            allin_call_mask = getattr(self, "allin_call_mask", None)
            if allin_call_mask is not None and allin_call_mask.shape[0] >= N:
                value_root_mask &= ~allin_call_mask[:N]
        value_node_indices = root_indices[value_root_mask]
        value_roots_only = True
        value_targets = source_values[value_node_indices].clamp(-1.0, 1.0)

        policy_encoder = self.policy_feature_encoder
        value_encoder = self.value_feature_encoder
        if include_policy_batch:
            actor_top = self.env.to_act[:top]
            valid_actor_top = (actor_top >= 0) & (actor_top < self.num_players)
            valid_top = (
                self.valid_mask[:top]
                & ~self.leaf_mask[:top]
                & valid_actor_top
                & self.legal_mask[:top].any(dim=-1)
            )
            valid_policy_indices = torch.where(valid_top)[0].contiguous()
            policy_targets = self._policy_targets_for_nodes(valid_policy_indices, top)
            nonempty_policy_targets = policy_targets.sum(dim=(1, 2)) > 0
            valid_policy_indices = valid_policy_indices[nonempty_policy_targets]
            policy_targets = policy_targets[nonempty_policy_targets]
            policy_features = policy_encoder.encode(
                self.beliefs_avg,
                pre_chance_node=False,
                indices=valid_policy_indices,
            )
        if value_roots_only:
            value_features_all = value_encoder.encode(
                self.beliefs_avg, pre_chance_node=False
            )[:top]
            value_features = value_features_all[value_node_indices]
        else:
            value_features_all = value_encoder.encode(
                self.beliefs_avg, pre_chance_node=False
            )
            value_features = value_features_all[value_node_indices]
        root_value_features = value_features_all[:N]
        legal_masks = self.legal_mask
        bin_amounts = None
        if include_policy_batch:
            bin_amounts, _ = self.env.legal_bins_amounts_and_mask()
        node_depth = torch.zeros(self.total_nodes, dtype=torch.long, device=self.device)
        for depth in range(len(self.depth_offsets) - 1):
            node_depth[self.depth_offsets[depth] : self.depth_offsets[depth + 1]] = (
                depth
            )

        statistics = {
            "to_act": self.env.to_act,
            "street": self.env.street,
            "stage": 2 * self.env.street,
            "board": self.env.board_indices,
            "pot": self.env.pot,
            "scale": self.env.scale,
            "node_depth": node_depth,
            "actions_this_round": self.env.actions_this_round,
        }
        if hasattr(self.env, "has_folded"):
            statistics["has_folded"] = self.env.has_folded
        if hasattr(self.env, "is_allin"):
            statistics["is_allin"] = self.env.is_allin

        exploit_stats = self._compute_exploitability()
        exploit_mbbg = self._local_exploitability_mbbg(
            exploit_stats.local_exploitability
        )
        root_leaf_counts = self._root_leaf_target_source_counts(N)

        value_statistics = {
            key: statistics[key][value_node_indices] for key in statistics
        }
        value_statistics["bet_amounts"] = (
            bin_amounts[value_node_indices]
            if bin_amounts is not None
            else self._legal_bin_amounts_for(value_node_indices)
        )
        value_statistics["target_source"] = torch.full(
            (value_node_indices.numel(),),
            TARGET_SOURCE_CFR_BACKUP,
            dtype=torch.long,
            device=self.device,
        )
        value_statistics["local_exploitability"] = exploit_stats.local_exploitability[
            value_node_indices
        ]
        value_statistics["local_exploitability_mbbg"] = exploit_mbbg[
            value_node_indices
        ]
        value_statistics["local_best_response_values"] = (
            exploit_stats.local_best_response_values[value_node_indices]
        )
        continuation_value_mask = torch.zeros(
            value_node_indices.numel(), dtype=torch.bool, device=self.device
        )
        value_statistics["continuation_value_target"] = continuation_value_mask
        for key, value in root_leaf_counts.items():
            value_statistics[key] = value[value_node_indices]
        backup_consistency_enabled = self._backup_consistency_enabled()
        if backup_consistency_enabled:
            value_statistics.update(
                self._backup_consistency_child_statistics(
                    value_node_indices,
                    value_features_all,
                    top,
                )
            )

        value_batch = RebelBatch(
            features=value_features,
            value_targets=value_targets,
            legal_masks=legal_masks[value_node_indices],
            statistics=value_statistics,
        )

        self.stats["value_target_count"] = float(value_node_indices.numel())
        street_root = self.env.street[:N]
        actions_root = self.env.actions_this_round[:N]
        if value_node_indices.numel() > 0:
            selected_streets = self.env.street[value_node_indices]
            selected_actions = self.env.actions_this_round[value_node_indices]
            mid_street_value_roots = (selected_streets < 4) & (selected_actions > 0)
            if mid_street_value_roots.any():
                count = int(mid_street_value_roots.sum().item())
                self.stats["mid_street_value_root_count"] = float(count)
                if (
                    not exclude_start
                    and not self._mid_street_value_roots_are_expected()
                ):
                    warnings.warn(
                        "training_data() received mid-street roots for value targets; "
                        "value supervision is intended for street-boundary roots only. "
                        f"count={count}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        if exclude_start:
            value_start_nodes = (
                (self.env.street[value_node_indices] == 0)
                & (self.env.actions_this_round[value_node_indices] == 0)
                & ~value_statistics["continuation_value_target"]
            )
            value_batch = value_batch[~value_start_nodes]

        policy_batch = None
        if include_policy_batch:
            # Policy batch gets all valid, non-leaf states.
            # Use valid_mask directly (works for both: sparse has all-ones,
            # dense has computed mask).
            policy_statistics = {
                key: statistics[key][valid_policy_indices] for key in statistics
            }
            assert bin_amounts is not None
            policy_statistics["bet_amounts"] = bin_amounts[valid_policy_indices]
            if self._should_record_policy_node_reach():
                policy_statistics["policy_node_reach"] = (
                    self._compute_policy_node_reach(top)[valid_policy_indices]
                )
            policy_batch = RebelBatch(
                features=policy_features,
                policy_targets=policy_targets,
                legal_masks=legal_masks[valid_policy_indices],
                statistics=policy_statistics,
            )

        if not include_pre_chance_value_batch:
            return value_batch, None, policy_batch

        pre_features_all = value_encoder.encode(self.beliefs, pre_chance_node=True)
        pre_features_root = pre_features_all[:N].clone()
        pre_beliefs = self.root_pre_chance_beliefs[:N].reshape(N, -1)
        pre_features_root.beliefs = pre_beliefs

        value_targets_pre = root_value_targets.clone()
        value_statistics_pre = {key: statistics[key][:N].clone() for key in statistics}
        value_statistics_pre["bet_amounts"] = (
            bin_amounts[:N].clone()
            if bin_amounts is not None
            else self._legal_bin_amounts_for(root_indices)
        )
        value_statistics_pre["target_source"] = torch.full(
            (N,),
            TARGET_SOURCE_CHANCE_EXPECTATION,
            dtype=torch.long,
            device=self.device,
        )
        value_statistics_pre["local_exploitability"] = (
            exploit_stats.local_exploitability.clone()
        )
        value_statistics_pre["local_exploitability_mbbg"] = exploit_mbbg.clone()
        value_statistics_pre["local_best_response_values"] = (
            exploit_stats.local_best_response_values.clone()
        )
        value_statistics_pre["continuation_value_target"] = torch.zeros(
            N, dtype=torch.bool, device=self.device
        )
        for key, value in root_leaf_counts.items():
            value_statistics_pre[key] = value.clone()
        if backup_consistency_enabled:
            value_statistics_pre.update(
                self._empty_backup_consistency_statistics(
                    root_indices,
                    root_value_features,
                )
            )
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
                root_value_features,
                self.root_pre_chance_beliefs,
                self.env.last_board_indices,
            )
            value_targets_pre[turn_river_mask] = expected_turn_river

        flop_mask = start_mask & (street_root == 1)
        if flop_mask.any():
            expected_flop = self.chance_helper.flop_chance_values(
                torch.where(flop_mask)[0],
                root_value_features,
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

    def _root_leaf_target_source_counts(
        self, num_roots: int
    ) -> dict[str, torch.Tensor]:
        """Count terminal/model/closing leaf sources under each root."""

        total_nodes = int(self.total_nodes)
        device = self.device
        counts = {
            "leaf_total_count": torch.zeros(num_roots, dtype=torch.long, device=device),
            f"leaf_target_source_{TARGET_SOURCE_CFR_BACKUP}_count": torch.zeros(
                num_roots, dtype=torch.long, device=device
            ),
            f"leaf_target_source_{TARGET_SOURCE_EXACT_TERMINAL}_count": torch.zeros(
                num_roots, dtype=torch.long, device=device
            ),
            f"leaf_target_source_{TARGET_SOURCE_CLOSING_NET}_count": torch.zeros(
                num_roots, dtype=torch.long, device=device
            ),
        }
        if num_roots == 0 or total_nodes == 0 or self.leaf_mask.numel() == 0:
            return counts

        get_root_index = getattr(self, "_get_root_index", None)
        if callable(get_root_index):
            root_owner = get_root_index()
        else:
            root_owner = torch.arange(total_nodes, dtype=torch.long, device=device)
            root_owner[:num_roots] = torch.arange(
                num_roots, dtype=torch.long, device=device
            )
            for level in range(1, len(self.depth_offsets) - 1):
                start = self.depth_offsets[level]
                end = self.depth_offsets[level + 1]
                if end <= start:
                    continue
                parents = self.parent_index[start:end].clamp(min=0)
                root_owner[start:end] = root_owner[parents]

        valid_leaf = self.leaf_mask
        if self.valid_mask.numel() == self.leaf_mask.numel():
            valid_leaf = valid_leaf & self.valid_mask
        valid_leaf = valid_leaf & (root_owner < num_roots)

        allin_leaf = torch.zeros_like(valid_leaf)
        allin_indices = getattr(self, "allin_call_indices", None)
        if allin_indices is not None and allin_indices.numel() > 0:
            allin_leaf[allin_indices] = True

        closing_leaf = valid_leaf & self.new_street_mask
        exact_terminal_leaf = valid_leaf & ~closing_leaf & (self.env.done | allin_leaf)
        same_street_model_leaf = valid_leaf & ~closing_leaf & ~exact_terminal_leaf

        one = torch.ones(total_nodes, dtype=torch.long, device=device)
        leaf_source_masks = {
            "leaf_total_count": valid_leaf,
            f"leaf_target_source_{TARGET_SOURCE_CFR_BACKUP}_count": same_street_model_leaf,
            f"leaf_target_source_{TARGET_SOURCE_EXACT_TERMINAL}_count": exact_terminal_leaf,
            f"leaf_target_source_{TARGET_SOURCE_CLOSING_NET}_count": closing_leaf,
        }
        for key, mask in leaf_source_masks.items():
            counts[key].scatter_add_(0, root_owner[mask], one[mask])
        return counts

    def evaluate_cfr(
        self, training_mode: bool = True, sample_continuation: bool = True
    ) -> PublicBeliefState | None:
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

        if not sample_continuation:
            return None
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

            env_state = type(self.env).from_proto(
                self.env, num_envs=tree_indices.numel()
            )
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

    def _summarize_action_mix(
        self, action_mix_by_node: torch.Tensor
    ) -> dict[str, float]:
        return {
            "fold": action_mix_by_node[:, 0].mean().item(),
            "call": action_mix_by_node[:, 1].mean().item(),
            "bet": action_mix_by_node[:, 2:-1].sum(dim=1).mean().item(),
            "allin": action_mix_by_node[:, -1].mean().item(),
        }
