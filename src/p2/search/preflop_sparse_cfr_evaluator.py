from __future__ import annotations

import torch

from p2.allin.oracle import PreflopAllIn169Oracle
from p2.core.structured_config import Config
from p2.env.card_utils import (
    NUM_HANDS,
    PREFLOP_HANDS,
    collapse_1326_to_169,
    preflop_class_compatibility_counts_tensor,
    preflop_class_multiplicity_tensor,
)
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.env.pbs_env import PBSEnv
from p2.models.base_mlp_model import BaseMLPModel
from p2.rl.target_provenance import (
    TARGET_SOURCE_CFR_BACKUP,
    TARGET_SOURCE_CLOSING_NET,
    TARGET_SOURCE_EXACT_TERMINAL,
)
from p2.search.cfr_evaluator import ExploitabilityStats
from p2.search.sparse_cfr_evaluator import SparseCFREvaluator


class PreflopSparseCFREvaluator(SparseCFREvaluator):
    """Sparse CFR boundary for the S_0 arbitrary preflop public-state model.

    This class intentionally draws a hard line around the preflop bootstrap
    path. It does not make the generic sparse evaluator multiway-correct by
    itself; it enforces the public-state contract that the dedicated preflop
    leaf classification, handoff, and all-in resolver build on.
    """

    def __init__(
        self,
        model: BaseMLPModel,
        device: torch.device,
        cfg: Config,
        generator: torch.Generator | None = None,
        closing_leaf_model: BaseMLPModel | None = None,
    ) -> None:
        if cfg.search.sparse_fused:
            raise ValueError("PreflopSparseCFREvaluator requires non-fused sparse CFR")
        super().__init__(
            model=model,
            device=device,
            cfg=cfg,
            generator=generator,
            closing_leaf_model=closing_leaf_model,
        )
        if int(getattr(self, "hand_dim", NUM_HANDS)) != PREFLOP_HANDS:
            raise ValueError(
                "PreflopSparseCFREvaluator is compact-only; attach a "
                f"{PREFLOP_HANDS}-hand preflop policy/value model"
        )
        self.warm_start_iterations = 0
        self._preflop_unblocked_projection: torch.Tensor | None = None
        self._preflop_unblocked_projection_key: tuple[torch.device, torch.dtype] | None = (
            None
        )
        self._preflop_ev_action_weights_buf: torch.Tensor | None = None
        self._preflop_ev_child_values_buf: torch.Tensor | None = None

    @property
    def _compact_preflop(self) -> bool:
        return True

    def _continuation_value_target_sampling_enabled(self) -> bool:
        return True

    def _continuation_value_target_streets(self) -> tuple[int, ...]:
        return (0,)

    def warm_start(self) -> None:
        return None

    def _preflop_unblocked_projection_for(
        self,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        key = (ref.device, ref.dtype)
        projection = getattr(self, "_preflop_unblocked_projection", None)
        if (
            projection is None
            or getattr(self, "_preflop_unblocked_projection_key", None) != key
        ):
            multiplicity = preflop_class_multiplicity_tensor(
                device=ref.device
            ).to(dtype=ref.dtype)
            compatibility = preflop_class_compatibility_counts_tensor(
                device=ref.device
            ).to(dtype=ref.dtype)
            inv_multiplicity = multiplicity.reciprocal()
            projection = compatibility.T * inv_multiplicity[:, None]
            self._preflop_unblocked_projection = projection.contiguous()
            self._preflop_unblocked_projection_key = key
        assert self._preflop_unblocked_projection is not None
        return self._preflop_unblocked_projection

    def _preflop_unblocked_mass(self, class_mass: torch.Tensor) -> torch.Tensor:
        if class_mass.shape[-1] != PREFLOP_HANDS:
            raise ValueError(
                f"expected final axis {PREFLOP_HANDS}, got {class_mass.shape[-1]}"
            )
        return class_mass @ self._preflop_unblocked_projection_for(class_mass)

    def _preflop_ev_buffers(
        self,
        rows: int,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (rows, self.num_players, PREFLOP_HANDS)
        action_buf = getattr(self, "_preflop_ev_action_weights_buf", None)
        value_buf = getattr(self, "_preflop_ev_child_values_buf", None)
        if (
            action_buf is None
            or action_buf.shape[0] < rows
            or action_buf.shape[1:] != shape[1:]
            or action_buf.dtype != dtype
            or action_buf.device != self.device
        ):
            action_buf = torch.empty(shape, dtype=dtype, device=self.device)
            self._preflop_ev_action_weights_buf = action_buf
        if (
            value_buf is None
            or value_buf.shape[0] < rows
            or value_buf.shape[1:] != shape[1:]
            or value_buf.dtype != dtype
            or value_buf.device != self.device
        ):
            value_buf = torch.empty(shape, dtype=dtype, device=self.device)
            self._preflop_ev_child_values_buf = value_buf
        return action_buf[:rows], value_buf[:rows]

    def initialize_subgame(
        self,
        src_env: HUNLTensorEnv | PBSEnv,
        src_indices: torch.Tensor,
        initial_beliefs: torch.Tensor | None = None,
    ) -> None:
        if not isinstance(src_env, PBSEnv):
            raise TypeError("PreflopSparseCFREvaluator requires PBSEnv roots")
        if src_indices.dim() != 1:
            raise AssertionError("src_indices must be 1-D")
        if src_indices.numel() == 0:
            raise AssertionError("must supply at least one root state")
        root_streets = src_env.street[src_indices]
        if not (root_streets == 0).all():
            raise ValueError("PreflopSparseCFREvaluator only accepts street-0 roots")
        root_boards = src_env.board_indices[src_indices]
        if (root_boards >= 0).any():
            raise ValueError("PreflopSparseCFREvaluator roots must have no public board")
        initial_beliefs = self._compact_initial_beliefs(
            src_indices.numel(), initial_beliefs
        )
        super().initialize_subgame(src_env, src_indices, initial_beliefs)
        self._validate_compact_shapes()

    def _compact_initial_beliefs(
        self,
        num_roots: int,
        initial_beliefs: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return combo-mass-preserving compact preflop beliefs."""
        prior = preflop_class_multiplicity_tensor(device=self.device).to(
            dtype=self.float_dtype
        )
        prior = prior / prior.sum().clamp(min=1.0)
        if initial_beliefs is None:
            return prior.expand(num_roots, self.num_players, PREFLOP_HANDS).clone()

        beliefs = initial_beliefs.to(device=self.device, dtype=self.float_dtype)
        if beliefs.shape[-1] == NUM_HANDS:
            beliefs = collapse_1326_to_169(beliefs, reduction="sum")
        elif beliefs.shape[-1] != PREFLOP_HANDS:
            raise ValueError(
                f"compact preflop beliefs must have final axis {PREFLOP_HANDS} "
                f"or {NUM_HANDS}; got {beliefs.shape[-1]}"
            )
        denom = beliefs.sum(dim=-1, keepdim=True)
        return torch.where(denom > 1e-8, beliefs / denom.clamp(min=1e-8), prior)

    def _validate_compact_shapes(self) -> None:
        tensor_names = (
            "beliefs",
            "beliefs_avg",
            "self_reach",
            "self_reach_avg",
            "latest_values",
            "values_avg",
            "policy_probs",
            "policy_probs_avg",
            "cumulative_regrets",
            "average_policy_numerator",
            "average_policy_denominator",
        )
        bad = [
            name
            for name in tensor_names
            if getattr(self, name).shape[-1] != PREFLOP_HANDS
        ]
        if bad:
            raise RuntimeError(f"compact preflop tensors are not 169-wide: {bad}")

    def _init_hand_rank_data(self) -> None:
        self.hand_rank_data = None

    def _empty_allin_call_partitions(self) -> None:
        empty = torch.empty(0, dtype=torch.long, device=self.device)
        empty_boards = torch.empty(0, 5, dtype=torch.long, device=self.device)
        self.allin_call_indices = empty
        self.allin_call_parent_indices = empty
        self.allin_call_mask = torch.zeros(
            self.total_nodes, dtype=torch.bool, device=self.device
        )
        self.allin_call_indices_by_street = (empty, empty, empty)
        self.allin_call_parent_indices_by_street = (empty, empty, empty)
        self.allin_call_boards_by_street = (
            empty_boards,
            empty_boards,
            empty_boards,
        )
        self.preflop_allin_indices_by_live_count = tuple(
            empty for _ in range(self.num_players + 1)
        )
        self.preflop_allin_parent_indices_by_live_count = tuple(
            empty for _ in range(self.num_players + 1)
        )
        self.preflop_allin_exact_indices = empty
        self.preflop_allin_model_indices = empty
        self.preflop_allin_live3_entry_rows = empty
        self.preflop_allin_live3_hero_players = empty
        self.preflop_allin_live3_opp0_players = empty
        self.preflop_allin_live3_opp1_players = empty

    def _cache_preflop_allin_live_partitions(self) -> None:
        empty = torch.empty(0, dtype=torch.long, device=self.device)
        if getattr(self, "allin_call_indices", empty).numel() == 0:
            self.preflop_allin_indices_by_live_count = tuple(
                empty for _ in range(self.num_players + 1)
            )
            self.preflop_allin_parent_indices_by_live_count = tuple(
                empty for _ in range(self.num_players + 1)
            )
            self.preflop_allin_exact_indices = empty
            self.preflop_allin_model_indices = empty
            self.preflop_allin_live3_entry_rows = empty
            self.preflop_allin_live3_hero_players = empty
            self.preflop_allin_live3_opp0_players = empty
            self.preflop_allin_live3_opp1_players = empty
            return

        live_counts = (~self.env.has_folded[self.allin_call_indices]).sum(dim=1)
        indices_by_count = []
        parents_by_count = []
        for live_count in range(self.num_players + 1):
            mask = live_counts == live_count
            indices_by_count.append(self.allin_call_indices[mask].contiguous())
            parents_by_count.append(
                self.allin_call_parent_indices[mask].contiguous()
            )
        self.preflop_allin_indices_by_live_count = tuple(indices_by_count)
        self.preflop_allin_parent_indices_by_live_count = tuple(parents_by_count)
        self.preflop_allin_exact_indices = (
            torch.cat(indices_by_count[2:4]).contiguous()
            if self.num_players >= 3
            else indices_by_count[2]
        )
        self.preflop_allin_model_indices = (
            torch.cat(indices_by_count[4:]).contiguous()
            if self.num_players >= 4
            else empty
        )
        live3_idx = indices_by_count[3] if self.num_players >= 3 else empty
        if live3_idx.numel() > 0:
            live3 = ~self.env.has_folded[live3_idx]
            rows, hero_players = torch.nonzero(live3, as_tuple=True)
            opp_mask = live3[rows].clone()
            entry = torch.arange(rows.shape[0], device=self.device)
            opp_mask[entry, hero_players] = False
            opps = torch.nonzero(opp_mask, as_tuple=False)[:, 1].reshape(-1, 2)
            self.preflop_allin_live3_entry_rows = rows.contiguous()
            self.preflop_allin_live3_hero_players = hero_players.contiguous()
            self.preflop_allin_live3_opp0_players = opps[:, 0].contiguous()
            self.preflop_allin_live3_opp1_players = opps[:, 1].contiguous()
        else:
            self.preflop_allin_live3_entry_rows = empty
            self.preflop_allin_live3_hero_players = empty
            self.preflop_allin_live3_opp0_players = empty
            self.preflop_allin_live3_opp1_players = empty

    def _cache_allin_call_street_partitions(
        self, parent_streets: torch.Tensor
    ) -> None:
        empty = torch.empty(0, dtype=torch.long, device=self.device)
        empty_boards = torch.empty(0, 5, dtype=torch.long, device=self.device)
        if self.allin_call_indices.numel() == 0:
            self.allin_call_indices_by_street = (empty, empty, empty)
            self.allin_call_parent_indices_by_street = (empty, empty, empty)
            self.allin_call_boards_by_street = (
                empty_boards,
                empty_boards,
                empty_boards,
            )
            self._cache_preflop_allin_live_partitions()
            return

        street0 = parent_streets == 0
        self.allin_call_indices_by_street = (
            self.allin_call_indices[street0].contiguous(),
            empty,
            empty,
        )
        self.allin_call_parent_indices_by_street = (
            self.allin_call_parent_indices[street0].contiguous(),
            empty,
            empty,
        )
        boards = self.env.board_indices[self.allin_call_parent_indices].long()
        self.allin_call_boards_by_street = (
            boards[street0].contiguous(),
            empty_boards,
            empty_boards,
        )
        self._cache_preflop_allin_live_partitions()

    def _allin_call_child_mask(
        self,
        parent_env: HUNLTensorEnv | PBSEnv,
        parent_local_indices: torch.Tensor,
        action_bins: torch.Tensor,
    ) -> torch.Tensor:
        if not self._allin_abstraction_enabled() or action_bins.numel() == 0:
            return torch.zeros_like(action_bins, dtype=torch.bool)
        actor = parent_env.to_act[parent_local_indices]
        rows = torch.arange(parent_local_indices.numel(), device=self.device)
        live = ~parent_env.has_folded[parent_local_indices]
        actor_onehot = torch.zeros_like(live)
        actor_onehot[rows, actor] = True
        other_live = live & ~actor_onehot
        other_allin = other_live & parent_env.is_allin[parent_local_indices]
        other_betting_eligible = (
            other_live
            & ~parent_env.is_allin[parent_local_indices]
            & (parent_env.stacks[parent_local_indices] > 0)
        )
        max_committed = parent_env.committed[parent_local_indices].amax(dim=1)
        actor_committed = parent_env.committed[parent_local_indices, actor]
        parent_to_call = max_committed - actor_committed
        parent_street = parent_env.street[parent_local_indices]
        return (
            (action_bins == 1)
            & actor_onehot.any(dim=1)
            & other_allin.any(dim=1)
            & ~other_betting_eligible.any(dim=1)
            & (parent_to_call > 0)
            & (parent_street == 0)
        )

    def _mark_allin_call_leaves(self) -> None:
        self._empty_allin_call_partitions()
        if not self._allin_abstraction_enabled() or self.total_nodes <= self.root_nodes:
            return

        child_indices = torch.arange(
            self.root_nodes, self.total_nodes, device=self.device
        )
        parent, action = self._parent_action_for_nodes(child_indices)
        mask = self._allin_call_child_mask(self.env, parent, action)
        indices = child_indices[mask]
        if indices.numel() == 0:
            self._cache_preflop_allin_live_partitions()
            return

        self.allin_call_indices = indices.contiguous()
        self.allin_call_parent_indices = parent[mask].contiguous()
        self.allin_call_mask[self.allin_call_indices] = True
        self.leaf_mask[self.allin_call_indices] = True
        self.new_street_mask[self.allin_call_indices] = False
        parent_street = self.env.street[self.allin_call_parent_indices].contiguous()
        self._cache_allin_call_street_partitions(parent_street)
        self._prune_allin_call_descendants()

    def _ensure_preflop_allin_169_oracle(self) -> PreflopAllIn169Oracle:
        oracle = getattr(self, "preflop_allin_169_oracle", None)
        if oracle is not None:
            return oracle
        search_cfg = getattr(getattr(self, "cfg", None), "search", None)
        oracle = PreflopAllIn169Oracle(
            device=self.device,
            exact_cache_path=getattr(
                search_cfg,
                "preflop_allin_169_cache_path",
                None,
            ),
            model_checkpoint_path=getattr(
                search_cfg,
                "preflop_allin_169_model_checkpoint",
                None,
            ),
            compile_model=getattr(
                search_cfg,
                "preflop_allin_169_compile_model",
                False,
            ),
        )
        self.preflop_allin_169_oracle = oracle
        return oracle

    def _set_allin_call_values(self, beliefs: torch.Tensor) -> None:
        indices = getattr(self, "allin_call_indices", None)
        if indices is None or indices.numel() == 0:
            return
        if not hasattr(self, "preflop_allin_indices_by_live_count"):
            self._cache_preflop_allin_live_partitions()
        oracle = self._ensure_preflop_allin_169_oracle()
        for live_players in range(2, min(3, self.num_players) + 1):
            node_idx = self.preflop_allin_indices_by_live_count[live_players]
            if node_idx.numel() == 0:
                continue
            beliefs_at_nodes = beliefs[node_idx]
            starting_stacks = self.env.starting_stacks[node_idx].to(torch.float32)
            committed = self.env.chips_placed[node_idx].to(torch.float32)
            stacks_after = self.env.stacks[node_idx].to(torch.float32)
            folded_mask = self.env.has_folded[node_idx]
            scale = self.env.scale[node_idx].to(torch.float32)
            if live_players == 3 and hasattr(oracle, "values_for_live3_entries"):
                values = oracle.values_for_live3_entries(
                    beliefs=beliefs_at_nodes,
                    starting_stacks=starting_stacks,
                    committed=committed,
                    stacks_after=stacks_after,
                    folded_mask=folded_mask,
                    scale=scale,
                    live_entry_rows=self.preflop_allin_live3_entry_rows,
                    hero_players=self.preflop_allin_live3_hero_players,
                    opp0_players=self.preflop_allin_live3_opp0_players,
                    opp1_players=self.preflop_allin_live3_opp1_players,
                )
            else:
                values = oracle.values(
                    beliefs=beliefs_at_nodes,
                    starting_stacks=starting_stacks,
                    committed=committed,
                    stacks_after=stacks_after,
                    allin_mask=self.env.is_allin[node_idx],
                    folded_mask=folded_mask,
                    scale=scale,
                    live_players=live_players,
                )
            self.latest_values[node_idx] = values.to(self.latest_values.dtype)

        node_idx = self.preflop_allin_model_indices
        if node_idx.numel() == 0:
            return
        values = oracle.values(
            beliefs=beliefs[node_idx],
            starting_stacks=self.env.starting_stacks[node_idx].to(torch.float32),
            committed=self.env.chips_placed[node_idx].to(torch.float32),
            stacks_after=self.env.stacks[node_idx].to(torch.float32),
            allin_mask=self.env.is_allin[node_idx],
            folded_mask=self.env.has_folded[node_idx],
            scale=self.env.scale[node_idx].to(torch.float32),
            live_players=self.num_players,
        )
        self.latest_values[node_idx] = values.to(self.latest_values.dtype)

    def _compute_policy_node_reach(self, top: int) -> torch.Tensor:
        allowed = self.allowed_hands[:top].to(dtype=self.float_dtype)
        reach = self.self_reach_avg[:top].to(dtype=self.float_dtype)
        unblocked = self._preflop_unblocked_mass(reach)
        player_ids = torch.arange(self.num_players, device=self.device)
        other_live = player_ids[None, :, None] != 0
        if hasattr(self.env, "has_folded"):
            other_live &= ~self.env.has_folded[:top, :, None]
        other_reach = torch.where(
            other_live,
            unblocked.clamp_min(1e-12),
            torch.ones_like(unblocked),
        ).prod(dim=1)
        numer = (reach[:, 0] * other_reach * allowed).sum(dim=-1)

        allowed_unblocked = self._preflop_unblocked_mass(allowed)
        other_allowed = allowed_unblocked.clamp_min(1e-12).pow(
            max(0, self.num_players - 1)
        )
        denom = (allowed * other_allowed).sum(dim=-1).clamp(min=1e-12)
        return (numer / denom).clamp(min=0.0, max=1.0)

    def compute_expected_values(
        self,
        policy: torch.Tensor | None = None,
        beliefs: torch.Tensor | None = None,
        leaf_values: torch.Tensor | None = None,
        values: torch.Tensor | None = None,
    ) -> None:
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

        for depth in range(self.tree_depth - 1, -1, -1):
            child_start = self.depth_offsets[depth + 1]
            child_end = self.depth_offsets[depth + 2]
            if child_end == child_start:
                continue
            child_rows = child_end - child_start
            parent = self.parent_index[child_start:child_end]
            prev_actor = self.prev_actor[child_start:child_end]
            child_policy = policy[child_start:child_end]

            actor_beliefs = beliefs[parent].gather(
                1,
                prev_actor[:, None, None].expand(-1, 1, PREFLOP_HANDS),
            ).squeeze(1)
            matchup_mass = self._preflop_unblocked_mass(actor_beliefs)
            policy_blocked = self._preflop_unblocked_mass(actor_beliefs * child_policy)
            opponent_conditioned_policy = torch.where(
                matchup_mass > 1e-8,
                policy_blocked / matchup_mass.clamp(min=1e-8),
                torch.zeros_like(policy_blocked),
            )
            action_weights, child_values = self._preflop_ev_buffers(
                child_rows,
                dtype=values.dtype,
            )
            action_weights.copy_(
                opponent_conditioned_policy[:, None, :].expand(
                    -1, self.num_players, -1
                )
            )
            action_weights.scatter_(
                1,
                prev_actor[:, None, None].expand(-1, 1, PREFLOP_HANDS),
                child_policy[:, None, :],
            )
            torch.mul(values[child_start:child_end], action_weights, out=child_values)
            self._pull_back_sum(child_values, values, level=depth)

    def compute_instantaneous_regrets(
        self, values_achieved: torch.Tensor, values_expected: torch.Tensor | None = None
    ) -> torch.Tensor:
        if values_expected is None:
            values_expected = values_achieved

        bottom = self.depth_offsets[1]
        beliefs = self.beliefs_avg if self.cfr_avg else self.beliefs
        regrets = torch.zeros_like(self.policy_probs)
        src_actor_indices = self.env.to_act[:, None, None].expand(
            -1, 1, PREFLOP_HANDS
        )
        prev_actor_indices = self.prev_actor[bottom:, None, None].expand(
            -1, 1, PREFLOP_HANDS
        )
        actor_values_expected = self._fan_out(
            values_expected.gather(1, src_actor_indices).squeeze(1)
        )
        actor_values_achieved = values_achieved[bottom:].gather(
            1, prev_actor_indices
        ).squeeze(1)
        unblocked_reach = self._preflop_unblocked_mass(beliefs)
        player_ids = torch.arange(self.num_players, device=self.device)
        other_live = player_ids[None, :, None] != self.env.to_act[:, None, None]
        if hasattr(self.env, "has_folded"):
            other_live &= ~self.env.has_folded[:, :, None]
        src_weights = torch.where(
            other_live,
            unblocked_reach.clamp_min(1e-12),
            torch.ones_like(unblocked_reach),
        ).prod(dim=1)
        src_weights *= self.allowed_hands.to(dtype=self.float_dtype)
        weights = self._fan_out(src_weights)

        regrets[bottom:] = weights * (actor_values_achieved - actor_values_expected)
        self._mask_invalid(regrets)
        return regrets

    def _compute_exploitability(self) -> ExploitabilityStats:
        """Skip heads-up exploitability diagnostics for multiway preflop solves."""

        local = torch.zeros(
            self.root_nodes, dtype=self.float_dtype, device=self.device
        )
        br_values = torch.zeros(
            self.root_nodes,
            self.num_players,
            dtype=self.float_dtype,
            device=self.device,
        )
        return ExploitabilityStats(
            local_exploitability=local,
            local_best_response_values=br_values,
        )

    def _root_leaf_target_source_counts(self, num_roots: int) -> dict[str, torch.Tensor]:
        if self.num_players == 2:
            return super()._root_leaf_target_source_counts(num_roots)

        device = self.device
        return {
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
