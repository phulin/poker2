from __future__ import annotations

import torch

from p2.core.structured_config import Config
from p2.env.card_utils import (
    NUM_HANDS,
    PREFLOP_HANDS,
    collapse_1326_to_169,
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
        self.warm_start_iterations = 0

    @property
    def _compact_preflop(self) -> bool:
        return int(getattr(self, "hand_dim", NUM_HANDS)) == PREFLOP_HANDS

    def _continuation_value_target_sampling_enabled(self) -> bool:
        return True

    def _continuation_value_target_replace_roots(self) -> bool:
        return True

    def _continuation_value_target_streets(self) -> tuple[int, ...]:
        return (0,)

    def warm_start(self) -> None:
        return None

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
        if self._compact_preflop:
            initial_beliefs = self._compact_initial_beliefs(
                src_indices.numel(), initial_beliefs
            )
        super().initialize_subgame(src_env, src_indices, initial_beliefs)
        if self._compact_preflop:
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
        if self._compact_preflop:
            self.hand_rank_data = None
            return
        super()._init_hand_rank_data()

    def _set_allin_call_values(self, beliefs: torch.Tensor) -> None:
        if self._compact_preflop and self.allin_call_indices.numel() > 0:
            raise NotImplementedError(
                "compact preflop all-in terminal values require a 169-class "
                "all-in resolver; disable allin_call_terminal_abstraction for now"
            )
        super()._set_allin_call_values(beliefs)

    def _compute_policy_node_reach(self, top: int) -> torch.Tensor:
        if not self._compact_preflop:
            return super()._compute_policy_node_reach(top)
        allowed = self.allowed_hands[:top].to(dtype=self.float_dtype)
        reach = self.self_reach_avg[:top].to(dtype=self.float_dtype)
        live_players = torch.ones(
            top, self.num_players, 1, dtype=torch.bool, device=self.device
        )
        if hasattr(self.env, "has_folded"):
            live_players = ~self.env.has_folded[:top, :, None]
        reach_product = torch.where(
            live_players,
            reach.clamp_min(1e-12),
            torch.ones_like(reach),
        ).prod(dim=1)
        numer = (reach_product * allowed).sum(dim=-1)
        denom = allowed.sum(dim=-1).clamp(min=1.0)
        return (numer / denom).clamp(min=0.0, max=1.0)

    def compute_expected_values(
        self,
        policy: torch.Tensor | None = None,
        beliefs: torch.Tensor | None = None,
        leaf_values: torch.Tensor | None = None,
        values: torch.Tensor | None = None,
    ) -> None:
        if not self._compact_preflop:
            return super().compute_expected_values(
                policy=policy,
                beliefs=beliefs,
                leaf_values=leaf_values,
                values=values,
            )
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

        player_ids = torch.arange(self.num_players, device=self.device)
        for depth in range(self.tree_depth - 1, -1, -1):
            child_start = self.depth_offsets[depth + 1]
            child_end = self.depth_offsets[depth + 2]
            if child_end == child_start:
                continue
            parent = self.parent_index[child_start:child_end]
            prev_actor = self.prev_actor[child_start:child_end]
            child_policy = policy[child_start:child_end]
            child_values = values[child_start:child_end].clone()

            actor_beliefs = beliefs[parent].gather(
                1,
                prev_actor[:, None, None].expand(-1, 1, PREFLOP_HANDS),
            ).squeeze(1)
            public_action_weight = (actor_beliefs * child_policy).sum(
                dim=-1
            ).clamp_min(0.0)
            is_actor = player_ids[None, :, None] == prev_actor[:, None, None]
            action_weights = torch.where(
                is_actor,
                child_policy[:, None, :],
                public_action_weight[:, None, None],
            )
            child_values *= action_weights
            self._pull_back_sum(child_values, values, level=depth)

    def compute_instantaneous_regrets(
        self, values_achieved: torch.Tensor, values_expected: torch.Tensor | None = None
    ) -> torch.Tensor:
        if not self._compact_preflop:
            return super().compute_instantaneous_regrets(
                values_achieved, values_expected=values_expected
            )
        if values_expected is None:
            values_expected = values_achieved

        bottom = self.depth_offsets[1]
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
        regrets[bottom:] = (
            actor_values_achieved - actor_values_expected
        ) * self.allowed_hands[bottom:].to(dtype=self.float_dtype)
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
