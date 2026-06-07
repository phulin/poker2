from __future__ import annotations

import torch

from p2.core.structured_config import Config
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
        super().initialize_subgame(src_env, src_indices, initial_beliefs)

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
