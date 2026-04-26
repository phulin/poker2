"""TrueSkill rating tracker for periodic model checkpoints.

Snapshots EMA weights at fixed fractions of the training run, then plays each
new snapshot against a recency-weighted sample of all prior snapshots using
public-belief CFR games. Ratings are updated per-game with the standard
TrueSkill 1v1 Bayesian update.

Snapshots are kept in CPU RAM (bfloat16 by default) — never written to disk.
A single pair of compiled model instances is reused for all matchups; per-pair
weights are bound in via parameter `.data` rebinding (same trick as EMAHelper)
so no recompilation is triggered.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from p2.core.structured_config import CFRType, Config, SearchConfig
from p2.rl.pbs_pool import PBSPool


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class TSSnapshot:
    step: int
    mu: float
    sigma: float
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    # name -> tensor on CPU, in snapshot dtype
    weights: Dict[str, torch.Tensor] = field(default_factory=dict)


@contextmanager
def _bind_weights(module: nn.Module, weights: Dict[str, torch.Tensor]):
    """Temporarily bind ``weights`` (CPU/any-dtype) into ``module`` parameters.

    Mirrors EMAHelper.swapped: rebinds param.data so a compiled module picks up
    the new weights without recompiling. We cast/move to each parameter's
    device+dtype lazily, caching on the snapshot tensor identity.
    """
    if isinstance(module, nn.DataParallel):
        module = module.module  # ty:ignore[invalid-assignment]
    saved: list[tuple[nn.Parameter, torch.Tensor]] = []
    try:
        for name, param in module.named_parameters():
            if name in weights:
                saved.append((param, param.data))
                src = weights[name]
                if src.device != param.device or src.dtype != param.dtype:
                    src = src.to(device=param.device, dtype=param.dtype)
                param.data = src
        yield module
    finally:
        for param, data in saved:
            param.data = data


class TrueSkillTracker:
    def __init__(
        self,
        cfg: Config,
        candidate_model: nn.Module,
        opponent_model: nn.Module,
        device: torch.device,
        generator: torch.Generator,
    ):
        self.cfg = cfg
        self.ts_cfg = cfg.trueskill
        self.candidate_model = candidate_model
        self.opponent_model = opponent_model
        self.device = device
        self.generator = generator
        self.snapshots: List[TSSnapshot] = []
        self.snapshot_dtype = _DTYPE_MAP[self.ts_cfg.snapshot_dtype]

        total_steps = max(1, cfg.num_steps)
        # Snapshot every snapshot_frac of the run, but at least every step.
        self.snapshot_interval = max(
            1, int(round(self.ts_cfg.snapshot_frac * total_steps))
        )
        # Number of snapshots we'll take over the run.
        n_snapshots = max(1, total_steps // self.snapshot_interval)

        # Total scheduled training "actions" ≈ steps * envs * actions_per_game.
        # Eval budget (in actions) = game_budget_frac * total. Convert to games
        # by dividing by actions_per_game, then split across snapshots. The
        # k-th snapshot (1-indexed) plays vs k-1 prior opponents, so total
        # eval games are roughly (n_snapshots - 1) * games_per_eval.
        total_actions = total_steps * cfg.num_envs * self.ts_cfg.actions_per_game
        eval_action_budget = self.ts_cfg.game_budget_frac * total_actions
        eval_game_budget = eval_action_budget / self.ts_cfg.actions_per_game
        # Spread across n_snapshots evals (the very first eval has no opponents
        # so it's free; that just gives us a small budget headroom).
        self.games_per_eval = max(1, int(eval_game_budget / max(1, n_snapshots)))

        print(
            f"[TrueSkill] enabled. snapshot_interval={self.snapshot_interval} "
            f"steps; ~{n_snapshots} snapshots planned; "
            f"games_per_eval={self.games_per_eval}"
        )

    # ------------------------------------------------------------------ utils

    def should_snapshot(self, step_public: int) -> bool:
        """Return True if step_public (1-indexed) is a snapshot point."""
        if not self.ts_cfg.enabled:
            return False
        return step_public % self.snapshot_interval == 0

    def _clone_weights_cpu(self, src_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for name, t in src_weights.items():
            out[name] = t.detach().to(device="cpu", dtype=self.snapshot_dtype).clone()
        return out

    # ----------------------------------------------------------- ts updates

    def _ts_update(
        self,
        mu_w: float,
        sig_w: float,
        mu_l: float,
        sig_l: float,
        draw: bool = False,
    ) -> Tuple[float, float, float, float]:
        """Standard TrueSkill 1v1 update. Returns (mu_w', sig_w', mu_l', sig_l')."""
        beta = self.ts_cfg.beta
        tau = self.ts_cfg.tau
        c2 = 2.0 * beta * beta + sig_w * sig_w + sig_l * sig_l
        c = math.sqrt(c2)
        t = (mu_w - mu_l) / c

        # Standard normal pdf/cdf
        def pdf(x: float) -> float:
            return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

        def cdf(x: float) -> float:
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

        if not draw:
            denom = cdf(t)
            if denom < 1e-12:
                denom = 1e-12
            v = pdf(t) / denom
            w = v * (v + t)
            mu_w_new = mu_w + (sig_w * sig_w / c) * v
            mu_l_new = mu_l - (sig_l * sig_l / c) * v
            sig_w_new = math.sqrt(
                max(1e-12, sig_w * sig_w * (1.0 - (sig_w * sig_w / c2) * w) + tau * tau)
            )
            sig_l_new = math.sqrt(
                max(1e-12, sig_l * sig_l * (1.0 - (sig_l * sig_l / c2) * w) + tau * tau)
            )
            return mu_w_new, sig_w_new, mu_l_new, sig_l_new
        else:
            # Symmetric draw update (eps = 0 for poker; near-zero rewards count as draws).
            denom = cdf(t) - cdf(-t)
            if abs(denom) < 1e-12:
                denom = 1e-12 if denom >= 0 else -1e-12
            v = (pdf(-t) - pdf(t)) / denom
            w = v * v + (t * pdf(t) + (-t) * pdf(-t)) / denom
            mu_w_new = mu_w + (sig_w * sig_w / c) * v
            mu_l_new = mu_l - (sig_l * sig_l / c) * v
            sig_w_new = math.sqrt(
                max(1e-12, sig_w * sig_w * (1.0 - (sig_w * sig_w / c2) * w) + tau * tau)
            )
            sig_l_new = math.sqrt(
                max(1e-12, sig_l * sig_l * (1.0 - (sig_l * sig_l / c2) * w) + tau * tau)
            )
            return mu_w_new, sig_w_new, mu_l_new, sig_l_new

    def _conservative_skill(self, snap: TSSnapshot) -> float:
        return snap.mu - 3.0 * snap.sigma

    # ------------------------------------------------------- opponent sampling

    def _allocate_games(self, n_opponents: int, total_games: int) -> List[int]:
        """Distribute ``total_games`` across ``n_opponents`` weighted toward
        most-recent (highest index = most recent)."""
        if n_opponents == 0 or total_games == 0:
            return [0] * n_opponents
        tau = max(1e-3, self.ts_cfg.recency_tau_frac) * n_opponents
        # weights for index i (0..n-1), with i = n-1 most recent
        raw = [math.exp(-(n_opponents - 1 - i) / tau) for i in range(n_opponents)]
        s = sum(raw)
        norm = [r / s for r in raw]
        alloc = [int(round(w * total_games)) for w in norm]
        # clamp
        lo, hi = self.ts_cfg.min_games_per_opponent, self.ts_cfg.max_games_per_opponent
        alloc = [max(lo, min(hi, a)) for a in alloc]
        return alloc

    # ------------------------------------------------------- main entrypoint

    def snapshot_and_evaluate(
        self,
        step: int,
        candidate_weights: Dict[str, torch.Tensor],
        wandb_run=None,
    ) -> Dict[str, float]:
        """Take a snapshot of ``candidate_weights`` (typically EMA shadow) and
        play it against a recency-weighted sample of prior snapshots, updating
        TrueSkill ratings on both sides.

        Returns a dict of metrics suitable for wandb.log.
        """
        # Each snapshot is a distinct player (different weights → different
        # strategy). Past snapshots' true skills are fixed; only the current
        # candidate is a *new* player joining the league, so it gets the
        # cold TrueSkill prior. Once these games are played, this snapshot's
        # posterior is frozen alongside its weights for future evals.
        new_snap = TSSnapshot(
            step=step,
            mu=self.ts_cfg.initial_mu,
            sigma=self.ts_cfg.initial_sigma,
            weights=self._clone_weights_cpu(candidate_weights),
        )

        opponents = list(self.snapshots)  # ordered oldest -> newest
        allocations = self._allocate_games(len(opponents), self.games_per_eval)

        # Search config for eval games (cheap / fast).
        search_cfg = SearchConfig()
        search_cfg.depth = 1
        search_cfg.iterations = 1
        search_cfg.warm_start_iterations = 0
        search_cfg.cfr_type = CFRType.linear
        search_cfg.cfr_avg = True
        bet_bins = list(self.cfg.env.bet_bins)

        total_games_played = 0
        total_reward = 0.0

        # Bind candidate weights into the candidate model once for the full eval.
        with _bind_weights(self.candidate_model, new_snap.weights):
            for opp, n_games in zip(opponents, allocations):
                if n_games <= 0:
                    continue
                with _bind_weights(self.opponent_model, opp.weights):
                    rewards = PBSPool._play_public_belief_games(
                        self.candidate_model,
                        self.opponent_model,
                        n_games,
                        bet_bins,
                        self.generator,
                        self.device,
                        search_cfg,
                    )

                # Per-game TrueSkill updates from candidate (=p0) perspective.
                rewards_list = rewards.detach().cpu().tolist()
                for r in rewards_list:
                    if r > 1e-6:
                        # candidate wins
                        mu_w, sig_w, mu_l, sig_l = self._ts_update(
                            new_snap.mu, new_snap.sigma, opp.mu, opp.sigma, draw=False
                        )
                        new_snap.mu, new_snap.sigma = mu_w, sig_w
                        opp.mu, opp.sigma = mu_l, sig_l
                        new_snap.wins += 1
                        opp.losses += 1
                    elif r < -1e-6:
                        # opponent wins
                        mu_w, sig_w, mu_l, sig_l = self._ts_update(
                            opp.mu, opp.sigma, new_snap.mu, new_snap.sigma, draw=False
                        )
                        opp.mu, opp.sigma = mu_w, sig_w
                        new_snap.mu, new_snap.sigma = mu_l, sig_l
                        new_snap.losses += 1
                        opp.wins += 1
                    else:
                        mu_w, sig_w, mu_l, sig_l = self._ts_update(
                            new_snap.mu, new_snap.sigma, opp.mu, opp.sigma, draw=True
                        )
                        new_snap.mu, new_snap.sigma = mu_w, sig_w
                        opp.mu, opp.sigma = mu_l, sig_l
                        new_snap.draws += 1
                        opp.draws += 1
                    new_snap.games += 1
                    opp.games += 1
                    total_games_played += 1
                total_reward += float(rewards.sum().item())

        self.snapshots.append(new_snap)

        # Build metrics.
        skill = self._conservative_skill(new_snap)
        metrics: Dict[str, float] = {
            "trueskill/mu": new_snap.mu,
            "trueskill/sigma": new_snap.sigma,
            "trueskill/skill": skill,
            "trueskill/games_played": total_games_played,
            "trueskill/snapshots": len(self.snapshots),
            "trueskill/avg_reward": (
                total_reward / total_games_played if total_games_played else 0.0
            ),
        }

        # Best (by conservative skill) across all snapshots so far.
        if self.snapshots:
            best = max(self.snapshots, key=self._conservative_skill)
            metrics["trueskill/best_step"] = best.step
            metrics["trueskill/best_skill"] = self._conservative_skill(best)

        if wandb_run is not None:
            try:
                wandb_run.log(metrics, step=step)
            except Exception:
                pass

        print(
            f"[TrueSkill] step={step} mu={new_snap.mu:.2f} sigma={new_snap.sigma:.2f} "
            f"skill={skill:.2f} games={total_games_played} "
            f"snapshots={len(self.snapshots)}"
        )
        return metrics
