"""TrueSkill rating tracker for periodic model checkpoints.

Snapshots EMA weights at fixed fractions of the training run, then plays each
new snapshot against a bounded recency-weighted sample of prior snapshots using
public-belief CFR games. Ratings are updated per-game with the standard
TrueSkill 1v1 Bayesian update. A second Gaussian reward rating is updated once
per opponent matchup from the mean stack-normalized PBS reward.

Snapshots are kept in CPU RAM (bfloat16 by default) — never written to disk.
A single pair of compiled model instances is reused for all matchups; per-pair
weights are bound in via parameter `.data` rebinding (same trick as EMAHelper)
so no recompilation is triggered.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from p2.core.structured_config import Config
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.rl.pbs_games import play_public_belief_games


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
    gaussian_mu: float = 0.0
    gaussian_sigma: float = 1.0
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
        trainer_evaluator,
    ):
        self.cfg = cfg
        self.ts_cfg = cfg.trueskill
        self.candidate_model = candidate_model
        self.opponent_model = opponent_model
        self.device = device
        self.generator = generator
        self.trainer_evaluator = trainer_evaluator
        self.snapshots: List[TSSnapshot] = []
        self.snapshot_dtype = _DTYPE_MAP[self.ts_cfg.snapshot_dtype]

        # Build a single-env prototype that matches the training env config.
        self.env_proto = HUNLTensorEnv(
            num_envs=1,
            starting_stack=cfg.env.stack,
            sb=cfg.env.sb,
            bb=cfg.env.bb,
            default_bet_bins=cfg.env.bet_bins,
            device=device,
            float_dtype=torch.float32,
            flop_showdown=cfg.env.flop_showdown,
            randomize_stacks=cfg.env.randomize_stacks,
            stack_mode=cfg.env.stack_mode,
            min_stack_bb=cfg.env.min_stack_bb,
            mid_stack_bb=cfg.env.mid_stack_bb,
            max_stack_bb=cfg.env.max_stack_bb,
            high_stack_mass_ratio=cfg.env.high_stack_mass_ratio,
        )

        # Build matchup evaluators that mirror the trainer's data-generator
        # config — same depth, same CFR variant, same DCFR params, same
        # warm-start. We sync the live-scheduled (iterations, warm_start,
        # dcfr_delay) values just before each eval round.
        self.evaluator_a = self._build_eval(candidate_model)
        self.evaluator_b = self._build_eval(opponent_model)

        total_steps = max(1, cfg.num_steps)
        # Snapshot every snapshot_frac of the run, but at least every step.
        self.snapshot_interval = max(
            1, int(round(self.ts_cfg.snapshot_frac * total_steps))
        )
        # Number of snapshots we'll take over the run.
        n_snapshots = max(1, total_steps // self.snapshot_interval)

        self.opponents_per_eval = max(1, int(self.ts_cfg.opponents_per_eval))
        self.games_per_opponent = self._compute_games_per_opponent(
            total_steps=total_steps,
            num_envs=cfg.num_envs,
            n_snapshots=n_snapshots,
            ts_cfg=self.ts_cfg,
            opponents_per_eval=self.opponents_per_eval,
        )

        print(
            f"[TrueSkill] enabled. snapshot_interval={self.snapshot_interval} "
            f"steps; ~{n_snapshots} snapshots planned; "
            f"opponents_per_eval={self.opponents_per_eval}; "
            f"games_per_opponent={self.games_per_opponent}; "
            f"eval_solves_per_action={self.ts_cfg.eval_solves_per_action:g}; "
            f"eval_inefficiency_factor={self.ts_cfg.eval_inefficiency_factor:g}"
        )

    # ------------------------------------------------------------- evaluators

    def _build_eval(self, model: nn.Module):
        """Build a CFR evaluator that mirrors the trainer's class + config."""
        cfg = self.cfg
        if not cfg.search.sparse:
            raise ValueError(
                "Dense RebelCFREvaluator has been removed; set search.sparse=true."
            )
        if cfg.search.sparse_fused:
            from p2.search.fused_sparse_cfr_evaluator import FusedSparseCFREvaluator

            ev = FusedSparseCFREvaluator(
                model=model,
                device=self.device,
                cfg=cfg,
                generator=self.generator,
            )
        else:
            from p2.search.sparse_cfr_evaluator import SparseCFREvaluator

            ev = SparseCFREvaluator(
                model=model,
                device=self.device,
                cfg=cfg,
                generator=self.generator,
            )
        return ev

    def _sync_live_schedule(self):
        """Copy the trainer evaluator's live-scheduled iteration counts onto
        our matchup evaluators so they search at the same fidelity the
        trainer is currently using."""
        for ev in (self.evaluator_a, self.evaluator_b):
            ev.cfr_iterations = self.trainer_evaluator.cfr_iterations
            ev.warm_start_iterations = self.trainer_evaluator.warm_start_iterations
            if hasattr(self.trainer_evaluator, "dcfr_delay"):
                ev.dcfr_delay = self.trainer_evaluator.dcfr_delay

    # ------------------------------------------------------------------ utils

    def should_snapshot(self, step_public: int) -> bool:
        """Return True if step_public (1-indexed) is a snapshot point."""
        if not self.ts_cfg.enabled:
            return False
        return step_public % self.snapshot_interval == 0

    def _clone_weights_cpu(
        self, src_weights: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
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

    def _gaussian_conservative_skill(self, snap: TSSnapshot) -> float:
        return snap.gaussian_mu - 3.0 * snap.gaussian_sigma

    def _gaussian_score_update(
        self,
        mu_a: float,
        sig_a: float,
        mu_b: float,
        sig_b: float,
        reward_mean: float,
        num_games: int,
    ) -> Tuple[float, float, float, float]:
        """Gaussian reward-rating update from stack-normalized matchup reward.

        Observation model:
            reward_mean ~ Normal(skill_a - skill_b, beta^2 / num_games)

        We keep only independent marginal variances after each update, matching
        the rest of the lightweight tracker implementation.
        """
        games = max(1, int(num_games))
        beta = max(1e-9, float(self.ts_cfg.gaussian_beta))
        tau = max(0.0, float(self.ts_cfg.gaussian_tau))
        var_a = sig_a * sig_a
        var_b = sig_b * sig_b
        obs_var = beta * beta / games
        c2 = max(1e-12, var_a + var_b + obs_var)
        residual = reward_mean - (mu_a - mu_b)

        mu_a_new = mu_a + (var_a / c2) * residual
        mu_b_new = mu_b - (var_b / c2) * residual
        sig_a_new = math.sqrt(max(1e-12, var_a - (var_a * var_a / c2) + tau * tau))
        sig_b_new = math.sqrt(max(1e-12, var_b - (var_b * var_b / c2) + tau * tau))
        return mu_a_new, sig_a_new, mu_b_new, sig_b_new

    @staticmethod
    def _compute_games_per_opponent(
        *,
        total_steps: int,
        num_envs: int,
        n_snapshots: int,
        ts_cfg,
        opponents_per_eval: int | None = None,
    ) -> int:
        """Convert the nominal eval action budget into games per sampled opponent."""
        total_actions = total_steps * num_envs * ts_cfg.actions_per_game
        eval_action_budget = ts_cfg.game_budget_frac * total_actions
        cost_per_eval_game = (
            ts_cfg.actions_per_game
            * max(1e-6, float(ts_cfg.eval_solves_per_action))
            * max(1e-6, float(ts_cfg.eval_inefficiency_factor))
        )
        eval_game_budget = eval_action_budget / cost_per_eval_game
        # Spread across n_snapshots evals. The first eval has no opponents, so
        # this leaves a small amount of budget headroom.
        games_per_eval = int(eval_game_budget / max(1, n_snapshots))
        if opponents_per_eval is None:
            opponents_per_eval = max(1, int(getattr(ts_cfg, "opponents_per_eval", 10)))
        games_per_opponent = games_per_eval // max(1, int(opponents_per_eval))
        lo = max(1, int(ts_cfg.min_games_per_opponent))
        hi = max(lo, int(ts_cfg.max_games_per_opponent))
        return min(hi, max(lo, games_per_opponent))

    @staticmethod
    def _compute_games_per_eval(
        *,
        total_steps: int,
        num_envs: int,
        n_snapshots: int,
        ts_cfg,
    ) -> int:
        """Convert the nominal eval action budget into total games per snapshot."""
        total_actions = total_steps * num_envs * ts_cfg.actions_per_game
        eval_action_budget = ts_cfg.game_budget_frac * total_actions
        cost_per_eval_game = (
            ts_cfg.actions_per_game
            * max(1e-6, float(ts_cfg.eval_solves_per_action))
            * max(1e-6, float(ts_cfg.eval_inefficiency_factor))
        )
        eval_game_budget = eval_action_budget / cost_per_eval_game
        return max(1, int(eval_game_budget / max(1, n_snapshots)))

    @staticmethod
    def _step_regression_metrics(
        snapshots: List[TSSnapshot],
        values: List[float],
        prefix: str,
    ) -> Dict[str, float]:
        """Fit snapshot values against snapshot step.

        For an intercept linear regression with one feature, R^2 is the squared
        Pearson correlation. Fewer than three snapshots are too thin to report
        a useful trend, so no trend metrics are emitted. Degenerate variance
        cases with at least three snapshots are reported as zero signal rather
        than NaN so log streams stay numeric.
        """
        n = len(snapshots)
        if n < 3:
            return {}

        xs = [float(s.step) for s in snapshots]
        ys = [float(y) for y in values]
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n

        ss_xx = 0.0
        ss_yy = 0.0
        ss_xy = 0.0
        for x, y in zip(xs, ys):
            dx = x - x_mean
            dy = y - y_mean
            ss_xx += dx * dx
            ss_yy += dy * dy
            ss_xy += dx * dy

        if ss_xx <= 0.0 or ss_yy <= 0.0:
            corr = 0.0
        else:
            corr = ss_xy / math.sqrt(ss_xx * ss_yy)
            corr = max(-1.0, min(1.0, corr))

        return {
            f"{prefix}/mu_step_correlation": corr,
            f"{prefix}/mu_step_r2": corr * corr,
        }

    @staticmethod
    def _mu_step_regression_metrics(snapshots: List[TSSnapshot]) -> Dict[str, float]:
        """Fit TrueSkill and Gaussian snapshot mu values against snapshot step."""
        metrics = TrueSkillTracker._step_regression_metrics(
            snapshots,
            [s.mu for s in snapshots],
            "trueskill",
        )
        metrics.update(
            TrueSkillTracker._step_regression_metrics(
                snapshots,
                [s.gaussian_mu for s in snapshots],
                "gaussian",
            )
        )
        return metrics

    # ------------------------------------------------------- opponent sampling

    def _opponent_sampling_weights(self, n_opponents: int) -> torch.Tensor:
        """Return recency weights for stored opponents, oldest -> newest."""
        tau = max(1e-3, self.ts_cfg.recency_tau_frac) * n_opponents
        return torch.tensor(
            [math.exp(-(n_opponents - 1 - i) / tau) for i in range(n_opponents)],
            dtype=torch.float32,
            device=self.device,
        )

    def _sample_opponent_indices(self, n_opponents: int) -> List[int]:
        """Sample a bounded set of opponent indices, sorted oldest -> newest."""
        if n_opponents <= 0:
            return []
        sample_count = min(n_opponents, self.opponents_per_eval)
        if sample_count == n_opponents:
            return list(range(n_opponents))

        weights = self._opponent_sampling_weights(n_opponents)
        sampled = torch.multinomial(
            weights,
            num_samples=sample_count,
            replacement=False,
            generator=self.generator,
        )
        return sorted(sampled.detach().cpu().tolist())

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
            gaussian_mu=self.ts_cfg.gaussian_initial_mu,
            gaussian_sigma=self.ts_cfg.gaussian_initial_sigma,
            weights=self._clone_weights_cpu(candidate_weights),
        )

        opponents = list(self.snapshots)  # ordered oldest -> newest
        opponent_indices = self._sample_opponent_indices(len(opponents))
        sampled_opponents = [opponents[i] for i in opponent_indices]

        # Use the same search fidelity the trainer is currently running with.
        self._sync_live_schedule()

        total_games_played = 0
        total_reward = 0.0
        total_reward_sq = 0.0
        eval_start = time.perf_counter()

        # Bind candidate weights into the candidate model once for the full eval.
        with _bind_weights(self.candidate_model, new_snap.weights):
            for opp in sampled_opponents:
                with _bind_weights(self.opponent_model, opp.weights):
                    rewards = play_public_belief_games(
                        self.evaluator_a,
                        self.evaluator_b,
                        self.env_proto,
                        self.games_per_opponent,
                        self.generator,
                        self.device,
                    )

                # Per-game TrueSkill updates from candidate (=p0) perspective.
                rewards_list = rewards.detach().cpu().tolist()
                matchup_games = len(rewards_list)
                if matchup_games > 0:
                    matchup_mean_reward = float(sum(rewards_list) / matchup_games)
                    (
                        new_snap.gaussian_mu,
                        new_snap.gaussian_sigma,
                        opp.gaussian_mu,
                        opp.gaussian_sigma,
                    ) = self._gaussian_score_update(
                        new_snap.gaussian_mu,
                        new_snap.gaussian_sigma,
                        opp.gaussian_mu,
                        opp.gaussian_sigma,
                        matchup_mean_reward,
                        matchup_games,
                    )
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
                total_reward += float(sum(rewards_list))
                total_reward_sq += float(sum(r * r for r in rewards_list))

        eval_elapsed = time.perf_counter() - eval_start

        self.snapshots.append(new_snap)

        # Build metrics.
        skill = self._conservative_skill(new_snap)
        gaussian_skill = self._gaussian_conservative_skill(new_snap)
        avg_reward = total_reward / total_games_played if total_games_played else 0.0
        reward_var = (
            max(0.0, total_reward_sq / total_games_played - avg_reward * avg_reward)
            if total_games_played
            else 0.0
        )
        reward_std = math.sqrt(reward_var)
        reward_se = reward_std / math.sqrt(total_games_played) if total_games_played else 0.0
        win_rate = new_snap.wins / total_games_played if total_games_played else 0.0
        loss_rate = new_snap.losses / total_games_played if total_games_played else 0.0
        draw_rate = new_snap.draws / total_games_played if total_games_played else 0.0
        metrics: Dict[str, float] = {
            "trueskill/mu": new_snap.mu,
            "trueskill/sigma": new_snap.sigma,
            "trueskill/skill": skill,
            "trueskill/win_rate": win_rate,
            "trueskill/loss_rate": loss_rate,
            "trueskill/draw_rate": draw_rate,
            "gaussian/mu": new_snap.gaussian_mu,
            "gaussian/sigma": new_snap.gaussian_sigma,
            "gaussian/skill": gaussian_skill,
            "gaussian/avg_reward": avg_reward,
            "gaussian/reward_std": reward_std,
            "gaussian/reward_se": reward_se,
            "trueskill/games_played": total_games_played,
            "trueskill/opponents_sampled": len(sampled_opponents),
            "trueskill/games_per_opponent": self.games_per_opponent,
            "trueskill/snapshots": len(self.snapshots),
            "trueskill/avg_reward": avg_reward,
        }
        metrics.update(self._mu_step_regression_metrics(self.snapshots))

        # Best (by conservative skill) across all snapshots so far.
        if self.snapshots:
            best = max(self.snapshots, key=self._conservative_skill)
            metrics["trueskill/best_step"] = best.step
            metrics["trueskill/best_skill"] = self._conservative_skill(best)
            best_gaussian = max(self.snapshots, key=self._gaussian_conservative_skill)
            metrics["gaussian/best_step"] = best_gaussian.step
            metrics["gaussian/best_skill"] = self._gaussian_conservative_skill(
                best_gaussian
            )

        if wandb_run is not None:
            try:
                wandb_run.log(metrics, step=step)
            except Exception:
                pass

        trend_suffix = ""
        if "trueskill/mu_step_correlation" in metrics:
            trend_suffix = (
                f" mu_step_corr={metrics['trueskill/mu_step_correlation']:.3f}"
                f" mu_step_r2={metrics['trueskill/mu_step_r2']:.3f}"
                f" gaussian_mu_step_corr={metrics['gaussian/mu_step_correlation']:.3f}"
                f" gaussian_mu_step_r2={metrics['gaussian/mu_step_r2']:.3f}"
            )
        print(
            f"[TrueSkill] step={step} mu={new_snap.mu:.2f} sigma={new_snap.sigma:.2f} "
            f"skill={skill:.2f} win_rate={win_rate:.3f} "
            f"avg_reward={avg_reward:.4f} reward_se={reward_se:.4f} "
            f"gaussian_mu={new_snap.gaussian_mu:.4f} "
            f"gaussian_sigma={new_snap.gaussian_sigma:.4f} "
            f"gaussian_skill={gaussian_skill:.4f} games={total_games_played} "
            f"opponents={len(sampled_opponents)} "
            f"snapshots={len(self.snapshots)}"
            f"{trend_suffix} "
            f"elapsed={eval_elapsed:.1f}s"
        )
        return metrics
