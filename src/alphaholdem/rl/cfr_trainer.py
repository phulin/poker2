from __future__ import annotations

import math
import os

import torch
import torch.nn as nn

from alphaholdem.core.structured_config import Config, ModelType
from alphaholdem.env.aggression_analyzer import AggressionAnalyzer
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.mlp import RebelFFN
from alphaholdem.models.mlp.better_features import context_length
from alphaholdem.models.mlp.better_ffn import BetterFFN
from alphaholdem.rl.losses import RebelSupervisedLoss
from alphaholdem.rl.pbs_pool import PBSPool
from alphaholdem.rl.rebel_replay import RebelBatch, RebelReplayBuffer
from alphaholdem.search.rebel_cfr_evaluator import T_WARM, RebelCFREvaluator
from alphaholdem.search.rebel_data_generator import RebelDataGenerator
from alphaholdem.utils.profiling import profile


class RebelCFRTrainer:
    """Trainer that couples DCFR search with a ReBeL-style FFN."""

    def __init__(self, cfg: Config, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.rng = torch.Generator(device=self.device)
        self.float_dtype = torch.float32
        self.search_cfg = cfg.search
        self.bet_bins = cfg.env.bet_bins
        self.num_bet_bins = len(self.bet_bins) + 3
        self.batch_size = cfg.train.batch_size
        self.buffer_device = torch.device("cpu")
        self.buffer_rng = torch.Generator(device=self.buffer_device)
        if cfg.seed is not None:
            self.rng.manual_seed(int(cfg.seed))
            self.buffer_rng.manual_seed(int(cfg.seed))
        self.num_actions = len(self.bet_bins) + 3
        self.num_players = 2

        if cfg.model.num_actions != self.num_actions:
            print(
                f"[RebelCFRTrainer] Overriding model.num_actions "
                f"({cfg.model.num_actions}) -> {self.num_actions} "
                f"to match bet bin configuration."
            )
            cfg.model.num_actions = self.num_actions

        # Environment used to provide root states for CFR search
        self.env = HUNLTensorEnv(
            num_envs=self.batch_size,
            starting_stack=cfg.env.stack,
            sb=cfg.env.sb,
            bb=cfg.env.bb,
            default_bet_bins=self.bet_bins,
            device=self.device,
            float_dtype=self.float_dtype,
            flop_showdown=cfg.env.flop_showdown,
        )
        self.env.reset()

        # Model
        if cfg.model.name == ModelType.better_ffn:
            self.model = BetterFFN(
                num_actions=self.num_actions,
                hidden_dim=cfg.model.hidden_dim,
                range_hidden_dim=cfg.model.range_hidden_dim,
                ffn_dim=cfg.model.ffn_dim,
                num_hidden_layers=cfg.model.num_hidden_layers,
                num_players=self.num_players,
            )
            num_context_features = context_length(self.num_players)
        else:
            self.model = RebelFFN(
                input_dim=cfg.model.input_dim,
                num_actions=self.num_actions,
                hidden_dim=cfg.model.hidden_dim,
                num_hidden_layers=cfg.model.num_hidden_layers,
                detach_value_head=cfg.model.detach_value_head,
                num_players=self.num_players,
            )
            num_context_features = 4

        cpu_rng = torch.Generator(device="cpu")
        if self.cfg.seed is not None:
            cpu_rng.manual_seed(self.cfg.seed)
        self.model.init_weights(cpu_rng)
        self.model.to(self.device)
        if self.device.type == "cuda" and cfg.model.compile:
            self.model.compile()

        # data generation rate per training step
        self.K_value = max(1, self.batch_size // self.cfg.train.value_reuse_goal)
        # approximate number of policy samples when collecting K_value value samples
        self.K_policy = max(
            1, self.K_value * (self.num_actions // 2) ** self.cfg.search.depth
        )

        C_over_K = self.cfg.train.replay_buffer_batches
        value_capacity = C_over_K * self.K_value
        policy_capacity = C_over_K * self.K_policy

        # Replay buffers
        self.value_buffer = RebelReplayBuffer(
            capacity=value_capacity,
            num_actions=self.num_actions,
            num_players=self.num_players,
            num_context_features=num_context_features,
            device=self.buffer_device,
            policy_targets=False,
        )
        # Larger policy buffer since we store more samples there
        self.policy_buffer = RebelReplayBuffer(
            capacity=policy_capacity,
            num_actions=self.num_actions,
            num_players=self.num_players,
            num_context_features=num_context_features,
            device=self.buffer_device,
            policy_targets=True,
            value_targets=False,
        )

        # Optimizer & loss
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
        self.loss_fn = RebelSupervisedLoss(
            policy_weight=1.0,
            value_weight=cfg.train.value_coef,
            entropy_coef=cfg.train.entropy_coef,
        )
        self.grad_clip = cfg.train.grad_clip

        self.cfr_evaluator = RebelCFREvaluator(
            search_batch_size=self.cfg.num_envs,
            env_proto=self.env,
            model=self.model,
            bet_bins=self.bet_bins,
            max_depth=max(1, self.cfg.search.depth),
            cfr_iterations=max(T_WARM + 1, self.cfg.search.iterations),
            device=self.device,
            float_dtype=self.float_dtype,
            generator=self.rng,
            warm_start_iterations=self.cfg.search.warm_start_iterations,
            cfr_type=self.cfg.search.cfr_type,
            cfr_avg=self.cfg.search.cfr_avg,
            dcfr_alpha=self.cfg.search.dcfr_alpha,
            dcfr_beta=self.cfg.search.dcfr_beta,
            dcfr_gamma=self.cfg.search.dcfr_gamma,
            dcfr_delay=self.cfg.search.dcfr_plus_delay,
        )
        self.data_generator = RebelDataGenerator(
            env_proto=self.env,
            evaluator=self.cfr_evaluator,
            value_buffer=self.value_buffer,
            policy_buffer=self.policy_buffer,
        )

        self.aggression_analyzer = AggressionAnalyzer(device=self.device)
        self.pbs_pool = PBSPool(pool_size=3, generator=self.rng)

    def _compute_entropy(self, probs: torch.Tensor) -> float:
        eps = 1e-8
        norm = probs.clamp_min(eps)
        entropy = -(norm * norm.log()).sum(dim=-1).mean()
        return float(entropy.item())

    def _compute_metrics(
        self,
        value_batch: RebelBatch,
        policy_batch: RebelBatch,
        value_loss: float,
        value_loss_all: torch.Tensor,
        policy_loss: float,
        policy_loss_all: torch.Tensor | None,
        entropy_loss: float,
        permutation_loss: float,
    ) -> dict[str, float]:
        grad_norm_clipped = torch.nn.utils.get_total_norm(
            p.grad for p in self.model.parameters()
        ).item()

        value_preflop = value_batch.features.street == 0
        value_flop = value_batch.features.street == 1
        value_turn = value_batch.features.street == 2
        value_river = value_batch.features.street == 3
        value_showdown = value_batch.features.street == 4

        def by_street(tensor: torch.Tensor, batch=value_batch) -> dict[str, float]:
            preflop = batch.features.street == 0
            flop = batch.features.street == 1
            turn = batch.features.street == 2
            river = batch.features.street == 3
            showdown = batch.features.street == 4

            result = {
                "preflop": tensor[preflop].mean().item(),
                "flop": tensor[flop].mean().item(),
                "turn": tensor[turn].mean().item(),
                "river": tensor[river].mean().item(),
                "showdown": tensor[showdown].mean().item(),
            }
            return {k: v for k, v in result.items() if not math.isnan(v)}

        return {
            "loss": self.cfg.train.value_coef * value_loss + policy_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "permutation_loss": permutation_loss,
            "value_buffer_size": len(self.value_buffer),
            "policy_buffer_size": len(self.policy_buffer),
            "grad_norm_clipped": grad_norm_clipped,
            "local_exploitability": value_batch.statistics["local_exploitability"]
            .mean()
            .item(),
            "local_exploitability_street": by_street(
                value_batch.statistics["local_exploitability"]
            ),
            "aggression_stats": {
                f"chunk_{i}": v
                for i, v in enumerate(
                    self.aggression_analyzer.analyze_batch(policy_batch)[
                        "group_avg_bets"
                    ].tolist()
                )
            },
            "value_batch_street": {
                "preflop": value_preflop.float().mean().item(),
                "flop": value_flop.float().mean().item(),
                "turn": value_turn.float().mean().item(),
                "river": value_river.float().mean().item(),
                "showdown": value_showdown.float().mean().item(),
            },
            "value_loss_street": by_street(value_loss_all),
            "policy_loss_street": by_street(policy_loss_all, policy_batch),
            **self.cfr_evaluator.stats,
        }

    def _get_stratify_streets(self, step: int) -> list[float] | None:
        configs = self.cfg.train.stratify_streets

        # Flat until the first threshold
        if step < configs[0].threshold:
            return configs[0].probabilities

        # Find the two thresholds that bracket the current step
        for i in range(len(configs) - 1):
            if configs[i].threshold <= step < configs[i + 1].threshold:
                # Linear interpolation between thresholds
                lower_threshold = configs[i].threshold
                upper_threshold = configs[i + 1].threshold
                lower_probs = configs[i].probabilities
                upper_probs = configs[i + 1].probabilities

                # Compute interpolation weight (0 at lower, 1 at upper)
                alpha = (step - lower_threshold) / (upper_threshold - lower_threshold)

                # Linearly interpolate each probability
                interpolated = [
                    lower * (1 - alpha) + upper * alpha
                    for lower, upper in zip(lower_probs, upper_probs)
                ]
                return interpolated

        # Step >= last threshold, return last config's probabilities
        return configs[-1].probabilities

    @profile
    def _update_model(self, step: int) -> dict[str, float]:
        self.data_generator.generate_data(self.K_value)

        # TODO: think about how to interleave these/ratio in a smarter way.
        # Might need to use different sizes for the two buffers.
        value_batch = self.value_buffer.sample(
            self.batch_size,
            stratify_streets=self._get_stratify_streets(step),
            generator=self.buffer_rng,
        ).to(self.device)
        policy_batch = self.policy_buffer.sample(
            self.batch_size,
            stratify_streets=self._get_stratify_streets(step),
            generator=self.buffer_rng,
        ).to(self.device)

        self.model.train()
        self.optimizer.zero_grad()

        value_loss, policy_loss, entropy_loss = None, None, None
        value_loss_all, policy_loss_all = None, None
        permutation_loss = 0.0
        for batch in [value_batch, policy_batch]:
            output = self.model(
                batch.features,
                permuted=batch.features.clone().permute_suits(generator=self.rng),
            )
            loss_dict = self.loss_fn(output, batch)
            loss = loss_dict["total_loss"]
            permutation_loss += loss_dict["permutation_loss"]
            if batch is value_batch:
                value_loss = loss_dict["value_loss"]
                value_loss_all = loss_dict["value_loss_all"]
            else:
                policy_loss = loss_dict["policy_loss"]
                policy_loss_all = loss_dict["policy_loss_all"]
                entropy_loss = loss_dict["entropy"]
            loss.backward()

        assert all(
            p.grad.isfinite().all() for p in self.model.parameters()
        ), "NaN/Inf in model gradients"

        if self.grad_clip is not None and self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        return self._compute_metrics(
            value_batch,
            policy_batch,
            value_loss,
            value_loss_all,
            policy_loss,
            policy_loss_all,
            entropy_loss,
            permutation_loss,
        )

    def train_step(self, step: int) -> dict[str, any]:
        step_public = step + 1

        update_info = self._update_model(step)
        update_info["step"] = step_public

        return update_info

    def train(self, num_steps: int | None = None) -> list[dict[str, any]]:
        total_steps = num_steps or self.cfg.num_steps
        history: list[dict[str, any]] = []

        for step in range(total_steps):
            update_info = self.train_step(step)
            history.append(update_info)

        return history

    def evaluate_against_pool(self, min_games: int) -> dict[str, float]:
        return self.pbs_pool.evaluate_model_against_pool(self.model, min_games)

    def save_checkpoint(
        self,
        path: str,
        step: int,
        wandb_run_id: str | None = None,
        save_optimizer: bool = True,
        save_dtype: torch.dtype | None = None,
    ) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # Convert model state to bfloat16 if requested
        model_state = self.model.state_dict()
        if save_dtype is not None:
            model_state = {
                k: v.to(save_dtype) if v.dtype.is_floating_point else v
                for k, v in model_state.items()
            }

        state = {
            "model": model_state,
            "step": step,
            "save_dtype": str(save_dtype) if save_dtype is not None else None,
            # Store wandb run ID for resumption
            "wandb_run_id": wandb_run_id,
        }

        # Only save optimizer and RNG state if requested
        if save_optimizer:
            state["optimizer"] = self.optimizer.state_dict()
            state["rng"] = self.rng.get_state()

        torch.save(state, path)

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)

        # Convert model state back to host dtype if it was saved in bfloat16
        save_dtype_str = ckpt.get("save_dtype")
        model_state = ckpt["model"]
        if save_dtype_str is not None and save_dtype_str != str(self.float_dtype):
            # Convert back to float32 for host dtype
            model_state = {
                k: v.to(self.float_dtype) if v.dtype.is_floating_point else v
                for k, v in model_state.items()
            }

        self.model.load_state_dict(model_state)

        # Only load optimizer if it exists in checkpoint
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])

        self.cfg.wandb_run_id = ckpt.get("wandb_run_id")
        # if "rng" in ckpt:
        #     self.rng.set_state(ckpt["rng"].to(self.device))
        return int(ckpt["step"])
