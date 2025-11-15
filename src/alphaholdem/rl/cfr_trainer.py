from __future__ import annotations

import math
import os
from dataclasses import asdict

import torch
import torch.nn as nn

from alphaholdem.core.structured_config import Config, ModelType
from alphaholdem.env.aggression_analyzer import AggressionAnalyzer
from alphaholdem.env.card_utils import NUM_HANDS, suit_permutations_tensor
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.mlp import RebelFFN
from alphaholdem.models.mlp.better_features import context_length
from alphaholdem.models.mlp.better_ffn import BetterFFN
from alphaholdem.models.model_output import ModelOutput
from alphaholdem.rl.losses import RebelSupervisedLoss
from alphaholdem.rl.pbs_pool import PBSPool
from alphaholdem.rl.rebel_batch import RebelBatch
from alphaholdem.rl.rebel_replay import RebelReplayBuffer
from alphaholdem.search.rebel_cfr_evaluator import T_WARM, RebelCFREvaluator
from alphaholdem.search.sparse_cfr_evaluator import SparseCFREvaluator
from alphaholdem.search.rebel_data_generator import RebelDataGenerator
from alphaholdem.utils.profiling import profile

STREETS = ["preflop", "flop", "turn", "river", "showdown"]


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
            num_envs=self.cfg.num_envs,
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
                num_policy_layers=cfg.model.num_policy_layers,
                num_value_layers=cfg.model.num_value_layers,
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
            1, round(self.K_value * (self.num_actions / 2) ** self.cfg.search.depth)
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
            permutation_weight=cfg.train.permutation_coef,
            num_players=self.num_players,
        )
        self.grad_clip = cfg.train.grad_clip

        if cfg.search.sparse:
            self.cfr_evaluator = SparseCFREvaluator(
                model=self.model,
                device=self.device,
                cfg=cfg,
            )
        else:
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
        value_output: ModelOutput,
        policy_output: ModelOutput,
        total_loss: float,
        value_loss: float,
        value_loss_all: torch.Tensor,
        value_weights: torch.Tensor,
        policy_loss: float,
        policy_loss_all: torch.Tensor,
        policy_weights: torch.Tensor,
        entropy_loss: float,
        permutation_loss: float,
        fresh_value_loss: float | None = None,
        fresh_value_batch: RebelBatch | None = None,
    ) -> dict[str, float]:
        grad_norm_clipped = torch.nn.utils.get_total_norm(
            p.grad for p in self.model.parameters() if p.grad is not None
        ).item()

        def by_street(
            tensor: torch.Tensor, batch=value_batch, street=None, weights=None
        ) -> dict[str, float]:
            if street is None:
                street = batch.features.street
            masks = {street_name: street == i for i, street_name in enumerate(STREETS)}
            if weights is not None:
                result = {
                    k: (tensor[mask] * weights[mask]).sum(dim=-1)
                    / weights[mask].sum(dim=-1).mean().item()
                    for k, mask in masks.items()
                }
            else:
                result = {k: tensor[mask].mean().item() for k, mask in masks.items()}
            return {k: v for k, v in result.items() if not math.isnan(v)}

        def street_count(street: torch.Tensor) -> dict[str, float]:
            return {
                street_name: (street == i).sum().item()
                for i, street_name in enumerate(STREETS)
            }

        value_buffer_streets_stats = street_count(
            self.value_buffer.features.street[: len(self.value_buffer)]
        )

        exploitability = value_batch.statistics["local_exploitability"]
        metrics = {
            "loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "permutation_loss": permutation_loss,
            "value_buffer": value_buffer_streets_stats,
            "value_buffer_size": len(self.value_buffer),
            "policy_buffer_size": len(self.policy_buffer),
            "value_buffer_mean_sample_count": (
                self.value_buffer.sample_count[: len(self.value_buffer)]
                .float()
                .mean()
                .item()
                if len(self.value_buffer) > 0
                else 0.0
            ),
            "value_buffer_target_mean_abs": (
                (
                    self.value_buffer.value_targets[: len(self.value_buffer)]
                    * self.value_buffer.features.beliefs[: len(self.value_buffer)].view(
                        -1, 2, NUM_HANDS
                    )
                )
            )
            .abs()
            .sum(dim=2)
            .mean()
            .item(),
            "value_buffer_target_mean_abs_street": by_street(
                (
                    self.value_buffer.value_targets[: len(self.value_buffer)]
                    * self.value_buffer.features.beliefs[: len(self.value_buffer)].view(
                        -1, 2, NUM_HANDS
                    )
                )
                .abs()
                .sum(dim=2)
                .mean(dim=1),
                street=self.value_buffer.features.street[: len(self.value_buffer)],
            ),
            "batch_value_target_mean_abs": value_batch.value_targets.abs()
            .mean()
            .item(),
            "batch_value_target_std": value_batch.value_targets.std().item(),
            "policy_buffer_mean_sample_count": (
                self.policy_buffer.sample_count[: len(self.policy_buffer)]
                .float()
                .mean()
                .item()
                if len(self.policy_buffer) > 0
                else 0.0
            ),
            "grad_norm_clipped": grad_norm_clipped,
            "local_exploitability": exploitability.mean().item(),
            "local_exploitability_street": by_street(
                exploitability,
                weights=torch.ones_like(exploitability),
            ),
            "aggression_stats": {
                f"chunk_{i}": v
                for i, v in enumerate(
                    self.aggression_analyzer.analyze_batch(policy_batch)[
                        "group_avg_bets"
                    ].tolist()
                )
            },
            "value_batch_street": street_count(value_batch.features.street),
            "value_loss_street": by_street(value_loss_all),
            "policy_loss_street": by_street(policy_loss_all, batch=policy_batch),
            "value_mean_std": value_output.value.std(dim=0).mean(),
            **self.cfr_evaluator.stats,
        }
        if fresh_value_loss is not None:
            metrics["fresh_value_loss"] = fresh_value_loss
        if fresh_value_batch is not None:
            metrics["fresh_value_batch_street"] = street_count(
                fresh_value_batch.features.street
            )
            metrics["fresh_value_target_mean_abs"] = (
                (
                    fresh_value_batch.value_targets
                    * fresh_value_batch.features.beliefs.view(-1, 2, NUM_HANDS)
                )
                .abs()
                .sum(dim=2)
                .mean()
                .item()
            )
            metrics["fresh_value_target_mean_abs_street"] = by_street(
                (
                    fresh_value_batch.value_targets
                    * fresh_value_batch.features.beliefs.view(-1, 2, NUM_HANDS)
                )
                .abs()
                .sum(dim=2)
                .mean(dim=1),
                batch=fresh_value_batch,
            )
        return metrics

    def _get_stratify_streets(self, step: int) -> list[float] | None:
        configs = self.cfg.train.stratify_streets
        if not configs:
            return None

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
        fresh_value_batch, fresh_policy_batch = self.data_generator.generate_data(
            self.K_value
        )
        # Warmup: make sure we have enough samples.
        while min(len(self.value_buffer), len(self.policy_buffer)) < self.batch_size:
            self.data_generator.generate_data(self.K_value)

        value_fullness = len(self.value_buffer) / self.value_buffer.capacity
        episodes = math.ceil(self.cfg.train.episodes_per_step * value_fullness)
        value_step_loss_all = torch.empty(
            self.batch_size * episodes, self.num_players, NUM_HANDS, device=self.device
        )
        policy_step_loss_all = torch.empty(
            self.batch_size * episodes, NUM_HANDS, self.num_actions, device=self.device
        )
        policy_step_loss, value_step_loss = 0.0, 0.0
        entropy_step_loss, permutation_step_loss = 0.0, 0.0
        total_step_loss = 0.0
        for episode in range(episodes):
            # TODO: think about how to interleave these/ratio in a smarter way.
            # Might need to use different sizes for the two batches.
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
            value_loss_episode, policy_loss_episode = None, None
            permutation_loss = 0.0

            value_output = self.model(value_batch.features)
            # Sample B suit permutations.
            suit_permutations_idxs = torch.randint(
                0, 24, (len(value_batch),), generator=self.rng, device=self.device
            )
            suit_permutations = suit_permutations_tensor(device=self.device)[
                suit_permutations_idxs
            ]
            permuted = value_batch.features.clone()
            permuted.permute_suits(suit_permutations)
            # Run model on permuted inputs [model(permute(features))]
            output_permuted = self.model(permuted)

            loss_dict = self.loss_fn(
                value_output,
                value_batch,
                output_permuted=output_permuted,
                suit_permutation_idxs=suit_permutations_idxs,
            )
            value_loss = loss_dict["value_loss"]
            value_loss_episode = loss_dict["value_loss_all"]
            value_step_loss_all[
                episode * self.batch_size : (episode + 1) * self.batch_size
            ] = value_loss_episode
            value_weights = loss_dict["value_weights"]
            permutation_loss = loss_dict["permutation_loss"]
            total_loss = loss_dict["total_loss"]

            policy_output = self.model(policy_batch.features)
            loss_dict = self.loss_fn(policy_output, policy_batch)
            policy_loss = loss_dict["policy_loss"]
            policy_loss_episode = loss_dict["policy_loss_all"]
            policy_step_loss_all[
                episode * self.batch_size : (episode + 1) * self.batch_size
            ] = policy_loss_episode
            policy_weights = loss_dict["policy_weights"]
            entropy_loss = loss_dict["entropy"]

            total_loss += loss_dict["total_loss"]
            total_loss.backward()

            total_step_loss += total_loss.item()
            policy_step_loss += policy_loss
            value_step_loss += value_loss
            entropy_step_loss += entropy_loss
            permutation_step_loss += permutation_loss

            assert all(
                p.grad.isfinite().all() for p in self.model.parameters()
            ), "NaN/Inf in model gradients"

            if self.grad_clip is not None and self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

        # After the loop, calculate loss on fresh data
        fresh_value_loss = None
        if fresh_value_batch:
            with torch.no_grad():
                self.model.eval()
                fresh_value_batch = fresh_value_batch.to(self.device)
                fresh_model_output = self.model(fresh_value_batch.features)
                fresh_loss_dict = self.loss_fn(fresh_model_output, fresh_value_batch)
                fresh_value_loss = fresh_loss_dict["value_loss"]

        metrics = self._compute_metrics(
            value_batch,
            policy_batch,
            value_output,
            policy_output,
            total_step_loss / episodes,
            value_step_loss / episodes,
            value_loss_episode,
            value_weights,
            policy_step_loss / episodes,
            policy_loss_episode,
            policy_weights,
            entropy_step_loss / episodes,
            permutation_step_loss / episodes,
            fresh_value_loss=fresh_value_loss,
            fresh_value_batch=fresh_value_batch,
        )
        metrics["episodes"] = episodes

        return metrics

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
            "config": asdict(self.cfg),
            # Store wandb run ID for resumption
            "wandb_run_id": wandb_run_id,
        }

        # Only save optimizer and RNG state if requested
        if save_optimizer:
            state["optimizer"] = self.optimizer.state_dict()
            state["rng"] = self.rng.get_state()

        torch.save(state, path)

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

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
