#!/usr/bin/env python3
"""Timing-only value-forward ablations for GPU-resident pregenerated batches."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Callable

import torch
import torch.nn as nn

from p2.env.card_utils import NUM_HANDS
from p2.models.model_output import ModelOutput
from p2.rl.cfr_trainer import RebelCFRTrainer
from run_value_arch_proposal import (
    PROPOSALS,
    _benchmark_no_grad_value_inference,
    _build_config,
    _dataset_dir,
    _load_gpu_value_epoch,
    _load_manifest,
    _resolve_value_batch_size,
)


DEFAULT_ROOT = Path("outputs/value_arch_proposals_500step_20260630")
DEFAULT_DATASET = Path(
    "outputs/rebel_postflop/river_value_500steps_512000_300it_20260630/manifest.json"
)


def _runner_args(args: argparse.Namespace, proposal: str, compile_mode: str):
    return SimpleNamespace(
        proposals=[proposal],
        dataset=args.dataset,
        output_root=args.fixed_root,
        steps=int(args.steps),
        value_batch_size=None,
        validation_interval=50,
        seed=int(args.seed),
        shuffle=not args.no_shuffle,
        compile_mode=compile_mode,
        timing_warmup_batches=int(args.timing_warmup_batches),
        timing_batches=int(args.timing_batches),
    )


def _zero_hand_values(module, features) -> torch.Tensor:
    return features.beliefs.new_zeros((len(features), module.num_players, NUM_HANDS))


def _as_output(hand_values: torch.Tensor) -> ModelOutput:
    return ModelOutput(value=hand_values.mean(dim=-1), hand_values=hand_values)


def _post_only_forward(module):
    def forward_value(
        self,
        features,
        latent=None,
        apply_zero_sum: bool = True,
        static_base_features=None,
        value_head: str = "auto",
    ) -> ModelOutput:
        del latent, value_head
        hand_values = self.forward_post(
            features,
            static_base_features=static_base_features,
            apply_zero_sum=apply_zero_sum,
        )
        return _as_output(hand_values)

    module.forward_value = MethodType(forward_value, module)


def _pre_only_forward(module):
    def forward_value(
        self,
        features,
        latent=None,
        apply_zero_sum: bool = True,
        static_base_features=None,
        value_head: str = "auto",
    ) -> ModelOutput:
        del latent, value_head
        hand_values = self.forward_pre(
            features,
            static_base_features=static_base_features,
            apply_zero_sum=apply_zero_sum,
        )
        return _as_output(hand_values)

    module.forward_value = MethodType(forward_value, module)


def _zero_output_forward(module):
    def forward_value(
        self,
        features,
        latent=None,
        apply_zero_sum: bool = True,
        static_base_features=None,
        value_head: str = "auto",
    ) -> ModelOutput:
        del latent, apply_zero_sum, static_base_features, value_head
        return _as_output(_zero_hand_values(self, features))

    module.forward_value = MethodType(forward_value, module)


def _base_only_forward(module):
    def forward_value(
        self,
        features,
        latent=None,
        apply_zero_sum: bool = True,
        static_base_features=None,
        value_head: str = "auto",
    ) -> ModelOutput:
        del latent, apply_zero_sum, value_head
        if static_base_features is None:
            _, _, x, _, _ = self._forward_base(features)
        else:
            _, _, x, _, _ = self._forward_base_from_static(
                features,
                static_base_features=static_base_features,
            )
        if x.dim() == 3:
            x = x[:, 0]
        hand_values = x[:, : self.num_players].unsqueeze(-1).expand(
            -1,
            self.num_players,
            NUM_HANDS,
        )
        return _as_output(hand_values)

    module.forward_value = MethodType(forward_value, module)


def _post_tower_only_forward(module):
    head = module.post_value_head
    if not isinstance(head, nn.Sequential) or len(head) < 2:
        raise TypeError("post_tower_only requires a sequential post value head")
    tower = nn.Sequential(*list(head.children())[:-1])
    module.post_value_tower_only = tower

    def forward_value(
        self,
        features,
        latent=None,
        apply_zero_sum: bool = True,
        static_base_features=None,
        value_head: str = "auto",
    ) -> ModelOutput:
        del latent, apply_zero_sum, value_head
        if static_base_features is None:
            _, _, x, _, _ = self._forward_base(features)
        else:
            _, _, x, _, _ = self._forward_base_from_static(
                features,
                static_base_features=static_base_features,
            )
        if x.dim() == 3:
            x = x[:, 0]
        tower_out = self.post_value_tower_only(x)
        hand_values = tower_out[:, : self.num_players].unsqueeze(-1).expand(
            -1,
            self.num_players,
            NUM_HANDS,
        )
        return _as_output(hand_values)

    module.forward_value = MethodType(forward_value, module)


def _post_final_only(module):
    head = module.post_value_head
    if not isinstance(head, nn.Sequential) or len(head) < 1:
        raise TypeError("post_final_only requires a sequential post value head")
    module.post_value_head = list(head.children())[-1]
    _post_only_forward(module)


def _post_head_zero_base_forward(module):
    def forward_value(
        self,
        features,
        latent=None,
        apply_zero_sum: bool = True,
        static_base_features=None,
        value_head: str = "auto",
    ) -> ModelOutput:
        del latent, static_base_features, value_head
        x = features.beliefs.new_zeros((len(features), self.hidden_dim))
        player_beliefs = features.beliefs.view(-1, self.num_players, NUM_HANDS)
        hand_emb = self._hand_embedding(None)
        hand_values = self._value_tensor_from_base(
            player_beliefs,
            x,
            hand_emb,
            self.post_value_head,
            features,
            apply_zero_sum=apply_zero_sum,
        )
        return _as_output(hand_values)

    module.forward_value = MethodType(forward_value, module)


def _post_final_zero_base(module):
    _post_final_only(module)
    _post_head_zero_base_forward(module)


def _post_no_belief_moments(module):
    def belief_moments(self, player_beliefs, hand_emb, board_context=None):
        del hand_emb, board_context
        shape = (player_beliefs.shape[0], self.num_players, self.belief_feature_dim)
        zeros = player_beliefs.new_zeros(shape)
        return zeros, None

    module._belief_moments = MethodType(belief_moments, module)
    _post_only_forward(module)


def _post_no_static_context(module):
    def static_feature_base(self, features):
        return features.beliefs.new_zeros((len(features), self.hidden_dim))

    module.static_feature_base = MethodType(static_feature_base, module)
    _post_only_forward(module)


ABLATIONS: dict[str, Callable[[Any], None]] = {
    "full_auto": lambda module: None,
    "post_only": _post_only_forward,
    "pre_only": _pre_only_forward,
    "base_only": _base_only_forward,
    "zero_output": _zero_output_forward,
    "post_tower_only": _post_tower_only_forward,
    "post_final_only": _post_final_only,
    "post_head_zero_base": _post_head_zero_base_forward,
    "post_final_zero_base": _post_final_zero_base,
    "post_no_belief_moments": _post_no_belief_moments,
    "post_no_static_context": _post_no_static_context,
}


def _compile_after_patch(trainer: RebelCFRTrainer, compile_mode: str) -> None:
    trainer.model.compile_forward_modes(
        dynamic=True,
        mode=compile_mode,
        policy_compile=False,
    )


def _summarize_timing(timing: dict[str, Any]) -> dict[str, Any]:
    runtimes = [float(v) for v in timing["runtime_s"]]
    return {
        "mean_s": float(statistics.fmean(runtimes)),
        "median_s": float(statistics.median(runtimes)),
        "min_s": float(min(runtimes)),
        "max_s": float(max(runtimes)),
        "count": len(runtimes),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--proposal", default="flops_hidden512_ffn1024_value6")
    parser.add_argument("--checkpoint-name", default="rebel_final.pt")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--timing-warmup-batches", type=int, default=3)
    parser.add_argument("--timing-batches", type=int, default=20)
    parser.add_argument(
        "ablations",
        nargs="*",
        default=None,
        choices=sorted(ABLATIONS),
    )
    args = parser.parse_args()
    if not args.ablations:
        args.ablations = list(ABLATIONS)
    return args


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Value Forward Ablation Timing",
        "",
        f"Proposal: `{payload['proposal']}`",
        f"Compile mode: `{payload['compile_mode']}`",
        f"Timing: {payload['timing_warmup_batches']} warmup and "
        f"{payload['timing_batches']} timed 4096-row batches.",
        "",
        "| Ablation | Mean 4096 forward | Median | Min | Max | Notes |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    notes = {
        "full_auto": "Original compiled auto value path.",
        "post_only": "Force post value head; avoids compiled pre+post auto dispatch.",
        "pre_only": "Force pre value head.",
        "base_only": "Run feature/range base, skip value head.",
        "zero_output": "Allocate correctly-shaped zero output only.",
        "post_tower_only": "Run base plus post residual tower, skip final H->2N projection.",
        "post_final_only": "Run base plus final H->2N projection only.",
        "post_head_zero_base": "Run full post head from zero trunk state, skip base/range path.",
        "post_final_zero_base": "Run final projection from zero trunk state only.",
        "post_no_belief_moments": "Force belief moments to zero, keeping the rest.",
        "post_no_static_context": "Force context/board static base to zero, keeping the rest.",
    }
    for result in payload["results"]:
        summary = result["summary"]
        lines.append(
            "| "
            f"`{result['ablation']}` | "
            f"{summary['mean_s'] * 1000:.3f}ms | "
            f"{summary['median_s'] * 1000:.3f}ms | "
            f"{summary['min_s'] * 1000:.3f}ms | "
            f"{summary['max_s'] * 1000:.3f}ms | "
            f"{notes.get(result['ablation'], '')} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.proposal not in PROPOSALS:
        raise ValueError(f"unknown proposal {args.proposal!r}")
    run_dir = args.fixed_root / args.proposal
    checkpoint = run_dir / "checkpoints" / args.checkpoint_name
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    dataset_dir = _dataset_dir(args.dataset)
    manifest = _load_manifest(dataset_dir)
    bootstrap_args = _runner_args(args, args.proposal, "off")
    value_batch_size = _resolve_value_batch_size(bootstrap_args, manifest)
    base_cfg, _, _, _ = _build_config(
        bootstrap_args,
        proposal=args.proposal,
        manifest=manifest,
        value_batch_size=value_batch_size,
    )
    device = torch.device(base_cfg.device)
    loaded_at = time.time()
    gpu_epoch = _load_gpu_value_epoch(
        dataset_dir=dataset_dir,
        manifest=manifest,
        device=device,
        batch_size=value_batch_size,
        steps=int(args.steps),
        shuffle_seed=int(args.seed),
        shuffle=not args.no_shuffle,
    )

    results: list[dict[str, Any]] = []
    for ablation in args.ablations:
        cfg_args = _runner_args(args, args.proposal, "off")
        cfg, _, _, _ = _build_config(
            cfg_args,
            proposal=args.proposal,
            manifest=manifest,
            value_batch_size=value_batch_size,
        )
        trainer = RebelCFRTrainer(cfg, torch.device(cfg.device))
        checkpoint_step = trainer.load_checkpoint(str(checkpoint), load_optimizer=False)
        value_model = getattr(trainer.model, "value_model", trainer.model)
        ABLATIONS[ablation](value_model)
        _compile_after_patch(trainer, str(args.compile_mode))
        timing = _benchmark_no_grad_value_inference(
            trainer=trainer,
            gpu_epoch=gpu_epoch,
            batch_size=4096,
            warmup_batches=int(args.timing_warmup_batches),
            timed_batches=int(args.timing_batches),
            value_head="auto" if ablation == "full_auto" else "post",
        )
        result = {
            "ablation": ablation,
            "checkpoint_step": int(checkpoint_step),
            "timing": timing,
            "summary": _summarize_timing(timing),
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "proposal": args.proposal,
        "checkpoint": str(checkpoint),
        "dataset": str(dataset_dir),
        "compile_mode": str(args.compile_mode),
        "timing_warmup_batches": int(args.timing_warmup_batches),
        "timing_batches": int(args.timing_batches),
        "shuffle": not args.no_shuffle,
        "shuffle_seed": int(args.seed),
        "gpu_epoch": {
            "examples": gpu_epoch.examples,
            "batch_size": gpu_epoch.batch_size,
            "tensor_bytes": gpu_epoch.tensor_bytes,
            "tensor_gib": gpu_epoch.tensor_bytes / (1024**3),
            "load_time_s": gpu_epoch.load_time_s,
        },
        "elapsed_s": time.time() - loaded_at,
        "results": results,
    }
    json_out = args.json_out or (
        args.fixed_root / f"{args.proposal}_value_forward_ablations.json"
    )
    md_out = args.md_out or (
        args.fixed_root / f"{args.proposal}_value_forward_ablations.md"
    )
    json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_markdown(md_out, payload)
    print(json.dumps({"json_out": str(json_out), "md_out": str(md_out)}))


if __name__ == "__main__":
    main()
