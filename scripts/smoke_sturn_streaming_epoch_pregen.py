#!/usr/bin/env python3
"""Exercise streaming-epoch value replay using pregenerated S-turn examples."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from p2.config.rebel_load import load_rebel_config_file
from p2.core.structured_config import PregeneratedDatasetConfig
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.rl.rebel_replay import StreamingEpochValueBuffer
from p2.rl.validation_set import RebelValueValidationSetEvaluator
from p2.stages.curriculum import _initialize_value_from_checkpoint
from run_value_arch_proposal import _dataset_dir, _load_gpu_value_epoch, _load_manifest


DEFAULT_DATASET = Path(
    "outputs/rebel_postflop/sturn_value_500steps_1024000_300it_eturn300k_20260711"
)
DEFAULT_VALIDATION = Path(
    "outputs/rebel_postflop/turn_holdout_32768_300it_eturn300k_seed777_20260711"
)
DEFAULT_INITIALIZATION = Path(
    "checkpoints-eturn-distill-lr-current-teb-300k-lr0p01-wandb-20260708/"
    "t001_lr0p01_300000st_b1024/promoted/E_turn.pt"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--initialization", type=Path, default=DEFAULT_INITIALIZATION)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/sturn_streaming_epoch_pregen_smoke_20260712.json"),
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--block-batches", type=int, default=20)
    parser.add_argument("--episodes-per-step", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset = _dataset_dir(args.dataset)
    manifest = _load_manifest(dataset)
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    block_batches = int(args.block_batches)
    episodes = int(args.episodes_per_step)
    total_updates = epochs * block_batches
    if total_updates % episodes != 0:
        raise ValueError("epochs * block_batches must divide by episodes-per-step")
    outer_steps = total_updates // episodes
    block_examples = batch_size * block_batches

    cfg = load_rebel_config_file("conf/config_rebel_curriculum_turn.yaml")
    cfg.num_steps = total_updates
    cfg.seed = int(args.seed)
    cfg.use_wandb = False
    cfg.model.compile = "reduce-overhead"
    cfg.train.batch_size = batch_size
    cfg.train.episodes_per_step = episodes
    cfg.train.learning_rate = 0.004
    cfg.train.learning_rate_final = 0.0004
    cfg.train.adamw_learning_rate = 0.0004
    cfg.train.replay_buffer_batches = 1
    cfg.train.policy_capacity_factor = 1
    cfg.train.replay_buffer_device = "cpu"
    cfg.train.save_replay_buffers = False
    cfg.curriculum.stages = []
    cfg.curriculum.substeps = {}
    cfg.data.mode = "pregenerated"
    cfg.data.pregenerated.value_batch_size = batch_size
    cfg.data.pregenerated.policy_batch_size = 0
    cfg.data.pregenerated.datasets = [
        PregeneratedDatasetConfig(path=str(dataset), policy_weight=0.0)
    ]
    cfg.validation_set.enabled = False

    device = torch.device(cfg.device)
    gpu_epoch = _load_gpu_value_epoch(
        dataset_dir=dataset,
        manifest=manifest,
        device=device,
        batch_size=batch_size,
        steps=2 * block_batches,
        shuffle_seed=int(args.seed),
        shuffle=True,
    )
    trainer = RebelCFRTrainer(cfg, device)
    _initialize_value_from_checkpoint(
        trainer,
        str(args.initialization),
        substep_name="sturn_streaming_epoch_pregen_smoke",
    )
    buffer = StreamingEpochValueBuffer(
        block_capacity=block_examples,
        epochs=epochs,
        num_actions=trainer.num_actions,
        num_players=trainer.num_players,
        num_context_features=gpu_epoch.batch.features.context.shape[1],
        device=device,
        hand_dim=gpu_epoch.batch.features.hand_dim,
        generator=torch.Generator(device=device).manual_seed(int(args.seed)),
    )

    started = time.time()
    buffer.add_batch(gpu_epoch.slice_batch(0, block_examples))
    if len(buffer) != block_examples:
        raise RuntimeError("first block did not seal before training")

    source_cursor = block_examples
    losses: list[float] = []
    update = 0
    generation_numerator = batch_size * episodes
    for outer in range(outer_steps):
        fresh_count = ((outer + 1) * generation_numerator) // epochs - (
            outer * generation_numerator
        ) // epochs
        buffer.add_batch(gpu_epoch.slice_batch(source_cursor, fresh_count))
        source_cursor += fresh_count
        for _ in range(episodes):
            trainer._apply_schedules(update)
            stats = trainer.train_value_batch(
                buffer.sample(batch_size),
                update,
                sync_inference_model=False,
            )
            losses.append(float(stats["value_loss"]))
            update += 1
    trainer._sync_inference_model()

    first_block_counts = buffer.sample_count[:block_examples]
    if not torch.all(first_block_counts == epochs):
        raise RuntimeError("first block did not receive exact epoch coverage")
    if buffer.read_block != 1 or buffer.write_size != 0 or buffer.pending_batches:
        raise RuntimeError("streaming epoch blocks did not swap cleanly")
    if source_cursor != 2 * block_examples:
        raise RuntimeError("replacement block generation count drifted")

    validation = RebelValueValidationSetEvaluator(
        trainer=trainer,
        cfg=cfg,
        dataset_path=str(args.validation),
        batch_size=1024,
    ).evaluate()
    result = {
        "batch_size": batch_size,
        "epochs": epochs,
        "block_batches": block_batches,
        "block_examples": block_examples,
        "outer_steps": outer_steps,
        "optimizer_updates": update,
        "generated_examples": source_cursor,
        "first_value_loss": losses[0],
        "final_value_loss": losses[-1],
        "mean_value_loss": sum(losses) / len(losses),
        "elapsed_s": time.time() - started,
        "buffer_metrics": buffer.epoch_metrics(),
        "validation": validation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
