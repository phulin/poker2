from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from p2.allin.sampler import (
    DEFAULT_CANONICAL_BOARD_RANKS,
    NUM_CANONICAL_FULL_BOARDS,
    NUM_FULL_BOARDS,
)
from p2.allin.training_data import AllInDataGenConfig, save_allin_training_dataset
from p2.env.card_utils import NUM_HANDS


@dataclass
class PregenerateConfig:
    output_dir: str = "outputs/allin_training_data"
    examples: int = 64_000
    shard_size: int = 1024
    generation_batch_size: int = 64
    seed: int = 0
    device: str = "cuda"
    players: int = 4
    min_allin_players: int = 2
    range_hand_dim: int = NUM_HANDS
    sample_count: int = 50_000
    board_samples: int = 256
    tuple_samples: int = 0
    all_boards: bool = False
    samples_per_board: int = 0
    tuple_tries: int = 4
    board_chunk: int = 8
    hand_chunk: int = 128
    bb: int = 100
    min_stack_bb: int = 10
    mid_stack_bb: int = 200
    max_stack_bb: int = 400
    high_stack_mass_ratio: float = 1.0 / 3.0
    concentration: float = 1.0
    folded_commit_max_frac: float = 0.35
    preflop_table_path: str = ""
    board_ranks_path: str = str(DEFAULT_CANONICAL_BOARD_RANKS)
    no_exact_two_player: bool = False


def _device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _data_config(cfg: PregenerateConfig) -> AllInDataGenConfig:
    sample_count: int | None = cfg.sample_count
    board_samples = cfg.board_samples
    tuple_samples = cfg.tuple_samples
    if cfg.all_boards:
        tuple_samples = (
            cfg.samples_per_board if cfg.samples_per_board > 0 else cfg.tuple_samples
        )
        if tuple_samples <= 0:
            raise ValueError(
                "--all-boards requires --samples-per-board or a positive "
                "--tuple-samples value"
            )
        sample_count = None
        board_samples = (
            NUM_CANONICAL_FULL_BOARDS if cfg.board_ranks_path else NUM_FULL_BOARDS
        )

    kwargs: dict[str, Any] = {
        "players": cfg.players,
        "min_allin_players": cfg.min_allin_players,
        "range_hand_dim": cfg.range_hand_dim,
        "sample_count": sample_count,
        "board_samples": board_samples,
        "tuple_samples": tuple_samples,
        "all_boards": cfg.all_boards,
        "tuple_tries": cfg.tuple_tries,
        "board_chunk": cfg.board_chunk,
        "hand_chunk": cfg.hand_chunk,
        "bb": cfg.bb,
        "min_stack_bb": cfg.min_stack_bb,
        "mid_stack_bb": cfg.mid_stack_bb,
        "max_stack_bb": cfg.max_stack_bb,
        "high_stack_mass_ratio": cfg.high_stack_mass_ratio,
        "concentration": cfg.concentration,
        "folded_commit_max_frac": cfg.folded_commit_max_frac,
        "board_ranks_path": cfg.board_ranks_path,
        "use_exact_two_player": not cfg.no_exact_two_player,
    }
    if cfg.preflop_table_path:
        kwargs["preflop_table_path"] = cfg.preflop_table_path
    return AllInDataGenConfig(**kwargs)


def parse_args() -> PregenerateConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    for field_name, field_def in PregenerateConfig.__dataclass_fields__.items():
        default = field_def.default
        arg = "--" + field_name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(arg, action="store_true", default=default)
        else:
            parser.add_argument(arg, type=type(default), default=default)
    ns = parser.parse_args()
    return PregenerateConfig(**vars(ns))


def main() -> None:
    cfg = parse_args()
    device = _device(cfg.device)
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    data_cfg = _data_config(cfg)
    output_dir = Path(cfg.output_dir)

    def progress(info: dict[str, Any]) -> None:
        diag = info["last_target_diag"]
        target_seconds = diag.get("target_seconds", 0.0)
        print(
            f"[{info['examples_done']:>8}/{info['examples_total']}] "
            f"wrote={info['shard']['file']} "
            f"shard={info['shard_seconds']:.2f}s "
            f"target={target_seconds:.2f}s "
            f"total={info['total_seconds'] / 60.0:.2f}m",
            flush=True,
        )

    manifest = save_allin_training_dataset(
        output_dir,
        examples=cfg.examples,
        shard_size=cfg.shard_size,
        generation_batch_size=cfg.generation_batch_size,
        cfg=data_cfg,
        device=device,
        generator=generator,
        progress=progress,
    )
    print(
        f"wrote {manifest['examples']} examples to {output_dir / 'manifest.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
