"""Save an untrained (at-initialization) model snapshot for use as an eval control.

The evaluation ladder needs a rung whose strength comes from search alone. A
freshly initialized net supplies exactly that: the CFR machinery is identical to
every other rung, but the leaf values carry no training signal. Anything that
scores at or below this rung is not demonstrating learned strength.

The snapshot is written in the same format ``p2.eval.checkpoints`` reads, so it
loads through the ordinary code path with no special-casing, and it is seeded so
the control is reproducible rather than a one-off random draw.

Usage:
    uv run python scripts/make_init_snapshot.py \
        --resolved-config eval_anchors/v3_resolved_config.json \
        --out eval_anchors/v3_init_seed0.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from p2.eval.checkpoints import SearchFidelity, load_eval_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved-config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    if args.out.exists():
        raise SystemExit(f"{args.out} already exists; refusing to overwrite a control")

    cfg = load_eval_config(
        checkpoint="<init>",
        resolved_config=args.resolved_config,
        device=args.device,
        fidelity=SearchFidelity(),
        num_envs=8,  # nothing plays here; keep the build cheap
    )
    cfg.seed = int(args.seed)

    from p2.rl.cfr_trainer import RebelCFRTrainer
    from p2.runtime.training_run import device_from_config, setup_torch_runtime

    torch_device = device_from_config(cfg)
    setup_torch_runtime(cfg, torch_device)
    torch.manual_seed(int(args.seed))
    trainer = RebelCFRTrainer(cfg=cfg, device=torch_device, pregeneration_only=True)

    state = {key: value.cpu() for key, value in trainer.model.state_dict().items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": state,
            "step": 0,
            "save_dtype": str(trainer.float_dtype),
            "init_seed": int(args.seed),
        },
        str(args.out),
    )
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB), seed={args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
