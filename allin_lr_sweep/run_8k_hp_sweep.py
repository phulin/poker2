from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = "outputs/allin_training_data_512k_s65536_b4096_bc64/manifest.json"
VAL = "outputs/allin_validation_data_4096_s16777216_b2097152_bc64/manifest.json"
LOG_DIR = ROOT / "allin_lr_sweep" / "logs_8k_hp"
CKPT_ROOT = ROOT / "allin_lr_sweep" / "checkpoints_8k_hp"
STEPS = 8000


CANDIDATES = [
    {
        "name": "mlp_cos4000_r015",
        "model_type": "mlp",
        "layers": 6,
        "lr_decay": "cosine",
        "cosine_lr_decay_steps": 4000,
    },
    {
        "name": "mlp_cos8000_r015",
        "model_type": "mlp",
        "layers": 6,
        "lr_decay": "cosine",
        "cosine_lr_decay_steps": 8000,
    },
    {
        "name": "mlp_linear8000_r015",
        "model_type": "mlp",
        "layers": 6,
        "lr_decay": "linear",
        "cosine_lr_decay_steps": 8000,
    },
    {
        "name": "xf10_cos4000_r015",
        "model_type": "player_transformer",
        "layers": 10,
        "lr_decay": "cosine",
        "cosine_lr_decay_steps": 4000,
    },
    {
        "name": "xf10_cos8000_r015",
        "model_type": "player_transformer",
        "layers": 10,
        "lr_decay": "cosine",
        "cosine_lr_decay_steps": 8000,
    },
    {
        "name": "xf10_linear8000_r015",
        "model_type": "player_transformer",
        "layers": 10,
        "lr_decay": "linear",
        "cosine_lr_decay_steps": 8000,
    },
]


TRAIN_RE = re.compile(
    r"\[(?P<step>\d+)/8000\] bs=512 mse=(?P<mse>[0-9.]+) "
    r"(?:value_mse=(?P<value_mse>[0-9.]+) )?"
    r"mae=(?P<mae>[0-9.]+) target=(?P<target>[0-9.]+)s "
    r"step=(?P<seconds>[0-9.]+)s"
)
EVAL_RE = re.compile(
    r"\[eval (?P<step>\d+)/8000\] mse=(?P<mse>[0-9.]+) "
    r"mae=(?P<mae>[0-9.]+) max_abs=(?P<max_abs>[0-9.]+) "
    r"seconds=(?P<seconds>[0-9.]+)s"
)


def parse_log(path: Path) -> dict[str, object]:
    train_rows: list[dict[str, float | int | None]] = []
    eval_rows: list[dict[str, float | int]] = []
    for line in path.read_text().splitlines():
        train_match = TRAIN_RE.search(line)
        if train_match:
            value_mse = train_match.group("value_mse")
            train_rows.append(
                {
                    "step": int(train_match.group("step")),
                    "mse": float(train_match.group("mse")),
                    "value_mse": float(value_mse) if value_mse is not None else None,
                    "mae": float(train_match.group("mae")),
                    "step_seconds": float(train_match.group("seconds")),
                }
            )
            continue
        eval_match = EVAL_RE.search(line)
        if eval_match:
            eval_rows.append(
                {
                    "step": int(eval_match.group("step")),
                    "mse": float(eval_match.group("mse")),
                    "mae": float(eval_match.group("mae")),
                    "max_abs": float(eval_match.group("max_abs")),
                    "seconds": float(eval_match.group("seconds")),
                }
            )
    step_rows = train_rows[1:] if len(train_rows) > 1 else train_rows
    return {
        "train_last": train_rows[-1] if train_rows else None,
        "eval_last": eval_rows[-1] if eval_rows else None,
        "eval_rows": eval_rows,
        "mean_logged_step_seconds": (
            sum(float(row["step_seconds"]) for row in step_rows) / len(step_rows)
            if step_rows
            else None
        ),
    }


def _base_cmd(candidate: dict[str, object], checkpoint_dir: Path) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "-m",
        "p2.allin.train",
        "players=6",
        "batch_size=512",
        f"steps={STEPS}",
        "no_wandb=true",
        "eval_interval=1000",
        "eval_batch_size=512",
        f"checkpoint_interval={STEPS}",
        f"checkpoint_dir={checkpoint_dir}",
        f"pregenerated_data={TRAIN}",
        f"validation_data={VAL}",
        "compile_model=true",
        "compile_dynamic=true",
        "log_interval=100",
        "target_mode=eligible_pot_share",
        "learning_rate=0.015",
        "adamw_learning_rate=0.024",
        "cosine_lr_decay_ratio=0.015",
        f"lr_decay={candidate['lr_decay']}",
        f"cosine_lr_decay_steps={candidate['cosine_lr_decay_steps']}",
        f"model_type={candidate['model_type']}",
        "hidden_dim=1024",
        "hand_dim=512",
        f"layers={candidate['layers']}",
        "film_rank=64",
        "transformer_heads=8",
        "dense_output_residual=false",
        "mask_folded_player_tokens=false",
    ]


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    results_path = LOG_DIR / "results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text())
    completed = {str(result["name"]) for result in results if result["return_code"] == 0}
    for candidate in CANDIDATES:
        name = str(candidate["name"])
        if name in completed:
            print(f"=== {name} already complete ===", flush=True)
            continue
        log_path = LOG_DIR / f"{name}.log"
        checkpoint_dir = CKPT_ROOT / name
        cmd = _base_cmd(candidate, checkpoint_dir)
        print(f"=== {name} ===", flush=True)
        print(" ".join(cmd), flush=True)
        started = time.perf_counter()
        with log_path.open("w") as log_file:
            process = subprocess.Popen(
                cmd,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
            return_code = process.wait()
        result = {
            "name": name,
            "return_code": return_code,
            "elapsed_seconds": time.perf_counter() - started,
            "params": {
                **candidate,
                "steps": STEPS,
                "target_mode": "eligible_pot_share",
                "validation_data": VAL,
                "pregenerated_data": TRAIN,
            },
            "log": str(log_path.relative_to(ROOT)),
            **parse_log(log_path),
        }
        results = [row for row in results if row.get("name") != name]
        results.append(result)
        results_path.write_text(json.dumps(results, indent=2) + "\n")
        if return_code != 0:
            print(f"{name} failed with return code {return_code}", file=sys.stderr)
            raise SystemExit(return_code)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
