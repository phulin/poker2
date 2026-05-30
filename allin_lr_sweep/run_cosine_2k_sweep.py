from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

from run_sweep import ROOT, TRAIN, VAL


STEPS = 2000
LOG_DIR = ROOT / "allin_lr_sweep" / "logs_cosine_2k"
CKPT_ROOT = ROOT / "allin_lr_sweep" / "checkpoints_cosine_2k"


CANDIDATES = [
    {
        "name": "lr015_cos1000_r015_steps2000",
        "learning_rate": 0.015,
        "adamw_learning_rate": 0.024,
        "cosine_lr_decay_ratio": 0.015,
        "cosine_lr_decay_steps": 1000,
    },
    {
        "name": "lr015_cos2000_r015_steps2000",
        "learning_rate": 0.015,
        "adamw_learning_rate": 0.024,
        "cosine_lr_decay_ratio": 0.015,
        "cosine_lr_decay_steps": 2000,
    },
]


TRAIN_RE = re.compile(
    r"\[(?P<step>\d+)/2000\] bs=512 mse=(?P<mse>[0-9.]+) "
    r"mae=(?P<mae>[0-9.]+) target=(?P<target>[0-9.]+)s "
    r"step=(?P<seconds>[0-9.]+)s"
)
EVAL_RE = re.compile(
    r"\[eval (?P<step>\d+)/2000\] mse=(?P<mse>[0-9.]+) "
    r"mae=(?P<mae>[0-9.]+) max_abs=(?P<max_abs>[0-9.]+) "
    r"seconds=(?P<seconds>[0-9.]+)s"
)


def parse_log(path: Path) -> dict[str, object]:
    train_rows = []
    eval_rows = []
    for line in path.read_text().splitlines():
        train_match = TRAIN_RE.search(line)
        if train_match:
            train_rows.append(
                {
                    "step": int(train_match.group("step")),
                    "mse": float(train_match.group("mse")),
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
    return {
        "train_last": train_rows[-1] if train_rows else None,
        "eval_last": eval_rows[-1] if eval_rows else None,
        "eval_rows": eval_rows,
        "mean_logged_step_seconds": (
            sum(row["step_seconds"] for row in train_rows[1:]) / max(len(train_rows) - 1, 1)
            if train_rows
            else None
        ),
    }


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for candidate in CANDIDATES:
        name = candidate["name"]
        log_path = LOG_DIR / f"{name}.log"
        checkpoint_dir = CKPT_ROOT / name
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "p2.allin.train",
            "players=6",
            "batch_size=512",
            f"steps={STEPS}",
            "no_wandb=true",
            "eval_interval=500",
            "eval_batch_size=512",
            f"checkpoint_interval={STEPS}",
            f"checkpoint_dir={checkpoint_dir}",
            f"pregenerated_data={TRAIN}",
            f"validation_data={VAL}",
            "compile_model=true",
            "compile_dynamic=true",
            "log_interval=100",
            f"learning_rate={candidate['learning_rate']}",
            f"adamw_learning_rate={candidate['adamw_learning_rate']}",
            "lr_decay=cosine",
            f"cosine_lr_decay_ratio={candidate['cosine_lr_decay_ratio']}",
            f"cosine_lr_decay_steps={candidate['cosine_lr_decay_steps']}",
        ]
        print(f"=== {name} ===", flush=True)
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
            "params": {"lr_decay": "cosine", **candidate},
            "log": str(log_path.relative_to(ROOT)),
            **parse_log(log_path),
        }
        results.append(result)
        (LOG_DIR / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        if return_code != 0:
            print(f"{name} failed with return code {return_code}", file=sys.stderr)
            raise SystemExit(return_code)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
