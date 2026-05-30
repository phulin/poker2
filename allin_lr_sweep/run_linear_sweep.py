from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from run_sweep import ROOT, STEPS, TRAIN, VAL, parse_log


LOG_DIR = ROOT / "allin_lr_sweep" / "logs_linear"
CKPT_ROOT = ROOT / "allin_lr_sweep" / "checkpoints_linear"


CANDIDATES = [
    {
        "name": "lr015_linear500_r015",
        "learning_rate": 0.015,
        "adamw_learning_rate": 0.024,
        "cosine_lr_decay_ratio": 0.015,
        "cosine_lr_decay_steps": 500,
    },
    {
        "name": "lr015_linear1000_r015",
        "learning_rate": 0.015,
        "adamw_learning_rate": 0.024,
        "cosine_lr_decay_ratio": 0.015,
        "cosine_lr_decay_steps": 1000,
    },
    {
        "name": "lr015_linear2000_r015",
        "learning_rate": 0.015,
        "adamw_learning_rate": 0.024,
        "cosine_lr_decay_ratio": 0.015,
        "cosine_lr_decay_steps": 2000,
    },
]


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
            "eval_interval=250",
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
            "lr_decay=linear",
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
            "params": {"lr_decay": "linear", **candidate},
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
