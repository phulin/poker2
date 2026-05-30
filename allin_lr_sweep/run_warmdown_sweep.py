from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from run_sweep import ROOT, STEPS, TRAIN, VAL, parse_log


LOG_DIR = ROOT / "allin_lr_sweep" / "logs_warmdown"
CKPT_ROOT = ROOT / "allin_lr_sweep" / "checkpoints_warmdown"
WARMDOWN_START = 800


CANDIDATES = [
    {
        "name": "lr012_wd800_final00018",
        "learning_rate": 0.012,
        "adamw_learning_rate": 0.0192,
        "final_learning_rate": 0.00018,
    },
    {
        "name": "lr015_wd800_final000225",
        "learning_rate": 0.015,
        "adamw_learning_rate": 0.024,
        "final_learning_rate": 0.000225,
    },
    {
        "name": "lr015_wd800_final00045",
        "learning_rate": 0.015,
        "adamw_learning_rate": 0.024,
        "final_learning_rate": 0.00045,
    },
    {
        "name": "lr018_wd800_final00027",
        "learning_rate": 0.018,
        "adamw_learning_rate": 0.0288,
        "final_learning_rate": 0.00027,
    },
    {
        "name": "lr018_wd800_final00054",
        "learning_rate": 0.018,
        "adamw_learning_rate": 0.0288,
        "final_learning_rate": 0.00054,
    },
]


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for candidate in CANDIDATES:
        name = candidate["name"]
        lr = float(candidate["learning_rate"])
        final_lr = float(candidate["final_learning_rate"])
        ratio = final_lr / lr
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
            "lr_decay=stable_warmdown",
            f"lr_warmdown_start_step={WARMDOWN_START}",
            f"cosine_lr_decay_ratio={ratio}",
            f"cosine_lr_decay_steps={STEPS}",
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
            "params": {
                "lr_decay": "stable_warmdown",
                "lr_warmdown_start_step": WARMDOWN_START,
                "cosine_lr_decay_steps": STEPS,
                "cosine_lr_decay_ratio": ratio,
                **candidate,
            },
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
