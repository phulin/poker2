"""Bounded pytest file runner for auditing the current test suite."""

from __future__ import annotations

import json
import re
import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"


SUMMARY_RE = re.compile(
    r"=+\s+(?P<body>.+?)\s+in\s+(?P<seconds>[0-9.]+)s\s*=+", re.S
)
DURATION_RE = re.compile(r"^(?P<seconds>[0-9.]+)s\s+call\s+(?P<node>tests/.*)$")


def parse_summary(output: str) -> dict[str, object]:
    match = SUMMARY_RE.search(output)
    if not match:
        return {}

    body = match.group("body").replace("\n", " ")
    counts: dict[str, int] = {}
    for part in body.split(","):
        part = part.strip()
        count_match = re.match(r"(?P<count>\d+)\s+(?P<name>[a-zA-Z_]+)", part)
        if count_match:
            counts[count_match.group("name")] = int(count_match.group("count"))
    return {"summary": body, "pytest_seconds": float(match.group("seconds")), **counts}


def parse_durations(output: str) -> list[dict[str, object]]:
    durations = []
    for line in output.splitlines():
        match = DURATION_RE.match(line.strip())
        if match:
            durations.append(
                {
                    "seconds": float(match.group("seconds")),
                    "node": match.group("node"),
                }
            )
    return durations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", help="Specific test files to run")
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--output", default="test_survey_results.json")
    args = parser.parse_args()

    if args.paths:
        files = [ROOT / path for path in args.paths]
    else:
        files = sorted(TESTS_DIR.glob("test_*.py"))

    results_path = ROOT / args.output
    results = []
    for index, path in enumerate(files, 1):
        rel = path.relative_to(ROOT).as_posix()
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            rel,
            "-q",
            "-rs",
            "--tb=short",
            "--disable-warnings",
            "--durations=10",
            "--durations-min=0.05",
        ]
        start = time.perf_counter()
        try:
            completed = subprocess.run(
                cmd,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                check=False,
            )
            elapsed = time.perf_counter() - start
            output = completed.stdout
            result = {
                "file": rel,
                "returncode": completed.returncode,
                "elapsed_seconds": elapsed,
                "timed_out": False,
                "parsed": parse_summary(output),
                "durations": parse_durations(output),
                "output_tail": "\n".join(output.splitlines()[-80:]),
            }
        except subprocess.TimeoutExpired as exc:
            elapsed = time.perf_counter() - start
            output = exc.stdout or ""
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
            result = {
                "file": rel,
                "returncode": None,
                "elapsed_seconds": elapsed,
                "timed_out": True,
                "parsed": parse_summary(output),
                "durations": parse_durations(output),
                "output_tail": "\n".join(output.splitlines()[-80:]),
            }
        results.append(result)
        status = "TIMEOUT" if result["timed_out"] else result["returncode"]
        print(f"[{index:02d}/{len(files)}] {rel} -> {status} {elapsed:.2f}s", flush=True)

    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
