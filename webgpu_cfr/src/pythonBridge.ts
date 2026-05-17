import { spawnSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { WebgpuCfrFixture } from "./types.js";

const here = dirname(fileURLToPath(import.meta.url));
const packageRoot = resolve(here, "..");
const repoRoot = resolve(packageRoot, "..");

export interface ReferenceOptions {
  snapshot: string;
  spot: number[];
  iterations: number;
}

export function loadPythonReference(options: ReferenceOptions): WebgpuCfrFixture {
  const spot = options.spot.join(",");
  const result = spawnSync(
    "uv",
    [
      "run",
      "python",
      "webgpu_cfr/python/reference.py",
      "--snapshot",
      options.snapshot,
      "--spot",
      spot,
      "--iterations",
      String(options.iterations),
    ],
    {
      cwd: repoRoot,
      encoding: "utf8",
      maxBuffer: 128 * 1024 * 1024,
    },
  );
  if (result.status !== 0) {
    throw new Error(result.stderr || `Python reference exited with ${result.status}`);
  }
  return JSON.parse(result.stdout) as WebgpuCfrFixture;
}
