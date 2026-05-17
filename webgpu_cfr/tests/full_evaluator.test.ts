import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { test } from "node:test";
import { BrowserCfrEvaluator } from "../src/browserEvaluator.js";
import { createDawnDevice } from "../src/gpu.js";
import { loadNodeModel } from "../src/nodeModel.js";
import { loadPythonReference } from "../src/pythonBridge.js";

function assertCloseArray(
  actual: Float32Array,
  expected: number[],
  atol: number,
  label: string,
): void {
  assert.equal(actual.length, expected.length, `${label} length`);
  let maxDiff = 0;
  for (let i = 0; i < actual.length; i += 1) {
    const diff = Math.abs(actual[i]! - expected[i]!);
    if (diff > maxDiff) maxDiff = diff;
  }
  assert.ok(maxDiff <= atol, `${label} max diff ${maxDiff} > ${atol}`);
}

let cachedExport: { manifest: string; weights: string } | undefined;

function exportModel(): { manifest: string; weights: string } {
  if (cachedExport) return cachedExport;
  const out = mkdtempSync(join(tmpdir(), "p2-webgpu-cfr-"));
  const result = spawnSync(
    "uv",
    [
      "run",
      "python",
      "webgpu_cfr/python/export_model.py",
      "--snapshot",
      "checkpoints-rebel/rebel_latest.pt",
      "--out",
      out,
    ],
    {
      cwd: join(import.meta.dirname, "..", ".."),
      encoding: "utf8",
      maxBuffer: 128 * 1024 * 1024,
    },
  );
  if (result.status !== 0) {
    throw new Error(result.stderr || `export_model.py exited ${result.status}`);
  }
  cachedExport = {
    manifest: join(out, "model.json"),
    weights: join(out, "weights.bin"),
  };
  return cachedExport;
}

test("exported BetterFFN WebGPU evaluator matches Python fixture for call spot", async () => {
  const exported = exportModel();
  const fixture = loadPythonReference({
    snapshot: "checkpoints-rebel/rebel_latest.pt",
    spot: [1],
    iterations: 2,
  });
  assert.ok(fixture.expected);

  const device = await createDawnDevice();
  const model = await loadNodeModel(device, exported.manifest, exported.weights);
  try {
    const result = await new BrowserCfrEvaluator(device, model).evaluateSpot({
      spot: [1],
      iterations: 2,
    });
    assertCloseArray(
      result.beliefsAtSpot,
      fixture.expected.beliefsAtSpot,
      2e-3,
      "beliefsAtSpot",
    );
    assertCloseArray(
      result.actionProbs,
      fixture.expected.actionProbs,
      2e-3,
      "actionProbs",
    );
  } finally {
    model.dispose();
    device.destroy();
  }
});

test("exported BetterFFN WebGPU evaluator handles a raise/call prefix", async () => {
  const exported = exportModel();
  const fixture = loadPythonReference({
    snapshot: "checkpoints-rebel/rebel_latest.pt",
    spot: [2, 1],
    iterations: 2,
  });
  assert.ok(fixture.expected);

  const device = await createDawnDevice();
  const model = await loadNodeModel(device, exported.manifest, exported.weights);
  try {
    const result = await new BrowserCfrEvaluator(device, model).evaluateSpot({
      spot: [2, 1],
      iterations: 2,
    });
    assertCloseArray(
      result.actionProbs,
      fixture.expected.actionProbs,
      2e-3,
      "actionProbs",
    );
  } finally {
    model.dispose();
    device.destroy();
  }
});
