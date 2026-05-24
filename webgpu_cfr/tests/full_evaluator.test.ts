import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { test, type TestContext } from "node:test";
import { BrowserCfrEvaluator } from "../src/browserEvaluator.js";
import { handComboIndex, parseCard } from "../src/cards.js";
import { createDawnDevice } from "../src/gpu.js";
import { NUM_HANDS } from "../src/hunlEnv.js";
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

let cachedExport:
  | { manifest: string; weights: string }
  | { skipReason: string }
  | undefined;

function exportModel(t: TestContext): { manifest: string; weights: string } | undefined {
  if (cachedExport) {
    if ("skipReason" in cachedExport) {
      t.skip(cachedExport.skipReason);
      return undefined;
    }
    return cachedExport;
  }
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
    if (result.stderr.includes("supports only BetterFFN checkpoints with leaky_relu")) {
      const skipReason =
        "checkpoints-rebel/rebel_latest.pt is not an exportable leaky_relu BetterFFN checkpoint";
      cachedExport = { skipReason };
      t.skip(skipReason);
      return undefined;
    }
    throw new Error(result.stderr || `export_model.py exited ${result.status}`);
  }
  cachedExport = {
    manifest: join(out, "model.json"),
    weights: join(out, "weights.bin.gz"),
  };
  return cachedExport;
}

test("exported BetterFFN WebGPU evaluator matches Python fixture for call spot", async (t) => {
  const exported = exportModel(t);
  if (!exported) return;
  const fixture = loadPythonReference({
    snapshot: "checkpoints-rebel/rebel_latest.pt",
    spot: [1],
    iterations: 2,
  });
  assert.ok(fixture.expected);

  const device = await createDawnDevice();
  const model = await loadNodeModel(device, exported.manifest, exported.weights);
  const evaluator = new BrowserCfrEvaluator(device, model);
  try {
    const result = await evaluator.evaluateSpot({
      spot: [1],
      iterations: 2,
      depth: 1,
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
    evaluator.dispose();
    model.dispose();
    device.destroy();
  }
});

test("exported BetterFFN WebGPU evaluator handles a raise/call prefix", async (t) => {
  const exported = exportModel(t);
  if (!exported) return;
  const fixture = loadPythonReference({
    snapshot: "checkpoints-rebel/rebel_latest.pt",
    spot: [3, 1],
    iterations: 2,
  });
  assert.ok(fixture.expected);

  const device = await createDawnDevice();
  const model = await loadNodeModel(device, exported.manifest, exported.weights);
  const evaluator = new BrowserCfrEvaluator(device, model);
  try {
    const result = await evaluator.evaluateSpot({
      spot: [3, 1],
      iterations: 2,
      depth: 1,
    });
    assertCloseArray(
      result.actionProbs,
      fixture.expected.actionProbs,
      2e-3,
      "actionProbs",
    );
  } finally {
    evaluator.dispose();
    model.dispose();
    device.destroy();
  }
});

test("exported BetterFFN WebGPU evaluator supports hero-only exact hand beliefs", async (t) => {
  const exported = exportModel(t);
  if (!exported) return;
  const device = await createDawnDevice();
  const model = await loadNodeModel(device, exported.manifest, exported.weights);
  const evaluator = new BrowserCfrEvaluator(device, model);
  const heroHand = [parseCard("As"), parseCard("Kd")] as [number, number];
  const publicCards = [parseCard("2c"), parseCard("7d"), parseCard("Th")];
  try {
    const result = await evaluator.evaluateSpot({
      spot: [],
      iterations: 1,
      heroPlayer: 1,
      heroHand,
      publicCards,
    });
    const heroIndex = handComboIndex(heroHand[0], heroHand[1]);
    const villainBlockedByHero = handComboIndex(parseCard("As"), parseCard("Ac"));
    const villainBlockedByBoard = handComboIndex(parseCard("2c"), parseCard("3c"));

    assert.equal(result.beliefsAtSpot[NUM_HANDS + heroIndex], 1);
    assert.equal(result.beliefsAtSpot[villainBlockedByHero], 0);
    assert.equal(result.beliefsAtSpot[villainBlockedByBoard], 0);
    assert.equal(result.legalMask.length, model.manifest.architecture.numActions);
  } finally {
    evaluator.dispose();
    model.dispose();
    device.destroy();
  }
});

test("exported BetterFFN WebGPU evaluator rejects duplicate hero/public cards", async (t) => {
  const exported = exportModel(t);
  if (!exported) return;
  const device = await createDawnDevice();
  const model = await loadNodeModel(device, exported.manifest, exported.weights);
  const evaluator = new BrowserCfrEvaluator(device, model);
  try {
    await assert.rejects(
      () =>
        evaluator.evaluateSpot({
          spot: [],
          iterations: 1,
          heroPlayer: 1,
          heroHand: [parseCard("As"), parseCard("Kd")],
          publicCards: [parseCard("As"), parseCard("7d"), parseCard("Th")],
        }),
      /duplicate card As/i,
    );
  } finally {
    evaluator.dispose();
    model.dispose();
    device.destroy();
  }
});
