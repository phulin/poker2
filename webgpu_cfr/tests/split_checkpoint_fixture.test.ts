import assert from "node:assert/strict";
import { existsSync, statSync } from "node:fs";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { test } from "node:test";
import { handComboIndex, parseCard } from "../src/cards.js";
import { createDawnDevice } from "../src/gpu.js";
import { initialUniformBeliefs, NUM_HANDS, PublicHunlEnv } from "../src/hunlEnv.js";
import { loadNodeModel } from "../src/nodeModel.js";

const ROOT = join(import.meta.dirname, "..", "..");
const FIXTURE_DIR = join(
  import.meta.dirname,
  "fixtures",
  "rebel_296_4000",
);
const CHECKPOINT = join(FIXTURE_DIR, "checkpoint.pt");
const MANIFEST = join(FIXTURE_DIR, "model.json");
const WEIGHTS = join(FIXTURE_DIR, "weights.bin.gz");
const PREFLOP_ALL_IN = join(FIXTURE_DIR, "allin", "preflop.i16");

interface SplitReference {
  schemaVersion: number;
  numHands: number;
  numActions: number;
  hands: Record<
    string,
    {
      index: number;
      policyLogits: number[];
      handValues: [number, number];
    }
  >;
}

function loadPythonReference(): SplitReference {
  const result = spawnSync(
    "uv",
    [
      "run",
      "python",
      "webgpu_cfr/python/split_reference.py",
      "--snapshot",
      CHECKPOINT,
      "--force-button",
      "1",
    ],
    {
      cwd: ROOT,
      encoding: "utf8",
      maxBuffer: 64 * 1024 * 1024,
    },
  );
  if (result.status !== 0) {
    throw new Error(result.stderr || `split_reference.py exited ${result.status}`);
  }
  return JSON.parse(result.stdout) as SplitReference;
}

function assertCloseArray(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
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

test("rebel_296_4000 fixture includes checkpoint, export, and all-in table", () => {
  assert.ok(existsSync(CHECKPOINT), "checkpoint fixture exists");
  assert.ok(existsSync(MANIFEST), "manifest fixture exists");
  assert.ok(existsSync(WEIGHTS), "weights fixture exists");
  assert.equal(statSync(PREFLOP_ALL_IN).size, NUM_HANDS * NUM_HANDS * 2);
});

test("rebel_296_4000 WebGPU export matches PyTorch split checkpoint on root PBS", async () => {
  const reference = loadPythonReference();
  assert.equal(reference.schemaVersion, 1);
  assert.equal(reference.numHands, NUM_HANDS);

  const device = await createDawnDevice();
  const model = await loadNodeModel(device, MANIFEST, WEIGHTS);
  const env = PublicHunlEnv.fromManifest(model.manifest);
  try {
    const allInTable = await model.allInTableProvider?.tableForRoot(env);
    assert.equal(allInTable?.table.length, NUM_HANDS * NUM_HANDS);
    const prediction = await model.predict(env, initialUniformBeliefs(), {
      includePolicy: true,
    });
    assert.ok(prediction.policyLogits, "policy logits are present");
    assert.equal(reference.numActions, model.manifest.architecture.numActions);

    for (const [label, expected] of Object.entries(reference.hands)) {
      assert.equal(
        expected.index,
        handComboIndex(parseCard(label.slice(0, 2)), parseCard(label.slice(2, 4))),
        `${label} combo index`,
      );
      const policyStart = expected.index * reference.numActions;
      assertCloseArray(
        prediction.policyLogits.subarray(
          policyStart,
          policyStart + reference.numActions,
        ),
        expected.policyLogits,
        3e-2,
        `${label} policy logits`,
      );
      assertCloseArray(
        [
          prediction.handValues[expected.index]!,
          prediction.handValues[NUM_HANDS + expected.index]!,
        ],
        expected.handValues,
        3e-2,
        `${label} hand values`,
      );
    }
  } finally {
    model.dispose();
    device.destroy();
  }
});
