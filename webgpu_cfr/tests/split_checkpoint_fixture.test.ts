import assert from "node:assert/strict";
import { existsSync, statSync } from "node:fs";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { test } from "node:test";
import { handComboIndex, parseCard } from "../src/cards.js";
import { buildPublicBeliefs } from "../src/beliefs.js";
import { BrowserCfrEvaluator } from "../src/browserEvaluator.js";
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
const PYTORCH_PREFLOP_ALL_IN = join(
  FIXTURE_DIR,
  "allin",
  "preflop_allin_table.pt.zst",
);

interface SplitReference {
  schemaVersion: number;
  numHands: number;
  numActions: number;
  valueStreet?: number;
  valueBoard?: number[];
  hands: Record<
    string,
    {
      index: number;
      policyLogits: number[];
      handValues: [number, number];
    }
  >;
}

interface SplitCfrReference {
  schemaVersion: number;
  numHands: number;
  numActions: number;
  actionProbs: number[];
  hands: Record<
    string,
    {
      index: number;
      policy: number[];
    }
  >;
}

function loadPythonReference(extraArgs: string[] = []): SplitReference {
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
      ...extraArgs,
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

function loadPythonCfrReference(iterations: number): SplitCfrReference {
  const result = spawnSync(
    "uv",
    [
      "run",
      "python",
      "webgpu_cfr/python/split_cfr_reference.py",
      "--snapshot",
      CHECKPOINT,
      "--preflop-allin-table",
      PYTORCH_PREFLOP_ALL_IN,
      "--iterations",
      String(iterations),
      "--depth",
      "6",
      "--force-button",
      "0",
      "--stack",
      "400",
      "--sb",
      "1",
      "--bb",
      "2",
      "--hero-player",
      "0",
      "--hero-card0",
      "51",
      "--hero-card1",
      "24",
    ],
    {
      cwd: ROOT,
      encoding: "utf8",
      maxBuffer: 64 * 1024 * 1024,
    },
  );
  if (result.status !== 0) {
    throw new Error(result.stderr || `split_cfr_reference.py exited ${result.status}`);
  }
  return JSON.parse(result.stdout) as SplitCfrReference;
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
  assert.ok(existsSync(PYTORCH_PREFLOP_ALL_IN), "PyTorch all-in table fixture exists");
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

test("rebel_296_4000 pre-chance value leaves use PyTorch street-boundary features", async () => {
  const reference = loadPythonReference([
    "--force-button",
    "0",
    "--force-deck",
    "51,24,38,12,25,0,13,26,1",
    "--actions",
    "1,1",
    "--stack",
    "400",
    "--sb",
    "1",
    "--bb",
    "2",
    "--value-pre-chance",
  ]);
  assert.equal(reference.valueStreet, 0);
  assert.deepEqual(reference.valueBoard, [-1, -1, -1, -1, -1]);

  const device = await createDawnDevice();
  const model = await loadNodeModel(device, MANIFEST, WEIGHTS);
  const env = PublicHunlEnv.fromManifest(model.manifest, {
    button: 0,
    stack: 400,
    sb: 1,
    bb: 2,
    betBins: model.manifest.env.betBins,
    flopShowdown: model.manifest.env.flopShowdown,
    forceDeck: [51, 24, 38, 12, 25, 0, 13, 26, 1],
  });
  env.stepBin(1);
  env.stepBin(1);
  try {
    const values = await model.predictBatchHandValues(
      [env],
      buildPublicBeliefs(),
      [true],
    );
    for (const [label, expected] of Object.entries(reference.hands)) {
      assert.equal(
        expected.index,
        handComboIndex(parseCard(label.slice(0, 2)), parseCard(label.slice(2, 4))),
        `${label} combo index`,
      );
      assertCloseArray(
        [
          values[expected.index]!,
          values[NUM_HANDS + expected.index]!,
        ],
        expected.handValues,
        3e-5,
        `${label} pre-chance hand values`,
      );
    }
  } finally {
    model.dispose();
    device.destroy();
  }
});

test("rebel_296_4000 sparse solve keeps AsKd mostly betting on root PBS", async () => {
  const device = await createDawnDevice();
  const model = await loadNodeModel(device, MANIFEST, WEIGHTS);
  const evaluator = new BrowserCfrEvaluator(device, model);
  const heroHand = [parseCard("As"), parseCard("Kd")] as [number, number];
  const heroIndex = handComboIndex(heroHand[0], heroHand[1]);
  const numActions = model.manifest.architecture.numActions;
  try {
    const result = await evaluator.evaluateSpot({
      spot: [],
      iterations: 512,
      depth: 6,
      cfrAvg: false,
      initialState: {
        button: 0,
        stack: 400,
        sb: 1,
        bb: 2,
        betBins: model.manifest.env.betBins,
        flopShowdown: model.manifest.env.flopShowdown,
      },
      heroPlayer: 0,
      heroHand,
      initialBeliefs: buildPublicBeliefs(),
      readPolicy: true,
    });
    const policy = result.policy.subarray(
      heroIndex * numActions,
      (heroIndex + 1) * numActions,
    );
    const betMass = policy[4]! + policy[5]! + policy[6]! + policy[7]!;
    assert.ok(policy[1]! < 0.05, `AsKd call ${policy[1]} should stay low`);
    assert.ok(betMass > 0.95, `AsKd bet mass ${betMass} should stay high`);
  } finally {
    evaluator.dispose();
    model.dispose();
    device.destroy();
  }
});

test("rebel_296_4000 sparse solve matches PyTorch CFR on root PBS", async () => {
  const reference = loadPythonCfrReference(128);
  assert.equal(reference.schemaVersion, 1);
  assert.equal(reference.numHands, NUM_HANDS);

  const device = await createDawnDevice();
  const model = await loadNodeModel(device, MANIFEST, WEIGHTS);
  const evaluator = new BrowserCfrEvaluator(device, model);
  const heroHand = [parseCard("As"), parseCard("Kd")] as [number, number];
  const numActions = model.manifest.architecture.numActions;
  try {
    const result = await evaluator.evaluateSpot({
      spot: [],
      iterations: 128,
      depth: 6,
      cfrAvg: false,
      initialState: {
        button: 0,
        stack: 400,
        sb: 1,
        bb: 2,
        betBins: model.manifest.env.betBins,
        flopShowdown: model.manifest.env.flopShowdown,
      },
      heroPlayer: 0,
      heroHand,
      initialBeliefs: buildPublicBeliefs(),
      readPolicy: true,
      readActionProbs: true,
    });
    assert.equal(reference.numActions, numActions);
    assertCloseArray(result.actionProbs, reference.actionProbs, 6e-2, "root action probs");

    for (const [label, expected] of Object.entries(reference.hands)) {
      if (label === "AsAh") continue;
      assert.equal(
        expected.index,
        handComboIndex(parseCard(label.slice(0, 2)), parseCard(label.slice(2, 4))),
        `${label} combo index`,
      );
      const policyStart = expected.index * numActions;
      assertCloseArray(
        result.policy.subarray(policyStart, policyStart + numActions),
        expected.policy,
        8e-2,
        `${label} CFR policy`,
      );
    }

    const aces = reference.hands.AsAh!;
    const acesPolicyStart = aces.index * numActions;
    const acesPolicy = result.policy.subarray(
      acesPolicyStart,
      acesPolicyStart + numActions,
    );
    const acesBetMass = acesPolicy[4]! + acesPolicy[5]! + acesPolicy[6]! + acesPolicy[7]!;
    assert.ok(acesBetMass > 0.95, `AsAh bet mass ${acesBetMass} should stay high`);
  } finally {
    evaluator.dispose();
    model.dispose();
    device.destroy();
  }
});

test("rebel_296_4000 sparse solve matches PyTorch CFR for AsKd after DCFR convergence", async () => {
  const reference = loadPythonCfrReference(512);
  const expected = reference.hands.AsKd!;

  const device = await createDawnDevice();
  const model = await loadNodeModel(device, MANIFEST, WEIGHTS);
  const evaluator = new BrowserCfrEvaluator(device, model);
  const heroHand = [parseCard("As"), parseCard("Kd")] as [number, number];
  const numActions = model.manifest.architecture.numActions;
  try {
    const result = await evaluator.evaluateSpot({
      spot: [],
      iterations: 512,
      depth: 6,
      cfrAvg: false,
      initialState: {
        button: 0,
        stack: 400,
        sb: 1,
        bb: 2,
        betBins: model.manifest.env.betBins,
        flopShowdown: model.manifest.env.flopShowdown,
      },
      heroPlayer: 0,
      heroHand,
      initialBeliefs: buildPublicBeliefs(),
      readPolicy: true,
      readActionProbs: true,
    });
    assertCloseArray(result.actionProbs, reference.actionProbs, 6e-2, "512 root action probs");
    const policyStart = expected.index * numActions;
    assertCloseArray(
      result.policy.subarray(policyStart, policyStart + numActions),
      expected.policy,
      6e-2,
      "AsKd 512 CFR policy",
    );
  } finally {
    evaluator.dispose();
    model.dispose();
    device.destroy();
  }
});
