import assert from "node:assert/strict";
import { test } from "node:test";
import { ALL_IN_I16_SCALE, type AllInTableProvider } from "../src/allInTables.js";
import type {
  BetterFfnPrediction,
  BetterFfnWebGpuModel,
  GpuHandValuePrediction,
} from "../src/betterFfnWebGpuModel.js";
import { BrowserCfrEvaluator } from "../src/browserEvaluator.js";
import { HAND_COMBOS } from "../src/cards.js";
import { createDawnDevice } from "../src/gpu.js";
import { makeEmptyStorageBuffer } from "../src/gpuBuffers.js";
import { DEFAULT_FORCE_DECK, NUM_HANDS, PublicHunlEnv } from "../src/hunlEnv.js";
import { SparseCfrResolver } from "../src/sparseResolver.js";
import type { SolveProgress } from "../src/types.js";

interface FakeModelCounters {
  singleLeafCalls: number;
  batchLeafSizes: number[];
}

function fakeModel(
  numActions: number,
  device?: GPUDevice,
  counters?: FakeModelCounters,
  allInTableProvider?: AllInTableProvider,
): BetterFfnWebGpuModel {
  return {
    ...(device ? { device } : {}),
    ...(allInTableProvider ? { allInTableProvider } : {}),
    manifest: {
      architecture: { numActions },
      env: {
        stack: 20,
        sb: 1,
        bb: 2,
        betBins: [0.5],
        flopShowdown: false,
        defaultButton: 1,
        defaultForceDeck: DEFAULT_FORCE_DECK,
      },
    },
    actionLabels: Array.from({ length: numActions }, (_, index) => String(index)),
    async predict(): Promise<BetterFfnPrediction> {
      return {
        handValues: new Float32Array(2 * NUM_HANDS),
        policyLogits: new Float32Array(NUM_HANDS * numActions),
      };
    },
    async predictBatch(envs: readonly PublicHunlEnv[]): Promise<BetterFfnPrediction> {
      return {
        handValues: new Float32Array(envs.length * 2 * NUM_HANDS),
        policyLogits: new Float32Array(envs.length * NUM_HANDS * numActions),
      };
    },
    async predictHandValues(): Promise<Float32Array<ArrayBuffer>> {
      if (counters) counters.singleLeafCalls += 1;
      return new Float32Array(2 * NUM_HANDS);
    },
    async predictBatchHandValues(
      envs: readonly PublicHunlEnv[],
    ): Promise<Float32Array<ArrayBuffer>> {
      if (counters) counters.batchLeafSizes.push(envs.length);
      return new Float32Array(envs.length * 2 * NUM_HANDS);
    },
    async predictBatchHandValuesGpu(
      envs: readonly PublicHunlEnv[],
      _beliefs: Float32Array<ArrayBufferLike> | GPUBuffer,
      _prepared?: unknown,
      beforeSubmit?: (encoder: GPUCommandEncoder, handValues: GPUBuffer) => void,
      _exactBelief?: unknown,
      beforeEncode?: (encoder: GPUCommandEncoder) => void,
    ): Promise<GpuHandValuePrediction> {
      if (!device) throw new Error("fake GPU model has no device");
      if (counters) counters.batchLeafSizes.push(envs.length);
      const buffer = makeEmptyStorageBuffer(device, envs.length * 2 * NUM_HANDS);
      if (beforeEncode || beforeSubmit) {
        const encoder = device.createCommandEncoder();
        beforeEncode?.(encoder);
        beforeSubmit?.(encoder, buffer);
        device.queue.submit([encoder.finish()]);
      }
      return {
        buffer,
        batch: envs.length,
        valuesPerSample: 2 * NUM_HANDS,
        dispose: () => buffer.destroy(),
      };
    },
  } as unknown as BetterFfnWebGpuModel;
}

function positiveAllInProvider(): AllInTableProvider {
  const table = new Int16Array(NUM_HANDS * NUM_HANDS);
  for (let h = 0; h < NUM_HANDS; h += 1) {
    const [h0, h1] = HAND_COMBOS[h]!;
    for (let o = 0; o < NUM_HANDS; o += 1) {
      const [o0, o1] = HAND_COMBOS[o]!;
      table[h * NUM_HANDS + o] =
        h0 === o0 || h0 === o1 || h1 === o0 || h1 === o1 ? 0 : ALL_IN_I16_SCALE - 1;
    }
  }
  return {
    async tableForRoot() {
      return { street: 0, table, scale: ALL_IN_I16_SCALE };
    },
  };
}

function uniformBeliefs(): Float32Array<ArrayBuffer> {
  const beliefs = new Float32Array(2 * NUM_HANDS);
  beliefs.fill(1 / NUM_HANDS);
  return beliefs;
}

function maxArrayDiff(a: Float32Array, b: Float32Array): number {
  assert.equal(a.length, b.length);
  let maxDiff = 0;
  for (let i = 0; i < a.length; i += 1) {
    maxDiff = Math.max(maxDiff, Math.abs(a[i]! - b[i]!));
  }
  return maxDiff;
}

test("sparse resolver supports depth greater than one", async () => {
  const betBins = [0.5];
  const numActions = betBins.length + 3;
  const env = new PublicHunlEnv({
    stack: 20,
    sb: 1,
    bb: 2,
    betBins,
    button: 1,
    forceDeck: DEFAULT_FORCE_DECK,
  });
  const resolver = new SparseCfrResolver(fakeModel(numActions));

  const result = await resolver.solve(env, uniformBeliefs(), {
    depth: 3,
    iterations: 2,
    selectedAction: 1,
  });

  assert.equal(result.policy.length, NUM_HANDS * numActions);
  assert.equal(result.actionProbs.length, numActions);
  assert.equal(result.beliefsAfter?.length, 2 * NUM_HANDS);

  const actionMass = Array.from(result.actionProbs).reduce((sum, value) => sum + value, 0);
  assert.ok(Math.abs(actionMass - 1) < 1e-5, `action mass ${actionMass}`);

  for (let player = 0; player < 2; player += 1) {
    const offset = player * NUM_HANDS;
    let beliefMass = 0;
    for (let hand = 0; hand < NUM_HANDS; hand += 1) {
      beliefMass += result.beliefsAfter![offset + hand]!;
    }
    assert.ok(Math.abs(beliefMass - 1) < 1e-5, `belief mass ${beliefMass}`);
  }
});

test("sparse resolver reconstructs dense root policy from child rows", async () => {
  const betBins = [0.5];
  const numActions = betBins.length + 3;
  const env = new PublicHunlEnv({
    stack: 20,
    sb: 1,
    bb: 2,
    betBins,
    button: 1,
    forceDeck: DEFAULT_FORCE_DECK,
  });
  const resolver = new SparseCfrResolver(fakeModel(numActions));

  const result = await resolver.solve(env, uniformBeliefs(), {
    depth: 1,
    iterations: 1,
  });

  const legalActions = [0, 1, numActions - 1];
  const illegalActions = [2];
  let hand = -1;
  for (let candidate = 0; candidate < NUM_HANDS; candidate += 1) {
    let mass = 0;
    for (const action of legalActions) {
      mass += result.policy[candidate * numActions + action]!;
    }
    if (mass > 0) {
      hand = candidate;
      break;
    }
  }
  assert.notEqual(hand, -1, "expected at least one unblocked hand policy");
  let legalMass = 0;
  for (let action = 0; action < numActions; action += 1) {
    const probability = result.policy[hand * numActions + action]!;
    if (legalActions.includes(action)) {
      legalMass += probability;
    } else if (illegalActions.includes(action)) {
      assert.equal(probability, 0, `expected illegal action ${action} to be zero`);
    }
  }
  assert.ok(Math.abs(legalMass - 1) < 1e-6, `root policy hand mass ${legalMass}`);
});

test("sparse resolver selected action uses root action lookup", async () => {
  const betBins = [0.5];
  const numActions = betBins.length + 3;
  const env = new PublicHunlEnv({
    stack: 20,
    sb: 1,
    bb: 2,
    betBins,
    button: 1,
    forceDeck: DEFAULT_FORCE_DECK,
  });
  const resolver = new SparseCfrResolver(fakeModel(numActions));

  const result = await resolver.solve(env, uniformBeliefs(), {
    depth: 2,
    iterations: 1,
    selectedAction: numActions - 1,
    readPolicy: false,
    readActionProbs: false,
  });

  assert.equal(result.beliefsAfter?.length, 2 * NUM_HANDS);
  for (let player = 0; player < 2; player += 1) {
    const offset = player * NUM_HANDS;
    let mass = 0;
    for (let hand = 0; hand < NUM_HANDS; hand += 1) {
      mass += result.beliefsAfter![offset + hand]!;
    }
    assert.ok(Math.abs(mass - 1) < 1e-5, `belief mass ${mass}`);
  }
});

test("public env steps exact custom raise-to amounts", () => {
  const env = new PublicHunlEnv({
    stack: 20,
    sb: 1,
    bb: 2,
    betBins: [0.5],
    button: 1,
    forceDeck: DEFAULT_FORCE_DECK,
  });

  env.stepRaiseTo(5);

  assert.equal(env.pot, 7);
  assert.deepEqual(env.committed, [2, 5]);
  assert.equal(env.minRaise, 3);
  assert.equal(env.toAct, 0);
});

test("sparse resolver exposes custom root raise action", async () => {
  const betBins = [1.0, 2.0];
  const numActions = betBins.length + 3;
  const env = new PublicHunlEnv({
    stack: 20,
    sb: 1,
    bb: 2,
    betBins,
    button: 1,
    forceDeck: DEFAULT_FORCE_DECK,
  });
  const resolver = new SparseCfrResolver(fakeModel(numActions));

  const result = await resolver.solve(env, uniformBeliefs(), {
    depth: 1,
    iterations: 1,
    rootCustomRaiseTo: 6,
  });

  const actionCount = numActions + 1;
  assert.equal(result.policy.length, NUM_HANDS * actionCount);
  assert.equal(result.actionProbs.length, actionCount);
  let hand = -1;
  for (let candidate = 0; candidate < NUM_HANDS; candidate += 1) {
    const mass = result.policy[candidate * actionCount + 1]!;
    if (mass > 0) {
      hand = candidate;
      break;
    }
  }
  assert.notEqual(hand, -1, "expected at least one unblocked hand policy");
  const custom = result.policy[hand * actionCount + numActions]!;
  const mass = Array.from(result.actionProbs).reduce((sum, value) => sum + value, 0);
  assert.ok(custom > 0, `custom ${custom}`);
  assert.ok(result.actionProbs[numActions]! > 0, `custom action ${result.actionProbs[numActions]}`);
  assert.ok(Math.abs(mass - 1) < 1e-5, `action mass ${mass}`);
});

test("browser evaluator applies custom first raise exactly", async () => {
  const betBins = [1.0, 2.0];
  const numActions = betBins.length + 3;
  const evaluator = new BrowserCfrEvaluator(
    undefined as unknown as GPUDevice,
    fakeModel(numActions),
  );

  const result = await evaluator.evaluateSpot({
    spot: [numActions],
    customFirstRaiseTo: 6,
    iterations: 1,
    depth: 1,
    initialState: {
      stack: 20,
      sb: 1,
      bb: 2,
      betBins,
      button: 1,
      forceDeck: DEFAULT_FORCE_DECK,
      flopShowdown: false,
    },
  });

  assert.equal(result.actor, 0);
  assert.equal(result.legalMask.length, numActions);
  assert.equal(result.actionProbs.length, numActions);
});

test("sparse resolver reports progress once per CFR iteration", async () => {
  const betBins = [0.5];
  const numActions = betBins.length + 3;
  const env = new PublicHunlEnv({
    stack: 20,
    sb: 1,
    bb: 2,
    betBins,
    button: 1,
    forceDeck: DEFAULT_FORCE_DECK,
  });
  const resolver = new SparseCfrResolver(fakeModel(numActions));
  const progress: number[] = [];

  await resolver.solve(env, uniformBeliefs(), {
    depth: 3,
    iterations: 4,
    readPolicy: false,
    readActionProbs: false,
    onProgress: ({ iteration, iterations }) => {
      assert.equal(iterations, 4);
      progress.push(iteration);
    },
  });

  assert.deepEqual(progress, [1, 2, 3, 4]);
});

test("browser evaluator aggregates sparse progress across prefix and final solves", async () => {
  const betBins = [0.5];
  const numActions = betBins.length + 3;
  const evaluator = new BrowserCfrEvaluator(
    undefined as unknown as GPUDevice,
    fakeModel(numActions),
  );
  const progress: SolveProgress[] = [];

  await evaluator.evaluateSpot({
    spot: [3, 1],
    iterations: 3,
    depth: 3,
    initialState: {
      stack: 20,
      sb: 1,
      bb: 2,
      betBins,
      button: 1,
      forceDeck: DEFAULT_FORCE_DECK,
      flopShowdown: false,
    },
    onProgress: (event) => progress.push(event),
  });

  assert.equal(progress.length, 9);
  assert.equal(progress[0]?.completedIterations, 1);
  assert.equal(progress[0]?.totalIterations, 9);
  assert.equal(progress[0]?.phase, "prefix");
  assert.equal(progress[3]?.solveIndex, 1);
  assert.equal(progress[3]?.completedIterations, 4);
  assert.equal(progress.at(-1)?.phase, "final");
  assert.equal(progress.at(-1)?.completedIterations, 9);
  assert.equal(progress.at(-1)?.percent, 100);
});

test("sparse resolver can route CFR tensor operations through WGSL kernels", async () => {
  const device = await createDawnDevice();
  try {
    const betBins = [0.5];
    const numActions = betBins.length + 3;
    const env = new PublicHunlEnv({
      stack: 20,
      sb: 1,
      bb: 2,
      betBins,
      button: 1,
      forceDeck: DEFAULT_FORCE_DECK,
    });
    const cpu = new SparseCfrResolver(fakeModel(numActions));
    const gpu = new SparseCfrResolver(fakeModel(numActions, device));

    for (const cfrAvg of [true, false]) {
      const [cpuResult, gpuResult] = await Promise.all([
        cpu.solve(env.clone(), uniformBeliefs(), {
          depth: 3,
          iterations: 2,
          cfrAvg,
          selectedAction: 1,
        }),
        gpu.solve(env.clone(), uniformBeliefs(), {
          depth: 3,
          iterations: 2,
          cfrAvg,
          selectedAction: 1,
        }),
      ]);

      assert.equal(gpuResult.policy.length, cpuResult.policy.length);
      assert.equal(gpuResult.actionProbs.length, cpuResult.actionProbs.length);
      let maxDiff = 0;
      for (let i = 0; i < gpuResult.actionProbs.length; i += 1) {
        maxDiff = Math.max(
          maxDiff,
          Math.abs(gpuResult.actionProbs[i]! - cpuResult.actionProbs[i]!),
        );
      }
      assert.ok(maxDiff < 1e-5, `cfrAvg=${cfrAvg} action prob max diff ${maxDiff}`);
    }
  } finally {
    device.destroy();
  }
});

test("combined sparse prefix command buffers match separate submissions", async () => {
  const device = await createDawnDevice();
  const original = process.env.P2_COMBINE_PREFIX_WITH_LEAF;
  try {
    const betBins = [0.5];
    const numActions = betBins.length + 3;
    const env = new PublicHunlEnv({
      stack: 20,
      sb: 1,
      bb: 2,
      betBins,
      button: 1,
      forceDeck: DEFAULT_FORCE_DECK,
    });
    const oldPath = new SparseCfrResolver(fakeModel(numActions, device));
    const combinedPath = new SparseCfrResolver(fakeModel(numActions, device));
    try {
      process.env.P2_COMBINE_PREFIX_WITH_LEAF = "0";
      const oldResult = await oldPath.solve(env.clone(), uniformBeliefs(), {
        depth: 3,
        iterations: 4,
        selectedAction: 1,
        readPolicy: true,
        readActionProbs: true,
        readBeliefs: true,
      });
      process.env.P2_COMBINE_PREFIX_WITH_LEAF = "1";
      const combinedResult = await combinedPath.solve(env.clone(), uniformBeliefs(), {
        depth: 3,
        iterations: 4,
        selectedAction: 1,
        readPolicy: true,
        readActionProbs: true,
        readBeliefs: true,
      });

      assert.equal(maxArrayDiff(combinedResult.policy, oldResult.policy), 0);
      assert.equal(maxArrayDiff(combinedResult.actionProbs, oldResult.actionProbs), 0);
      assert.ok(combinedResult.beliefsAfter);
      assert.ok(oldResult.beliefsAfter);
      assert.equal(maxArrayDiff(combinedResult.beliefsAfter, oldResult.beliefsAfter), 0);
    } finally {
      oldPath.dispose();
      combinedPath.dispose();
    }
  } finally {
    if (original === undefined) {
      delete process.env.P2_COMBINE_PREFIX_WITH_LEAF;
    } else {
      process.env.P2_COMBINE_PREFIX_WITH_LEAF = original;
    }
    device.destroy();
  }
});

test("sparse resolver batches nonterminal leaf value evaluation", async () => {
  const betBins = [0.5];
  const numActions = betBins.length + 3;
  const env = new PublicHunlEnv({
    stack: 20,
    sb: 1,
    bb: 2,
    betBins,
    button: 1,
    forceDeck: DEFAULT_FORCE_DECK,
  });
  const counters: FakeModelCounters = {
    singleLeafCalls: 0,
    batchLeafSizes: [],
  };
  const resolver = new SparseCfrResolver(fakeModel(numActions, undefined, counters));

  await resolver.solve(env, uniformBeliefs(), {
    depth: 3,
    iterations: 2,
    readPolicy: false,
    readActionProbs: false,
  });

  assert.equal(counters.singleLeafCalls, 0);
  assert.ok(counters.batchLeafSizes.length > 0, "expected batched leaf calls");
  assert.ok(
    counters.batchLeafSizes.some((size) => size > 1),
    `expected at least one multi-leaf batch, saw ${counters.batchLeafSizes.join(",")}`,
  );
});

test("sparse resolver table-values all-in call leaves without model inference", async () => {
  const betBins: number[] = [];
  const numActions = betBins.length + 3;
  const env = new PublicHunlEnv({
    stack: 20,
    sb: 1,
    bb: 2,
    betBins,
    button: 1,
    forceDeck: DEFAULT_FORCE_DECK,
  });
  env.stepBin(numActions - 1);
  const counters: FakeModelCounters = {
    singleLeafCalls: 0,
    batchLeafSizes: [],
  };
  const resolver = new SparseCfrResolver(
    fakeModel(numActions, undefined, counters, positiveAllInProvider()),
  );

  const result = await resolver.solve(env, uniformBeliefs(), {
    depth: 1,
    iterations: 3,
  });

  assert.equal(counters.singleLeafCalls, 0);
  assert.deepEqual(counters.batchLeafSizes, []);
  assert.equal(result.actionProbs[0]! < result.actionProbs[1]!, true);
});
