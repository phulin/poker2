import assert from "node:assert/strict";
import { test } from "node:test";
import type { BetterFfnWebGpuModel, BetterFfnPrediction } from "../src/betterFfnWebGpuModel.js";
import { createDawnDevice } from "../src/gpu.js";
import { DEFAULT_FORCE_DECK, NUM_HANDS, PublicHunlEnv } from "../src/hunlEnv.js";
import { SparseCfrResolver } from "../src/sparseResolver.js";

function fakeModel(numActions: number, device?: GPUDevice): BetterFfnWebGpuModel {
  return {
    ...(device ? { device } : {}),
    manifest: {
      architecture: { numActions },
    },
    async predict(): Promise<BetterFfnPrediction> {
      return {
        handValues: new Float32Array(2 * NUM_HANDS),
        policyLogits: new Float32Array(NUM_HANDS * numActions),
      };
    },
    async predictHandValues(): Promise<Float32Array<ArrayBuffer>> {
      return new Float32Array(2 * NUM_HANDS);
    },
  } as unknown as BetterFfnWebGpuModel;
}

function uniformBeliefs(): Float32Array<ArrayBuffer> {
  const beliefs = new Float32Array(2 * NUM_HANDS);
  beliefs.fill(1 / NUM_HANDS);
  return beliefs;
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

    const [cpuResult, gpuResult] = await Promise.all([
      cpu.solve(env.clone(), uniformBeliefs(), {
        depth: 3,
        iterations: 2,
        selectedAction: 1,
      }),
      gpu.solve(env.clone(), uniformBeliefs(), {
        depth: 3,
        iterations: 2,
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
    assert.ok(maxDiff < 1e-5, `action prob max diff ${maxDiff}`);
  } finally {
    device.destroy();
  }
});
