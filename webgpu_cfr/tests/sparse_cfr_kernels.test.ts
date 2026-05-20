import assert from "node:assert/strict";
import { test } from "node:test";
import {
  makeEmptyStorageBuffer,
  makeStorageBuffer,
  readFloatBuffer,
} from "../src/gpuBuffers.js";
import { createDawnDevice } from "../src/gpu.js";
import { SparseCfrGpuKernels } from "../src/sparseCfrKernels.js";

function assertCloseArray(
  actual: Float32Array<ArrayBufferLike>,
  expected: readonly number[],
  atol: number,
  label: string,
): void {
  assert.equal(actual.length, expected.length, `${label} length`);
  let maxDiff = 0;
  for (let i = 0; i < actual.length; i += 1) {
    maxDiff = Math.max(maxDiff, Math.abs(actual[i]! - expected[i]!));
  }
  assert.ok(maxDiff <= atol, `${label} max diff ${maxDiff} > ${atol}`);
}

test("sparse WGSL kernels regret-match and propagate beliefs by depth", async () => {
  const device = await createDawnDevice();
  try {
    const kernels = new SparseCfrGpuKernels(device);
    const tree = kernels.createTreeBuffers({
      nodeCount: 3,
      numHands: 4,
      childOffsets: new Uint32Array([0, 2, 2]),
      childCount: new Uint32Array([2, 0, 0]),
      childIndices: new Uint32Array([1, 2]),
      parentIndex: new Uint32Array([0, 0, 0]),
      prevActor: new Uint32Array([0, 0, 0]),
      toAct: new Uint32Array([0, 1, 1]),
      allowedMask: new Uint32Array([
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
      ]),
      allowedProb: new Float32Array([
        0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25,
      ]),
    });

    const regrets = makeStorageBuffer(
      device,
      new Float32Array([
        0, 0, 0, 0,
        1, 0, 3, 0,
        3, 0, 1, 0,
      ]),
    );
    const policy = makeEmptyStorageBuffer(device, 12);
    const beliefs = makeStorageBuffer(
      device,
      new Float32Array([
        0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
      ]),
    );
    const reach = makeStorageBuffer(
      device,
      new Float32Array([
        1, 1, 1, 1,
        1, 1, 1, 1,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
      ]),
    );
    const numerator = makeEmptyStorageBuffer(device, 12);
    const denominator = makeEmptyStorageBuffer(device, 12);
    const policyAvg = makeEmptyStorageBuffer(device, 12);
    const denom = makeEmptyStorageBuffer(device, 6);
    const opponentPolicy = makeStorageBuffer(
      device,
      new Float32Array([
        0, 0, 0, 0,
        0.4, 0.4, 0.4, 0.4,
        0.6, 0.6, 0.6, 0.6,
      ]),
    );
    const values = makeStorageBuffer(
      device,
      new Float32Array([
        0, 0, 0, 0,
        0, 0, 0, 0,
        1, 2, 3, 4,
        -1, -2, -3, -4,
        5, 6, 7, 8,
        -5, -6, -7, -8,
      ]),
    );
    const regretWeights = makeStorageBuffer(
      device,
      new Float32Array([
        1, 1, 1, 1,
        0, 0, 0, 0,
        0, 0, 0, 0,
      ]),
    );
    const tailRegrets = makeEmptyStorageBuffer(device, 12);

    const encoder = device.createCommandEncoder();
    const paramsA = kernels.encodeRegretMatch(encoder, tree, regrets, policy);
    const paramsReach = kernels.encodePropagateReachDepth(
      encoder,
      tree,
      policy,
      reach,
      1,
      3,
    );
    const paramsAvg = kernels.encodeUpdateAveragePolicyRange(
      encoder,
      tree,
      reach,
      policy,
      numerator,
      denominator,
      policyAvg,
      1,
      3,
    );
    const paramsB = kernels.encodePropagateBeliefsDepth(
      encoder,
      tree,
      policy,
      beliefs,
      denom,
      1,
      3,
    );
    const paramsBackup = kernels.encodeBackupDepth(
      encoder,
      tree,
      policy,
      opponentPolicy,
      values,
      0,
      1,
    );
    const paramsRegret = kernels.encodeAccumulateRegretsRange(
      encoder,
      tree,
      regretWeights,
      values,
      tailRegrets,
      1,
      3,
    );
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    paramsA.destroy();
    paramsReach.destroy();
    paramsAvg.destroy();
    paramsB.destroy();
    paramsBackup.destroy();
    paramsRegret.destroy();

    assertCloseArray(
      await readFloatBuffer(device, policy, 12),
      [
        0, 0, 0, 0,
        0.25, 0.5, 0.75, 0.5,
        0.75, 0.5, 0.25, 0.5,
      ],
      1e-6,
      "policy",
    );
    assertCloseArray(
      await readFloatBuffer(device, beliefs, 24),
      [
        0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25,
        0.125, 0.25, 0.375, 0.25,
        0.25, 0.25, 0.25, 0.25,
        0.375, 0.25, 0.125, 0.25,
        0.25, 0.25, 0.25, 0.25,
      ],
      1e-6,
      "beliefs",
    );
    assertCloseArray(
      await readFloatBuffer(device, reach, 24),
      [
        1, 1, 1, 1,
        1, 1, 1, 1,
        0.25, 0.5, 0.75, 0.5,
        1, 1, 1, 1,
        0.75, 0.5, 0.25, 0.5,
        1, 1, 1, 1,
      ],
      1e-6,
      "reach",
    );
    assertCloseArray(
      await readFloatBuffer(device, policyAvg, 12),
      [
        0, 0, 0, 0,
        0.25, 0.5, 0.75, 0.5,
        0.75, 0.5, 0.25, 0.5,
      ],
      1e-6,
      "policyAvg",
    );
    assertCloseArray(
      await readFloatBuffer(device, values, 24),
      [
        4, 4, 4, 6,
        -3.4, -4.4, -5.4, -6.4,
        1, 2, 3, 4,
        -1, -2, -3, -4,
        5, 6, 7, 8,
        -5, -6, -7, -8,
      ],
      1e-6,
      "values",
    );
    assertCloseArray(
      await readFloatBuffer(device, regrets, 12),
      [
        0, 0, 0, 0,
        1, 0, 3, 0,
        3, 0, 1, 0,
      ],
      1e-6,
      "source regrets",
    );
    assertCloseArray(
      await readFloatBuffer(device, tailRegrets, 12),
      [
        0, 0, 0, 0,
        -3, -2, -1, -2,
        1, 2, 3, 2,
      ],
      1e-6,
      "tailRegrets",
    );

    tree.dispose();
    regrets.destroy();
    policy.destroy();
    beliefs.destroy();
    reach.destroy();
    numerator.destroy();
    denominator.destroy();
    policyAvg.destroy();
    denom.destroy();
    opponentPolicy.destroy();
    values.destroy();
    regretWeights.destroy();
    tailRegrets.destroy();
  } finally {
    device.destroy();
  }
});
