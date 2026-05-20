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
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    paramsA.destroy();
    paramsReach.destroy();
    paramsAvg.destroy();
    paramsB.destroy();

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

    tree.dispose();
    regrets.destroy();
    policy.destroy();
    beliefs.destroy();
    reach.destroy();
    numerator.destroy();
    denominator.destroy();
    policyAvg.destroy();
    denom.destroy();
  } finally {
    device.destroy();
  }
});
