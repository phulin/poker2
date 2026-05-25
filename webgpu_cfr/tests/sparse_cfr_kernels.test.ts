import assert from "node:assert/strict";
import { test } from "node:test";
import {
  makeEmptyStorageBuffer,
  makeStorageBuffer,
  readFloatBuffer,
} from "../src/gpuBuffers.js";
import { createDawnDevice } from "../src/gpu.js";
import { SparseCfrGpuKernels } from "../src/sparseCfrKernels.js";
import { packInt16Table } from "../src/allInTables.js";

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
      handCard0: new Uint32Array([0, 0, 3, 5]),
      handCard1: new Uint32Array([1, 2, 4, 6]),
      overlapHands: new Uint32Array([
        0, 1, 0, 0,
        0, 1, 0, 0,
        2, 0, 0, 0,
        3, 0, 0, 0,
      ]),
      overlapCounts: new Uint32Array([2, 2, 1, 1]),
      overlapSlots: 4,
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
    const opponentPolicy = makeEmptyStorageBuffer(device, 12);
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
    const regretWeights = makeEmptyStorageBuffer(device, 12);
    const tailRegrets = makeEmptyStorageBuffer(device, 12);
    const nodeIndices = makeStorageBuffer(device, new Uint32Array([1, 2]));
    const gatheredBeliefs = makeEmptyStorageBuffer(device, 16);
    const scatteredValues = makeEmptyStorageBuffer(device, 24);
    const showdownNode = makeStorageBuffer(device, new Uint32Array([1]));
    const showdownRanks = makeStorageBuffer(device, new Uint32Array([4, 3, 2, 1]));
    const showdownPayoffs = makeStorageBuffer(device, new Float32Array([10, -5, 2]));
    const showdownValues = makeEmptyStorageBuffer(device, 24);

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
    const paramsGather = kernels.encodeGatherNodeBeliefs(
      encoder,
      tree,
      nodeIndices,
      beliefs,
      gatheredBeliefs,
      2,
    );
    const paramsScatter = kernels.encodeScatterNodeValues(
      encoder,
      tree,
      nodeIndices,
      gatheredBeliefs,
      scatteredValues,
      2,
    );
    const paramsShowdown = kernels.encodeShowdownValues(
      encoder,
      tree,
      showdownNode,
      showdownRanks,
      showdownPayoffs,
      beliefs,
      showdownValues,
      1,
    );
    const paramsOpponent = kernels.encodeComputeOpponentPolicyReferenceRange(
      encoder,
      tree,
      beliefs,
      policy,
      opponentPolicy,
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
    const paramsWeights = kernels.encodeComputeRegretWeightsReferenceRange(
      encoder,
      tree,
      beliefs,
      regretWeights,
      0,
      3,
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
    for (const params of paramsGather) params.destroy();
    for (const params of paramsScatter) params.destroy();
    for (const params of paramsShowdown) params.destroy();
    paramsOpponent.destroy();
    paramsBackup.destroy();
    paramsWeights.destroy();
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
      await readFloatBuffer(device, gatheredBeliefs, 16),
      [
        0.125, 0.25, 0.375, 0.25,
        0.25, 0.25, 0.25, 0.25,
        0.375, 0.25, 0.125, 0.25,
        0.25, 0.25, 0.25, 0.25,
      ],
      1e-6,
      "gatheredBeliefs",
    );
    assertCloseArray(
      await readFloatBuffer(device, scatteredValues, 24),
      [
        0, 0, 0, 0,
        0, 0, 0, 0,
        0.125, 0.25, 0.375, 0.25,
        0.25, 0.25, 0.25, 0.25,
        0.375, 0.25, 0.125, 0.25,
        0.25, 0.25, 0.25, 0.25,
      ],
      1e-6,
      "scatteredValues",
    );
    assertCloseArray(
      await readFloatBuffer(device, showdownValues, 24),
      [
        0, 0, 0, 0,
        0, 0, 0, 0,
        10, 10, 0, -5,
        5, 5, -4, -10,
        0, 0, 0, 0,
        0, 0, 0, 0,
      ],
      1e-6,
      "showdownValues",
    );
    assertCloseArray(
      await readFloatBuffer(device, opponentPolicy, 12),
      [
        0, 0, 0, 0,
        0.625, 0.625, 0.4166667, 0.5,
        0.375, 0.375, 0.5833333, 0.5,
      ],
      1e-6,
      "opponentPolicy",
    );
    assertCloseArray(
      await readFloatBuffer(device, regretWeights, 12),
      [
        0.5, 0.5, 0.75, 0.75,
        0.625, 0.625, 0.625, 0.75,
        0.375, 0.375, 0.875, 0.75,
      ],
      1e-6,
      "regretWeights",
    );
    assertCloseArray(
      await readFloatBuffer(device, values, 24),
      [
        4, 4, 4, 6,
        -2.5, -3.5, -5.3333335, -6,
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
        -1.5, -1, -0.75, -1.5,
        0.5, 1, 2.25, 1.5,
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
    nodeIndices.destroy();
    gatheredBeliefs.destroy();
    scatteredValues.destroy();
    showdownNode.destroy();
    showdownRanks.destroy();
    showdownPayoffs.destroy();
    showdownValues.destroy();
  } finally {
    device.destroy();
  }
});

test("sparse gather chunks leaf batches beyond WebGPU dispatch limits", async () => {
  const device = await createDawnDevice();
  try {
    const kernels = new SparseCfrGpuKernels(device);
    const numHands = 1326;
    const tree = kernels.createTreeBuffers({
      nodeCount: 1,
      numHands,
      childOffsets: new Uint32Array([0]),
      childCount: new Uint32Array([0]),
      childIndices: new Uint32Array([]),
      parentIndex: new Uint32Array([0]),
      prevActor: new Uint32Array([0]),
      toAct: new Uint32Array([0]),
      allowedMask: new Uint32Array(numHands).fill(1),
      allowedProb: new Float32Array(numHands).fill(1 / numHands),
      handCard0: new Uint32Array(numHands),
      handCard1: new Uint32Array(numHands),
      overlapHands: new Uint32Array(numHands),
      overlapCounts: new Uint32Array(numHands),
      overlapSlots: 1,
    });
    const batch = 1600;
    const nodeIndices = makeStorageBuffer(device, new Uint32Array(batch));
    const beliefData = new Float32Array(2 * numHands);
    for (let hand = 0; hand < numHands; hand += 1) {
      beliefData[hand] = hand;
      beliefData[numHands + hand] = 10000 + hand;
    }
    const beliefs = makeStorageBuffer(device, beliefData);
    const out = makeEmptyStorageBuffer(device, batch * 2 * numHands);

    const encoder = device.createCommandEncoder();
    const params = kernels.encodeGatherNodeBeliefs(
      encoder,
      tree,
      nodeIndices,
      beliefs,
      out,
      batch,
    );
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    for (const param of params) param.destroy();

    const actual = await readFloatBuffer(device, out, batch * 2 * numHands);
    const boundary = 1536 * 2 * numHands;
    assert.equal(actual[0], 0);
    assert.equal(actual[numHands], 10000);
    assert.equal(actual[boundary], 0);
    assert.equal(actual[boundary + numHands], 10000);

    tree.dispose();
    nodeIndices.destroy();
    beliefs.destroy();
    out.destroy();
  } finally {
    device.destroy();
  }
});

test("sparse WGSL all-in table values match blocker-aware CPU reference", async () => {
  const device = await createDawnDevice();
  try {
    const kernels = new SparseCfrGpuKernels(device);
    const handCard0 = new Uint32Array([0, 0, 3, 5]);
    const handCard1 = new Uint32Array([1, 2, 4, 6]);
    const tree = kernels.createTreeBuffers({
      nodeCount: 2,
      numHands: 4,
      childOffsets: new Uint32Array([0, 0]),
      childCount: new Uint32Array([0, 0]),
      childIndices: new Uint32Array([]),
      parentIndex: new Uint32Array([0, 0]),
      prevActor: new Uint32Array([0, 0]),
      toAct: new Uint32Array([0, 1]),
      allowedMask: new Uint32Array([
        1, 1, 1, 1,
        1, 1, 1, 1,
      ]),
      allowedProb: new Float32Array([
        0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25,
      ]),
      handCard0,
      handCard1,
      overlapHands: new Uint32Array([
        0, 1, 0, 0,
        0, 1, 0, 0,
        2, 0, 0, 0,
        3, 0, 0, 0,
      ]),
      overlapCounts: new Uint32Array([2, 2, 1, 1]),
      overlapSlots: 4,
    });
    const scale = 100;
    const table = new Int16Array(16);
    for (let h = 0; h < 4; h += 1) {
      for (let o = 0; o < 4; o += 1) {
        const overlaps =
          handCard0[h] === handCard0[o] ||
          handCard0[h] === handCard1[o] ||
          handCard1[h] === handCard0[o] ||
          handCard1[h] === handCard1[o];
        table[h * 4 + o] = overlaps ? 0 : (h + 1) * 10 - (o + 1);
      }
    }
    const beliefsData = new Float32Array([
      0, 0, 0, 0,
      0, 0, 0, 0,
      0.1, 0.2, 0.3, 0.4,
      0.4, 0.3, 0.2, 0.1,
    ]);
    const nodeIndices = makeStorageBuffer(device, new Uint32Array([1]));
    const tableBuffer = makeStorageBuffer(device, packInt16Table(table));
    const comboPerms = makeStorageBuffer(device, new Uint32Array([0, 1, 2, 3]));
    const scaleFactors = makeStorageBuffer(device, new Float32Array([2, 3]));
    const beliefs = makeStorageBuffer(device, beliefsData);
    const values = makeEmptyStorageBuffer(device, 16);

    const encoder = device.createCommandEncoder();
    const params = kernels.encodeAllInTableValuesReference(
      encoder,
      tree,
      nodeIndices,
      tableBuffer,
      comboPerms,
      scaleFactors,
      beliefs,
      values,
      1,
      scale,
      0,
      false,
    );
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    params.destroy();

    const expected = new Array(16).fill(0);
    for (let player = 0; player < 2; player += 1) {
      const oppBase = (2 + (1 - player)) * 4;
      for (let hand = 0; hand < 4; hand += 1) {
        let denom = 0;
        let numer = 0;
        for (let opp = 0; opp < 4; opp += 1) {
          const b = beliefsData[oppBase + opp]!;
          numer += (table[hand * 4 + opp]! / scale) * b;
          const overlaps =
            handCard0[hand] === handCard0[opp] ||
            handCard0[hand] === handCard1[opp] ||
            handCard1[hand] === handCard0[opp] ||
            handCard1[hand] === handCard1[opp];
          if (!overlaps) denom += b;
        }
        expected[8 + player * 4 + hand] =
          denom > 1.0e-8 ? (numer / denom) * (player === 0 ? 2 : 3) : 0;
      }
    }
    assertCloseArray(await readFloatBuffer(device, values, 16), expected, 1e-6, "allInValues");

    tree.dispose();
    nodeIndices.destroy();
    tableBuffer.destroy();
    comboPerms.destroy();
    scaleFactors.destroy();
    beliefs.destroy();
    values.destroy();
  } finally {
    device.destroy();
  }
});
