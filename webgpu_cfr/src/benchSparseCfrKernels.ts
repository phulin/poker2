import { performance } from "node:perf_hooks";
import {
  makeEmptyStorageBuffer,
  makeStorageBuffer,
  readFloatBuffer,
} from "./gpuBuffers.js";
import { createDawnDevice } from "./gpu.js";
import {
  SparseCfrGpuKernels,
  type SparseGpuTreeBuffers,
  type SparseGpuTreeData,
} from "./sparseCfrKernels.js";
import { NUM_HANDS } from "./hunlEnv.js";

interface Options {
  warmups: number;
  runs: number;
  dispatches: number;
  variant?: string;
  validate: boolean;
}

interface BenchOp {
  name: string;
  output: GPUBuffer;
  outputElements: number;
  encode: (encoder: GPUCommandEncoder) => GPUBuffer | GPUBuffer[] | undefined;
  validate?: () => Promise<ValidationResult>;
}

interface ValidationResult {
  reference: string;
  maxDiff: number;
  maxIndex?: number;
  referenceValue?: number;
  candidateValue?: number;
}

interface SparseBenchData {
  depthOffsets: number[];
  treeData: SparseGpuTreeData;
  tree: SparseGpuTreeBuffers;
  policy: GPUBuffer;
  policyAvg: GPUBuffer;
  regrets: GPUBuffer;
  values: GPUBuffer;
  beliefs: GPUBuffer;
  reach: GPUBuffer;
  denom: GPUBuffer;
  avgNumerator: GPUBuffer;
  avgDenominator: GPUBuffer;
  opponentPolicy: GPUBuffer;
  opponentPolicyDirect: GPUBuffer;
  opponentAggregates: GPUBuffer;
  regretWeights: GPUBuffer;
  regretWeightsDirect: GPUBuffer;
  regretWeightAggregates: GPUBuffer;
  nodeIndices: GPUBuffer;
  leafValues: GPUBuffer;
  gatheredBeliefs: GPUBuffer;
  rankOrdinals: GPUBuffer;
  rankCounts: GPUBuffer;
  rankHandOffsets: GPUBuffer;
  rankHandCounts: GPUBuffer;
  rankHands: GPUBuffer;
  payoffs: GPUBuffer;
  rankMass: GPUBuffer;
  rankPrefixLess: GPUBuffer;
  rankTotal: GPUBuffer;
  tablePacked: GPUBuffer;
  comboPerms: GPUBuffer;
  scaleFactors: GPUBuffer;
  nodeCount: number;
  policyCount: number;
  valueCount: number;
  leafBatch: number;
  maxRanks: number;
  tableScale: number;
}

const DEFAULT_LAYERS = [1, 4, 16, 64];
const LEAF_BATCH = 24;
const MAX_RANKS = 96;
const TABLE_SCALE = 1024;

function getArg(args: string[], name: string, fallback?: string): string {
  const index = args.indexOf(name);
  const value = index >= 0 ? args[index + 1] : undefined;
  if (value) return value;
  if (fallback !== undefined) return fallback;
  throw new Error(`Missing ${name}`);
}

function hasArg(args: string[], name: string): boolean {
  return args.includes(name);
}

function intArg(args: string[], name: string, fallback: number): number {
  const value = Number.parseInt(getArg(args, name, String(fallback)), 10);
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer`);
  }
  return value;
}

function readOptions(): Options {
  const args = process.argv.slice(2);
  const variant = getArg(args, "--variant", "");
  return {
    warmups: intArg(args, "--warmups", 5),
    runs: intArg(args, "--runs", 20),
    dispatches: intArg(args, "--dispatches", 20),
    ...(variant ? { variant } : {}),
    validate: !hasArg(args, "--no-validate"),
  };
}

function stats(samples: number[]): {
  meanMs: number;
  medianMs: number;
  minMs: number;
  maxMs: number;
} {
  const sorted = [...samples].sort((a, b) => a - b);
  return {
    meanMs: samples.reduce((sum, sample) => sum + sample, 0) / samples.length,
    medianMs: sorted[Math.floor(sorted.length / 2)]!,
    minMs: sorted[0]!,
    maxMs: sorted[sorted.length - 1]!,
  };
}

function fillFloat(length: number, scale: number, offset = 0): Float32Array<ArrayBuffer> {
  const out = new Float32Array(length);
  for (let i = 0; i < length; i += 1) {
    const raw = (((i + offset) * 1664525 + 1013904223) >>> 0) % 2001;
    out[i] = Math.fround((raw - 1000) * scale);
  }
  return out;
}

function handCards(): { card0: Uint32Array<ArrayBuffer>; card1: Uint32Array<ArrayBuffer> } {
  const card0 = new Uint32Array(NUM_HANDS);
  const card1 = new Uint32Array(NUM_HANDS);
  let hand = 0;
  for (let a = 0; a < 52; a += 1) {
    for (let b = a + 1; b < 52; b += 1) {
      card0[hand] = a;
      card1[hand] = b;
      hand += 1;
    }
  }
  if (hand !== NUM_HANDS) throw new Error(`expected ${NUM_HANDS} hands, got ${hand}`);
  return { card0, card1 };
}

function overlapData(
  card0: Uint32Array<ArrayBuffer>,
  card1: Uint32Array<ArrayBuffer>,
): {
  overlapHands: Uint32Array<ArrayBuffer>;
  overlapCounts: Uint32Array<ArrayBuffer>;
  overlapSlots: number;
} {
  const perHand: number[][] = [];
  let overlapSlots = 0;
  for (let hand = 0; hand < NUM_HANDS; hand += 1) {
    const overlaps: number[] = [];
    const a = card0[hand]!;
    const b = card1[hand]!;
    for (let other = 0; other < NUM_HANDS; other += 1) {
      if (
        card0[other] === a ||
        card0[other] === b ||
        card1[other] === a ||
        card1[other] === b
      ) {
        overlaps.push(other);
      }
    }
    perHand.push(overlaps);
    overlapSlots = Math.max(overlapSlots, overlaps.length);
  }
  const overlapHands = new Uint32Array(NUM_HANDS * overlapSlots);
  const overlapCounts = new Uint32Array(NUM_HANDS);
  for (let hand = 0; hand < NUM_HANDS; hand += 1) {
    const overlaps = perHand[hand]!;
    overlapCounts[hand] = overlaps.length;
    overlapHands.set(overlaps, hand * overlapSlots);
  }
  return { overlapHands, overlapCounts, overlapSlots };
}

function makeTreeData(): { data: SparseGpuTreeData; depthOffsets: number[] } {
  const depthOffsets = [0];
  for (const count of DEFAULT_LAYERS) {
    depthOffsets.push(depthOffsets[depthOffsets.length - 1]! + count);
  }
  const nodeCount = depthOffsets[depthOffsets.length - 1]!;
  const childOffsets = new Uint32Array(nodeCount);
  const childCount = new Uint32Array(nodeCount);
  const parentIndex = new Uint32Array(nodeCount);
  const prevActor = new Uint32Array(nodeCount);
  const toAct = new Uint32Array(nodeCount);
  const childIndices: number[] = [];
  parentIndex[0] = 0;
  prevActor[0] = 0;
  for (let depth = 0; depth < DEFAULT_LAYERS.length; depth += 1) {
    const start = depthOffsets[depth]!;
    const end = depthOffsets[depth + 1]!;
    for (let node = start; node < end; node += 1) {
      toAct[node] = depth % 2;
    }
    if (depth + 1 >= DEFAULT_LAYERS.length) continue;
    const nextStart = depthOffsets[depth + 1]!;
    const width = DEFAULT_LAYERS[depth + 1]! / DEFAULT_LAYERS[depth]!;
    for (let local = 0; local < DEFAULT_LAYERS[depth]!; local += 1) {
      const node = start + local;
      childOffsets[node] = childIndices.length;
      childCount[node] = width;
      for (let i = 0; i < width; i += 1) {
        const child = nextStart + local * width + i;
        childIndices.push(child);
        parentIndex[child] = node;
        prevActor[child] = toAct[node]!;
      }
    }
  }
  const allowedMask = new Uint32Array(nodeCount * NUM_HANDS);
  const allowedProb = new Float32Array(nodeCount * NUM_HANDS);
  for (let node = 0; node < nodeCount; node += 1) {
    let allowed = 0;
    for (let hand = 0; hand < NUM_HANDS; hand += 1) {
      const ok = ((hand * 17 + node * 31) % 97) !== 0;
      allowedMask[node * NUM_HANDS + hand] = ok ? 1 : 0;
      if (ok) allowed += 1;
    }
    const p = 1 / allowed;
    for (let hand = 0; hand < NUM_HANDS; hand += 1) {
      if (allowedMask[node * NUM_HANDS + hand]) {
        allowedProb[node * NUM_HANDS + hand] = p;
      }
    }
  }
  const { card0, card1 } = handCards();
  const { overlapHands, overlapCounts, overlapSlots } = overlapData(card0, card1);
  return {
    depthOffsets,
    data: {
      nodeCount,
      numHands: NUM_HANDS,
      childOffsets,
      childCount,
      childIndices: new Uint32Array(childIndices),
      parentIndex,
      prevActor,
      toAct,
      allowedMask,
      allowedProb,
      handCard0: card0,
      handCard1: card1,
      overlapHands,
      overlapCounts,
      overlapSlots,
    },
  };
}

function normalizeBeliefs(data: SparseGpuTreeData): Float32Array<ArrayBuffer> {
  const out = new Float32Array(data.nodeCount * 2 * NUM_HANDS);
  for (let node = 0; node < data.nodeCount; node += 1) {
    for (let player = 0; player < 2; player += 1) {
      let sum = 0;
      const base = (node * 2 + player) * NUM_HANDS;
      for (let hand = 0; hand < NUM_HANDS; hand += 1) {
        const value = data.allowedMask[node * NUM_HANDS + hand]
          ? 0.5 + (((hand * 13 + node * 7 + player * 19) % 1000) / 1000)
          : 0;
        out[base + hand] = value;
        sum += value;
      }
      for (let hand = 0; hand < NUM_HANDS; hand += 1) {
        out[base + hand] = Math.fround(out[base + hand]! / sum);
      }
    }
  }
  return out;
}

function makePolicy(data: SparseGpuTreeData): Float32Array<ArrayBuffer> {
  const out = new Float32Array(data.nodeCount * NUM_HANDS);
  for (let parent = 0; parent < data.nodeCount; parent += 1) {
    const count = data.childCount[parent]!;
    if (count === 0) continue;
    const offset = data.childOffsets[parent]!;
    for (let hand = 0; hand < NUM_HANDS; hand += 1) {
      let total = 0;
      for (let i = 0; i < count; i += 1) {
        total += 1 + ((hand * 17 + parent * 13 + i * 7) % 11);
      }
      for (let i = 0; i < count; i += 1) {
        const child = data.childIndices[offset + i]!;
        const weight = 1 + ((hand * 17 + parent * 13 + i * 7) % 11);
        out[child * NUM_HANDS + hand] = Math.fround(weight / total);
      }
    }
  }
  return out;
}

function makeRankOrdinals(batch: number, maxRanks: number): {
  ordinals: Uint32Array<ArrayBuffer>;
  counts: Uint32Array<ArrayBuffer>;
  handOffsets: Uint32Array<ArrayBuffer>;
  handCounts: Uint32Array<ArrayBuffer>;
  hands: Uint32Array<ArrayBuffer>;
} {
  const ordinals = new Uint32Array(batch * NUM_HANDS);
  const counts = new Uint32Array(batch);
  const rankHandsBySample: number[][][] = [];
  for (let sample = 0; sample < batch; sample += 1) {
    const rankCount = maxRanks - (sample % 7);
    counts[sample] = rankCount;
    const rankHands = Array.from({ length: rankCount }, () => [] as number[]);
    for (let hand = 0; hand < NUM_HANDS; hand += 1) {
      const rank = (hand * 29 + sample * 11) % rankCount;
      ordinals[sample * NUM_HANDS + hand] = rank;
      rankHands[rank]!.push(hand);
    }
    rankHandsBySample.push(rankHands);
  }
  const handOffsets = new Uint32Array(batch * maxRanks);
  const handCounts = new Uint32Array(batch * maxRanks);
  const hands: number[] = [];
  for (let sample = 0; sample < batch; sample += 1) {
    const rankHands = rankHandsBySample[sample]!;
    for (let rank = 0; rank < maxRanks; rank += 1) {
      const outIdx = sample * maxRanks + rank;
      const list = rankHands[rank] ?? [];
      handOffsets[outIdx] = hands.length;
      handCounts[outIdx] = list.length;
      hands.push(...list);
    }
  }
  return { ordinals, counts, handOffsets, handCounts, hands: new Uint32Array(hands) };
}

function makePackedTable(): Uint32Array<ArrayBuffer> {
  const entries = NUM_HANDS * NUM_HANDS;
  const packed = new Uint32Array(Math.ceil(entries / 2));
  for (let i = 0; i < entries; i += 1) {
    const signed = ((i * 37) % 2001) - 1000;
    const raw = signed < 0 ? signed + 65536 : signed;
    const word = i >> 1;
    if ((i & 1) === 0) {
      packed[word] = (packed[word]! & 0xffff0000) | raw;
    } else {
      packed[word] = (packed[word]! & 0x0000ffff) | (raw << 16);
    }
  }
  return packed;
}

function makeBenchData(device: GPUDevice): SparseBenchData {
  const { data: treeData, depthOffsets } = makeTreeData();
  const kernels = new SparseCfrGpuKernels(device);
  const tree = kernels.createTreeBuffers(treeData);
  const nodeCount = treeData.nodeCount;
  const policyCount = nodeCount * NUM_HANDS;
  const valueCount = nodeCount * 2 * NUM_HANDS;
  const leafStart = depthOffsets[depthOffsets.length - 2]!;
  const leafNodes = new Uint32Array(LEAF_BATCH);
  for (let i = 0; i < LEAF_BATCH; i += 1) {
    leafNodes[i] = leafStart + ((i * 7) % (nodeCount - leafStart));
  }
  const { ordinals, counts, handOffsets, handCounts, hands } = makeRankOrdinals(
    LEAF_BATCH,
    MAX_RANKS,
  );
  const comboPerms = new Uint32Array(NUM_HANDS);
  for (let hand = 0; hand < NUM_HANDS; hand += 1) comboPerms[hand] = hand;
  return {
    depthOffsets,
    treeData,
    tree,
    policy: makeStorageBuffer(device, makePolicy(treeData)),
    policyAvg: makeStorageBuffer(device, makePolicy(treeData)),
    regrets: makeStorageBuffer(device, fillFloat(policyCount, 0.01, 3)),
    values: makeStorageBuffer(device, fillFloat(valueCount, 0.002, 4)),
    beliefs: makeStorageBuffer(device, normalizeBeliefs(treeData)),
    reach: makeStorageBuffer(device, normalizeBeliefs(treeData)),
    denom: makeEmptyStorageBuffer(device, nodeCount * 2),
    avgNumerator: makeStorageBuffer(device, fillFloat(policyCount, 0.0001, 5)),
    avgDenominator: makeStorageBuffer(device, fillFloat(policyCount, 0.0001, 6)),
    opponentPolicy: makeEmptyStorageBuffer(device, policyCount),
    opponentPolicyDirect: makeEmptyStorageBuffer(device, policyCount),
    opponentAggregates: makeEmptyStorageBuffer(device, nodeCount * 106),
    regretWeights: makeEmptyStorageBuffer(device, policyCount),
    regretWeightsDirect: makeEmptyStorageBuffer(device, policyCount),
    regretWeightAggregates: makeEmptyStorageBuffer(device, nodeCount * 53),
    nodeIndices: makeStorageBuffer(device, leafNodes),
    leafValues: makeStorageBuffer(device, fillFloat(LEAF_BATCH * 2 * NUM_HANDS, 0.003, 7)),
    gatheredBeliefs: makeEmptyStorageBuffer(device, LEAF_BATCH * 2 * NUM_HANDS),
    rankOrdinals: makeStorageBuffer(device, ordinals),
    rankCounts: makeStorageBuffer(device, counts),
    rankHandOffsets: makeStorageBuffer(device, handOffsets),
    rankHandCounts: makeStorageBuffer(device, handCounts),
    rankHands: makeStorageBuffer(device, hands),
    payoffs: makeStorageBuffer(device, fillFloat(LEAF_BATCH * 3, 0.01, 8)),
    rankMass: makeEmptyStorageBuffer(device, LEAF_BATCH * 2 * MAX_RANKS),
    rankPrefixLess: makeEmptyStorageBuffer(device, LEAF_BATCH * 2 * MAX_RANKS),
    rankTotal: makeEmptyStorageBuffer(device, LEAF_BATCH * 2),
    tablePacked: makeStorageBuffer(device, makePackedTable()),
    comboPerms: makeStorageBuffer(device, comboPerms),
    scaleFactors: makeStorageBuffer(device, fillFloat(LEAF_BATCH * 2, 0.01, 9)),
    nodeCount,
    policyCount,
    valueCount,
    leafBatch: LEAF_BATCH,
    maxRanks: MAX_RANKS,
    tableScale: TABLE_SCALE,
  };
}

function diffStats(a: Float32Array<ArrayBufferLike>, b: Float32Array<ArrayBufferLike>): {
  maxDiff: number;
  maxIndex: number;
  referenceValue: number;
  candidateValue: number;
} {
  let diff = 0;
  let maxIndex = 0;
  for (let i = 0; i < a.length; i += 1) {
    const current = Math.abs(a[i]! - b[i]!);
    if (current > diff) {
      diff = current;
      maxIndex = i;
    }
  }
  return {
    maxDiff: diff,
    maxIndex,
    referenceValue: a[maxIndex]!,
    candidateValue: b[maxIndex]!,
  };
}

function destroyParams(params: GPUBuffer | GPUBuffer[] | undefined): void {
  if (!params) return;
  for (const param of Array.isArray(params) ? params : [params]) {
    param.destroy();
  }
}

async function runOnce(
  device: GPUDevice,
  encode: (encoder: GPUCommandEncoder) => GPUBuffer | GPUBuffer[] | undefined,
): Promise<void> {
  const encoder = device.createCommandEncoder();
  const params = encode(encoder);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  destroyParams(params);
}

async function timeOp(
  device: GPUDevice,
  op: BenchOp,
  warmups: number,
  runs: number,
  dispatches: number,
): Promise<number[]> {
  for (let i = 0; i < warmups; i += 1) {
    await runOnce(device, op.encode);
  }
  const samples: number[] = [];
  for (let run = 0; run < runs; run += 1) {
    const encoder = device.createCommandEncoder();
    const params: GPUBuffer[] = [];
    for (let i = 0; i < dispatches; i += 1) {
      const encoded = op.encode(encoder);
      if (encoded) params.push(...(Array.isArray(encoded) ? encoded : [encoded]));
    }
    const start = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const elapsed = performance.now() - start;
    for (const param of params) param.destroy();
    samples.push(elapsed / dispatches);
  }
  return samples;
}

function buildOps(device: GPUDevice, kernels: SparseCfrGpuKernels, data: SparseBenchData): BenchOp[] {
  const start = 1;
  const end = data.nodeCount;
  const lastParentStart = data.depthOffsets[data.depthOffsets.length - 3]!;
  const lastParentEnd = data.depthOffsets[data.depthOffsets.length - 2]!;
  const leafStart = data.depthOffsets[data.depthOffsets.length - 2]!;
  const leafEnd = data.nodeCount;
  const validatePair = async (
    referenceBuffer: GPUBuffer,
    candidateBuffer: GPUBuffer,
    elements: number,
    reference: string,
  ): Promise<ValidationResult> => {
    const referenceValues = await readFloatBuffer(device, referenceBuffer, elements);
    const candidateValues = await readFloatBuffer(device, candidateBuffer, elements);
    return { reference, ...diffStats(referenceValues, candidateValues) };
  };
  return [
    {
      name: "regret_match",
      output: data.policy,
      outputElements: data.policyCount,
      encode: (encoder) =>
        kernels.encodeRegretMatch(encoder, data.tree, data.regrets, data.policy),
    },
    {
      name: "regret_weight_direct",
      output: data.regretWeightsDirect,
      outputElements: data.policyCount,
      encode: (encoder) =>
        kernels.encodeComputeRegretWeightsRange(
          encoder,
          data.tree,
          data.beliefs,
          data.regretWeightsDirect,
          start,
          end,
        ),
    },
    {
      name: "regret_weight_aggregate",
      output: data.regretWeights,
      outputElements: data.policyCount,
      encode: (encoder) =>
        kernels.encodeComputeRegretWeightsAggregateRange(
          encoder,
          data.tree,
          data.beliefs,
          data.regretWeightAggregates,
          data.regretWeights,
          start,
          end,
        ),
      validate: async () => {
        await runOnce(device, (encoder) =>
          kernels.encodeComputeRegretWeightsRange(
            encoder,
            data.tree,
            data.beliefs,
            data.regretWeightsDirect,
            start,
            end,
          ),
        );
        await runOnce(device, (encoder) =>
          kernels.encodeComputeRegretWeightsAggregateRange(
            encoder,
            data.tree,
            data.beliefs,
            data.regretWeightAggregates,
            data.regretWeights,
            start,
            end,
          ),
        );
        return validatePair(
          data.regretWeightsDirect,
          data.regretWeights,
          data.policyCount,
          "regret_weight_direct",
        );
      },
    },
    {
      name: "regret_weight_aggregate_parallel",
      output: data.regretWeights,
      outputElements: data.policyCount,
      encode: (encoder) =>
        kernels.encodeComputeRegretWeightsAggregateParallelRange(
          encoder,
          data.tree,
          data.beliefs,
          data.regretWeightAggregates,
          data.regretWeights,
          start,
          end,
        ),
      validate: async () => {
        const reference = makeEmptyStorageBuffer(device, data.policyCount);
        const candidate = makeEmptyStorageBuffer(device, data.policyCount);
        const referenceAggregates = makeEmptyStorageBuffer(device, data.nodeCount * 53);
        const candidateAggregates = makeEmptyStorageBuffer(device, data.nodeCount * 53);
        await runOnce(device, (encoder) =>
          kernels.encodeComputeRegretWeightsAggregateRange(
            encoder,
            data.tree,
            data.beliefs,
            referenceAggregates,
            reference,
            start,
            end,
          ),
        );
        await runOnce(device, (encoder) =>
          kernels.encodeComputeRegretWeightsAggregateParallelRange(
            encoder,
            data.tree,
            data.beliefs,
            candidateAggregates,
            candidate,
            start,
            end,
          ),
        );
        return validatePair(
          reference,
          candidate,
          data.policyCount,
          "regret_weight_aggregate",
        );
      },
    },
    {
      name: "accumulate_regrets",
      output: data.regrets,
      outputElements: data.policyCount,
      encode: (encoder) =>
        kernels.encodeAccumulateRegretsRange(
          encoder,
          data.tree,
          data.regretWeights,
          data.values,
          data.regrets,
          start,
          end,
        ),
    },
    {
      name: "belief_propagate_leaf_depth_split",
      output: data.beliefs,
      outputElements: data.valueCount,
      encode: (encoder) =>
        kernels.encodePropagateBeliefsDepthSplit(
          encoder,
          data.tree,
          data.policy,
          data.beliefs,
          data.denom,
          leafStart,
          leafEnd,
        ),
    },
    {
      name: "belief_propagate_leaf_depth_fused",
      output: data.beliefs,
      outputElements: data.valueCount,
      encode: (encoder) =>
        kernels.encodePropagateBeliefsDepthFused(
          encoder,
          data.tree,
          data.policy,
          data.beliefs,
          data.denom,
          leafStart,
          leafEnd,
        ),
      validate: async () => {
        const splitBeliefs = makeStorageBuffer(device, normalizeBeliefs(data.treeData));
        const fusedBeliefs = makeStorageBuffer(device, normalizeBeliefs(data.treeData));
        const splitDenom = makeEmptyStorageBuffer(device, data.nodeCount * 2);
        const fusedDenom = makeEmptyStorageBuffer(device, data.nodeCount * 2);
        await runOnce(device, (encoder) =>
          kernels.encodePropagateBeliefsDepthSplit(
            encoder,
            data.tree,
            data.policy,
            splitBeliefs,
            splitDenom,
            leafStart,
            leafEnd,
          ),
        );
        await runOnce(device, (encoder) =>
          kernels.encodePropagateBeliefsDepthFused(
            encoder,
            data.tree,
            data.policy,
            fusedBeliefs,
            fusedDenom,
            leafStart,
            leafEnd,
          ),
        );
        return validatePair(
          splitBeliefs,
          fusedBeliefs,
          data.valueCount,
          "belief_propagate_leaf_depth_split",
        );
      },
    },
    {
      name: "reach_propagate_leaf_depth",
      output: data.reach,
      outputElements: data.valueCount,
      encode: (encoder) =>
        kernels.encodePropagateReachDepth(
          encoder,
          data.tree,
          data.policy,
          data.reach,
          leafStart,
          leafEnd,
        ),
    },
    {
      name: "average_policy",
      output: data.policyAvg,
      outputElements: data.policyCount,
      encode: (encoder) =>
        kernels.encodeUpdateAveragePolicyRange(
          encoder,
          data.tree,
          data.reach,
          data.policy,
          data.avgNumerator,
          data.avgDenominator,
          data.policyAvg,
          start,
          end,
        ),
    },
    {
      name: "opponent_policy_direct",
      output: data.opponentPolicyDirect,
      outputElements: data.policyCount,
      encode: (encoder) =>
        kernels.encodeComputeOpponentPolicyRange(
          encoder,
          data.tree,
          data.beliefs,
          data.policy,
          data.opponentPolicyDirect,
          start,
          end,
        ),
    },
    {
      name: "opponent_policy_aggregate",
      output: data.opponentPolicy,
      outputElements: data.policyCount,
      encode: (encoder) =>
        kernels.encodeComputeOpponentPolicyAggregateRange(
          encoder,
          data.tree,
          data.beliefs,
          data.policy,
          data.opponentAggregates,
          data.opponentPolicy,
          start,
          end,
        ),
      validate: async () => {
        await runOnce(device, (encoder) =>
          kernels.encodeComputeOpponentPolicyRange(
            encoder,
            data.tree,
            data.beliefs,
            data.policy,
            data.opponentPolicyDirect,
            start,
            end,
          ),
        );
        await runOnce(device, (encoder) =>
          kernels.encodeComputeOpponentPolicyAggregateRange(
            encoder,
            data.tree,
            data.beliefs,
            data.policy,
            data.opponentAggregates,
            data.opponentPolicy,
            start,
            end,
          ),
        );
        return validatePair(
          data.opponentPolicyDirect,
          data.opponentPolicy,
          data.policyCount,
          "opponent_policy_direct",
        );
      },
    },
    {
      name: "backup_depth",
      output: data.values,
      outputElements: data.valueCount,
      encode: (encoder) =>
        kernels.encodeBackupDepth(
          encoder,
          data.tree,
          data.policy,
          data.opponentPolicy,
          data.values,
          lastParentStart,
          lastParentEnd,
        ),
    },
    {
      name: "gather_node_beliefs",
      output: data.gatheredBeliefs,
      outputElements: data.leafBatch * 2 * NUM_HANDS,
      encode: (encoder) =>
        kernels.encodeGatherNodeBeliefs(
          encoder,
          data.tree,
          data.nodeIndices,
          data.beliefs,
          data.gatheredBeliefs,
          data.leafBatch,
        ),
    },
    {
      name: "scatter_node_values",
      output: data.values,
      outputElements: data.valueCount,
      encode: (encoder) =>
        kernels.encodeScatterNodeValues(
          encoder,
          data.tree,
          data.nodeIndices,
          data.leafValues,
          data.values,
          data.leafBatch,
        ),
    },
    {
      name: "showdown_rank_mass",
      output: data.rankMass,
      outputElements: data.leafBatch * 2 * data.maxRanks,
      encode: (encoder) =>
        kernels.encodeShowdownRankMass(
          encoder,
          data.tree,
          data.nodeIndices,
          data.rankOrdinals,
          data.rankCounts,
          data.beliefs,
          data.rankMass,
          data.leafBatch,
          data.maxRanks,
        ),
    },
    {
      name: "showdown_rank_mass_by_hands",
      output: data.rankMass,
      outputElements: data.leafBatch * 2 * data.maxRanks,
      encode: (encoder) =>
        kernels.encodeShowdownRankMassByHands(
          encoder,
          data.tree,
          data.nodeIndices,
          data.rankHandOffsets,
          data.rankHandCounts,
          data.rankHands,
          data.rankCounts,
          data.beliefs,
          data.rankMass,
          data.leafBatch,
          data.maxRanks,
        ),
      validate: async () => {
        const reference = makeEmptyStorageBuffer(device, data.leafBatch * 2 * data.maxRanks);
        const candidate = makeEmptyStorageBuffer(device, data.leafBatch * 2 * data.maxRanks);
        await runOnce(device, (encoder) =>
          kernels.encodeShowdownRankMass(
            encoder,
            data.tree,
            data.nodeIndices,
            data.rankOrdinals,
            data.rankCounts,
            data.beliefs,
            reference,
            data.leafBatch,
            data.maxRanks,
          ),
        );
        await runOnce(device, (encoder) =>
          kernels.encodeShowdownRankMassByHands(
            encoder,
            data.tree,
            data.nodeIndices,
            data.rankHandOffsets,
            data.rankHandCounts,
            data.rankHands,
            data.rankCounts,
            data.beliefs,
            candidate,
            data.leafBatch,
            data.maxRanks,
          ),
        );
        return validatePair(
          reference,
          candidate,
          data.leafBatch * 2 * data.maxRanks,
          "showdown_rank_mass",
        );
      },
    },
    {
      name: "showdown_rank_prefix",
      output: data.rankPrefixLess,
      outputElements: data.leafBatch * 2 * data.maxRanks,
      encode: (encoder) =>
        kernels.encodeShowdownRankPrefix(
          encoder,
          data.rankCounts,
          data.rankMass,
          data.rankPrefixLess,
          data.rankTotal,
          data.leafBatch,
          data.maxRanks,
        ),
    },
    {
      name: "showdown_values_from_rank_aggregates",
      output: data.values,
      outputElements: data.valueCount,
      encode: (encoder) =>
        kernels.encodeShowdownValuesFromRanks(
          encoder,
          data.tree,
          data.nodeIndices,
          data.rankOrdinals,
          data.payoffs,
          data.beliefs,
          data.values,
          data.rankMass,
          data.rankPrefixLess,
          data.rankTotal,
          data.leafBatch,
          data.maxRanks,
        ),
    },
    {
      name: "showdown_values_from_ranks",
      output: data.values,
      outputElements: data.valueCount,
      encode: (encoder) =>
        kernels.encodeShowdownValuesFromRankAggregates(
          encoder,
          data.tree,
          data.nodeIndices,
          data.rankOrdinals,
          data.rankCounts,
          data.payoffs,
          data.beliefs,
          data.values,
          data.rankMass,
          data.rankPrefixLess,
          data.rankTotal,
          data.leafBatch,
          data.maxRanks,
        ),
    },
    {
      name: "showdown_values_from_ranks_by_hands",
      output: data.values,
      outputElements: data.valueCount,
      encode: (encoder) =>
        kernels.encodeShowdownValuesFromRankAggregates(
          encoder,
          data.tree,
          data.nodeIndices,
          data.rankOrdinals,
          data.rankCounts,
          data.payoffs,
          data.beliefs,
          data.values,
          data.rankMass,
          data.rankPrefixLess,
          data.rankTotal,
          data.leafBatch,
          data.maxRanks,
          data.rankHandOffsets,
          data.rankHandCounts,
          data.rankHands,
        ),
      validate: async () => {
        const reference = makeStorageBuffer(device, fillFloat(data.valueCount, 0.002, 4));
        const candidate = makeStorageBuffer(device, fillFloat(data.valueCount, 0.002, 4));
        const referenceRankMass = makeEmptyStorageBuffer(
          device,
          data.leafBatch * 2 * data.maxRanks,
        );
        const referenceRankPrefix = makeEmptyStorageBuffer(
          device,
          data.leafBatch * 2 * data.maxRanks,
        );
        const referenceRankTotal = makeEmptyStorageBuffer(device, data.leafBatch * 2);
        const candidateRankMass = makeEmptyStorageBuffer(
          device,
          data.leafBatch * 2 * data.maxRanks,
        );
        const candidateRankPrefix = makeEmptyStorageBuffer(
          device,
          data.leafBatch * 2 * data.maxRanks,
        );
        const candidateRankTotal = makeEmptyStorageBuffer(device, data.leafBatch * 2);
        await runOnce(device, (encoder) =>
          kernels.encodeShowdownValuesFromRankAggregates(
            encoder,
            data.tree,
            data.nodeIndices,
            data.rankOrdinals,
            data.rankCounts,
            data.payoffs,
            data.beliefs,
            reference,
            referenceRankMass,
            referenceRankPrefix,
            referenceRankTotal,
            data.leafBatch,
            data.maxRanks,
          ),
        );
        await runOnce(device, (encoder) =>
          kernels.encodeShowdownValuesFromRankAggregates(
            encoder,
            data.tree,
            data.nodeIndices,
            data.rankOrdinals,
            data.rankCounts,
            data.payoffs,
            data.beliefs,
            candidate,
            candidateRankMass,
            candidateRankPrefix,
            candidateRankTotal,
            data.leafBatch,
            data.maxRanks,
            data.rankHandOffsets,
            data.rankHandCounts,
            data.rankHands,
          ),
        );
        return validatePair(
          reference,
          candidate,
          data.valueCount,
          "showdown_values_from_ranks",
        );
      },
    },
    {
      name: "allin_table_values",
      output: data.values,
      outputElements: data.valueCount,
      encode: (encoder) =>
        kernels.encodeAllInTableValues(
          encoder,
          data.tree,
          data.nodeIndices,
          data.tablePacked,
          data.comboPerms,
          data.scaleFactors,
          data.beliefs,
          data.values,
          Math.min(4, data.leafBatch),
          data.tableScale,
          0,
          false,
        ),
    },
  ];
}

async function main(): Promise<void> {
  const options = readOptions();
  const device = await createDawnDevice();
  const kernels = new SparseCfrGpuKernels(device);
  const data = makeBenchData(device);
  const ops = buildOps(device, kernels, data).filter((op) =>
    options.variant ? op.name === options.variant : true,
  );
  if (ops.length === 0) {
    throw new Error(`No sparse CFR operation matched ${options.variant}`);
  }
  const results = [];
  for (const op of ops) {
    const validation = options.validate && op.validate ? await op.validate() : undefined;
    const samples = await timeOp(
      device,
      op,
      options.warmups,
      options.runs,
      options.dispatches,
    );
    results.push({
      name: op.name,
      outputElements: op.outputElements,
      ...(validation ? { validation } : {}),
      ...stats(samples),
      samplesMs: samples,
    });
  }
  console.log(
    JSON.stringify(
      {
        backend: process.env.WEBGPU_BACKEND ?? "auto",
        nodeCount: data.nodeCount,
        numHands: NUM_HANDS,
        leafBatch: data.leafBatch,
        warmups: options.warmups,
        runs: options.runs,
        dispatches: options.dispatches,
        results,
      },
      null,
      2,
    ),
  );
}

await main();
