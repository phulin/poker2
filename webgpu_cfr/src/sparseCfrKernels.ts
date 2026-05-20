import {
  makeStorageBuffer,
  makeUniformBuffer,
} from "./gpuBuffers.js";

export const SPARSE_REGRET_MATCH_WGSL = /* wgsl */ `
struct Params {
  nodeCount: u32,
  numHands: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> childOffsets: array<u32>;
@group(0) @binding(1) var<storage, read> childCount: array<u32>;
@group(0) @binding(2) var<storage, read> childIndices: array<u32>;
@group(0) @binding(3) var<storage, read> regrets: array<f32>;
@group(0) @binding(4) var<storage, read_write> policy: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let linear = gid.x;
  let total = params.nodeCount * params.numHands;
  if (linear >= total) {
    return;
  }
  let node = linear / params.numHands;
  let hand = linear - node * params.numHands;
  let count = childCount[node];
  if (count == 0u) {
    return;
  }

  let offset = childOffsets[node];
  var positiveSum = 0.0;
  for (var i = 0u; i < count; i = i + 1u) {
    let child = childIndices[offset + i];
    positiveSum = positiveSum + max(regrets[child * params.numHands + hand], 0.0);
  }

  let uniform = 1.0 / f32(count);
  for (var i = 0u; i < count; i = i + 1u) {
    let child = childIndices[offset + i];
    let idx = child * params.numHands + hand;
    if (positiveSum > 1.0e-8) {
      policy[idx] = max(regrets[idx], 0.0) / positiveSum;
    } else {
      policy[idx] = uniform;
    }
  }
}
`;

export const SPARSE_BELIEF_APPLY_WGSL = /* wgsl */ `
struct Params {
  numHands: u32,
  start: u32,
  end: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> parentIndex: array<u32>;
@group(0) @binding(1) var<storage, read> prevActor: array<u32>;
@group(0) @binding(2) var<storage, read> allowedMask: array<u32>;
@group(0) @binding(3) var<storage, read> policy: array<f32>;
@group(0) @binding(4) var<storage, read_write> beliefs: array<f32>;
@group(0) @binding(5) var<storage, read_write> denom: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let child = params.start + wid.x;
  let player = wid.y;
  let lane = lid.x;
  if (child >= params.end || player >= 2u) {
    return;
  }

  let parent = parentIndex[child];
  let actor = prevActor[child];
  var sum = 0.0;
  for (var hand = lane; hand < params.numHands; hand = hand + 256u) {
    let allowed = allowedMask[child * params.numHands + hand] != 0u;
    let parentIdx = (parent * 2u + player) * params.numHands + hand;
    let childIdx = (child * 2u + player) * params.numHands + hand;
    var value = beliefs[parentIdx];
    if (player == actor) {
      value = value * policy[child * params.numHands + hand];
    }
    if (!allowed) {
      value = 0.0;
    }
    beliefs[childIdx] = value;
    sum = sum + value;
  }

  partial[lane] = sum;
  workgroupBarrier();

  var stride = 128u;
  loop {
    if (lane < stride) {
      partial[lane] = partial[lane] + partial[lane + stride];
    }
    workgroupBarrier();
    if (stride == 1u) {
      break;
    }
    stride = stride / 2u;
  }

  if (lane == 0u) {
    denom[child * 2u + player] = partial[0];
  }
}
`;

export const SPARSE_BELIEF_NORMALIZE_WGSL = /* wgsl */ `
struct Params {
  numHands: u32,
  start: u32,
  end: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> allowedProb: array<f32>;
@group(0) @binding(1) var<storage, read> denom: array<f32>;
@group(0) @binding(2) var<storage, read_write> beliefs: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let linear = gid.x;
  let nodeCount = params.end - params.start;
  let total = nodeCount * 2u * params.numHands;
  if (linear >= total) {
    return;
  }

  let hand = linear % params.numHands;
  let player = (linear / params.numHands) % 2u;
  let localNode = linear / (2u * params.numHands);
  let node = params.start + localNode;
  let beliefIdx = (node * 2u + player) * params.numHands + hand;
  let d = denom[node * 2u + player];
  if (d > 1.0e-8) {
    beliefs[beliefIdx] = beliefs[beliefIdx] / d;
  } else {
    beliefs[beliefIdx] = allowedProb[node * params.numHands + hand];
  }
}
`;

export const SPARSE_REACH_APPLY_WGSL = /* wgsl */ `
struct Params {
  numHands: u32,
  start: u32,
  end: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> parentIndex: array<u32>;
@group(0) @binding(1) var<storage, read> prevActor: array<u32>;
@group(0) @binding(2) var<storage, read> allowedMask: array<u32>;
@group(0) @binding(3) var<storage, read> policy: array<f32>;
@group(0) @binding(4) var<storage, read_write> reach: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let linear = gid.x;
  let nodeCount = params.end - params.start;
  let total = nodeCount * 2u * params.numHands;
  if (linear >= total) {
    return;
  }

  let hand = linear % params.numHands;
  let player = (linear / params.numHands) % 2u;
  let localNode = linear / (2u * params.numHands);
  let child = params.start + localNode;
  let parent = parentIndex[child];
  let actor = prevActor[child];
  var value = reach[(parent * 2u + player) * params.numHands + hand];
  if (player == actor) {
    value = value * policy[child * params.numHands + hand];
  }
  if (allowedMask[child * params.numHands + hand] == 0u) {
    value = 0.0;
  }
  reach[(child * 2u + player) * params.numHands + hand] = value;
}
`;

export const SPARSE_AVERAGE_POLICY_WGSL = /* wgsl */ `
struct Params {
  numHands: u32,
  start: u32,
  end: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> parentIndex: array<u32>;
@group(0) @binding(1) var<storage, read> prevActor: array<u32>;
@group(0) @binding(2) var<storage, read> reach: array<f32>;
@group(0) @binding(3) var<storage, read> policy: array<f32>;
@group(0) @binding(4) var<storage, read_write> numerator: array<f32>;
@group(0) @binding(5) var<storage, read_write> denominator: array<f32>;
@group(0) @binding(6) var<storage, read_write> policyAvg: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let linear = gid.x;
  let nodeCount = params.end - params.start;
  let total = nodeCount * params.numHands;
  if (linear >= total) {
    return;
  }
  let hand = linear % params.numHands;
  let child = params.start + linear / params.numHands;
  let parent = parentIndex[child];
  let actor = prevActor[child];
  let weight = reach[(parent * 2u + actor) * params.numHands + hand];
  let idx = child * params.numHands + hand;
  numerator[idx] = numerator[idx] + weight * policy[idx];
  denominator[idx] = denominator[idx] + weight;
  if (denominator[idx] > 1.0e-8) {
    policyAvg[idx] = numerator[idx] / denominator[idx];
  } else {
    policyAvg[idx] = policy[idx];
  }
}
`;

export const SPARSE_BACKUP_DEPTH_WGSL = /* wgsl */ `
struct Params {
  numHands: u32,
  start: u32,
  end: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> childOffsets: array<u32>;
@group(0) @binding(1) var<storage, read> childCount: array<u32>;
@group(0) @binding(2) var<storage, read> childIndices: array<u32>;
@group(0) @binding(3) var<storage, read> toAct: array<u32>;
@group(0) @binding(4) var<storage, read> policy: array<f32>;
@group(0) @binding(5) var<storage, read> opponentPolicy: array<f32>;
@group(0) @binding(6) var<storage, read_write> values: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let linear = gid.x;
  let nodeCount = params.end - params.start;
  let total = nodeCount * 2u * params.numHands;
  if (linear >= total) {
    return;
  }

  let hand = linear % params.numHands;
  let player = (linear / params.numHands) % 2u;
  let parent = params.start + linear / (2u * params.numHands);
  let count = childCount[parent];
  if (count == 0u) {
    return;
  }

  let actor = toAct[parent];
  let offset = childOffsets[parent];
  var sum = 0.0;
  for (var i = 0u; i < count; i = i + 1u) {
    let child = childIndices[offset + i];
    let weight = select(
      opponentPolicy[child * params.numHands + hand],
      policy[child * params.numHands + hand],
      player == actor,
    );
    sum = sum + weight * values[(child * 2u + player) * params.numHands + hand];
  }
  values[(parent * 2u + player) * params.numHands + hand] = sum;
}
`;

export const SPARSE_REGRET_TAIL_WGSL = /* wgsl */ `
struct Params {
  numHands: u32,
  start: u32,
  end: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> parentIndex: array<u32>;
@group(0) @binding(1) var<storage, read> prevActor: array<u32>;
@group(0) @binding(2) var<storage, read> regretWeights: array<f32>;
@group(0) @binding(3) var<storage, read> values: array<f32>;
@group(0) @binding(4) var<storage, read_write> regrets: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let linear = gid.x;
  let nodeCount = params.end - params.start;
  let total = nodeCount * params.numHands;
  if (linear >= total) {
    return;
  }

  let hand = linear % params.numHands;
  let child = params.start + linear / params.numHands;
  let parent = parentIndex[child];
  let actor = prevActor[child];
  let weight = regretWeights[parent * params.numHands + hand];
  let childValue = values[(child * 2u + actor) * params.numHands + hand];
  let parentValue = values[(parent * 2u + actor) * params.numHands + hand];
  let idx = child * params.numHands + hand;
  regrets[idx] = regrets[idx] + weight * (childValue - parentValue);
}
`;

export const SPARSE_OPPONENT_POLICY_WGSL = /* wgsl */ `
struct Params {
  numHands: u32,
  start: u32,
  end: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> parentIndex: array<u32>;
@group(0) @binding(1) var<storage, read> prevActor: array<u32>;
@group(0) @binding(2) var<storage, read> handCard0: array<u32>;
@group(0) @binding(3) var<storage, read> handCard1: array<u32>;
@group(0) @binding(4) var<storage, read> beliefs: array<f32>;
@group(0) @binding(5) var<storage, read> policy: array<f32>;
@group(0) @binding(6) var<storage, read_write> opponentPolicy: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

fn overlaps(a: u32, b: u32) -> bool {
  return handCard0[a] == handCard0[b] ||
    handCard0[a] == handCard1[b] ||
    handCard1[a] == handCard0[b] ||
    handCard1[a] == handCard1[b];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let linear = gid.x;
  let nodeCount = params.end - params.start;
  let total = nodeCount * params.numHands;
  if (linear >= total) {
    return;
  }

  let hand = linear % params.numHands;
  let child = params.start + linear / params.numHands;
  let parent = parentIndex[child];
  let actor = prevActor[child];
  let beliefBase = (parent * 2u + actor) * params.numHands;
  let policyBase = child * params.numHands;
  var denom = 0.0;
  var numer = 0.0;
  for (var other = 0u; other < params.numHands; other = other + 1u) {
    if (!overlaps(hand, other)) {
      let belief = beliefs[beliefBase + other];
      denom = denom + belief;
      numer = numer + belief * policy[policyBase + other];
    }
  }
  if (denom > 1.0e-8) {
    opponentPolicy[policyBase + hand] = numer / denom;
  } else {
    opponentPolicy[policyBase + hand] = 0.0;
  }
}
`;

export const SPARSE_REGRET_WEIGHT_WGSL = /* wgsl */ `
struct Params {
  numHands: u32,
  start: u32,
  end: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> toAct: array<u32>;
@group(0) @binding(1) var<storage, read> allowedMask: array<u32>;
@group(0) @binding(2) var<storage, read> handCard0: array<u32>;
@group(0) @binding(3) var<storage, read> handCard1: array<u32>;
@group(0) @binding(4) var<storage, read> beliefs: array<f32>;
@group(0) @binding(5) var<storage, read_write> regretWeights: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

fn overlaps(a: u32, b: u32) -> bool {
  return handCard0[a] == handCard0[b] ||
    handCard0[a] == handCard1[b] ||
    handCard1[a] == handCard0[b] ||
    handCard1[a] == handCard1[b];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let linear = gid.x;
  let nodeCount = params.end - params.start;
  let total = nodeCount * params.numHands;
  if (linear >= total) {
    return;
  }

  let hand = linear % params.numHands;
  let node = params.start + linear / params.numHands;
  let actor = toAct[node];
  let opp = 1u - actor;
  let beliefBase = (node * 2u + opp) * params.numHands;
  var weight = 0.0;
  if (allowedMask[node * params.numHands + hand] != 0u) {
    for (var other = 0u; other < params.numHands; other = other + 1u) {
      if (!overlaps(hand, other)) {
        weight = weight + beliefs[beliefBase + other];
      }
    }
  }
  regretWeights[node * params.numHands + hand] = weight;
}
`;

export const SPARSE_GATHER_NODE_BELIEFS_WGSL = /* wgsl */ `
struct Params {
  numHands: u32,
  batch: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> nodeIndices: array<u32>;
@group(0) @binding(1) var<storage, read> beliefs: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let linear = gid.x;
  let total = params.batch * 2u * params.numHands;
  if (linear >= total) {
    return;
  }
  let hand = linear % params.numHands;
  let player = (linear / params.numHands) % 2u;
  let sample = linear / (2u * params.numHands);
  let node = nodeIndices[sample];
  out[linear] = beliefs[(node * 2u + player) * params.numHands + hand];
}
`;

export const SPARSE_SCATTER_NODE_VALUES_WGSL = /* wgsl */ `
struct Params {
  numHands: u32,
  batch: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> nodeIndices: array<u32>;
@group(0) @binding(1) var<storage, read> sourceValues: array<f32>;
@group(0) @binding(2) var<storage, read_write> values: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let linear = gid.x;
  let total = params.batch * 2u * params.numHands;
  if (linear >= total) {
    return;
  }
  let hand = linear % params.numHands;
  let player = (linear / params.numHands) % 2u;
  let sample = linear / (2u * params.numHands);
  let node = nodeIndices[sample];
  values[(node * 2u + player) * params.numHands + hand] = sourceValues[linear];
}
`;

export interface SparseGpuTreeData {
  nodeCount: number;
  numHands: number;
  childOffsets: Uint32Array<ArrayBufferLike>;
  childCount: Uint32Array<ArrayBufferLike>;
  childIndices: Uint32Array<ArrayBufferLike>;
  parentIndex: Uint32Array<ArrayBufferLike>;
  prevActor: Uint32Array<ArrayBufferLike>;
  toAct: Uint32Array<ArrayBufferLike>;
  allowedMask: Uint32Array<ArrayBufferLike>;
  allowedProb: Float32Array<ArrayBufferLike>;
  handCard0: Uint32Array<ArrayBufferLike>;
  handCard1: Uint32Array<ArrayBufferLike>;
}

export interface SparseGpuTreeBuffers {
  nodeCount: number;
  numHands: number;
  childOffsets: GPUBuffer;
  childCount: GPUBuffer;
  childIndices: GPUBuffer;
  parentIndex: GPUBuffer;
  prevActor: GPUBuffer;
  toAct: GPUBuffer;
  allowedMask: GPUBuffer;
  allowedProb: GPUBuffer;
  handCard0: GPUBuffer;
  handCard1: GPUBuffer;
  dispose: () => void;
}

export class SparseCfrGpuKernels {
  readonly device: GPUDevice;
  private readonly regretMatchPipeline: GPUComputePipeline;
  private readonly beliefApplyPipeline: GPUComputePipeline;
  private readonly beliefNormalizePipeline: GPUComputePipeline;
  private readonly reachApplyPipeline: GPUComputePipeline;
  private readonly averagePolicyPipeline: GPUComputePipeline;
  private readonly backupDepthPipeline: GPUComputePipeline;
  private readonly regretTailPipeline: GPUComputePipeline;
  private readonly opponentPolicyPipeline: GPUComputePipeline;
  private readonly regretWeightPipeline: GPUComputePipeline;
  private readonly gatherNodeBeliefsPipeline: GPUComputePipeline;
  private readonly scatterNodeValuesPipeline: GPUComputePipeline;

  constructor(device: GPUDevice) {
    this.device = device;
    this.regretMatchPipeline = this.pipeline(
      SPARSE_REGRET_MATCH_WGSL,
      "sparse-cfr-regret-match",
    );
    this.beliefApplyPipeline = this.pipeline(
      SPARSE_BELIEF_APPLY_WGSL,
      "sparse-cfr-belief-apply",
    );
    this.beliefNormalizePipeline = this.pipeline(
      SPARSE_BELIEF_NORMALIZE_WGSL,
      "sparse-cfr-belief-normalize",
    );
    this.reachApplyPipeline = this.pipeline(
      SPARSE_REACH_APPLY_WGSL,
      "sparse-cfr-reach-apply",
    );
    this.averagePolicyPipeline = this.pipeline(
      SPARSE_AVERAGE_POLICY_WGSL,
      "sparse-cfr-average-policy",
    );
    this.backupDepthPipeline = this.pipeline(
      SPARSE_BACKUP_DEPTH_WGSL,
      "sparse-cfr-backup-depth",
    );
    this.regretTailPipeline = this.pipeline(
      SPARSE_REGRET_TAIL_WGSL,
      "sparse-cfr-regret-tail",
    );
    this.opponentPolicyPipeline = this.pipeline(
      SPARSE_OPPONENT_POLICY_WGSL,
      "sparse-cfr-opponent-policy",
    );
    this.regretWeightPipeline = this.pipeline(
      SPARSE_REGRET_WEIGHT_WGSL,
      "sparse-cfr-regret-weight",
    );
    this.gatherNodeBeliefsPipeline = this.pipeline(
      SPARSE_GATHER_NODE_BELIEFS_WGSL,
      "sparse-cfr-gather-node-beliefs",
    );
    this.scatterNodeValuesPipeline = this.pipeline(
      SPARSE_SCATTER_NODE_VALUES_WGSL,
      "sparse-cfr-scatter-node-values",
    );
  }

  createTreeBuffers(data: SparseGpuTreeData): SparseGpuTreeBuffers {
    const buffers = [
      makeStorageBuffer(this.device, data.childOffsets),
      makeStorageBuffer(this.device, data.childCount),
      makeStorageBuffer(this.device, data.childIndices),
      makeStorageBuffer(this.device, data.parentIndex),
      makeStorageBuffer(this.device, data.prevActor),
      makeStorageBuffer(this.device, data.toAct),
      makeStorageBuffer(this.device, data.allowedMask),
      makeStorageBuffer(this.device, data.allowedProb),
      makeStorageBuffer(this.device, data.handCard0),
      makeStorageBuffer(this.device, data.handCard1),
    ] as const;
    return {
      nodeCount: data.nodeCount,
      numHands: data.numHands,
      childOffsets: buffers[0],
      childCount: buffers[1],
      childIndices: buffers[2],
      parentIndex: buffers[3],
      prevActor: buffers[4],
      toAct: buffers[5],
      allowedMask: buffers[6],
      allowedProb: buffers[7],
      handCard0: buffers[8],
      handCard1: buffers[9],
      dispose: () => {
        for (const buffer of buffers) {
          buffer.destroy();
        }
      },
    };
  }

  encodeRegretMatch(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    regrets: GPUBuffer,
    policy: GPUBuffer,
  ): GPUBuffer {
    const params = makeUniformBuffer(
      this.device,
      new Uint32Array([tree.nodeCount, tree.numHands, 0, 0]),
    );
    const bindGroup = this.device.createBindGroup({
      layout: this.regretMatchPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tree.childOffsets } },
        { binding: 1, resource: { buffer: tree.childCount } },
        { binding: 2, resource: { buffer: tree.childIndices } },
        { binding: 3, resource: { buffer: regrets } },
        { binding: 4, resource: { buffer: policy } },
        { binding: 5, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.regretMatchPipeline,
      bindGroup,
      Math.ceil((tree.nodeCount * tree.numHands) / 64),
    );
    return params;
  }

  encodePropagateBeliefsDepth(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    policy: GPUBuffer,
    beliefs: GPUBuffer,
    denom: GPUBuffer,
    start: number,
    end: number,
  ): GPUBuffer {
    const params = makeUniformBuffer(
      this.device,
      new Uint32Array([tree.numHands, start, end, 0]),
    );
    const applyBindGroup = this.device.createBindGroup({
      layout: this.beliefApplyPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tree.parentIndex } },
        { binding: 1, resource: { buffer: tree.prevActor } },
        { binding: 2, resource: { buffer: tree.allowedMask } },
        { binding: 3, resource: { buffer: policy } },
        { binding: 4, resource: { buffer: beliefs } },
        { binding: 5, resource: { buffer: denom } },
        { binding: 6, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.beliefApplyPipeline,
      applyBindGroup,
      Math.max(0, end - start),
      2,
    );

    const normalizeBindGroup = this.device.createBindGroup({
      layout: this.beliefNormalizePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tree.allowedProb } },
        { binding: 1, resource: { buffer: denom } },
        { binding: 2, resource: { buffer: beliefs } },
        { binding: 3, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.beliefNormalizePipeline,
      normalizeBindGroup,
      Math.ceil(((end - start) * 2 * tree.numHands) / 64),
    );
    return params;
  }

  encodePropagateReachDepth(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    policy: GPUBuffer,
    reach: GPUBuffer,
    start: number,
    end: number,
  ): GPUBuffer {
    const params = makeUniformBuffer(
      this.device,
      new Uint32Array([tree.numHands, start, end, 0]),
    );
    const bindGroup = this.device.createBindGroup({
      layout: this.reachApplyPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tree.parentIndex } },
        { binding: 1, resource: { buffer: tree.prevActor } },
        { binding: 2, resource: { buffer: tree.allowedMask } },
        { binding: 3, resource: { buffer: policy } },
        { binding: 4, resource: { buffer: reach } },
        { binding: 5, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.reachApplyPipeline,
      bindGroup,
      Math.ceil(((end - start) * 2 * tree.numHands) / 64),
    );
    return params;
  }

  encodeUpdateAveragePolicyRange(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    reach: GPUBuffer,
    policy: GPUBuffer,
    numerator: GPUBuffer,
    denominator: GPUBuffer,
    policyAvg: GPUBuffer,
    start: number,
    end: number,
  ): GPUBuffer {
    const params = makeUniformBuffer(
      this.device,
      new Uint32Array([tree.numHands, start, end, 0]),
    );
    const bindGroup = this.device.createBindGroup({
      layout: this.averagePolicyPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tree.parentIndex } },
        { binding: 1, resource: { buffer: tree.prevActor } },
        { binding: 2, resource: { buffer: reach } },
        { binding: 3, resource: { buffer: policy } },
        { binding: 4, resource: { buffer: numerator } },
        { binding: 5, resource: { buffer: denominator } },
        { binding: 6, resource: { buffer: policyAvg } },
        { binding: 7, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.averagePolicyPipeline,
      bindGroup,
      Math.ceil(((end - start) * tree.numHands) / 64),
    );
    return params;
  }

  encodeBackupDepth(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    policy: GPUBuffer,
    opponentPolicy: GPUBuffer,
    values: GPUBuffer,
    start: number,
    end: number,
  ): GPUBuffer {
    const params = makeUniformBuffer(
      this.device,
      new Uint32Array([tree.numHands, start, end, 0]),
    );
    const bindGroup = this.device.createBindGroup({
      layout: this.backupDepthPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tree.childOffsets } },
        { binding: 1, resource: { buffer: tree.childCount } },
        { binding: 2, resource: { buffer: tree.childIndices } },
        { binding: 3, resource: { buffer: tree.toAct } },
        { binding: 4, resource: { buffer: policy } },
        { binding: 5, resource: { buffer: opponentPolicy } },
        { binding: 6, resource: { buffer: values } },
        { binding: 7, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.backupDepthPipeline,
      bindGroup,
      Math.ceil(((end - start) * 2 * tree.numHands) / 64),
    );
    return params;
  }

  encodeAccumulateRegretsRange(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    regretWeights: GPUBuffer,
    values: GPUBuffer,
    regrets: GPUBuffer,
    start: number,
    end: number,
  ): GPUBuffer {
    const params = makeUniformBuffer(
      this.device,
      new Uint32Array([tree.numHands, start, end, 0]),
    );
    const bindGroup = this.device.createBindGroup({
      layout: this.regretTailPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tree.parentIndex } },
        { binding: 1, resource: { buffer: tree.prevActor } },
        { binding: 2, resource: { buffer: regretWeights } },
        { binding: 3, resource: { buffer: values } },
        { binding: 4, resource: { buffer: regrets } },
        { binding: 5, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.regretTailPipeline,
      bindGroup,
      Math.ceil(((end - start) * tree.numHands) / 64),
    );
    return params;
  }

  encodeComputeOpponentPolicyRange(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    beliefs: GPUBuffer,
    policy: GPUBuffer,
    opponentPolicy: GPUBuffer,
    start: number,
    end: number,
  ): GPUBuffer {
    const params = makeUniformBuffer(
      this.device,
      new Uint32Array([tree.numHands, start, end, 0]),
    );
    const bindGroup = this.device.createBindGroup({
      layout: this.opponentPolicyPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tree.parentIndex } },
        { binding: 1, resource: { buffer: tree.prevActor } },
        { binding: 2, resource: { buffer: tree.handCard0 } },
        { binding: 3, resource: { buffer: tree.handCard1 } },
        { binding: 4, resource: { buffer: beliefs } },
        { binding: 5, resource: { buffer: policy } },
        { binding: 6, resource: { buffer: opponentPolicy } },
        { binding: 7, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.opponentPolicyPipeline,
      bindGroup,
      Math.ceil(((end - start) * tree.numHands) / 64),
    );
    return params;
  }

  encodeComputeRegretWeightsRange(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    beliefs: GPUBuffer,
    regretWeights: GPUBuffer,
    start: number,
    end: number,
  ): GPUBuffer {
    const params = makeUniformBuffer(
      this.device,
      new Uint32Array([tree.numHands, start, end, 0]),
    );
    const bindGroup = this.device.createBindGroup({
      layout: this.regretWeightPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tree.toAct } },
        { binding: 1, resource: { buffer: tree.allowedMask } },
        { binding: 2, resource: { buffer: tree.handCard0 } },
        { binding: 3, resource: { buffer: tree.handCard1 } },
        { binding: 4, resource: { buffer: beliefs } },
        { binding: 5, resource: { buffer: regretWeights } },
        { binding: 6, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.regretWeightPipeline,
      bindGroup,
      Math.ceil(((end - start) * tree.numHands) / 64),
    );
    return params;
  }

  encodeGatherNodeBeliefs(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    nodeIndices: GPUBuffer,
    beliefs: GPUBuffer,
    out: GPUBuffer,
    batch: number,
  ): GPUBuffer {
    const params = makeUniformBuffer(
      this.device,
      new Uint32Array([tree.numHands, batch, 0, 0]),
    );
    const bindGroup = this.device.createBindGroup({
      layout: this.gatherNodeBeliefsPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: nodeIndices } },
        { binding: 1, resource: { buffer: beliefs } },
        { binding: 2, resource: { buffer: out } },
        { binding: 3, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.gatherNodeBeliefsPipeline,
      bindGroup,
      Math.ceil((batch * 2 * tree.numHands) / 64),
    );
    return params;
  }

  encodeScatterNodeValues(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    nodeIndices: GPUBuffer,
    sourceValues: GPUBuffer,
    values: GPUBuffer,
    batch: number,
  ): GPUBuffer {
    const params = makeUniformBuffer(
      this.device,
      new Uint32Array([tree.numHands, batch, 0, 0]),
    );
    const bindGroup = this.device.createBindGroup({
      layout: this.scatterNodeValuesPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: nodeIndices } },
        { binding: 1, resource: { buffer: sourceValues } },
        { binding: 2, resource: { buffer: values } },
        { binding: 3, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.scatterNodeValuesPipeline,
      bindGroup,
      Math.ceil((batch * 2 * tree.numHands) / 64),
    );
    return params;
  }

  private encode(
    encoder: GPUCommandEncoder,
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    x: number,
    y = 1,
  ): void {
    if (x <= 0 || y <= 0) return;
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(x, y);
    pass.end();
  }

  private pipeline(source: string, label: string): GPUComputePipeline {
    return this.device.createComputePipeline({
      label,
      layout: "auto",
      compute: {
        module: this.device.createShaderModule({ label: `${label}.wgsl`, code: source }),
        entryPoint: "main",
      },
    });
  }
}
