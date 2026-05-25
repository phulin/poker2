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

export const SPARSE_BELIEF_PROPAGATE_FUSED_WGSL = /* wgsl */ `
struct Params {
  numHands: u32,
  start: u32,
  end: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> parentIndex: array<u32>;
@group(0) @binding(1) var<storage, read> prevActor: array<u32>;
@group(0) @binding(2) var<storage, read> allowedMask: array<u32>;
@group(0) @binding(3) var<storage, read> allowedProb: array<f32>;
@group(0) @binding(4) var<storage, read> policy: array<f32>;
@group(0) @binding(5) var<storage, read_write> beliefs: array<f32>;
@group(0) @binding(6) var<storage, read_write> denom: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

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

  let d = partial[0];
  if (lane == 0u) {
    denom[child * 2u + player] = d;
  }

  for (var hand = lane; hand < params.numHands; hand = hand + 256u) {
    let childIdx = (child * 2u + player) * params.numHands + hand;
    let allowed = allowedMask[child * params.numHands + hand] != 0u;
    let parentIdx = (parent * 2u + player) * params.numHands + hand;
    var value = beliefs[parentIdx];
    if (player == actor) {
      value = value * policy[child * params.numHands + hand];
    }
    if (!allowed) {
      value = 0.0;
    }
    if (d > 1.0e-8) {
      beliefs[childIdx] = value / d;
    } else {
      beliefs[childIdx] = allowedProb[child * params.numHands + hand];
    }
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

@compute @workgroup_size(128)
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
