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

export const SPARSE_BACKUP_DEPTH_1326_WGSL = SPARSE_BACKUP_DEPTH_WGSL
  .replace("numHands: u32,", "_numHands: u32,")
  .replaceAll("nodeCount * 2u * params.numHands", "nodeCount * 2652u")
  .replaceAll("linear % params.numHands", "linear % 1326u")
  .replaceAll("(linear / params.numHands) % 2u", "(linear / 1326u) % 2u")
  .replaceAll("linear / (2u * params.numHands)", "linear / 2652u")
  .replaceAll("child * params.numHands + hand", "child * 1326u + hand")
  .replaceAll(
    "(child * 2u + player) * params.numHands + hand",
    "child * 2652u + player * 1326u + hand",
  )
  .replaceAll(
    "(parent * 2u + player) * params.numHands + hand",
    "parent * 2652u + player * 1326u + hand",
  );

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

@compute @workgroup_size(128)
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
  let hand0 = handCard0[hand];
  let hand1 = handCard1[hand];
  var denom = 0.0;
  var numer = 0.0;
  for (var other = 0u; other < params.numHands; other = other + 1u) {
    let other0 = handCard0[other];
    let other1 = handCard1[other];
    if (!(hand0 == other0 || hand0 == other1 || hand1 == other0 || hand1 == other1)) {
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

export const SPARSE_OPPONENT_POLICY_AGGREGATE_WGSL = /* wgsl */ `
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
@group(0) @binding(6) var<storage, read_write> aggregates: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let child = params.start + gid.x;
  if (child >= params.end) {
    return;
  }

  var denomCards: array<f32, 52>;
  var numerCards: array<f32, 52>;
  for (var card = 0u; card < 52u; card = card + 1u) {
    denomCards[card] = 0.0;
    numerCards[card] = 0.0;
  }

  let parent = parentIndex[child];
  let actor = prevActor[child];
  let beliefBase = (parent * 2u + actor) * params.numHands;
  let policyBase = child * params.numHands;
  var denomTotal = 0.0;
  var numerTotal = 0.0;
  for (var hand = 0u; hand < params.numHands; hand = hand + 1u) {
    let belief = beliefs[beliefBase + hand];
    let numer = belief * policy[policyBase + hand];
    denomTotal = denomTotal + belief;
    numerTotal = numerTotal + numer;
    let card0 = handCard0[hand];
    let card1 = handCard1[hand];
    denomCards[card0] = denomCards[card0] + belief;
    denomCards[card1] = denomCards[card1] + belief;
    numerCards[card0] = numerCards[card0] + numer;
    numerCards[card1] = numerCards[card1] + numer;
  }

  let outBase = child * 106u;
  aggregates[outBase] = denomTotal;
  for (var card = 0u; card < 52u; card = card + 1u) {
    aggregates[outBase + 1u + card] = denomCards[card];
  }
  aggregates[outBase + 53u] = numerTotal;
  for (var card = 0u; card < 52u; card = card + 1u) {
    aggregates[outBase + 54u + card] = numerCards[card];
  }
}
`;

export const SPARSE_OPPONENT_POLICY_AGGREGATE_PARALLEL_WGSL = /* wgsl */ `
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
@group(0) @binding(6) var<storage, read_write> aggregates: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

var<workgroup> partial: array<f32, 3392>;

@compute @workgroup_size(32)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let child = params.start + wid.x;
  let lane = lid.x;
  if (child >= params.end) {
    return;
  }

  let localBase = lane * 106u;
  for (var slot = 0u; slot < 106u; slot = slot + 1u) {
    partial[localBase + slot] = 0.0;
  }
  workgroupBarrier();

  let parent = parentIndex[child];
  let actor = prevActor[child];
  let beliefBase = (parent * 2u + actor) * params.numHands;
  let policyBase = child * params.numHands;
  for (var hand = lane; hand < params.numHands; hand = hand + 32u) {
    let belief = beliefs[beliefBase + hand];
    let numer = belief * policy[policyBase + hand];
    let card0 = handCard0[hand];
    let card1 = handCard1[hand];
    partial[localBase] = partial[localBase] + belief;
    partial[localBase + 1u + card0] = partial[localBase + 1u + card0] + belief;
    partial[localBase + 1u + card1] = partial[localBase + 1u + card1] + belief;
    partial[localBase + 53u] = partial[localBase + 53u] + numer;
    partial[localBase + 54u + card0] = partial[localBase + 54u + card0] + numer;
    partial[localBase + 54u + card1] = partial[localBase + 54u + card1] + numer;
  }
  workgroupBarrier();

  var stride = 16u;
  loop {
    if (lane < stride) {
      let otherBase = (lane + stride) * 106u;
      for (var slot = 0u; slot < 106u; slot = slot + 1u) {
        partial[localBase + slot] = partial[localBase + slot] + partial[otherBase + slot];
      }
    }
    workgroupBarrier();
    if (stride == 1u) {
      break;
    }
    stride = stride / 2u;
  }

  if (lane == 0u) {
    let outBase = child * 106u;
    for (var slot = 0u; slot < 106u; slot = slot + 1u) {
      aggregates[outBase + slot] = partial[slot];
    }
  }
}
`;

export const SPARSE_OPPONENT_POLICY_FROM_AGGREGATE_WGSL = /* wgsl */ `
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
@group(0) @binding(6) var<storage, read> aggregates: array<f32>;
@group(0) @binding(7) var<storage, read_write> opponentPolicy: array<f32>;
@group(0) @binding(8) var<uniform> params: Params;

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
  let aggregateBase = child * 106u;
  let hand0 = handCard0[hand];
  let hand1 = handCard1[hand];
  let belief = beliefs[beliefBase + hand];
  let weighted = belief * policy[policyBase + hand];
  let denom =
    aggregates[aggregateBase] -
    aggregates[aggregateBase + 1u + hand0] -
    aggregates[aggregateBase + 1u + hand1] +
    belief;
  let numer =
    aggregates[aggregateBase + 53u] -
    aggregates[aggregateBase + 54u + hand0] -
    aggregates[aggregateBase + 54u + hand1] +
    weighted;
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

@compute @workgroup_size(128)
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
  let hand0 = handCard0[hand];
  let hand1 = handCard1[hand];
  var weight = 0.0;
  if (allowedMask[node * params.numHands + hand] != 0u) {
    for (var other = 0u; other < params.numHands; other = other + 1u) {
      let other0 = handCard0[other];
      let other1 = handCard1[other];
      if (!(hand0 == other0 || hand0 == other1 || hand1 == other0 || hand1 == other1)) {
        weight = weight + beliefs[beliefBase + other];
      }
    }
  }
  regretWeights[node * params.numHands + hand] = weight;
}
`;

export const SPARSE_REGRET_WEIGHT_AGGREGATE_WGSL = /* wgsl */ `
struct Params {
  numHands: u32,
  start: u32,
  end: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> toAct: array<u32>;
@group(0) @binding(1) var<storage, read> handCard0: array<u32>;
@group(0) @binding(2) var<storage, read> handCard1: array<u32>;
@group(0) @binding(3) var<storage, read> beliefs: array<f32>;
@group(0) @binding(4) var<storage, read_write> aggregates: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let node = params.start + gid.x;
  if (node >= params.end) {
    return;
  }

  var cardSums: array<f32, 52>;
  for (var card = 0u; card < 52u; card = card + 1u) {
    cardSums[card] = 0.0;
  }

  let actor = toAct[node];
  let opp = 1u - actor;
  let beliefBase = (node * 2u + opp) * params.numHands;
  var total = 0.0;
  for (var hand = 0u; hand < params.numHands; hand = hand + 1u) {
    let value = beliefs[beliefBase + hand];
    total = total + value;
    cardSums[handCard0[hand]] = cardSums[handCard0[hand]] + value;
    cardSums[handCard1[hand]] = cardSums[handCard1[hand]] + value;
  }

  let outBase = node * 53u;
  aggregates[outBase] = total;
  for (var card = 0u; card < 52u; card = card + 1u) {
    aggregates[outBase + 1u + card] = cardSums[card];
  }
}
`;

export const SPARSE_REGRET_WEIGHT_AGGREGATE_PARALLEL_WGSL = /* wgsl */ `
struct Params {
  numHands: u32,
  start: u32,
  end: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> toAct: array<u32>;
@group(0) @binding(1) var<storage, read> handCard0: array<u32>;
@group(0) @binding(2) var<storage, read> handCard1: array<u32>;
@group(0) @binding(3) var<storage, read> beliefs: array<f32>;
@group(0) @binding(4) var<storage, read_write> aggregates: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> partial: array<f32, 3392>;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let node = params.start + wid.x;
  let lane = lid.x;
  if (node >= params.end) {
    return;
  }

  let localBase = lane * 53u;
  for (var slot = 0u; slot < 53u; slot = slot + 1u) {
    partial[localBase + slot] = 0.0;
  }
  workgroupBarrier();

  let actor = toAct[node];
  let opp = 1u - actor;
  let beliefBase = (node * 2u + opp) * params.numHands;
  for (var hand = lane; hand < params.numHands; hand = hand + 64u) {
    let value = beliefs[beliefBase + hand];
    partial[localBase] = partial[localBase] + value;
    partial[localBase + 1u + handCard0[hand]] =
      partial[localBase + 1u + handCard0[hand]] + value;
    partial[localBase + 1u + handCard1[hand]] =
      partial[localBase + 1u + handCard1[hand]] + value;
  }
  workgroupBarrier();

  var stride = 32u;
  loop {
    if (lane < stride) {
      let otherBase = (lane + stride) * 53u;
      for (var slot = 0u; slot < 53u; slot = slot + 1u) {
        partial[localBase + slot] = partial[localBase + slot] + partial[otherBase + slot];
      }
    }
    workgroupBarrier();
    if (stride == 1u) {
      break;
    }
    stride = stride / 2u;
  }

  if (lane == 0u) {
    let outBase = node * 53u;
    for (var slot = 0u; slot < 53u; slot = slot + 1u) {
      aggregates[outBase + slot] = partial[slot];
    }
  }
}
`;

export const SPARSE_REGRET_WEIGHT_FROM_AGGREGATE_WGSL = /* wgsl */ `
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
@group(0) @binding(5) var<storage, read> aggregates: array<f32>;
@group(0) @binding(6) var<storage, read_write> regretWeights: array<f32>;
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
  let node = params.start + linear / params.numHands;
  let outIdx = node * params.numHands + hand;
  var weight = 0.0;
  if (allowedMask[outIdx] != 0u) {
    let actor = toAct[node];
    let opp = 1u - actor;
    let beliefBase = (node * 2u + opp) * params.numHands;
    let aggregateBase = node * 53u;
    let hand0 = handCard0[hand];
    let hand1 = handCard1[hand];
    weight =
      aggregates[aggregateBase] -
      aggregates[aggregateBase + 1u + hand0] -
      aggregates[aggregateBase + 1u + hand1] +
      beliefs[beliefBase + hand];
  }
  regretWeights[outIdx] = weight;
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

export const SPARSE_SHOWDOWN_VALUES_WGSL = /* wgsl */ `
struct Params {
  numHands: u32,
  batch: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> nodeIndices: array<u32>;
@group(0) @binding(1) var<storage, read> allowedMask: array<u32>;
@group(0) @binding(2) var<storage, read> handCard0: array<u32>;
@group(0) @binding(3) var<storage, read> handCard1: array<u32>;
@group(0) @binding(4) var<storage, read> rankCodes: array<u32>;
@group(0) @binding(5) var<storage, read> payoffs: array<f32>;
@group(0) @binding(6) var<storage, read> beliefs: array<f32>;
@group(0) @binding(7) var<storage, read_write> values: array<f32>;
@group(0) @binding(8) var<uniform> params: Params;

fn overlaps(a: u32, b: u32) -> bool {
  return handCard0[a] == handCard0[b] ||
    handCard0[a] == handCard1[b] ||
    handCard1[a] == handCard0[b] ||
    handCard1[a] == handCard1[b];
}

@compute @workgroup_size(128)
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
  let valueIdx = (node * 2u + player) * params.numHands + hand;
  if (allowedMask[node * params.numHands + hand] == 0u) {
    values[valueIdx] = 0.0;
    return;
  }

  let rank = rankCodes[sample * params.numHands + hand];
  let winValue = payoffs[sample * 3u];
  let loseValue = payoffs[sample * 3u + 1u];
  let tieValue = payoffs[sample * 3u + 2u];
  let hand0 = handCard0[hand];
  let hand1 = handCard1[hand];
  var value = 0.0;
  var mass = 0.0;
  for (var opp = 0u; opp < params.numHands; opp = opp + 1u) {
    let opp0 = handCard0[opp];
    let opp1 = handCard1[opp];
    if (
      allowedMask[node * params.numHands + opp] == 0u ||
      hand0 == opp0 ||
      hand0 == opp1 ||
      hand1 == opp0 ||
      hand1 == opp1
    ) {
      continue;
    }
    let oppRank = rankCodes[sample * params.numHands + opp];
    if (player == 0u) {
      let belief = beliefs[(node * 2u + 1u) * params.numHands + opp];
      let payoff = select(select(tieValue, loseValue, rank < oppRank), winValue, rank > oppRank);
      value = value + belief * payoff;
      mass = mass + belief;
    } else {
      let belief = beliefs[node * 2u * params.numHands + opp];
      let p0Payoff = select(select(tieValue, loseValue, oppRank < rank), winValue, oppRank > rank);
      value = value + belief * -p0Payoff;
      mass = mass + belief;
    }
  }
  values[valueIdx] = select(0.0, value / mass, mass > 1.0e-12);
}
`;

export const SPARSE_SHOWDOWN_RANK_MASS_WGSL = /* wgsl */ `
struct Params {
  numHands: u32,
  batch: u32,
  maxRanks: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> nodeIndices: array<u32>;
@group(0) @binding(1) var<storage, read> allowedMask: array<u32>;
@group(0) @binding(2) var<storage, read> rankOrdinals: array<u32>;
@group(0) @binding(3) var<storage, read> rankCounts: array<u32>;
@group(0) @binding(4) var<storage, read> beliefs: array<f32>;
@group(0) @binding(5) var<storage, read_write> rankMass: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

var<workgroup> partial: array<f32, 128>;

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let rank = wid.x;
  let samplePlayer = wid.y;
  let lane = lid.x;
  if (rank >= params.maxRanks || samplePlayer >= params.batch * 2u) {
    return;
  }

  let sample = samplePlayer / 2u;
  let player = samplePlayer % 2u;
  let node = nodeIndices[sample];
  let rankCount = rankCounts[sample];
  var sum = 0.0;
  if (rank < rankCount) {
  for (var hand = lane; hand < params.numHands; hand = hand + 128u) {
    let handIdx = node * params.numHands + hand;
    if (
      allowedMask[handIdx] != 0u &&
      rankOrdinals[sample * params.numHands + hand] == rank
    ) {
      sum = sum + beliefs[(node * 2u + player) * params.numHands + hand];
    }
  }
  }

  partial[lane] = sum;
  workgroupBarrier();

  var stride = 64u;
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
    rankMass[samplePlayer * params.maxRanks + rank] = partial[0];
  }
}
`;

export const SPARSE_SHOWDOWN_RANK_PREFIX_WGSL = /* wgsl */ `
struct Params {
  numHands: u32,
  batch: u32,
  maxRanks: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> rankCounts: array<u32>;
@group(0) @binding(1) var<storage, read> rankMass: array<f32>;
@group(0) @binding(2) var<storage, read_write> rankPrefixLess: array<f32>;
@group(0) @binding(3) var<storage, read_write> rankTotal: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let samplePlayer = gid.x;
  if (samplePlayer >= params.batch * 2u) {
    return;
  }
  let sample = samplePlayer / 2u;
  let rankCount = rankCounts[sample];
  let base = samplePlayer * params.maxRanks;
  var sum = 0.0;
  for (var rank = 0u; rank < params.maxRanks; rank = rank + 1u) {
    rankPrefixLess[base + rank] = sum;
    if (rank < rankCount) {
      sum = sum + rankMass[base + rank];
    }
  }
  rankTotal[samplePlayer] = sum;
}
`;

export const SPARSE_SHOWDOWN_RANK_MASS_BY_HANDS_WGSL = /* wgsl */ `
struct Params {
  numHands: u32,
  batch: u32,
  maxRanks: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> nodeIndices: array<u32>;
@group(0) @binding(1) var<storage, read> allowedMask: array<u32>;
@group(0) @binding(2) var<storage, read> rankHandOffsets: array<u32>;
@group(0) @binding(3) var<storage, read> rankHandCounts: array<u32>;
@group(0) @binding(4) var<storage, read> rankHands: array<u32>;
@group(0) @binding(5) var<storage, read> rankCounts: array<u32>;
@group(0) @binding(6) var<storage, read> beliefs: array<f32>;
@group(0) @binding(7) var<storage, read_write> rankMass: array<f32>;
@group(0) @binding(8) var<uniform> params: Params;

var<workgroup> partial: array<f32, 128>;

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let rank = wid.x;
  let samplePlayer = wid.y;
  let lane = lid.x;
  if (rank >= params.maxRanks || samplePlayer >= params.batch * 2u) {
    return;
  }

  let sample = samplePlayer / 2u;
  let player = samplePlayer % 2u;
  let node = nodeIndices[sample];
  let rankCount = rankCounts[sample];
  var sum = 0.0;
  if (rank < rankCount) {
    let rankIdx = sample * params.maxRanks + rank;
    let offset = rankHandOffsets[rankIdx];
    let count = rankHandCounts[rankIdx];
    for (var i = lane; i < count; i = i + 128u) {
      let hand = rankHands[offset + i];
      if (allowedMask[node * params.numHands + hand] != 0u) {
        sum = sum + beliefs[(node * 2u + player) * params.numHands + hand];
      }
    }
  }

  partial[lane] = sum;
  workgroupBarrier();

  var stride = 64u;
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
    rankMass[samplePlayer * params.maxRanks + rank] = partial[0];
  }
}
`;

export const SPARSE_SHOWDOWN_RANK_MASS_BY_HANDS_BOTH_PLAYERS_WGSL = /* wgsl */ `
struct Params {
  numHands: u32,
  batch: u32,
  maxRanks: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> nodeIndices: array<u32>;
@group(0) @binding(1) var<storage, read> allowedMask: array<u32>;
@group(0) @binding(2) var<storage, read> rankHandOffsets: array<u32>;
@group(0) @binding(3) var<storage, read> rankHandCounts: array<u32>;
@group(0) @binding(4) var<storage, read> rankHands: array<u32>;
@group(0) @binding(5) var<storage, read> rankCounts: array<u32>;
@group(0) @binding(6) var<storage, read> beliefs: array<f32>;
@group(0) @binding(7) var<storage, read_write> rankMass: array<f32>;
@group(0) @binding(8) var<uniform> params: Params;

var<workgroup> partial0: array<f32, 128>;
var<workgroup> partial1: array<f32, 128>;

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let rank = wid.x;
  let sample = wid.y;
  let lane = lid.x;
  if (rank >= params.maxRanks || sample >= params.batch) {
    return;
  }

  let node = nodeIndices[sample];
  let rankCount = rankCounts[sample];
  var sum0 = 0.0;
  var sum1 = 0.0;
  if (rank < rankCount) {
    let rankIdx = sample * params.maxRanks + rank;
    let offset = rankHandOffsets[rankIdx];
    let count = rankHandCounts[rankIdx];
    let allowedBase = node * params.numHands;
    let beliefBase = node * 2u * params.numHands;
    for (var i = lane; i < count; i = i + 128u) {
      let hand = rankHands[offset + i];
      if (allowedMask[allowedBase + hand] != 0u) {
        sum0 = sum0 + beliefs[beliefBase + hand];
        sum1 = sum1 + beliefs[beliefBase + params.numHands + hand];
      }
    }
  }

  partial0[lane] = sum0;
  partial1[lane] = sum1;
  workgroupBarrier();

  var stride = 64u;
  loop {
    if (lane < stride) {
      partial0[lane] = partial0[lane] + partial0[lane + stride];
      partial1[lane] = partial1[lane] + partial1[lane + stride];
    }
    workgroupBarrier();
    if (stride == 1u) {
      break;
    }
    stride = stride / 2u;
  }

  if (lane == 0u) {
    let outBase = sample * 2u * params.maxRanks + rank;
    rankMass[outBase] = partial0[0];
    rankMass[outBase + params.maxRanks] = partial1[0];
  }
}
`;

export const SPARSE_SHOWDOWN_VALUES_FROM_RANKS_WGSL = /* wgsl */ `
struct Params {
  numHands: u32,
  batch: u32,
  maxRanks: u32,
  overlapSlots: u32,
};

@group(0) @binding(0) var<storage, read> nodeIndices: array<u32>;
@group(0) @binding(1) var<storage, read> rankOrdinals: array<u32>;
@group(0) @binding(2) var<storage, read> overlapHands: array<u32>;
@group(0) @binding(3) var<storage, read> overlapCounts: array<u32>;
@group(0) @binding(4) var<storage, read> payoffs: array<f32>;
@group(0) @binding(5) var<storage, read> beliefs: array<f32>;
@group(0) @binding(6) var<storage, read> rankMass: array<f32>;
@group(0) @binding(7) var<storage, read> rankPrefixLess: array<f32>;
@group(0) @binding(8) var<storage, read> rankTotal: array<f32>;
@group(0) @binding(9) var<storage, read_write> values: array<f32>;
@group(0) @binding(10) var<uniform> params: Params;

@compute @workgroup_size(128)
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
  let valueIdx = (node * 2u + player) * params.numHands + hand;
  let rank = rankOrdinals[sample * params.numHands + hand];
  if (rank >= params.maxRanks) {
    values[valueIdx] = 0.0;
    return;
  }

  let opponent = 1u - player;
  let aggregateBase = (sample * 2u + opponent) * params.maxRanks;
  var lower = rankPrefixLess[aggregateBase + rank];
  var equal = rankMass[aggregateBase + rank];
  var higher = rankTotal[sample * 2u + opponent] - lower - equal;

  let beliefBase = (node * 2u + opponent) * params.numHands;
  let overlapBase = hand * params.overlapSlots;
  let overlapCount = overlapCounts[hand];
  for (var i = 0u; i < overlapCount; i = i + 1u) {
    let opp = overlapHands[overlapBase + i];
    let oppRank = rankOrdinals[sample * params.numHands + opp];
    if (oppRank >= params.maxRanks) {
      continue;
    }
    let oppBelief = beliefs[beliefBase + opp];
    if (oppRank < rank) {
      lower = lower - oppBelief;
    } else if (oppRank > rank) {
      higher = higher - oppBelief;
    } else {
      equal = equal - oppBelief;
    }
  }

  let winValue = payoffs[sample * 3u];
  let loseValue = payoffs[sample * 3u + 1u];
  let tieValue = payoffs[sample * 3u + 2u];
  let mass = lower + equal + higher;
  var value: f32;
  if (player == 0u) {
    value = lower * winValue + equal * tieValue + higher * loseValue;
  } else {
    value = lower * -loseValue + equal * -tieValue + higher * -winValue;
  }
  values[valueIdx] = select(0.0, value / mass, mass > 1.0e-12);
}
`;

export const SPARSE_SHOWDOWN_VALUES_FROM_RANKS_1326_WGSL =
  SPARSE_SHOWDOWN_VALUES_FROM_RANKS_WGSL
    .replace("numHands: u32,", "_numHands: u32,")
    .replace("overlapSlots: u32,", "_overlapSlots: u32,")
    .replaceAll("params.batch * 2u * params.numHands", "params.batch * 2652u")
    .replaceAll("linear % params.numHands", "linear % 1326u")
    .replaceAll("(linear / params.numHands) % 2u", "(linear / 1326u) % 2u")
    .replaceAll("linear / (2u * params.numHands)", "linear / 2652u")
    .replaceAll("sample * params.numHands + hand", "sample * 1326u + hand")
    .replaceAll("sample * params.numHands + opp", "sample * 1326u + opp")
    .replaceAll("(node * 2u + player) * params.numHands + hand", "node * 2652u + player * 1326u + hand")
    .replaceAll("(node * 2u + opponent) * params.numHands", "node * 2652u + opponent * 1326u")
    .replaceAll("hand * params.overlapSlots", "hand * 101u")
    .replaceAll("batch * 2 * tree.numHands", "batch * 2652");

export const SPARSE_SHOWDOWN_VALUES_FROM_RANKS_1326_BOTH_PLAYERS_WGSL = /* wgsl */ `
struct Params {
  _numHands: u32,
  batch: u32,
  maxRanks: u32,
  _overlapSlots: u32,
};

@group(0) @binding(0) var<storage, read> nodeIndices: array<u32>;
@group(0) @binding(1) var<storage, read> rankOrdinals: array<u32>;
@group(0) @binding(2) var<storage, read> overlapHands: array<u32>;
@group(0) @binding(3) var<storage, read> overlapCounts: array<u32>;
@group(0) @binding(4) var<storage, read> payoffs: array<f32>;
@group(0) @binding(5) var<storage, read> beliefs: array<f32>;
@group(0) @binding(6) var<storage, read> rankMass: array<f32>;
@group(0) @binding(7) var<storage, read> rankPrefixLess: array<f32>;
@group(0) @binding(8) var<storage, read> rankTotal: array<f32>;
@group(0) @binding(9) var<storage, read_write> values: array<f32>;
@group(0) @binding(10) var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let linear = gid.x;
  let total = params.batch * 1326u;
  if (linear >= total) {
    return;
  }

  let hand = linear % 1326u;
  let sample = linear / 1326u;
  let node = nodeIndices[sample];
  let valueBase = node * 2652u;
  let rank = rankOrdinals[sample * 1326u + hand];
  if (rank >= params.maxRanks) {
    values[valueBase + hand] = 0.0;
    values[valueBase + 1326u + hand] = 0.0;
    return;
  }

  let aggregateBase0 = (sample * 2u + 1u) * params.maxRanks;
  var lower0 = rankPrefixLess[aggregateBase0 + rank];
  var equal0 = rankMass[aggregateBase0 + rank];
  var higher0 = rankTotal[sample * 2u + 1u] - lower0 - equal0;

  let aggregateBase1 = (sample * 2u) * params.maxRanks;
  var lower1 = rankPrefixLess[aggregateBase1 + rank];
  var equal1 = rankMass[aggregateBase1 + rank];
  var higher1 = rankTotal[sample * 2u] - lower1 - equal1;

  let beliefBase0 = valueBase + 1326u;
  let beliefBase1 = valueBase;
  let overlapBase = hand * 101u;
  let overlapCount = overlapCounts[hand];
  for (var i = 0u; i < overlapCount; i = i + 1u) {
    let opp = overlapHands[overlapBase + i];
    let oppRank = rankOrdinals[sample * 1326u + opp];
    if (oppRank >= params.maxRanks) {
      continue;
    }
    let oppBelief0 = beliefs[beliefBase0 + opp];
    let oppBelief1 = beliefs[beliefBase1 + opp];
    if (oppRank < rank) {
      lower0 = lower0 - oppBelief0;
      lower1 = lower1 - oppBelief1;
    } else if (oppRank > rank) {
      higher0 = higher0 - oppBelief0;
      higher1 = higher1 - oppBelief1;
    } else {
      equal0 = equal0 - oppBelief0;
      equal1 = equal1 - oppBelief1;
    }
  }

  let winValue = payoffs[sample * 3u];
  let loseValue = payoffs[sample * 3u + 1u];
  let tieValue = payoffs[sample * 3u + 2u];
  let mass0 = lower0 + equal0 + higher0;
  let value0 = lower0 * winValue + equal0 * tieValue + higher0 * loseValue;
  values[valueBase + hand] = select(0.0, value0 / mass0, mass0 > 1.0e-12);

  let mass1 = lower1 + equal1 + higher1;
  let value1 = lower1 * -loseValue + equal1 * -tieValue + higher1 * -winValue;
  values[valueBase + 1326u + hand] = select(0.0, value1 / mass1, mass1 > 1.0e-12);
}
`;

export const SPARSE_ALLIN_TABLE_VALUES_WGSL = /* wgsl */ `
struct Params {
  numHands: u32,
  batch: u32,
  permId: u32,
  hasPerm: u32,
  tableScale: f32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> nodeIndices: array<u32>;
@group(0) @binding(1) var<storage, read> allowedMask: array<u32>;
@group(0) @binding(2) var<storage, read> handCard0: array<u32>;
@group(0) @binding(3) var<storage, read> handCard1: array<u32>;
@group(0) @binding(4) var<storage, read> tablePacked: array<u32>;
@group(0) @binding(5) var<storage, read> comboPerms: array<u32>;
@group(0) @binding(6) var<storage, read> scaleFactors: array<f32>;
@group(0) @binding(7) var<storage, read> beliefs: array<f32>;
@group(0) @binding(8) var<storage, read_write> values: array<f32>;
@group(0) @binding(9) var<uniform> params: Params;

fn overlaps(a: u32, b: u32) -> bool {
  return handCard0[a] == handCard0[b] ||
    handCard0[a] == handCard1[b] ||
    handCard1[a] == handCard0[b] ||
    handCard1[a] == handCard1[b];
}

fn table_hand(hand: u32) -> u32 {
  if (params.hasPerm != 0u) {
    return comboPerms[params.permId * params.numHands + hand];
  }
  return hand;
}

fn table_value(hero: u32, opp: u32) -> f32 {
  let idx = table_hand(hero) * params.numHands + table_hand(opp);
  let word = tablePacked[idx / 2u];
  let raw = select(word >> 16u, word & 0xffffu, (idx & 1u) == 0u);
  let signed = select(i32(raw), i32(raw) - 65536, raw >= 32768u);
  return f32(signed) / params.tableScale;
}

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
  let valueIdx = (node * 2u + player) * params.numHands + hand;
  if (allowedMask[node * params.numHands + hand] == 0u) {
    values[valueIdx] = 0.0;
    return;
  }

  let oppPlayer = 1u - player;
  let oppBase = (node * 2u + oppPlayer) * params.numHands;
  var numer = 0.0;
  var denom = 0.0;
  for (var opp = 0u; opp < params.numHands; opp = opp + 1u) {
    let belief = beliefs[oppBase + opp];
    numer = numer + table_value(hand, opp) * belief;
    if (
      allowedMask[node * params.numHands + opp] != 0u &&
      !overlaps(hand, opp)
    ) {
      denom = denom + belief;
    }
  }
  let rawEv = select(0.0, numer / denom, denom > 1.0e-8);
  values[valueIdx] = rawEv * scaleFactors[sample * 2u + player];
}
`;

export const SPARSE_ALLIN_TABLE_VALUES_1326_NOPERM_WGSL =
  SPARSE_ALLIN_TABLE_VALUES_WGSL
    .replace("numHands: u32,", "_numHands: u32,")
    .replace("permId: u32,", "_permId: u32,")
    .replace("hasPerm: u32,", "_hasPerm: u32,")
    .replace(
      `fn table_hand(hand: u32) -> u32 {
  if (params.hasPerm != 0u) {
    return comboPerms[params.permId * params.numHands + hand];
  }
  return hand;
}`,
      `fn table_hand(hand: u32) -> u32 {
  return hand;
}`,
    )
    .replaceAll("params.numHands", "1326u");

export const SPARSE_ALLIN_TABLE_VALUES_1326_NOPERM_BOTH_PLAYERS_WGSL = /* wgsl */ `
struct Params {
  _numHands: u32,
  batch: u32,
  _permId: u32,
  _hasPerm: u32,
  tableScale: f32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> nodeIndices: array<u32>;
@group(0) @binding(1) var<storage, read> allowedMask: array<u32>;
@group(0) @binding(2) var<storage, read> handCard0: array<u32>;
@group(0) @binding(3) var<storage, read> handCard1: array<u32>;
@group(0) @binding(4) var<storage, read> tablePacked: array<u32>;
@group(0) @binding(6) var<storage, read> scaleFactors: array<f32>;
@group(0) @binding(7) var<storage, read> beliefs: array<f32>;
@group(0) @binding(8) var<storage, read_write> values: array<f32>;
@group(0) @binding(9) var<uniform> params: Params;

fn table_value(hero: u32, opp: u32) -> f32 {
  let idx = hero * 1326u + opp;
  let word = tablePacked[idx / 2u];
  let raw = select(word >> 16u, word & 0xffffu, (idx & 1u) == 0u);
  let signed = select(i32(raw), i32(raw) - 65536, raw >= 32768u);
  return f32(signed) / params.tableScale;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let linear = gid.x;
  let total = params.batch * 1326u;
  if (linear >= total) {
    return;
  }
  let hand = linear % 1326u;
  let sample = linear / 1326u;
  let node = nodeIndices[sample];
  let allowedBase = node * 1326u;
  let valueBase = node * 2652u;
  if (allowedMask[allowedBase + hand] == 0u) {
    values[valueBase + hand] = 0.0;
    values[valueBase + 1326u + hand] = 0.0;
    return;
  }

  let p0OppBase = valueBase + 1326u;
  let p1OppBase = valueBase;
  let hand0 = handCard0[hand];
  let hand1 = handCard1[hand];
  var numer0 = 0.0;
  var denom0 = 0.0;
  var numer1 = 0.0;
  var denom1 = 0.0;
  for (var opp = 0u; opp < 1326u; opp = opp + 1u) {
    let tableEv = table_value(hand, opp);
    let belief0 = beliefs[p0OppBase + opp];
    let belief1 = beliefs[p1OppBase + opp];
    numer0 = numer0 + tableEv * belief0;
    numer1 = numer1 + tableEv * belief1;
    let opp0 = handCard0[opp];
    let opp1 = handCard1[opp];
    if (
      allowedMask[allowedBase + opp] != 0u &&
      !(hand0 == opp0 || hand0 == opp1 || hand1 == opp0 || hand1 == opp1)
    ) {
      denom0 = denom0 + belief0;
      denom1 = denom1 + belief1;
    }
  }
  let rawEv0 = select(0.0, numer0 / denom0, denom0 > 1.0e-8);
  let rawEv1 = select(0.0, numer1 / denom1, denom1 > 1.0e-8);
  values[valueBase + hand] = rawEv0 * scaleFactors[sample * 2u];
  values[valueBase + 1326u + hand] = rawEv1 * scaleFactors[sample * 2u + 1u];
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
  overlapHands: Uint32Array<ArrayBufferLike>;
  overlapCounts: Uint32Array<ArrayBufferLike>;
  overlapSlots: number;
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
  overlapHands: GPUBuffer;
  overlapCounts: GPUBuffer;
  overlapSlots: number;
  dispose: () => void;
}

export class SparseCfrGpuKernels {
  readonly device: GPUDevice;
  private readonly regretMatchPipeline: GPUComputePipeline;
  private readonly beliefApplyPipeline: GPUComputePipeline;
  private readonly beliefNormalizePipeline: GPUComputePipeline;
  private readonly beliefPropagateFusedPipeline: GPUComputePipeline;
  private readonly reachApplyPipeline: GPUComputePipeline;
  private readonly averagePolicyPipeline: GPUComputePipeline;
  private readonly backupDepthPipeline: GPUComputePipeline;
  private readonly backupDepth1326Pipeline: GPUComputePipeline;
  private readonly regretTailPipeline: GPUComputePipeline;
  private readonly opponentPolicyPipeline: GPUComputePipeline;
  private readonly opponentPolicyAggregatePipeline: GPUComputePipeline;
  private readonly opponentPolicyAggregateParallelPipeline: GPUComputePipeline;
  private readonly opponentPolicyFromAggregatePipeline: GPUComputePipeline;
  private readonly regretWeightPipeline: GPUComputePipeline;
  private readonly regretWeightAggregatePipeline: GPUComputePipeline;
  private readonly regretWeightAggregateParallelPipeline: GPUComputePipeline;
  private readonly regretWeightFromAggregatePipeline: GPUComputePipeline;
  private readonly gatherNodeBeliefsPipeline: GPUComputePipeline;
  private readonly scatterNodeValuesPipeline: GPUComputePipeline;
  private readonly showdownValuesPipeline: GPUComputePipeline;
  private readonly showdownRankMassPipeline: GPUComputePipeline;
  private readonly showdownRankMassByHandsPipeline: GPUComputePipeline;
  private readonly showdownRankMassByHandsBothPlayersPipeline: GPUComputePipeline;
  private readonly showdownRankPrefixPipeline: GPUComputePipeline;
  private readonly showdownValuesFromRanksPipeline: GPUComputePipeline;
  private readonly showdownValuesFromRanks1326Pipeline: GPUComputePipeline;
  private readonly showdownValuesFromRanks1326BothPlayersPipeline: GPUComputePipeline;
  private readonly allInTableValuesPipeline: GPUComputePipeline;
  private readonly allInTableValues1326NoPermPipeline: GPUComputePipeline;
  private readonly allInTableValues1326NoPermBothPlayersPipeline: GPUComputePipeline;

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
    this.beliefPropagateFusedPipeline = this.pipeline(
      SPARSE_BELIEF_PROPAGATE_FUSED_WGSL,
      "sparse-cfr-belief-propagate-fused",
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
    this.backupDepth1326Pipeline = this.pipeline(
      SPARSE_BACKUP_DEPTH_1326_WGSL,
      "sparse-cfr-backup-depth-1326",
    );
    this.regretTailPipeline = this.pipeline(
      SPARSE_REGRET_TAIL_WGSL,
      "sparse-cfr-regret-tail",
    );
    this.opponentPolicyPipeline = this.pipeline(
      SPARSE_OPPONENT_POLICY_WGSL,
      "sparse-cfr-opponent-policy",
    );
    this.opponentPolicyAggregatePipeline = this.pipeline(
      SPARSE_OPPONENT_POLICY_AGGREGATE_WGSL,
      "sparse-cfr-opponent-policy-aggregate",
    );
    this.opponentPolicyAggregateParallelPipeline = this.pipeline(
      SPARSE_OPPONENT_POLICY_AGGREGATE_PARALLEL_WGSL,
      "sparse-cfr-opponent-policy-aggregate-parallel",
    );
    this.opponentPolicyFromAggregatePipeline = this.pipeline(
      SPARSE_OPPONENT_POLICY_FROM_AGGREGATE_WGSL,
      "sparse-cfr-opponent-policy-from-aggregate",
    );
    this.regretWeightPipeline = this.pipeline(
      SPARSE_REGRET_WEIGHT_WGSL,
      "sparse-cfr-regret-weight",
    );
    this.regretWeightAggregatePipeline = this.pipeline(
      SPARSE_REGRET_WEIGHT_AGGREGATE_WGSL,
      "sparse-cfr-regret-weight-aggregate",
    );
    this.regretWeightAggregateParallelPipeline = this.pipeline(
      SPARSE_REGRET_WEIGHT_AGGREGATE_PARALLEL_WGSL,
      "sparse-cfr-regret-weight-aggregate-parallel",
    );
    this.regretWeightFromAggregatePipeline = this.pipeline(
      SPARSE_REGRET_WEIGHT_FROM_AGGREGATE_WGSL,
      "sparse-cfr-regret-weight-from-aggregate",
    );
    this.gatherNodeBeliefsPipeline = this.pipeline(
      SPARSE_GATHER_NODE_BELIEFS_WGSL,
      "sparse-cfr-gather-node-beliefs",
    );
    this.scatterNodeValuesPipeline = this.pipeline(
      SPARSE_SCATTER_NODE_VALUES_WGSL,
      "sparse-cfr-scatter-node-values",
    );
    this.showdownValuesPipeline = this.pipeline(
      SPARSE_SHOWDOWN_VALUES_WGSL,
      "sparse-cfr-showdown-values",
    );
    this.showdownRankMassPipeline = this.pipeline(
      SPARSE_SHOWDOWN_RANK_MASS_WGSL,
      "sparse-cfr-showdown-rank-mass",
    );
    this.showdownRankMassByHandsPipeline = this.pipeline(
      SPARSE_SHOWDOWN_RANK_MASS_BY_HANDS_WGSL,
      "sparse-cfr-showdown-rank-mass-by-hands",
    );
    this.showdownRankMassByHandsBothPlayersPipeline = this.pipeline(
      SPARSE_SHOWDOWN_RANK_MASS_BY_HANDS_BOTH_PLAYERS_WGSL,
      "sparse-cfr-showdown-rank-mass-by-hands-both-players",
    );
    this.showdownRankPrefixPipeline = this.pipeline(
      SPARSE_SHOWDOWN_RANK_PREFIX_WGSL,
      "sparse-cfr-showdown-rank-prefix",
    );
    this.showdownValuesFromRanksPipeline = this.pipeline(
      SPARSE_SHOWDOWN_VALUES_FROM_RANKS_WGSL,
      "sparse-cfr-showdown-values-from-ranks",
    );
    this.showdownValuesFromRanks1326Pipeline = this.pipeline(
      SPARSE_SHOWDOWN_VALUES_FROM_RANKS_1326_WGSL,
      "sparse-cfr-showdown-values-from-ranks-1326",
    );
    this.showdownValuesFromRanks1326BothPlayersPipeline = this.pipeline(
      SPARSE_SHOWDOWN_VALUES_FROM_RANKS_1326_BOTH_PLAYERS_WGSL,
      "sparse-cfr-showdown-values-from-ranks-1326-both-players",
    );
    this.allInTableValuesPipeline = this.pipeline(
      SPARSE_ALLIN_TABLE_VALUES_WGSL,
      "sparse-cfr-allin-table-values",
    );
    this.allInTableValues1326NoPermPipeline = this.pipeline(
      SPARSE_ALLIN_TABLE_VALUES_1326_NOPERM_WGSL,
      "sparse-cfr-allin-table-values-1326-noperm",
    );
    this.allInTableValues1326NoPermBothPlayersPipeline = this.pipeline(
      SPARSE_ALLIN_TABLE_VALUES_1326_NOPERM_BOTH_PLAYERS_WGSL,
      "sparse-cfr-allin-table-values-1326-noperm-both-players",
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
      makeStorageBuffer(this.device, data.overlapHands),
      makeStorageBuffer(this.device, data.overlapCounts),
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
      overlapHands: buffers[10],
      overlapCounts: buffers[11],
      overlapSlots: data.overlapSlots,
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
    if (globalThis.process?.env?.P2_DISABLE_FUSED_BELIEF_PROPAGATE !== "1") {
      return this.encodePropagateBeliefsDepthFused(
        encoder,
        tree,
        policy,
        beliefs,
        denom,
        start,
        end,
      );
    }
    return this.encodePropagateBeliefsDepthSplit(
      encoder,
      tree,
      policy,
      beliefs,
      denom,
      start,
      end,
    );
  }

  encodePropagateBeliefsDepthSplit(
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

  encodePropagateBeliefsDepthFused(
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
    const bindGroup = this.device.createBindGroup({
      layout: this.beliefPropagateFusedPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tree.parentIndex } },
        { binding: 1, resource: { buffer: tree.prevActor } },
        { binding: 2, resource: { buffer: tree.allowedMask } },
        { binding: 3, resource: { buffer: tree.allowedProb } },
        { binding: 4, resource: { buffer: policy } },
        { binding: 5, resource: { buffer: beliefs } },
        { binding: 6, resource: { buffer: denom } },
        { binding: 7, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.beliefPropagateFusedPipeline,
      bindGroup,
      Math.max(0, end - start),
      2,
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
    const pipeline =
      tree.numHands === 1326
        ? this.backupDepth1326Pipeline
        : this.backupDepthPipeline;
    const params = makeUniformBuffer(
      this.device,
      new Uint32Array([tree.numHands, start, end, 0]),
    );
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
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
      pipeline,
      bindGroup,
      Math.ceil(((end - start) * 2 * tree.numHands) / 128),
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
      Math.ceil(((end - start) * tree.numHands) / 128),
    );
    return params;
  }

  encodeComputeOpponentPolicyAggregateRange(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    beliefs: GPUBuffer,
    policy: GPUBuffer,
    aggregates: GPUBuffer,
    opponentPolicy: GPUBuffer,
    start: number,
    end: number,
  ): GPUBuffer[] {
    if (globalThis.process?.env?.P2_DISABLE_PARALLEL_OPPONENT_POLICY_AGGREGATE !== "1") {
      return this.encodeComputeOpponentPolicyAggregateParallelRange(
        encoder,
        tree,
        beliefs,
        policy,
        aggregates,
        opponentPolicy,
        start,
        end,
      );
    }
    const params = makeUniformBuffer(
      this.device,
      new Uint32Array([tree.numHands, start, end, 0]),
    );
    const aggregateBindGroup = this.device.createBindGroup({
      layout: this.opponentPolicyAggregatePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tree.parentIndex } },
        { binding: 1, resource: { buffer: tree.prevActor } },
        { binding: 2, resource: { buffer: tree.handCard0 } },
        { binding: 3, resource: { buffer: tree.handCard1 } },
        { binding: 4, resource: { buffer: beliefs } },
        { binding: 5, resource: { buffer: policy } },
        { binding: 6, resource: { buffer: aggregates } },
        { binding: 7, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.opponentPolicyAggregatePipeline,
      aggregateBindGroup,
      Math.max(0, end - start),
    );

    const applyBindGroup = this.device.createBindGroup({
      layout: this.opponentPolicyFromAggregatePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tree.parentIndex } },
        { binding: 1, resource: { buffer: tree.prevActor } },
        { binding: 2, resource: { buffer: tree.handCard0 } },
        { binding: 3, resource: { buffer: tree.handCard1 } },
        { binding: 4, resource: { buffer: beliefs } },
        { binding: 5, resource: { buffer: policy } },
        { binding: 6, resource: { buffer: aggregates } },
        { binding: 7, resource: { buffer: opponentPolicy } },
        { binding: 8, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.opponentPolicyFromAggregatePipeline,
      applyBindGroup,
      Math.ceil(((end - start) * tree.numHands) / 64),
    );
    return [params];
  }

  encodeComputeOpponentPolicyAggregateParallelRange(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    beliefs: GPUBuffer,
    policy: GPUBuffer,
    aggregates: GPUBuffer,
    opponentPolicy: GPUBuffer,
    start: number,
    end: number,
  ): GPUBuffer[] {
    const params = makeUniformBuffer(
      this.device,
      new Uint32Array([tree.numHands, start, end, 0]),
    );
    const aggregateBindGroup = this.device.createBindGroup({
      layout: this.opponentPolicyAggregateParallelPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tree.parentIndex } },
        { binding: 1, resource: { buffer: tree.prevActor } },
        { binding: 2, resource: { buffer: tree.handCard0 } },
        { binding: 3, resource: { buffer: tree.handCard1 } },
        { binding: 4, resource: { buffer: beliefs } },
        { binding: 5, resource: { buffer: policy } },
        { binding: 6, resource: { buffer: aggregates } },
        { binding: 7, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.opponentPolicyAggregateParallelPipeline,
      aggregateBindGroup,
      Math.max(0, end - start),
    );

    const applyBindGroup = this.device.createBindGroup({
      layout: this.opponentPolicyFromAggregatePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tree.parentIndex } },
        { binding: 1, resource: { buffer: tree.prevActor } },
        { binding: 2, resource: { buffer: tree.handCard0 } },
        { binding: 3, resource: { buffer: tree.handCard1 } },
        { binding: 4, resource: { buffer: beliefs } },
        { binding: 5, resource: { buffer: policy } },
        { binding: 6, resource: { buffer: aggregates } },
        { binding: 7, resource: { buffer: opponentPolicy } },
        { binding: 8, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.opponentPolicyFromAggregatePipeline,
      applyBindGroup,
      Math.ceil(((end - start) * tree.numHands) / 64),
    );
    return [params];
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
      Math.ceil(((end - start) * tree.numHands) / 128),
    );
    return params;
  }

  encodeComputeRegretWeightsAggregateRange(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    beliefs: GPUBuffer,
    aggregates: GPUBuffer,
    regretWeights: GPUBuffer,
    start: number,
    end: number,
  ): GPUBuffer[] {
    if (globalThis.process?.env?.P2_DISABLE_PARALLEL_REGRET_WEIGHT_AGGREGATE !== "1") {
      return this.encodeComputeRegretWeightsAggregateParallelRange(
        encoder,
        tree,
        beliefs,
        aggregates,
        regretWeights,
        start,
        end,
      );
    }
    const params = makeUniformBuffer(
      this.device,
      new Uint32Array([tree.numHands, start, end, 0]),
    );
    const aggregateBindGroup = this.device.createBindGroup({
      layout: this.regretWeightAggregatePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tree.toAct } },
        { binding: 1, resource: { buffer: tree.handCard0 } },
        { binding: 2, resource: { buffer: tree.handCard1 } },
        { binding: 3, resource: { buffer: beliefs } },
        { binding: 4, resource: { buffer: aggregates } },
        { binding: 5, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.regretWeightAggregatePipeline,
      aggregateBindGroup,
      Math.max(0, end - start),
    );

    const applyBindGroup = this.device.createBindGroup({
      layout: this.regretWeightFromAggregatePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tree.toAct } },
        { binding: 1, resource: { buffer: tree.allowedMask } },
        { binding: 2, resource: { buffer: tree.handCard0 } },
        { binding: 3, resource: { buffer: tree.handCard1 } },
        { binding: 4, resource: { buffer: beliefs } },
        { binding: 5, resource: { buffer: aggregates } },
        { binding: 6, resource: { buffer: regretWeights } },
        { binding: 7, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.regretWeightFromAggregatePipeline,
      applyBindGroup,
      Math.ceil(((end - start) * tree.numHands) / 64),
    );
    return [params];
  }

  encodeComputeRegretWeightsAggregateParallelRange(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    beliefs: GPUBuffer,
    aggregates: GPUBuffer,
    regretWeights: GPUBuffer,
    start: number,
    end: number,
  ): GPUBuffer[] {
    const params = makeUniformBuffer(
      this.device,
      new Uint32Array([tree.numHands, start, end, 0]),
    );
    const aggregateBindGroup = this.device.createBindGroup({
      layout: this.regretWeightAggregateParallelPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tree.toAct } },
        { binding: 1, resource: { buffer: tree.handCard0 } },
        { binding: 2, resource: { buffer: tree.handCard1 } },
        { binding: 3, resource: { buffer: beliefs } },
        { binding: 4, resource: { buffer: aggregates } },
        { binding: 5, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.regretWeightAggregateParallelPipeline,
      aggregateBindGroup,
      Math.max(0, end - start),
    );

    const applyBindGroup = this.device.createBindGroup({
      layout: this.regretWeightFromAggregatePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tree.toAct } },
        { binding: 1, resource: { buffer: tree.allowedMask } },
        { binding: 2, resource: { buffer: tree.handCard0 } },
        { binding: 3, resource: { buffer: tree.handCard1 } },
        { binding: 4, resource: { buffer: beliefs } },
        { binding: 5, resource: { buffer: aggregates } },
        { binding: 6, resource: { buffer: regretWeights } },
        { binding: 7, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.regretWeightFromAggregatePipeline,
      applyBindGroup,
      Math.ceil(((end - start) * tree.numHands) / 64),
    );
    return [params];
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

  encodeShowdownValues(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    nodeIndices: GPUBuffer,
    rankCodes: GPUBuffer,
    payoffs: GPUBuffer,
    beliefs: GPUBuffer,
    values: GPUBuffer,
    batch: number,
  ): GPUBuffer {
    const params = makeUniformBuffer(
      this.device,
      new Uint32Array([tree.numHands, batch, 0, 0]),
    );
    const bindGroup = this.device.createBindGroup({
      layout: this.showdownValuesPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: nodeIndices } },
        { binding: 1, resource: { buffer: tree.allowedMask } },
        { binding: 2, resource: { buffer: tree.handCard0 } },
        { binding: 3, resource: { buffer: tree.handCard1 } },
        { binding: 4, resource: { buffer: rankCodes } },
        { binding: 5, resource: { buffer: payoffs } },
        { binding: 6, resource: { buffer: beliefs } },
        { binding: 7, resource: { buffer: values } },
        { binding: 8, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.showdownValuesPipeline,
      bindGroup,
      Math.ceil((batch * 2 * tree.numHands) / 128),
    );
    return params;
  }

  encodeShowdownValuesFromRankAggregates(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    nodeIndices: GPUBuffer,
    rankOrdinals: GPUBuffer,
    rankCounts: GPUBuffer,
    payoffs: GPUBuffer,
    beliefs: GPUBuffer,
    values: GPUBuffer,
    rankMass: GPUBuffer,
    rankPrefixLess: GPUBuffer,
    rankTotal: GPUBuffer,
    batch: number,
    maxRanks: number,
    rankHandOffsets?: GPUBuffer,
    rankHandCounts?: GPUBuffer,
    rankHands?: GPUBuffer,
  ): GPUBuffer {
    const params =
      rankHandOffsets &&
      rankHandCounts &&
      rankHands &&
      globalThis.process?.env?.P2_DISABLE_SHOWDOWN_RANK_HANDS !== "1"
        ? globalThis.process?.env?.P2_DISABLE_SHOWDOWN_RANK_MASS_BOTH_PLAYERS === "1"
          ? this.encodeShowdownRankMassByHands(
              encoder,
              tree,
              nodeIndices,
              rankHandOffsets,
              rankHandCounts,
              rankHands,
              rankCounts,
              beliefs,
              rankMass,
              batch,
              maxRanks,
            )
          : this.encodeShowdownRankMassByHandsBothPlayers(
            encoder,
            tree,
            nodeIndices,
            rankHandOffsets,
            rankHandCounts,
            rankHands,
            rankCounts,
            beliefs,
            rankMass,
            batch,
            maxRanks,
          )
        : this.encodeShowdownRankMass(
            encoder,
            tree,
            nodeIndices,
            rankOrdinals,
            rankCounts,
            beliefs,
            rankMass,
            batch,
            maxRanks,
          );
    this.encodeShowdownRankPrefix(
      encoder,
      rankCounts,
      rankMass,
      rankPrefixLess,
      rankTotal,
      batch,
      maxRanks,
      params,
    );
    if (globalThis.process?.env?.P2_DISABLE_SHOWDOWN_VALUES_1326 === "1") {
      this.encodeShowdownValuesFromRanks(
        encoder,
        tree,
        nodeIndices,
        rankOrdinals,
        payoffs,
        beliefs,
        values,
        rankMass,
        rankPrefixLess,
        rankTotal,
        batch,
        maxRanks,
        params,
      );
    } else if (
      globalThis.process?.env?.P2_DISABLE_SHOWDOWN_VALUES_1326_BOTH_PLAYERS === "1"
    ) {
      this.encodeShowdownValuesFromRanks1326(
        encoder,
        tree,
        nodeIndices,
        rankOrdinals,
        payoffs,
        beliefs,
        values,
        rankMass,
        rankPrefixLess,
        rankTotal,
        batch,
        maxRanks,
        params,
      );
    } else {
      this.encodeShowdownValuesFromRanks1326BothPlayers(
        encoder,
        tree,
        nodeIndices,
        rankOrdinals,
        payoffs,
        beliefs,
        values,
        rankMass,
        rankPrefixLess,
        rankTotal,
        batch,
        maxRanks,
        params,
      );
    }
    return params;
  }

  encodeShowdownRankMassByHandsBothPlayers(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    nodeIndices: GPUBuffer,
    rankHandOffsets: GPUBuffer,
    rankHandCounts: GPUBuffer,
    rankHands: GPUBuffer,
    rankCounts: GPUBuffer,
    beliefs: GPUBuffer,
    rankMass: GPUBuffer,
    batch: number,
    maxRanks: number,
    existingParams?: GPUBuffer,
  ): GPUBuffer {
    const params =
      existingParams ??
      makeUniformBuffer(
        this.device,
        new Uint32Array([tree.numHands, batch, maxRanks, tree.overlapSlots]),
      );
    const bindGroup = this.device.createBindGroup({
      layout: this.showdownRankMassByHandsBothPlayersPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: nodeIndices } },
        { binding: 1, resource: { buffer: tree.allowedMask } },
        { binding: 2, resource: { buffer: rankHandOffsets } },
        { binding: 3, resource: { buffer: rankHandCounts } },
        { binding: 4, resource: { buffer: rankHands } },
        { binding: 5, resource: { buffer: rankCounts } },
        { binding: 6, resource: { buffer: beliefs } },
        { binding: 7, resource: { buffer: rankMass } },
        { binding: 8, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.showdownRankMassByHandsBothPlayersPipeline,
      bindGroup,
      maxRanks,
      batch,
    );
    return params;
  }

  encodeShowdownRankMassByHands(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    nodeIndices: GPUBuffer,
    rankHandOffsets: GPUBuffer,
    rankHandCounts: GPUBuffer,
    rankHands: GPUBuffer,
    rankCounts: GPUBuffer,
    beliefs: GPUBuffer,
    rankMass: GPUBuffer,
    batch: number,
    maxRanks: number,
    existingParams?: GPUBuffer,
  ): GPUBuffer {
    const params =
      existingParams ??
      makeUniformBuffer(
        this.device,
        new Uint32Array([tree.numHands, batch, maxRanks, tree.overlapSlots]),
      );
    const bindGroup = this.device.createBindGroup({
      layout: this.showdownRankMassByHandsPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: nodeIndices } },
        { binding: 1, resource: { buffer: tree.allowedMask } },
        { binding: 2, resource: { buffer: rankHandOffsets } },
        { binding: 3, resource: { buffer: rankHandCounts } },
        { binding: 4, resource: { buffer: rankHands } },
        { binding: 5, resource: { buffer: rankCounts } },
        { binding: 6, resource: { buffer: beliefs } },
        { binding: 7, resource: { buffer: rankMass } },
        { binding: 8, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.showdownRankMassByHandsPipeline,
      bindGroup,
      maxRanks,
      batch * 2,
    );
    return params;
  }

  encodeShowdownRankMass(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    nodeIndices: GPUBuffer,
    rankOrdinals: GPUBuffer,
    rankCounts: GPUBuffer,
    beliefs: GPUBuffer,
    rankMass: GPUBuffer,
    batch: number,
    maxRanks: number,
    existingParams?: GPUBuffer,
  ): GPUBuffer {
    const params =
      existingParams ??
      makeUniformBuffer(
        this.device,
        new Uint32Array([tree.numHands, batch, maxRanks, tree.overlapSlots]),
      );
    const massBindGroup = this.device.createBindGroup({
      layout: this.showdownRankMassPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: nodeIndices } },
        { binding: 1, resource: { buffer: tree.allowedMask } },
        { binding: 2, resource: { buffer: rankOrdinals } },
        { binding: 3, resource: { buffer: rankCounts } },
        { binding: 4, resource: { buffer: beliefs } },
        { binding: 5, resource: { buffer: rankMass } },
        { binding: 6, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.showdownRankMassPipeline,
      massBindGroup,
      maxRanks,
      batch * 2,
    );
    return params;
  }

  encodeShowdownRankPrefix(
    encoder: GPUCommandEncoder,
    rankCounts: GPUBuffer,
    rankMass: GPUBuffer,
    rankPrefixLess: GPUBuffer,
    rankTotal: GPUBuffer,
    batch: number,
    maxRanks: number,
    existingParams?: GPUBuffer,
  ): GPUBuffer {
    const params =
      existingParams ??
      makeUniformBuffer(
        this.device,
        new Uint32Array([0, batch, maxRanks, 0]),
      );
    const prefixBindGroup = this.device.createBindGroup({
      layout: this.showdownRankPrefixPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: rankCounts } },
        { binding: 1, resource: { buffer: rankMass } },
        { binding: 2, resource: { buffer: rankPrefixLess } },
        { binding: 3, resource: { buffer: rankTotal } },
        { binding: 4, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.showdownRankPrefixPipeline,
      prefixBindGroup,
      batch * 2,
    );
    return params;
  }

  encodeShowdownValuesFromRanks(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    nodeIndices: GPUBuffer,
    rankOrdinals: GPUBuffer,
    payoffs: GPUBuffer,
    beliefs: GPUBuffer,
    values: GPUBuffer,
    rankMass: GPUBuffer,
    rankPrefixLess: GPUBuffer,
    rankTotal: GPUBuffer,
    batch: number,
    maxRanks: number,
    existingParams?: GPUBuffer,
  ): GPUBuffer {
    const params =
      existingParams ??
      makeUniformBuffer(
        this.device,
        new Uint32Array([tree.numHands, batch, maxRanks, tree.overlapSlots]),
      );
    const valuesBindGroup = this.device.createBindGroup({
      layout: this.showdownValuesFromRanksPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: nodeIndices } },
        { binding: 1, resource: { buffer: rankOrdinals } },
        { binding: 2, resource: { buffer: tree.overlapHands } },
        { binding: 3, resource: { buffer: tree.overlapCounts } },
        { binding: 4, resource: { buffer: payoffs } },
        { binding: 5, resource: { buffer: beliefs } },
        { binding: 6, resource: { buffer: rankMass } },
        { binding: 7, resource: { buffer: rankPrefixLess } },
        { binding: 8, resource: { buffer: rankTotal } },
        { binding: 9, resource: { buffer: values } },
        { binding: 10, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.showdownValuesFromRanksPipeline,
      valuesBindGroup,
      Math.ceil((batch * 2 * tree.numHands) / 128),
    );
    return params;
  }

  encodeShowdownValuesFromRanks1326(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    nodeIndices: GPUBuffer,
    rankOrdinals: GPUBuffer,
    payoffs: GPUBuffer,
    beliefs: GPUBuffer,
    values: GPUBuffer,
    rankMass: GPUBuffer,
    rankPrefixLess: GPUBuffer,
    rankTotal: GPUBuffer,
    batch: number,
    maxRanks: number,
    existingParams?: GPUBuffer,
  ): GPUBuffer {
    if (tree.numHands !== 1326 || tree.overlapSlots !== 101) {
      return this.encodeShowdownValuesFromRanks(
        encoder,
        tree,
        nodeIndices,
        rankOrdinals,
        payoffs,
        beliefs,
        values,
        rankMass,
        rankPrefixLess,
        rankTotal,
        batch,
        maxRanks,
        existingParams,
      );
    }
    const params =
      existingParams ??
      makeUniformBuffer(
        this.device,
        new Uint32Array([tree.numHands, batch, maxRanks, tree.overlapSlots]),
      );
    const valuesBindGroup = this.device.createBindGroup({
      layout: this.showdownValuesFromRanks1326Pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: nodeIndices } },
        { binding: 1, resource: { buffer: rankOrdinals } },
        { binding: 2, resource: { buffer: tree.overlapHands } },
        { binding: 3, resource: { buffer: tree.overlapCounts } },
        { binding: 4, resource: { buffer: payoffs } },
        { binding: 5, resource: { buffer: beliefs } },
        { binding: 6, resource: { buffer: rankMass } },
        { binding: 7, resource: { buffer: rankPrefixLess } },
        { binding: 8, resource: { buffer: rankTotal } },
        { binding: 9, resource: { buffer: values } },
        { binding: 10, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.showdownValuesFromRanks1326Pipeline,
      valuesBindGroup,
      Math.ceil((batch * 2652) / 128),
    );
    return params;
  }

  encodeShowdownValuesFromRanks1326BothPlayers(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    nodeIndices: GPUBuffer,
    rankOrdinals: GPUBuffer,
    payoffs: GPUBuffer,
    beliefs: GPUBuffer,
    values: GPUBuffer,
    rankMass: GPUBuffer,
    rankPrefixLess: GPUBuffer,
    rankTotal: GPUBuffer,
    batch: number,
    maxRanks: number,
    existingParams?: GPUBuffer,
  ): GPUBuffer {
    if (tree.numHands !== 1326 || tree.overlapSlots !== 101) {
      return this.encodeShowdownValuesFromRanks(
        encoder,
        tree,
        nodeIndices,
        rankOrdinals,
        payoffs,
        beliefs,
        values,
        rankMass,
        rankPrefixLess,
        rankTotal,
        batch,
        maxRanks,
        existingParams,
      );
    }
    const params =
      existingParams ??
      makeUniformBuffer(
        this.device,
        new Uint32Array([tree.numHands, batch, maxRanks, tree.overlapSlots]),
      );
    const valuesBindGroup = this.device.createBindGroup({
      layout: this.showdownValuesFromRanks1326BothPlayersPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: nodeIndices } },
        { binding: 1, resource: { buffer: rankOrdinals } },
        { binding: 2, resource: { buffer: tree.overlapHands } },
        { binding: 3, resource: { buffer: tree.overlapCounts } },
        { binding: 4, resource: { buffer: payoffs } },
        { binding: 5, resource: { buffer: beliefs } },
        { binding: 6, resource: { buffer: rankMass } },
        { binding: 7, resource: { buffer: rankPrefixLess } },
        { binding: 8, resource: { buffer: rankTotal } },
        { binding: 9, resource: { buffer: values } },
        { binding: 10, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.showdownValuesFromRanks1326BothPlayersPipeline,
      valuesBindGroup,
      Math.ceil((batch * 1326) / 128),
    );
    return params;
  }

  encodeAllInTableValues(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    nodeIndices: GPUBuffer,
    tablePacked: GPUBuffer,
    comboPerms: GPUBuffer,
    scaleFactors: GPUBuffer,
    beliefs: GPUBuffer,
    values: GPUBuffer,
    batch: number,
    tableScale: number,
    permId: number,
    hasPerm: boolean,
  ): GPUBuffer {
    if (
      tree.numHands === 1326 &&
      !hasPerm &&
      globalThis.process?.env?.P2_DISABLE_ALLIN_1326_NOPERM !== "1"
    ) {
      return this.encodeAllInTableValues1326NoPermBothPlayers(
        encoder,
        tree,
        nodeIndices,
        tablePacked,
        comboPerms,
        scaleFactors,
        beliefs,
        values,
        batch,
        tableScale,
        permId,
        hasPerm,
      );
    }
    return this.encodeAllInTableValuesGeneric(
      encoder,
      tree,
      nodeIndices,
      tablePacked,
      comboPerms,
      scaleFactors,
      beliefs,
      values,
      batch,
      tableScale,
      permId,
      hasPerm,
    );
  }

  encodeAllInTableValuesGeneric(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    nodeIndices: GPUBuffer,
    tablePacked: GPUBuffer,
    comboPerms: GPUBuffer,
    scaleFactors: GPUBuffer,
    beliefs: GPUBuffer,
    values: GPUBuffer,
    batch: number,
    tableScale: number,
    permId: number,
    hasPerm: boolean,
  ): GPUBuffer {
    const words = new ArrayBuffer(32);
    const u32 = new Uint32Array(words);
    const f32 = new Float32Array(words);
    u32[0] = tree.numHands;
    u32[1] = batch;
    u32[2] = permId;
    u32[3] = hasPerm ? 1 : 0;
    f32[4] = tableScale;
    const params = makeUniformBuffer(this.device, u32);
    const bindGroup = this.device.createBindGroup({
      layout: this.allInTableValuesPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: nodeIndices } },
        { binding: 1, resource: { buffer: tree.allowedMask } },
        { binding: 2, resource: { buffer: tree.handCard0 } },
        { binding: 3, resource: { buffer: tree.handCard1 } },
        { binding: 4, resource: { buffer: tablePacked } },
        { binding: 5, resource: { buffer: comboPerms } },
        { binding: 6, resource: { buffer: scaleFactors } },
        { binding: 7, resource: { buffer: beliefs } },
        { binding: 8, resource: { buffer: values } },
        { binding: 9, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.allInTableValuesPipeline,
      bindGroup,
      Math.ceil((batch * 2 * tree.numHands) / 64),
    );
    return params;
  }

  encodeAllInTableValues1326NoPerm(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    nodeIndices: GPUBuffer,
    tablePacked: GPUBuffer,
    comboPerms: GPUBuffer,
    scaleFactors: GPUBuffer,
    beliefs: GPUBuffer,
    values: GPUBuffer,
    batch: number,
    tableScale: number,
    permId: number,
    hasPerm: boolean,
  ): GPUBuffer {
    if (tree.numHands !== 1326 || hasPerm) {
      return this.encodeAllInTableValuesGeneric(
        encoder,
        tree,
        nodeIndices,
        tablePacked,
        comboPerms,
        scaleFactors,
        beliefs,
        values,
        batch,
        tableScale,
        permId,
        hasPerm,
      );
    }
    const words = new ArrayBuffer(32);
    const u32 = new Uint32Array(words);
    const f32 = new Float32Array(words);
    u32[0] = tree.numHands;
    u32[1] = batch;
    u32[2] = permId;
    u32[3] = hasPerm ? 1 : 0;
    f32[4] = tableScale;
    const params = makeUniformBuffer(this.device, u32);
    const bindGroup = this.device.createBindGroup({
      layout: this.allInTableValues1326NoPermPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: nodeIndices } },
        { binding: 1, resource: { buffer: tree.allowedMask } },
        { binding: 2, resource: { buffer: tree.handCard0 } },
        { binding: 3, resource: { buffer: tree.handCard1 } },
        { binding: 4, resource: { buffer: tablePacked } },
        { binding: 6, resource: { buffer: scaleFactors } },
        { binding: 7, resource: { buffer: beliefs } },
        { binding: 8, resource: { buffer: values } },
        { binding: 9, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.allInTableValues1326NoPermPipeline,
      bindGroup,
      Math.ceil((batch * 2652) / 64),
    );
    return params;
  }

  encodeAllInTableValues1326NoPermBothPlayers(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    nodeIndices: GPUBuffer,
    tablePacked: GPUBuffer,
    comboPerms: GPUBuffer,
    scaleFactors: GPUBuffer,
    beliefs: GPUBuffer,
    values: GPUBuffer,
    batch: number,
    tableScale: number,
    permId: number,
    hasPerm: boolean,
  ): GPUBuffer {
    if (tree.numHands !== 1326 || hasPerm) {
      return this.encodeAllInTableValuesGeneric(
        encoder,
        tree,
        nodeIndices,
        tablePacked,
        comboPerms,
        scaleFactors,
        beliefs,
        values,
        batch,
        tableScale,
        permId,
        hasPerm,
      );
    }
    const words = new ArrayBuffer(32);
    const u32 = new Uint32Array(words);
    const f32 = new Float32Array(words);
    u32[0] = tree.numHands;
    u32[1] = batch;
    u32[2] = permId;
    u32[3] = hasPerm ? 1 : 0;
    f32[4] = tableScale;
    const params = makeUniformBuffer(this.device, u32);
    const bindGroup = this.device.createBindGroup({
      layout: this.allInTableValues1326NoPermBothPlayersPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: nodeIndices } },
        { binding: 1, resource: { buffer: tree.allowedMask } },
        { binding: 2, resource: { buffer: tree.handCard0 } },
        { binding: 3, resource: { buffer: tree.handCard1 } },
        { binding: 4, resource: { buffer: tablePacked } },
        { binding: 6, resource: { buffer: scaleFactors } },
        { binding: 7, resource: { buffer: beliefs } },
        { binding: 8, resource: { buffer: values } },
        { binding: 9, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.allInTableValues1326NoPermBothPlayersPipeline,
      bindGroup,
      Math.ceil((batch * 1326) / 64),
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
