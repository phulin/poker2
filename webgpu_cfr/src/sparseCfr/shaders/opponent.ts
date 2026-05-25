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
