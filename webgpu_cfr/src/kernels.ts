export const REGRET_MATCH_WGSL = /* wgsl */ `
struct Params {
  numHands: u32,
  numActions: u32,
  actor: u32,
  selectedAction: u32,
  iterations: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> legalMask: array<u32>;
@group(0) @binding(1) var<storage, read> childValues: array<f32>;
@group(0) @binding(2) var<storage, read_write> regrets: array<f32>;
@group(0) @binding(3) var<storage, read_write> policy: array<f32>;
@group(0) @binding(4) var<storage, read_write> avgPolicy: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let h = gid.x;
  if (h >= params.numHands) {
    return;
  }

  var positiveSum = 0.0;
  var legalCount = 0u;
  for (var a = 0u; a < params.numActions; a = a + 1u) {
    if (legalMask[a] != 0u) {
      legalCount = legalCount + 1u;
      let r = max(regrets[h * params.numActions + a], 0.0);
      positiveSum = positiveSum + r;
    }
  }

  let uniform = select(0.0, 1.0 / f32(max(legalCount, 1u)), legalCount > 0u);
  for (var a = 0u; a < params.numActions; a = a + 1u) {
    let outIdx = h * params.numActions + a;
    if (legalMask[a] == 0u) {
      policy[outIdx] = 0.0;
    } else if (positiveSum > 1.0e-8) {
      policy[outIdx] = max(regrets[outIdx], 0.0) / positiveSum;
    } else {
      policy[outIdx] = uniform;
    }
  }
}
`;

export const ACCUMULATE_REGRET_WGSL = /* wgsl */ `
struct Params {
  numHands: u32,
  numActions: u32,
  actor: u32,
  selectedAction: u32,
  iterations: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> legalMask: array<u32>;
@group(0) @binding(1) var<storage, read> childValues: array<f32>;
@group(0) @binding(2) var<storage, read_write> regrets: array<f32>;
@group(0) @binding(3) var<storage, read_write> policy: array<f32>;
@group(0) @binding(4) var<storage, read_write> avgPolicy: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

fn value_at(action: u32, hand: u32) -> f32 {
  let idx = (action * 2u + params.actor) * params.numHands + hand;
  return childValues[idx];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let h = gid.x;
  if (h >= params.numHands) {
    return;
  }

  var expected = 0.0;
  for (var a = 0u; a < params.numActions; a = a + 1u) {
    let idx = h * params.numActions + a;
    expected = expected + policy[idx] * value_at(a, h);
  }

  for (var a = 0u; a < params.numActions; a = a + 1u) {
    let idx = h * params.numActions + a;
    if (legalMask[a] != 0u) {
      regrets[idx] = regrets[idx] + value_at(a, h) - expected;
      avgPolicy[idx] = avgPolicy[idx] + policy[idx];
    } else {
      policy[idx] = 0.0;
    }
  }
}
`;

export const FINALIZE_POLICY_WGSL = /* wgsl */ `
struct Params {
  numHands: u32,
  numActions: u32,
  actor: u32,
  selectedAction: u32,
  iterations: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> legalMask: array<u32>;
@group(0) @binding(1) var<storage, read> childValues: array<f32>;
@group(0) @binding(2) var<storage, read_write> regrets: array<f32>;
@group(0) @binding(3) var<storage, read_write> policy: array<f32>;
@group(0) @binding(4) var<storage, read_write> avgPolicy: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.numHands * params.numActions;
  if (idx >= total) {
    return;
  }
  policy[idx] = avgPolicy[idx] / f32(max(params.iterations, 1u));
}
`;

export const BELIEF_APPLY_WGSL = /* wgsl */ `
struct Params {
  numHands: u32,
  numActions: u32,
  actor: u32,
  selectedAction: u32,
  iterations: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> beliefsIn: array<f32>;
@group(0) @binding(1) var<storage, read> policy: array<f32>;
@group(0) @binding(2) var<storage, read_write> beliefsOut: array<f32>;
@group(0) @binding(3) var<storage, read_write> denom: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let lane = lid.x;
  var sum = 0.0;
  let other = 1u - params.actor;

  for (var h = lane; h < params.numHands; h = h + 256u) {
    let weight = policy[h * params.numActions + params.selectedAction];
    let updated = beliefsIn[params.actor * params.numHands + h] * weight;
    beliefsOut[params.actor * params.numHands + h] = updated;
    beliefsOut[other * params.numHands + h] = beliefsIn[other * params.numHands + h];
    sum = sum + updated;
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
    denom[0] = partial[0];
  }
}
`;

export const BELIEF_NORMALIZE_WGSL = /* wgsl */ `
struct Params {
  numHands: u32,
  numActions: u32,
  actor: u32,
  selectedAction: u32,
  iterations: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> beliefsIn: array<f32>;
@group(0) @binding(1) var<storage, read> policy: array<f32>;
@group(0) @binding(2) var<storage, read_write> beliefsOut: array<f32>;
@group(0) @binding(3) var<storage, read_write> denom: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let h = gid.x;
  if (h >= params.numHands) {
    return;
  }
  let d = denom[0];
  let idx = params.actor * params.numHands + h;
  if (d > 1.0e-12) {
    beliefsOut[idx] = beliefsOut[idx] / d;
  } else {
    beliefsOut[idx] = beliefsIn[idx];
  }
}
`;

export const ACTION_PROBS_WGSL = /* wgsl */ `
struct Params {
  numHands: u32,
  numActions: u32,
  actor: u32,
  selectedAction: u32,
  iterations: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> beliefs: array<f32>;
@group(0) @binding(1) var<storage, read> policy: array<f32>;
@group(0) @binding(2) var<storage, read_write> actionProbs: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let action = wid.x;
  let lane = lid.x;
  var sum = 0.0;
  for (var h = lane; h < params.numHands; h = h + 256u) {
    let b = beliefs[params.actor * params.numHands + h];
    let p = policy[h * params.numActions + action];
    sum = sum + b * p;
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
    actionProbs[action] = partial[0];
  }
}
`;
