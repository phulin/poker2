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

export const SPARSE_SHOWDOWN_RANK_PREFIX_PACKED_WGSL = /* wgsl */ `
struct Params {
  numHands: u32,
  batch: u32,
  maxRanks: u32,
  _pad0: u32,
  rankMassOffset: u32,
  rankPrefixOffset: u32,
  rankTotalOffset: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> rankCounts: array<u32>;
@group(0) @binding(1) var<storage, read_write> rankScratch: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

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
    rankScratch[params.rankPrefixOffset + base + rank] = sum;
    if (rank < rankCount) {
      sum = sum + rankScratch[params.rankMassOffset + base + rank];
    }
  }
  rankScratch[params.rankTotalOffset + samplePlayer] = sum;
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
  SPARSE_SHOWDOWN_VALUES_FROM_RANKS_WGSL.replace("numHands: u32,", "_numHands: u32,")
    .replace("overlapSlots: u32,", "_overlapSlots: u32,")
    .replaceAll("params.batch * 2u * params.numHands", "params.batch * 2652u")
    .replaceAll("linear % params.numHands", "linear % 1326u")
    .replaceAll("(linear / params.numHands) % 2u", "(linear / 1326u) % 2u")
    .replaceAll("linear / (2u * params.numHands)", "linear / 2652u")
    .replaceAll("sample * params.numHands + hand", "sample * 1326u + hand")
    .replaceAll("sample * params.numHands + opp", "sample * 1326u + opp")
    .replaceAll(
      "(node * 2u + player) * params.numHands + hand",
      "node * 2652u + player * 1326u + hand",
    )
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

export const SPARSE_SHOWDOWN_VALUES_FROM_RANKS_1326_BOTH_PLAYERS_PACKED_WGSL =
  SPARSE_SHOWDOWN_VALUES_FROM_RANKS_1326_BOTH_PLAYERS_WGSL.replace(
    `  _overlapSlots: u32,
};`,
    `  _overlapSlots: u32,
  rankMassOffset: u32,
  rankPrefixOffset: u32,
  rankTotalOffset: u32,
  _pad0: u32,
};`,
  )
    .replace("@group(0) @binding(6) var<storage, read> rankMass: array<f32>;\n", "")
    .replace("@group(0) @binding(7) var<storage, read> rankPrefixLess: array<f32>;\n", "")
    .replace("@group(0) @binding(8) var<storage, read> rankTotal: array<f32>;\n", "")
    .replace(
      "@group(0) @binding(9) var<storage, read_write> values: array<f32>;",
      "@group(0) @binding(6) var<storage, read> rankScratch: array<f32>;\n@group(0) @binding(7) var<storage, read_write> values: array<f32>;",
    )
    .replace(
      "@group(0) @binding(10) var<uniform> params: Params;",
      "@group(0) @binding(8) var<uniform> params: Params;",
    )
    .replaceAll("rankPrefixLess[", "rankScratch[params.rankPrefixOffset + ")
    .replaceAll("rankMass[", "rankScratch[params.rankMassOffset + ")
    .replaceAll("rankTotal[", "rankScratch[params.rankTotalOffset + ");
