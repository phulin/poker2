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

export const SPARSE_ALLIN_TABLE_VALUES_1326_NOPERM_WGSL = SPARSE_ALLIN_TABLE_VALUES_WGSL.replace(
  "numHands: u32,",
  "_numHands: u32,",
)
  .replace("permId: u32,", "_permId: u32,")
  .replace("hasPerm: u32,", "_hasPerm: u32,")
  .replace("@group(0) @binding(5) var<storage, read> comboPerms: array<u32>;\n", "")
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
    if (!(hand0 == opp0 || hand0 == opp1 || hand1 == opp0 || hand1 == opp1)) {
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
