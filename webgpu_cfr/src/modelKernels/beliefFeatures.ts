export const PLAYER_BOARD_HADAMARD_WGSL = /* wgsl */ `
struct Params {
  elements: u32,
  interactionDim: u32,
  numPlayers: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> pairLow: array<f32>;
@group(0) @binding(1) var<storage, read> boardLow: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.elements) {
    return;
  }
  let rowDim = params.numPlayers * params.interactionDim;
  let batch = idx / rowDim;
  let r = idx % params.interactionDim;
  output[idx] = pairLow[idx] * boardLow[batch * params.interactionDim + r];
}
`;

export const FILL_EXACT_PAIR_MASS_WGSL = /* wgsl */ `
struct Params {
  rows: u32,
  batch: u32,
  numPlayers: u32,
  player: u32,
  hand: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> pairOneHotT: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let linear = gid.x;
  let total = params.batch * params.rows;
  if (linear >= total) {
    return;
  }
  let row = linear % params.rows;
  let batch = linear / params.rows;
  let outputBase = (batch * params.numPlayers + params.player) * params.rows;
  output[outputBase + row] = pairOneHotT[row * 1326u + params.hand];
}
`;


export const MAT_VEC_BATCH_EXACT_BELIEF_LINEAR_IN_512_BATCH2_SUBGROUP_WGSL = /* wgsl */ `
enable subgroups;

struct Params {
  rows: u32,
  cols: u32,
  batch: u32,
  inputStride: u32,
  outputStride: u32,
  exactHand: u32,
  numHands: u32,
  contributionRows: u32,
};

@group(0) @binding(0) var<storage, read> matrix: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> invRms: array<f32>;
@group(0) @binding(3) var<storage, read> contribution: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> subgroupPartial00: array<f32, 256>;
var<workgroup> subgroupPartial01: array<f32, 256>;
var<workgroup> subgroupPartial02: array<f32, 256>;
var<workgroup> subgroupPartial03: array<f32, 256>;
var<workgroup> subgroupPartial10: array<f32, 256>;
var<workgroup> subgroupPartial11: array<f32, 256>;
var<workgroup> subgroupPartial12: array<f32, 256>;
var<workgroup> subgroupPartial13: array<f32, 256>;

fn exact_contribution(batch: u32, row: u32) -> f32 {
  if (params.contributionRows != 0u) {
    return contribution[batch * params.rows + row];
  }
  return contribution[row * params.numHands + params.exactHand];
}

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) subgroupLane: u32,
  @builtin(subgroup_size) subgroupSize: u32,
) {
  let row0 = wid.x * 4u;
  let row1 = row0 + 1u;
  let row2 = row0 + 2u;
  let row3 = row0 + 3u;
  let batch0 = wid.y * 2u;
  let batch1 = batch0 + 1u;
  let lane = lid.x;
  let inputBase0 = batch0 * params.inputStride;
  let inputBase1 = batch1 * params.inputStride;

  let col0 = lane;
  let x00 = input[inputBase0 + col0];
  var x10 = 0.0;
  if (batch1 < params.batch) {
    x10 = input[inputBase1 + col0];
  }
  let m00 = matrix[row0 * 512u + col0];
  let m10 = matrix[row1 * 512u + col0];
  let m20 = matrix[row2 * 512u + col0];
  let m30 = matrix[row3 * 512u + col0];
  var sum00 = m00 * x00;
  var sum01 = m10 * x00;
  var sum02 = m20 * x00;
  var sum03 = m30 * x00;
  var sum10 = m00 * x10;
  var sum11 = m10 * x10;
  var sum12 = m20 * x10;
  var sum13 = m30 * x10;

  let col1 = lane + 256u;
  let x01 = input[inputBase0 + col1];
  var x11 = 0.0;
  if (batch1 < params.batch) {
    x11 = input[inputBase1 + col1];
  }
  let m01 = matrix[row0 * 512u + col1];
  let m11 = matrix[row1 * 512u + col1];
  let m21 = matrix[row2 * 512u + col1];
  let m31 = matrix[row3 * 512u + col1];
  sum00 = sum00 + m01 * x01;
  sum01 = sum01 + m11 * x01;
  sum02 = sum02 + m21 * x01;
  sum03 = sum03 + m31 * x01;
  sum10 = sum10 + m01 * x11;
  sum11 = sum11 + m11 * x11;
  sum12 = sum12 + m21 * x11;
  sum13 = sum13 + m31 * x11;

  let reduced00 = subgroupAdd(sum00);
  let reduced01 = subgroupAdd(sum01);
  let reduced02 = subgroupAdd(sum02);
  let reduced03 = subgroupAdd(sum03);
  let reduced10 = subgroupAdd(sum10);
  let reduced11 = subgroupAdd(sum11);
  let reduced12 = subgroupAdd(sum12);
  let reduced13 = subgroupAdd(sum13);
  let subgroupIndex = lane / subgroupSize;
  if (subgroupLane == 0u) {
    subgroupPartial00[subgroupIndex] = reduced00;
    subgroupPartial01[subgroupIndex] = reduced01;
    subgroupPartial02[subgroupIndex] = reduced02;
    subgroupPartial03[subgroupIndex] = reduced03;
    subgroupPartial10[subgroupIndex] = reduced10;
    subgroupPartial11[subgroupIndex] = reduced11;
    subgroupPartial12[subgroupIndex] = reduced12;
    subgroupPartial13[subgroupIndex] = reduced13;
  }
  workgroupBarrier();

  if (lane == 0u) {
    let subgroupCount = (256u + subgroupSize - 1u) / subgroupSize;
    var out00 = 0.0;
    var out01 = 0.0;
    var out02 = 0.0;
    var out03 = 0.0;
    var out10 = 0.0;
    var out11 = 0.0;
    var out12 = 0.0;
    var out13 = 0.0;
    for (var i = 0u; i < subgroupCount; i = i + 1u) {
      out00 = out00 + subgroupPartial00[i];
      out01 = out01 + subgroupPartial01[i];
      out02 = out02 + subgroupPartial02[i];
      out03 = out03 + subgroupPartial03[i];
      out10 = out10 + subgroupPartial10[i];
      out11 = out11 + subgroupPartial11[i];
      out12 = out12 + subgroupPartial12[i];
      out13 = out13 + subgroupPartial13[i];
    }
    let contrib0 = invRms[batch0];
    let outputBase0 = batch0 * params.outputStride;
    output[outputBase0 + row0] =
      out00 + contrib0 * exact_contribution(batch0, row0);
    output[outputBase0 + row1] =
      out01 + contrib0 * exact_contribution(batch0, row1);
    output[outputBase0 + row2] =
      out02 + contrib0 * exact_contribution(batch0, row2);
    output[outputBase0 + row3] =
      out03 + contrib0 * exact_contribution(batch0, row3);
    if (batch1 < params.batch) {
      let contrib1 = invRms[batch1];
      let outputBase1 = batch1 * params.outputStride;
      output[outputBase1 + row0] =
        out10 + contrib1 * exact_contribution(batch1, row0);
      output[outputBase1 + row1] =
        out11 + contrib1 * exact_contribution(batch1, row1);
      output[outputBase1 + row2] =
        out12 + contrib1 * exact_contribution(batch1, row2);
      output[outputBase1 + row3] =
        out13 + contrib1 * exact_contribution(batch1, row3);
    }
  }
}
`;
