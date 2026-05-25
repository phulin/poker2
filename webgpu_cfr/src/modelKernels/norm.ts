import { REDUCE_PARTIAL_SQ_256_WGSL } from "./reductions.js";

export const RMS_NORM_WGSL = /* wgsl */ `
struct Params {
  dim: u32,
  inputOffset: u32,
  outputOffset: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partialSq: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let lane = lid.x;
  var sq = 0.0;
  for (var i = lane; i < params.dim; i = i + 256u) {
    let value = input[params.inputOffset + i];
    sq = sq + value * value;
  }
  partialSq[lane] = sq;
  workgroupBarrier();
${REDUCE_PARTIAL_SQ_256_WGSL}

  let invRms = inverseSqrt(partialSq[0] / f32(params.dim) + 1.0e-5);
  for (var i = lane; i < params.dim; i = i + 256u) {
    output[params.outputOffset + i] =
      input[params.inputOffset + i] * invRms * weight[i];
  }
}
`;

export const RMS_NORM_BATCH_WGSL = /* wgsl */ `
struct Params {
  dim: u32,
  batch: u32,
  inputStride: u32,
  outputStride: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partialSq: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let batch = wid.x;
  let lane = lid.x;
  if (batch >= params.batch) {
    return;
  }
  let inputBase = batch * params.inputStride;
  let outputBase = batch * params.outputStride;
  var sq = 0.0;
  for (var i = lane; i < params.dim; i = i + 256u) {
    let value = input[inputBase + i];
    sq = sq + value * value;
  }
  partialSq[lane] = sq;
  workgroupBarrier();
${REDUCE_PARTIAL_SQ_256_WGSL}

  let invRms = inverseSqrt(partialSq[0] / f32(params.dim) + 1.0e-5);
  for (var i = lane; i < params.dim; i = i + 256u) {
    output[outputBase + i] = input[inputBase + i] * invRms * weight[i];
  }
}
`;

export const RMS_NORM_BELIEF_EXACT_WGSL = /* wgsl */ `
struct Params {
  dim: u32,
  batch: u32,
  inputStride: u32,
  outputStride: u32,
  exactOffset: u32,
  exactHand: u32,
  numHands: u32,
  hidden: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> handEmbeddingT: array<f32>;

var<workgroup> partialSq: array<f32, 256>;

fn belief_input(inputBase: u32, i: u32) -> f32 {
  if (i >= params.exactOffset && i < params.exactOffset + params.hidden) {
    let dim = i - params.exactOffset;
    return handEmbeddingT[dim * params.numHands + params.exactHand];
  }
  return input[inputBase + i];
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let batch = wid.x;
  let lane = lid.x;
  if (batch >= params.batch) {
    return;
  }
  let inputBase = batch * params.inputStride;
  let outputBase = batch * params.outputStride;
  var sq = 0.0;
  for (var i = lane; i < params.dim; i = i + 256u) {
    let value = belief_input(inputBase, i);
    sq = sq + value * value;
  }
  partialSq[lane] = sq;
  workgroupBarrier();
${REDUCE_PARTIAL_SQ_256_WGSL}

  let invRms = inverseSqrt(partialSq[0] / f32(params.dim) + 1.0e-5);
  for (var i = lane; i < params.dim; i = i + 256u) {
    output[outputBase + i] = belief_input(inputBase, i) * invRms * weight[i];
  }
}
`;

export const RMS_NORM_BELIEF_EXACT_HALF_WGSL = /* wgsl */ `
struct Params {
  dim: u32,
  batch: u32,
  inputStride: u32,
  exactOffset: u32,
  exactHand: u32,
  numHands: u32,
  hidden: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> opponentOutput: array<f32>;
@group(0) @binding(3) var<storage, read_write> invRmsOutput: array<f32>;
@group(0) @binding(4) var<storage, read> handEmbeddingT: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> partialSq: array<f32, 256>;

fn belief_input(inputBase: u32, i: u32) -> f32 {
  if (i >= params.exactOffset && i < params.exactOffset + params.hidden) {
    let dim = i - params.exactOffset;
    return handEmbeddingT[dim * params.numHands + params.exactHand];
  }
  return input[inputBase + i];
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let batch = wid.x;
  let lane = lid.x;
  if (batch >= params.batch) {
    return;
  }
  let inputBase = batch * params.inputStride;
  var sq = 0.0;
  for (var i = lane; i < params.dim; i = i + 256u) {
    let value = belief_input(inputBase, i);
    sq = sq + value * value;
  }
  partialSq[lane] = sq;
  workgroupBarrier();
${REDUCE_PARTIAL_SQ_256_WGSL}

  let invRms = inverseSqrt(partialSq[0] / f32(params.dim) + 1.0e-5);
  if (lane == 0u) {
    invRmsOutput[batch] = invRms;
  }
  let opponentOffset = params.hidden - params.exactOffset;
  for (var i = lane; i < params.hidden; i = i + 256u) {
    let source = opponentOffset + i;
    opponentOutput[batch * params.hidden + i] =
      input[inputBase + source] * invRms * weight[source];
  }
}
`;

export const RMS_NORM_BATCH_SMALL_WGSL = /* wgsl */ `
struct Params {
  dim: u32,
  batch: u32,
  inputStride: u32,
  outputStride: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let batch = gid.x;
  if (batch >= params.batch) {
    return;
  }
  let inputBase = batch * params.inputStride;
  let outputBase = batch * params.outputStride;
  var sq = 0.0;
  for (var i = 0u; i < params.dim; i = i + 1u) {
    let value = input[inputBase + i];
    sq = sq + value * value;
  }
  let invRms = inverseSqrt(sq / f32(params.dim) + 1.0e-5);
  for (var i = 0u; i < params.dim; i = i + 1u) {
    output[outputBase + i] = input[inputBase + i] * invRms * weight[i];
  }
}
`;
