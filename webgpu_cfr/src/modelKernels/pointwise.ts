export const SCALED_RESIDUAL_ADD_WGSL = /* wgsl */ `
struct Params {
  dim: u32,
  _pad0: u32,
  alpha: f32,
  _pad1: f32,
};

@group(0) @binding(0) var<storage, read> residual: array<f32>;
@group(0) @binding(1) var<storage, read> inner: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) {
    return;
  }
  output[i] = residual[i] + params.alpha * inner[i];
}
`;

export const ADD3_WGSL = /* wgsl */ `
struct Params {
  dim: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> c: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) {
    return;
  }
  output[i] = a[i] + b[i] + c[i];
}
`;

export const REPEAT_ROWS_WGSL = /* wgsl */ `
struct Params {
  dim: u32,
  batch: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.dim * params.batch;
  if (idx >= total) {
    return;
  }
  output[idx] = input[idx % params.dim];
}
`;

export const ZERO_SUM_BATCH_WGSL = /* wgsl */ `
struct Params {
  numHands: u32,
  batch: u32,
  beliefStride: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> values: array<f32>;
@group(0) @binding(1) var<storage, read> beliefs: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> partial0: array<f32, 256>;
var<workgroup> partial1: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let sample = wid.x;
  let lane = lid.x;
  if (sample >= params.batch) {
    return;
  }
  let base = sample * 2u * params.numHands;
  let beliefBase = sample * params.beliefStride;

  var s0 = 0.0;
  var s1 = 0.0;
  for (var h = lane; h < params.numHands; h = h + 256u) {
    s0 = s0 + values[base + h] * beliefs[beliefBase + h];
    s1 = s1 + values[base + params.numHands + h] * beliefs[beliefBase + params.numHands + h];
  }
  partial0[lane] = s0;
  partial1[lane] = s1;
  workgroupBarrier();

  var stride = 128u;
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

  let offset = 0.5 * (partial0[0] + partial1[0]);
  for (var i = lane; i < 2u * params.numHands; i = i + 256u) {
    values[base + i] = values[base + i] - offset;
  }
}
`;
