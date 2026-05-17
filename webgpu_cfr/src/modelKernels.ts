export const MAT_VEC_WGSL = /* wgsl */ `
struct Params {
  rows: u32,
  cols: u32,
  inputOffset: u32,
  outputOffset: u32,
  biasPresent: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> matrix: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wid.x;
  let lane = lid.x;
  var sum = 0.0;
  let rowOffset = row * params.cols;
  for (var col = lane; col < params.cols; col = col + 256u) {
    sum = sum + matrix[rowOffset + col] * input[params.inputOffset + col];
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
    var out = partial[0];
    if (params.biasPresent != 0u) {
      out = out + bias[row];
    }
    output[params.outputOffset + row] = out;
  }
}
`;

export const MAT_VEC_BATCH_WGSL = /* wgsl */ `
struct Params {
  rows: u32,
  cols: u32,
  batch: u32,
  inputStride: u32,
  outputStride: u32,
  inputOffset: u32,
  outputOffset: u32,
  biasPresent: u32,
};

@group(0) @binding(0) var<storage, read> matrix: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wid.x;
  let batch = wid.y;
  let lane = lid.x;
  var sum = 0.0;
  let rowOffset = row * params.cols;
  let inputBase = batch * params.inputStride + params.inputOffset;
  for (var col = lane; col < params.cols; col = col + 256u) {
    sum = sum + matrix[rowOffset + col] * input[inputBase + col];
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
    var out = partial[0];
    if (params.biasPresent != 0u) {
      out = out + bias[row];
    }
    output[batch * params.outputStride + params.outputOffset + row] = out;
  }
}
`;

export const SWIGLU_DOWN_BATCH_WGSL = /* wgsl */ `
struct Params {
  rows: u32,
  cols: u32,
  batch: u32,
  inputStride: u32,
  outputStride: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> down: array<f32>;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<storage, read> up: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> partial: array<f32, 256>;

fn silu(x: f32) -> f32 {
  return x / (1.0 + exp(-x));
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wid.x;
  let batch = wid.y;
  let lane = lid.x;
  var sum = 0.0;
  let rowOffset = row * params.cols;
  let inputBase = batch * params.inputStride;
  for (var col = lane; col < params.cols; col = col + 256u) {
    let g = gate[inputBase + col];
    let gated = silu(g) * up[inputBase + col];
    sum = sum + down[rowOffset + col] * gated;
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
    output[batch * params.outputStride + row] = partial[0];
  }
}
`;

export const GELU_MAT_VEC_BATCH_WGSL = /* wgsl */ `
struct Params {
  rows: u32,
  cols: u32,
  batch: u32,
  inputStride: u32,
  outputStride: u32,
  biasPresent: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> matrix: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> partial: array<f32, 256>;

fn erf_approx(x: f32) -> f32 {
  let sign = select(-1.0, 1.0, x >= 0.0);
  let ax = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * ax);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-ax * ax);
  return sign * y;
}

fn gelu(x: f32) -> f32 {
  return 0.5 * x * (1.0 + erf_approx(x * 0.7071067811865476));
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wid.x;
  let batch = wid.y;
  let lane = lid.x;
  var sum = 0.0;
  let rowOffset = row * params.cols;
  let inputBase = batch * params.inputStride;
  for (var col = lane; col < params.cols; col = col + 256u) {
    sum = sum + matrix[rowOffset + col] * gelu(input[inputBase + col]);
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
    var out = partial[0];
    if (params.biasPresent != 0u) {
      out = out + bias[row];
    }
    output[batch * params.outputStride + row] = out;
  }
}
`;

export const LAYER_NORM_WGSL = /* wgsl */ `
struct Params {
  dim: u32,
  inputOffset: u32,
  outputOffset: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> partialSum: array<f32, 256>;
var<workgroup> partialSq: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let lane = lid.x;
  var sum = 0.0;
  for (var i = lane; i < params.dim; i = i + 256u) {
    sum = sum + input[params.inputOffset + i];
  }
  partialSum[lane] = sum;
  workgroupBarrier();

  var stride = 128u;
  loop {
    if (lane < stride) {
      partialSum[lane] = partialSum[lane] + partialSum[lane + stride];
    }
    workgroupBarrier();
    if (stride == 1u) {
      break;
    }
    stride = stride / 2u;
  }

  let mean = partialSum[0] / f32(params.dim);
  var sq = 0.0;
  for (var i = lane; i < params.dim; i = i + 256u) {
    let centered = input[params.inputOffset + i] - mean;
    sq = sq + centered * centered;
  }
  partialSq[lane] = sq;
  workgroupBarrier();

  stride = 128u;
  loop {
    if (lane < stride) {
      partialSq[lane] = partialSq[lane] + partialSq[lane + stride];
    }
    workgroupBarrier();
    if (stride == 1u) {
      break;
    }
    stride = stride / 2u;
  }

  let invStd = inverseSqrt(partialSq[0] / f32(params.dim) + 1.0e-5);
  for (var i = lane; i < params.dim; i = i + 256u) {
    output[params.outputOffset + i] =
      (input[params.inputOffset + i] - mean) * invStd * weight[i] + bias[i];
  }
}
`;

export const LAYER_NORM_BATCH_WGSL = /* wgsl */ `
struct Params {
  dim: u32,
  batch: u32,
  inputStride: u32,
  outputStride: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> partialSum: array<f32, 256>;
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
  var sum = 0.0;
  for (var i = lane; i < params.dim; i = i + 256u) {
    sum = sum + input[inputBase + i];
  }
  partialSum[lane] = sum;
  workgroupBarrier();

  var stride = 128u;
  loop {
    if (lane < stride) {
      partialSum[lane] = partialSum[lane] + partialSum[lane + stride];
    }
    workgroupBarrier();
    if (stride == 1u) {
      break;
    }
    stride = stride / 2u;
  }

  let mean = partialSum[0] / f32(params.dim);
  var sq = 0.0;
  for (var i = lane; i < params.dim; i = i + 256u) {
    let centered = input[inputBase + i] - mean;
    sq = sq + centered * centered;
  }
  partialSq[lane] = sq;
  workgroupBarrier();

  stride = 128u;
  loop {
    if (lane < stride) {
      partialSq[lane] = partialSq[lane] + partialSq[lane + stride];
    }
    workgroupBarrier();
    if (stride == 1u) {
      break;
    }
    stride = stride / 2u;
  }

  let invStd = inverseSqrt(partialSq[0] / f32(params.dim) + 1.0e-5);
  for (var i = lane; i < params.dim; i = i + 256u) {
    output[outputBase + i] = (input[inputBase + i] - mean) * invStd * weight[i] + bias[i];
  }
}
`;

export const SILU_MUL_WGSL = /* wgsl */ `
struct Params {
  dim: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) {
    return;
  }
  let g = gate[i];
  output[i] = (g / (1.0 + exp(-g))) * up[i];
}
`;

export const GELU_WGSL = /* wgsl */ `
struct Params {
  dim: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn erf_approx(x: f32) -> f32 {
  let sign = select(-1.0, 1.0, x >= 0.0);
  let ax = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * ax);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-ax * ax);
  return sign * y;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) {
    return;
  }
  let x = input[i];
  output[i] = 0.5 * x * (1.0 + erf_approx(x * 0.7071067811865476));
}
`;

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

export const ZERO_SUM_BATCH_WGSL = /* wgsl */ `
struct Params {
  numHands: u32,
  batch: u32,
  _pad0: u32,
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

  var s0 = 0.0;
  var s1 = 0.0;
  for (var h = lane; h < params.numHands; h = h + 256u) {
    s0 = s0 + values[base + h] * beliefs[base + h];
    s1 = s1 + values[base + params.numHands + h] * beliefs[base + params.numHands + h];
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
