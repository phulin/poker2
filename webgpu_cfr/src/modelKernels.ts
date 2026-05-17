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

const REDUCE_4X_256_WGSL = /* wgsl */ `
  if (lane < 128u) {
    partial0[lane] = partial0[lane] + partial0[lane + 128u];
    partial1[lane] = partial1[lane] + partial1[lane + 128u];
    partial2[lane] = partial2[lane] + partial2[lane + 128u];
    partial3[lane] = partial3[lane] + partial3[lane + 128u];
  }
  workgroupBarrier();
  if (lane < 64u) {
    partial0[lane] = partial0[lane] + partial0[lane + 64u];
    partial1[lane] = partial1[lane] + partial1[lane + 64u];
    partial2[lane] = partial2[lane] + partial2[lane + 64u];
    partial3[lane] = partial3[lane] + partial3[lane + 64u];
  }
  workgroupBarrier();
  if (lane < 32u) {
    partial0[lane] = partial0[lane] + partial0[lane + 32u];
    partial1[lane] = partial1[lane] + partial1[lane + 32u];
    partial2[lane] = partial2[lane] + partial2[lane + 32u];
    partial3[lane] = partial3[lane] + partial3[lane + 32u];
  }
  workgroupBarrier();
  if (lane < 16u) {
    partial0[lane] = partial0[lane] + partial0[lane + 16u];
    partial1[lane] = partial1[lane] + partial1[lane + 16u];
    partial2[lane] = partial2[lane] + partial2[lane + 16u];
    partial3[lane] = partial3[lane] + partial3[lane + 16u];
  }
  workgroupBarrier();
  if (lane < 8u) {
    partial0[lane] = partial0[lane] + partial0[lane + 8u];
    partial1[lane] = partial1[lane] + partial1[lane + 8u];
    partial2[lane] = partial2[lane] + partial2[lane + 8u];
    partial3[lane] = partial3[lane] + partial3[lane + 8u];
  }
  workgroupBarrier();
  if (lane < 4u) {
    partial0[lane] = partial0[lane] + partial0[lane + 4u];
    partial1[lane] = partial1[lane] + partial1[lane + 4u];
    partial2[lane] = partial2[lane] + partial2[lane + 4u];
    partial3[lane] = partial3[lane] + partial3[lane + 4u];
  }
  workgroupBarrier();
  if (lane < 2u) {
    partial0[lane] = partial0[lane] + partial0[lane + 2u];
    partial1[lane] = partial1[lane] + partial1[lane + 2u];
    partial2[lane] = partial2[lane] + partial2[lane + 2u];
    partial3[lane] = partial3[lane] + partial3[lane + 2u];
  }
  workgroupBarrier();
  if (lane < 1u) {
    partial0[lane] = partial0[lane] + partial0[lane + 1u];
    partial1[lane] = partial1[lane] + partial1[lane + 1u];
    partial2[lane] = partial2[lane] + partial2[lane + 1u];
    partial3[lane] = partial3[lane] + partial3[lane + 1u];
  }
`;

const REDUCE_PARTIAL_SUM_256_WGSL = /* wgsl */ `
  if (lane < 128u) {
    partialSum[lane] = partialSum[lane] + partialSum[lane + 128u];
  }
  workgroupBarrier();
  if (lane < 64u) {
    partialSum[lane] = partialSum[lane] + partialSum[lane + 64u];
  }
  workgroupBarrier();
  if (lane < 32u) {
    partialSum[lane] = partialSum[lane] + partialSum[lane + 32u];
  }
  workgroupBarrier();
  if (lane < 16u) {
    partialSum[lane] = partialSum[lane] + partialSum[lane + 16u];
  }
  workgroupBarrier();
  if (lane < 8u) {
    partialSum[lane] = partialSum[lane] + partialSum[lane + 8u];
  }
  workgroupBarrier();
  if (lane < 4u) {
    partialSum[lane] = partialSum[lane] + partialSum[lane + 4u];
  }
  workgroupBarrier();
  if (lane < 2u) {
    partialSum[lane] = partialSum[lane] + partialSum[lane + 2u];
  }
  workgroupBarrier();
  if (lane < 1u) {
    partialSum[lane] = partialSum[lane] + partialSum[lane + 1u];
  }
  workgroupBarrier();
`;

const REDUCE_PARTIAL_SQ_256_WGSL = /* wgsl */ `
  if (lane < 128u) {
    partialSq[lane] = partialSq[lane] + partialSq[lane + 128u];
  }
  workgroupBarrier();
  if (lane < 64u) {
    partialSq[lane] = partialSq[lane] + partialSq[lane + 64u];
  }
  workgroupBarrier();
  if (lane < 32u) {
    partialSq[lane] = partialSq[lane] + partialSq[lane + 32u];
  }
  workgroupBarrier();
  if (lane < 16u) {
    partialSq[lane] = partialSq[lane] + partialSq[lane + 16u];
  }
  workgroupBarrier();
  if (lane < 8u) {
    partialSq[lane] = partialSq[lane] + partialSq[lane + 8u];
  }
  workgroupBarrier();
  if (lane < 4u) {
    partialSq[lane] = partialSq[lane] + partialSq[lane + 4u];
  }
  workgroupBarrier();
  if (lane < 2u) {
    partialSq[lane] = partialSq[lane] + partialSq[lane + 2u];
  }
  workgroupBarrier();
  if (lane < 1u) {
    partialSq[lane] = partialSq[lane] + partialSq[lane + 1u];
  }
  workgroupBarrier();
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

var<workgroup> partial0: array<f32, 256>;
var<workgroup> partial1: array<f32, 256>;
var<workgroup> partial2: array<f32, 256>;
var<workgroup> partial3: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row0 = wid.x * 4u;
  let row1 = row0 + 1u;
  let row2 = row0 + 2u;
  let row3 = row0 + 3u;
  let batch = wid.y;
  let lane = lid.x;
  var sum0 = 0.0;
  var sum1 = 0.0;
  var sum2 = 0.0;
  var sum3 = 0.0;
  let inputBase = batch * params.inputStride + params.inputOffset;
  for (var col = lane; col < params.cols; col = col + 256u) {
    let x = input[inputBase + col];
    if (row0 < params.rows) {
      sum0 = sum0 + matrix[row0 * params.cols + col] * x;
    }
    if (row1 < params.rows) {
      sum1 = sum1 + matrix[row1 * params.cols + col] * x;
    }
    if (row2 < params.rows) {
      sum2 = sum2 + matrix[row2 * params.cols + col] * x;
    }
    if (row3 < params.rows) {
      sum3 = sum3 + matrix[row3 * params.cols + col] * x;
    }
  }
  partial0[lane] = sum0;
  partial1[lane] = sum1;
  partial2[lane] = sum2;
  partial3[lane] = sum3;
  workgroupBarrier();
${REDUCE_4X_256_WGSL}

  if (lane == 0u) {
    let outputBase = batch * params.outputStride + params.outputOffset;
    if (row0 < params.rows) {
      var out0 = partial0[0];
      if (params.biasPresent != 0u) {
        out0 = out0 + bias[row0];
      }
      output[outputBase + row0] = out0;
    }
    if (row1 < params.rows) {
      var out1 = partial1[0];
      if (params.biasPresent != 0u) {
        out1 = out1 + bias[row1];
      }
      output[outputBase + row1] = out1;
    }
    if (row2 < params.rows) {
      var out2 = partial2[0];
      if (params.biasPresent != 0u) {
        out2 = out2 + bias[row2];
      }
      output[outputBase + row2] = out2;
    }
    if (row3 < params.rows) {
      var out3 = partial3[0];
      if (params.biasPresent != 0u) {
        out3 = out3 + bias[row3];
      }
      output[outputBase + row3] = out3;
    }
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

var<workgroup> partial0: array<f32, 256>;
var<workgroup> partial1: array<f32, 256>;
var<workgroup> partial2: array<f32, 256>;
var<workgroup> partial3: array<f32, 256>;

fn silu(x: f32) -> f32 {
  return x / (1.0 + exp(-x));
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row0 = wid.x * 4u;
  let row1 = row0 + 1u;
  let row2 = row0 + 2u;
  let row3 = row0 + 3u;
  let batch = wid.y;
  let lane = lid.x;
  var sum0 = 0.0;
  var sum1 = 0.0;
  var sum2 = 0.0;
  var sum3 = 0.0;
  let inputBase = batch * params.inputStride;
  for (var col = lane; col < params.cols; col = col + 256u) {
    let g = gate[inputBase + col];
    let gated = silu(g) * up[inputBase + col];
    if (row0 < params.rows) {
      sum0 = sum0 + down[row0 * params.cols + col] * gated;
    }
    if (row1 < params.rows) {
      sum1 = sum1 + down[row1 * params.cols + col] * gated;
    }
    if (row2 < params.rows) {
      sum2 = sum2 + down[row2 * params.cols + col] * gated;
    }
    if (row3 < params.rows) {
      sum3 = sum3 + down[row3 * params.cols + col] * gated;
    }
  }
  partial0[lane] = sum0;
  partial1[lane] = sum1;
  partial2[lane] = sum2;
  partial3[lane] = sum3;
  workgroupBarrier();
${REDUCE_4X_256_WGSL}

  if (lane == 0u) {
    let outputBase = batch * params.outputStride;
    if (row0 < params.rows) {
      output[outputBase + row0] = partial0[0];
    }
    if (row1 < params.rows) {
      output[outputBase + row1] = partial1[0];
    }
    if (row2 < params.rows) {
      output[outputBase + row2] = partial2[0];
    }
    if (row3 < params.rows) {
      output[outputBase + row3] = partial3[0];
    }
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

var<workgroup> partial0: array<f32, 256>;
var<workgroup> partial1: array<f32, 256>;
var<workgroup> partial2: array<f32, 256>;
var<workgroup> partial3: array<f32, 256>;

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
  let row0 = wid.x * 4u;
  let row1 = row0 + 1u;
  let row2 = row0 + 2u;
  let row3 = row0 + 3u;
  let batch = wid.y;
  let lane = lid.x;
  var sum0 = 0.0;
  var sum1 = 0.0;
  var sum2 = 0.0;
  var sum3 = 0.0;
  let inputBase = batch * params.inputStride;
  for (var col = lane; col < params.cols; col = col + 256u) {
    let x = gelu(input[inputBase + col]);
    if (row0 < params.rows) {
      sum0 = sum0 + matrix[row0 * params.cols + col] * x;
    }
    if (row1 < params.rows) {
      sum1 = sum1 + matrix[row1 * params.cols + col] * x;
    }
    if (row2 < params.rows) {
      sum2 = sum2 + matrix[row2 * params.cols + col] * x;
    }
    if (row3 < params.rows) {
      sum3 = sum3 + matrix[row3 * params.cols + col] * x;
    }
  }
  partial0[lane] = sum0;
  partial1[lane] = sum1;
  partial2[lane] = sum2;
  partial3[lane] = sum3;
  workgroupBarrier();
${REDUCE_4X_256_WGSL}

  if (lane == 0u) {
    let outputBase = batch * params.outputStride;
    if (row0 < params.rows) {
      var out0 = partial0[0];
      if (params.biasPresent != 0u) {
        out0 = out0 + bias[row0];
      }
      output[outputBase + row0] = out0;
    }
    if (row1 < params.rows) {
      var out1 = partial1[0];
      if (params.biasPresent != 0u) {
        out1 = out1 + bias[row1];
      }
      output[outputBase + row1] = out1;
    }
    if (row2 < params.rows) {
      var out2 = partial2[0];
      if (params.biasPresent != 0u) {
        out2 = out2 + bias[row2];
      }
      output[outputBase + row2] = out2;
    }
    if (row3 < params.rows) {
      var out3 = partial3[0];
      if (params.biasPresent != 0u) {
        out3 = out3 + bias[row3];
      }
      output[outputBase + row3] = out3;
    }
  }
}
`;

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
