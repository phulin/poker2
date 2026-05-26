import { REDUCE_4X_256_WGSL } from "./reductions.js";

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

export const MAT_VEC_BATCH_SMALL_COLS_WGSL = /* wgsl */ `
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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.rows * params.batch;
  if (idx >= total) {
    return;
  }
  let row = idx % params.rows;
  let batch = idx / params.rows;
  let inputBase = batch * params.inputStride + params.inputOffset;
  var sum = 0.0;
  for (var col = 0u; col < params.cols; col = col + 1u) {
    sum = sum + matrix[row * params.cols + col] * input[inputBase + col];
  }
  if (params.biasPresent != 0u) {
    sum = sum + bias[row];
  }
  output[batch * params.outputStride + params.outputOffset + row] = sum;
}
`;

export const LEAKY_RELU_MAT_VEC_BATCH_WGSL = /* wgsl */ `
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

fn leaky_relu(x: f32) -> f32 {
  return select(0.01 * x, x, x >= 0.0);
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
    let x = leaky_relu(input[inputBase + col]);
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
