export const MAT_VEC_BATCH_TILED_GEMM_WGSL = /* wgsl */ `
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

@group(0) @binding(0) var<storage, read> packedMatrix: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> inputTile: array<f32, 256>;
var<workgroup> weightTile: array<f32, 256>;

@compute @workgroup_size(8, 8)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(local_invocation_index) localIndex: u32,
) {
  let row = wid.x * 8u + lid.x;
  let batch = wid.y * 8u + lid.y;
  let kTiles = (params.cols + 31u) / 32u;
  var sum = 0.0;

  for (var kTile = 0u; kTile < kTiles; kTile = kTile + 1u) {
    let kBase = kTile * 32u;
    for (var i = localIndex; i < 256u; i = i + 64u) {
      let batchLocal = i / 32u;
      let kLocal = i % 32u;
      let sourceBatch = wid.y * 8u + batchLocal;
      let col = kBase + kLocal;
      var x = 0.0;
      if (sourceBatch < params.batch && col < params.cols) {
        x = input[sourceBatch * params.inputStride + params.inputOffset + col];
      }
      inputTile[i] = x;
    }
    for (var i = localIndex; i < 256u; i = i + 64u) {
      let kLocal = i / 8u;
      let rowLocal = i % 8u;
      weightTile[i] = packedMatrix[((wid.x * kTiles + kTile) * 32u + kLocal) * 8u + rowLocal];
    }
    workgroupBarrier();

    for (var kLocal = 0u; kLocal < 32u; kLocal = kLocal + 1u) {
      sum = sum + inputTile[lid.y * 32u + kLocal] * weightTile[kLocal * 8u + lid.x];
    }
    workgroupBarrier();
  }

  if (row < params.rows && batch < params.batch) {
    var out = sum;
    if (params.biasPresent != 0u) {
      out = out + bias[row];
    }
    output[batch * params.outputStride + params.outputOffset + row] = out;
  }
}
`;

export const LEAKY_RELU_MAT_VEC_BATCH_TILED_GEMM_WGSL = /* wgsl */ `
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

@group(0) @binding(0) var<storage, read> packedMatrix: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> inputTile: array<f32, 256>;
var<workgroup> weightTile: array<f32, 256>;

fn leaky_relu(x: f32) -> f32 {
  return select(0.01 * x, x, x >= 0.0);
}

@compute @workgroup_size(8, 8)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(local_invocation_index) localIndex: u32,
) {
  let row = wid.x * 8u + lid.x;
  let batch = wid.y * 8u + lid.y;
  let kTiles = (params.cols + 31u) / 32u;
  var sum = 0.0;

  for (var kTile = 0u; kTile < kTiles; kTile = kTile + 1u) {
    let kBase = kTile * 32u;
    for (var i = localIndex; i < 256u; i = i + 64u) {
      let batchLocal = i / 32u;
      let kLocal = i % 32u;
      let sourceBatch = wid.y * 8u + batchLocal;
      let col = kBase + kLocal;
      var x = 0.0;
      if (sourceBatch < params.batch && col < params.cols) {
        x = leaky_relu(input[sourceBatch * params.inputStride + col]);
      }
      inputTile[i] = x;
    }
    for (var i = localIndex; i < 256u; i = i + 64u) {
      let kLocal = i / 8u;
      let rowLocal = i % 8u;
      weightTile[i] = packedMatrix[((wid.x * kTiles + kTile) * 32u + kLocal) * 8u + rowLocal];
    }
    workgroupBarrier();

    for (var kLocal = 0u; kLocal < 32u; kLocal = kLocal + 1u) {
      sum = sum + inputTile[lid.y * 32u + kLocal] * weightTile[kLocal * 8u + lid.x];
    }
    workgroupBarrier();
  }

  if (row < params.rows && batch < params.batch) {
    var out = sum;
    if (params.biasPresent != 0u) {
      out = out + bias[row];
    }
    output[batch * params.outputStride + row] = out;
  }
}
`;

export const LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_TILED_GEMM_WGSL = /* wgsl */ `
struct Params {
  rows: u32,
  cols: u32,
  batch: u32,
  inputStride: u32,
  outputStride: u32,
  biasPresent: u32,
  alphaBits: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> packedMatrix: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<storage, read> residual: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> inputTile: array<f32, 256>;
var<workgroup> weightTile: array<f32, 256>;

fn leaky_relu(x: f32) -> f32 {
  return select(0.01 * x, x, x >= 0.0);
}

@compute @workgroup_size(8, 8)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(local_invocation_index) localIndex: u32,
) {
  let row = wid.x * 8u + lid.x;
  let batch = wid.y * 8u + lid.y;
  let kTiles = (params.cols + 31u) / 32u;
  var sum = 0.0;

  for (var kTile = 0u; kTile < kTiles; kTile = kTile + 1u) {
    let kBase = kTile * 32u;
    for (var i = localIndex; i < 256u; i = i + 64u) {
      let batchLocal = i / 32u;
      let kLocal = i % 32u;
      let sourceBatch = wid.y * 8u + batchLocal;
      let col = kBase + kLocal;
      var x = 0.0;
      if (sourceBatch < params.batch && col < params.cols) {
        x = leaky_relu(input[sourceBatch * params.inputStride + col]);
      }
      inputTile[i] = x;
    }
    for (var i = localIndex; i < 256u; i = i + 64u) {
      let kLocal = i / 8u;
      let rowLocal = i % 8u;
      weightTile[i] = packedMatrix[((wid.x * kTiles + kTile) * 32u + kLocal) * 8u + rowLocal];
    }
    workgroupBarrier();

    for (var kLocal = 0u; kLocal < 32u; kLocal = kLocal + 1u) {
      sum = sum + inputTile[lid.y * 32u + kLocal] * weightTile[kLocal * 8u + lid.x];
    }
    workgroupBarrier();
  }

  if (row < params.rows && batch < params.batch) {
    var out = sum;
    if (params.biasPresent != 0u) {
      out = out + bias[row];
    }
    let outputBase = batch * params.outputStride + row;
    output[outputBase] = residual[outputBase] + bitcast<f32>(params.alphaBits) * out;
  }
}
`;
