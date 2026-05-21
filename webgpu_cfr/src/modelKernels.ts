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

function removeMatVecBatchRowBounds(source: string): string {
  return source
    .replace(
      `    if (row0 < params.rows) {
      sum0 = sum0 + matrix[row0 * params.cols + col] * x;
    }`,
      `    sum0 = sum0 + matrix[row0 * params.cols + col] * x;`,
    )
    .replace(
      `    if (row1 < params.rows) {
      sum1 = sum1 + matrix[row1 * params.cols + col] * x;
    }`,
      `    sum1 = sum1 + matrix[row1 * params.cols + col] * x;`,
    )
    .replace(
      `    if (row2 < params.rows) {
      sum2 = sum2 + matrix[row2 * params.cols + col] * x;
    }`,
      `    sum2 = sum2 + matrix[row2 * params.cols + col] * x;`,
    )
    .replace(
      `    if (row3 < params.rows) {
      sum3 = sum3 + matrix[row3 * params.cols + col] * x;
    }`,
      `    sum3 = sum3 + matrix[row3 * params.cols + col] * x;`,
    )
    .replace(
      `    if (row0 < params.rows) {
      var out0 = partial0[0];
      if (params.biasPresent != 0u) {
        out0 = out0 + bias[row0];
      }
      output[outputBase + row0] = out0;
    }`,
      `    var out0 = partial0[0];
    if (params.biasPresent != 0u) {
      out0 = out0 + bias[row0];
    }
    output[outputBase + row0] = out0;`,
    )
    .replace(
      `    if (row1 < params.rows) {
      var out1 = partial1[0];
      if (params.biasPresent != 0u) {
        out1 = out1 + bias[row1];
      }
      output[outputBase + row1] = out1;
    }`,
      `    var out1 = partial1[0];
    if (params.biasPresent != 0u) {
      out1 = out1 + bias[row1];
    }
    output[outputBase + row1] = out1;`,
    )
    .replace(
      `    if (row2 < params.rows) {
      var out2 = partial2[0];
      if (params.biasPresent != 0u) {
        out2 = out2 + bias[row2];
      }
      output[outputBase + row2] = out2;
    }`,
      `    var out2 = partial2[0];
    if (params.biasPresent != 0u) {
      out2 = out2 + bias[row2];
    }
    output[outputBase + row2] = out2;`,
    )
    .replace(
      `    if (row3 < params.rows) {
      var out3 = partial3[0];
      if (params.biasPresent != 0u) {
        out3 = out3 + bias[row3];
      }
      output[outputBase + row3] = out3;
    }`,
      `    var out3 = partial3[0];
    if (params.biasPresent != 0u) {
      out3 = out3 + bias[row3];
    }
    output[outputBase + row3] = out3;`,
    );
}

export const MAT_VEC_BATCH_EXACT_ROWS_WGSL =
  removeMatVecBatchRowBounds(MAT_VEC_BATCH_WGSL);

export const LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_WGSL =
  removeMatVecBatchRowBounds(LEAKY_RELU_MAT_VEC_BATCH_WGSL);

function unrollMatVecBatchColumns(
  source: string,
  cols: 512 | 1024,
  applyLeakyRelu: boolean,
): string {
  const xExpression = applyLeakyRelu
    ? "leaky_relu(input[inputBase + col])"
    : "input[inputBase + col]";
  const loopBlock = `  for (var col = lane; col < params.cols; col = col + 256u) {
    let x = ${xExpression};
    sum0 = sum0 + matrix[row0 * params.cols + col] * x;
    sum1 = sum1 + matrix[row1 * params.cols + col] * x;
    sum2 = sum2 + matrix[row2 * params.cols + col] * x;
    sum3 = sum3 + matrix[row3 * params.cols + col] * x;
  }`;
  const chunks: string[] = [];
  for (let offset = 0; offset < cols; offset += 256) {
    const suffix = offset === 0 ? "" : String(offset);
    const colExpr = offset === 0 ? "lane" : `lane + ${offset}u`;
    const rowStride = `${cols}u`;
    const valueExpr = applyLeakyRelu
      ? `leaky_relu(input[inputBase + col${suffix}])`
      : `input[inputBase + col${suffix}]`;
    chunks.push(`  let col${suffix} = ${colExpr};
  let x${suffix} = ${valueExpr};
  sum0 = sum0 + matrix[row0 * ${rowStride} + col${suffix}] * x${suffix};
  sum1 = sum1 + matrix[row1 * ${rowStride} + col${suffix}] * x${suffix};
  sum2 = sum2 + matrix[row2 * ${rowStride} + col${suffix}] * x${suffix};
  sum3 = sum3 + matrix[row3 * ${rowStride} + col${suffix}] * x${suffix};`);
  }
  return source.replace(loopBlock, chunks.join("\n"));
}

export const MAT_VEC_BATCH_EXACT_ROWS_COLS_512_WGSL =
  unrollMatVecBatchColumns(MAT_VEC_BATCH_EXACT_ROWS_WGSL, 512, false);
export const MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_WGSL =
  unrollMatVecBatchColumns(MAT_VEC_BATCH_EXACT_ROWS_WGSL, 1024, false);

export const MAT_VEC_BATCH_EXACT_ROWS_COLS_512_SUBGROUP_WGSL = /* wgsl */ `
enable subgroups;

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

var<workgroup> subgroupPartial0: array<f32, 256>;
var<workgroup> subgroupPartial1: array<f32, 256>;
var<workgroup> subgroupPartial2: array<f32, 256>;
var<workgroup> subgroupPartial3: array<f32, 256>;

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
  let batch = wid.y;
  let lane = lid.x;
  let inputBase = batch * params.inputStride + params.inputOffset;

  let col0 = lane;
  let x0 = input[inputBase + col0];
  var sum0 = matrix[row0 * 512u + col0] * x0;
  var sum1 = matrix[row1 * 512u + col0] * x0;
  var sum2 = matrix[row2 * 512u + col0] * x0;
  var sum3 = matrix[row3 * 512u + col0] * x0;

  let col1 = lane + 256u;
  let x1 = input[inputBase + col1];
  sum0 = sum0 + matrix[row0 * 512u + col1] * x1;
  sum1 = sum1 + matrix[row1 * 512u + col1] * x1;
  sum2 = sum2 + matrix[row2 * 512u + col1] * x1;
  sum3 = sum3 + matrix[row3 * 512u + col1] * x1;

  let reduced0 = subgroupAdd(sum0);
  let reduced1 = subgroupAdd(sum1);
  let reduced2 = subgroupAdd(sum2);
  let reduced3 = subgroupAdd(sum3);
  let subgroupIndex = lane / subgroupSize;
  if (subgroupLane == 0u) {
    subgroupPartial0[subgroupIndex] = reduced0;
    subgroupPartial1[subgroupIndex] = reduced1;
    subgroupPartial2[subgroupIndex] = reduced2;
    subgroupPartial3[subgroupIndex] = reduced3;
  }
  workgroupBarrier();

  if (lane == 0u) {
    let subgroupCount = (256u + subgroupSize - 1u) / subgroupSize;
    var out0 = 0.0;
    var out1 = 0.0;
    var out2 = 0.0;
    var out3 = 0.0;
    for (var i = 0u; i < subgroupCount; i = i + 1u) {
      out0 = out0 + subgroupPartial0[i];
      out1 = out1 + subgroupPartial1[i];
      out2 = out2 + subgroupPartial2[i];
      out3 = out3 + subgroupPartial3[i];
    }
    if (params.biasPresent != 0u) {
      out0 = out0 + bias[row0];
      out1 = out1 + bias[row1];
      out2 = out2 + bias[row2];
      out3 = out3 + bias[row3];
    }
    let outputBase = batch * params.outputStride + params.outputOffset;
    output[outputBase + row0] = out0;
    output[outputBase + row1] = out1;
    output[outputBase + row2] = out2;
    output[outputBase + row3] = out3;
  }
}
`;

export const LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_COLS_512_WGSL =
  unrollMatVecBatchColumns(LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_WGSL, 512, true);
export const LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_WGSL =
  unrollMatVecBatchColumns(LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_WGSL, 1024, true);

function addLeakyReluResidualOutput(source: string): string {
  return source
    .replace(
      `  biasPresent: u32,
  _pad0: u32,
  _pad1: u32,`,
      `  biasPresent: u32,
  alpha: f32,
  _pad0: u32,`,
    )
    .replace(
      "@group(0) @binding(4) var<uniform> params: Params;",
      `@group(0) @binding(4) var<storage, read> residual: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;`,
    )
    .replaceAll(
      "output[outputBase + row0] = out0;",
      "output[outputBase + row0] = residual[outputBase + row0] + params.alpha * out0;",
    )
    .replaceAll(
      "output[outputBase + row1] = out1;",
      "output[outputBase + row1] = residual[outputBase + row1] + params.alpha * out1;",
    )
    .replaceAll(
      "output[outputBase + row2] = out2;",
      "output[outputBase + row2] = residual[outputBase + row2] + params.alpha * out2;",
    )
    .replaceAll(
      "output[outputBase + row3] = out3;",
      "output[outputBase + row3] = residual[outputBase + row3] + params.alpha * out3;",
    );
}

export const LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_WGSL =
  addLeakyReluResidualOutput(LEAKY_RELU_MAT_VEC_BATCH_WGSL);
export const LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_EXACT_ROWS_WGSL =
  addLeakyReluResidualOutput(LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_WGSL);
export const LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_EXACT_ROWS_COLS_512_WGSL =
  addLeakyReluResidualOutput(LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_COLS_512_WGSL);
export const LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_WGSL =
  addLeakyReluResidualOutput(LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_WGSL);

export const LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_SUBGROUP_WGSL = /* wgsl */ `
enable subgroups;

struct Params {
  rows: u32,
  cols: u32,
  batch: u32,
  inputStride: u32,
  outputStride: u32,
  biasPresent: u32,
  alpha: f32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> matrix: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<storage, read> residual: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> subgroupPartial0: array<f32, 256>;
var<workgroup> subgroupPartial1: array<f32, 256>;
var<workgroup> subgroupPartial2: array<f32, 256>;
var<workgroup> subgroupPartial3: array<f32, 256>;

fn leaky_relu(x: f32) -> f32 {
  return select(0.01 * x, x, x >= 0.0);
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
  let batch = wid.y;
  let lane = lid.x;
  let inputBase = batch * params.inputStride;

  let col0 = lane;
  let x0 = leaky_relu(input[inputBase + col0]);
  var sum0 = matrix[row0 * 1024u + col0] * x0;
  var sum1 = matrix[row1 * 1024u + col0] * x0;
  var sum2 = matrix[row2 * 1024u + col0] * x0;
  var sum3 = matrix[row3 * 1024u + col0] * x0;

  let col1 = lane + 256u;
  let x1 = leaky_relu(input[inputBase + col1]);
  sum0 = sum0 + matrix[row0 * 1024u + col1] * x1;
  sum1 = sum1 + matrix[row1 * 1024u + col1] * x1;
  sum2 = sum2 + matrix[row2 * 1024u + col1] * x1;
  sum3 = sum3 + matrix[row3 * 1024u + col1] * x1;

  let col2 = lane + 512u;
  let x2 = leaky_relu(input[inputBase + col2]);
  sum0 = sum0 + matrix[row0 * 1024u + col2] * x2;
  sum1 = sum1 + matrix[row1 * 1024u + col2] * x2;
  sum2 = sum2 + matrix[row2 * 1024u + col2] * x2;
  sum3 = sum3 + matrix[row3 * 1024u + col2] * x2;

  let col3 = lane + 768u;
  let x3 = leaky_relu(input[inputBase + col3]);
  sum0 = sum0 + matrix[row0 * 1024u + col3] * x3;
  sum1 = sum1 + matrix[row1 * 1024u + col3] * x3;
  sum2 = sum2 + matrix[row2 * 1024u + col3] * x3;
  sum3 = sum3 + matrix[row3 * 1024u + col3] * x3;

  let reduced0 = subgroupAdd(sum0);
  let reduced1 = subgroupAdd(sum1);
  let reduced2 = subgroupAdd(sum2);
  let reduced3 = subgroupAdd(sum3);
  let subgroupIndex = lane / subgroupSize;
  if (subgroupLane == 0u) {
    subgroupPartial0[subgroupIndex] = reduced0;
    subgroupPartial1[subgroupIndex] = reduced1;
    subgroupPartial2[subgroupIndex] = reduced2;
    subgroupPartial3[subgroupIndex] = reduced3;
  }
  workgroupBarrier();

  if (lane == 0u) {
    let subgroupCount = (256u + subgroupSize - 1u) / subgroupSize;
    var out0 = 0.0;
    var out1 = 0.0;
    var out2 = 0.0;
    var out3 = 0.0;
    for (var i = 0u; i < subgroupCount; i = i + 1u) {
      out0 = out0 + subgroupPartial0[i];
      out1 = out1 + subgroupPartial1[i];
      out2 = out2 + subgroupPartial2[i];
      out3 = out3 + subgroupPartial3[i];
    }
    if (params.biasPresent != 0u) {
      out0 = out0 + bias[row0];
      out1 = out1 + bias[row1];
      out2 = out2 + bias[row2];
      out3 = out3 + bias[row3];
    }
    let outputBase = batch * params.outputStride;
    output[outputBase + row0] = residual[outputBase + row0] + params.alpha * out0;
    output[outputBase + row1] = residual[outputBase + row1] + params.alpha * out1;
    output[outputBase + row2] = residual[outputBase + row2] + params.alpha * out2;
    output[outputBase + row3] = residual[outputBase + row3] + params.alpha * out3;
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
