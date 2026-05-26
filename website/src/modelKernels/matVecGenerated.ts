import { LEAKY_RELU_MAT_VEC_BATCH_WGSL, MAT_VEC_BATCH_WGSL } from "./matVec.js";
import { REDUCE_4X_256_WGSL } from "./reductions.js";

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

export const MAT_VEC_BATCH_EXACT_ROWS_WGSL = removeMatVecBatchRowBounds(MAT_VEC_BATCH_WGSL);

export const LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_WGSL = removeMatVecBatchRowBounds(
  LEAKY_RELU_MAT_VEC_BATCH_WGSL,
);

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

export const MAT_VEC_BATCH_EXACT_ROWS_COLS_512_WGSL = unrollMatVecBatchColumns(
  MAT_VEC_BATCH_EXACT_ROWS_WGSL,
  512,
  false,
);
export const MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_WGSL = unrollMatVecBatchColumns(
  MAT_VEC_BATCH_EXACT_ROWS_WGSL,
  1024,
  false,
);

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

export const MAT_VEC_BATCH_EXACT_ROWS_COLS_512_BATCH2_SUBGROUP_WGSL = /* wgsl */ `
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

var<workgroup> subgroupPartial00: array<f32, 64>;
var<workgroup> subgroupPartial01: array<f32, 64>;
var<workgroup> subgroupPartial02: array<f32, 64>;
var<workgroup> subgroupPartial03: array<f32, 64>;
var<workgroup> subgroupPartial10: array<f32, 64>;
var<workgroup> subgroupPartial11: array<f32, 64>;
var<workgroup> subgroupPartial12: array<f32, 64>;
var<workgroup> subgroupPartial13: array<f32, 64>;

@compute @workgroup_size(64)
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
  let inputBase0 = batch0 * params.inputStride + params.inputOffset;
  let inputBase1 = batch1 * params.inputStride + params.inputOffset;

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

  let col1 = lane + 64u;
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

  let col2 = lane + 128u;
  let x02 = input[inputBase0 + col2];
  var x12 = 0.0;
  if (batch1 < params.batch) {
    x12 = input[inputBase1 + col2];
  }
  let m02 = matrix[row0 * 512u + col2];
  let m12 = matrix[row1 * 512u + col2];
  let m22 = matrix[row2 * 512u + col2];
  let m32 = matrix[row3 * 512u + col2];
  sum00 = sum00 + m02 * x02;
  sum01 = sum01 + m12 * x02;
  sum02 = sum02 + m22 * x02;
  sum03 = sum03 + m32 * x02;
  sum10 = sum10 + m02 * x12;
  sum11 = sum11 + m12 * x12;
  sum12 = sum12 + m22 * x12;
  sum13 = sum13 + m32 * x12;

  let col3 = lane + 192u;
  let x03 = input[inputBase0 + col3];
  var x13 = 0.0;
  if (batch1 < params.batch) {
    x13 = input[inputBase1 + col3];
  }
  let m03 = matrix[row0 * 512u + col3];
  let m13 = matrix[row1 * 512u + col3];
  let m23 = matrix[row2 * 512u + col3];
  let m33 = matrix[row3 * 512u + col3];
  sum00 = sum00 + m03 * x03;
  sum01 = sum01 + m13 * x03;
  sum02 = sum02 + m23 * x03;
  sum03 = sum03 + m33 * x03;
  sum10 = sum10 + m03 * x13;
  sum11 = sum11 + m13 * x13;
  sum12 = sum12 + m23 * x13;
  sum13 = sum13 + m33 * x13;

  let col4 = lane + 256u;
  let x04 = input[inputBase0 + col4];
  var x14 = 0.0;
  if (batch1 < params.batch) {
    x14 = input[inputBase1 + col4];
  }
  let m04 = matrix[row0 * 512u + col4];
  let m14 = matrix[row1 * 512u + col4];
  let m24 = matrix[row2 * 512u + col4];
  let m34 = matrix[row3 * 512u + col4];
  sum00 = sum00 + m04 * x04;
  sum01 = sum01 + m14 * x04;
  sum02 = sum02 + m24 * x04;
  sum03 = sum03 + m34 * x04;
  sum10 = sum10 + m04 * x14;
  sum11 = sum11 + m14 * x14;
  sum12 = sum12 + m24 * x14;
  sum13 = sum13 + m34 * x14;

  let col5 = lane + 320u;
  let x05 = input[inputBase0 + col5];
  var x15 = 0.0;
  if (batch1 < params.batch) {
    x15 = input[inputBase1 + col5];
  }
  let m05 = matrix[row0 * 512u + col5];
  let m15 = matrix[row1 * 512u + col5];
  let m25 = matrix[row2 * 512u + col5];
  let m35 = matrix[row3 * 512u + col5];
  sum00 = sum00 + m05 * x05;
  sum01 = sum01 + m15 * x05;
  sum02 = sum02 + m25 * x05;
  sum03 = sum03 + m35 * x05;
  sum10 = sum10 + m05 * x15;
  sum11 = sum11 + m15 * x15;
  sum12 = sum12 + m25 * x15;
  sum13 = sum13 + m35 * x15;

  let col6 = lane + 384u;
  let x06 = input[inputBase0 + col6];
  var x16 = 0.0;
  if (batch1 < params.batch) {
    x16 = input[inputBase1 + col6];
  }
  let m06 = matrix[row0 * 512u + col6];
  let m16 = matrix[row1 * 512u + col6];
  let m26 = matrix[row2 * 512u + col6];
  let m36 = matrix[row3 * 512u + col6];
  sum00 = sum00 + m06 * x06;
  sum01 = sum01 + m16 * x06;
  sum02 = sum02 + m26 * x06;
  sum03 = sum03 + m36 * x06;
  sum10 = sum10 + m06 * x16;
  sum11 = sum11 + m16 * x16;
  sum12 = sum12 + m26 * x16;
  sum13 = sum13 + m36 * x16;

  let col7 = lane + 448u;
  let x07 = input[inputBase0 + col7];
  var x17 = 0.0;
  if (batch1 < params.batch) {
    x17 = input[inputBase1 + col7];
  }
  let m07 = matrix[row0 * 512u + col7];
  let m17 = matrix[row1 * 512u + col7];
  let m27 = matrix[row2 * 512u + col7];
  let m37 = matrix[row3 * 512u + col7];
  sum00 = sum00 + m07 * x07;
  sum01 = sum01 + m17 * x07;
  sum02 = sum02 + m27 * x07;
  sum03 = sum03 + m37 * x07;
  sum10 = sum10 + m07 * x17;
  sum11 = sum11 + m17 * x17;
  sum12 = sum12 + m27 * x17;
  sum13 = sum13 + m37 * x17;

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
    let subgroupCount = (64u + subgroupSize - 1u) / subgroupSize;
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
    if (params.biasPresent != 0u) {
      out00 = out00 + bias[row0];
      out01 = out01 + bias[row1];
      out02 = out02 + bias[row2];
      out03 = out03 + bias[row3];
      out10 = out10 + bias[row0];
      out11 = out11 + bias[row1];
      out12 = out12 + bias[row2];
      out13 = out13 + bias[row3];
    }
    let outputBase0 = batch0 * params.outputStride + params.outputOffset;
    output[outputBase0 + row0] = out00;
    output[outputBase0 + row1] = out01;
    output[outputBase0 + row2] = out02;
    output[outputBase0 + row3] = out03;
    if (batch1 < params.batch) {
      let outputBase1 = batch1 * params.outputStride + params.outputOffset;
      output[outputBase1 + row0] = out10;
      output[outputBase1 + row1] = out11;
      output[outputBase1 + row2] = out12;
      output[outputBase1 + row3] = out13;
    }
  }
}
`;

function makeMatVecBatchExactRowsCols512BatchSubgroup(batchCount: number): string {
  const rows = [0, 1, 2, 3];
  const batches = Array.from({ length: batchCount }, (_, batch) => batch);
  const cells = batches.flatMap((batch) =>
    rows.map((row) => ({ batch, row, name: `${batch}${row}` })),
  );
  const partials = cells
    .map(({ name }) => `var<workgroup> subgroupPartial${name}: array<f32, 64>;`)
    .join("\n");
  const rowLets = rows
    .slice(1)
    .map((row) => `  let row${row} = row0 + ${row}u;`)
    .join("\n");
  const batchLets = batches
    .map((batch) =>
      batch === 0
        ? `  let batch0 = wid.y * ${batchCount}u;`
        : `  let batch${batch} = batch0 + ${batch}u;`,
    )
    .join("\n");
  const inputBaseLets = batches
    .map(
      (batch) =>
        `  let inputBase${batch} = batch${batch} * params.inputStride + params.inputOffset;`,
    )
    .join("\n");
  const chunkBlocks = Array.from({ length: 8 }, (_, chunk) => {
    const inputs = batches
      .map((batch) =>
        batch === 0
          ? `  let x${batch}${chunk} = input[inputBase${batch} + col${chunk}];`
          : `  var x${batch}${chunk} = 0.0;
  if (batch${batch} < params.batch) {
    x${batch}${chunk} = input[inputBase${batch} + col${chunk}];
  }`,
      )
      .join("\n");
    const matrixLoads = rows
      .map((row) => `  let m${row}${chunk} = matrix[row${row} * 512u + col${chunk}];`)
      .join("\n");
    const ops = cells
      .map(({ batch, row, name }) =>
        chunk === 0
          ? `  var sum${name} = m${row}${chunk} * x${batch}${chunk};`
          : `  sum${name} = sum${name} + m${row}${chunk} * x${batch}${chunk};`,
      )
      .join("\n");
    return `  let col${chunk} = lane + ${chunk * 64}u;
${inputs}
${matrixLoads}
${ops}`;
  }).join("\n\n");
  const reductions = cells
    .map(({ name }) => `  let reduced${name} = subgroupAdd(sum${name});`)
    .join("\n");
  const partialWrites = cells
    .map(({ name }) => `    subgroupPartial${name}[subgroupIndex] = reduced${name};`)
    .join("\n");
  const outDecls = cells.map(({ name }) => `    var out${name} = 0.0;`).join("\n");
  const outAdds = cells
    .map(({ name }) => `      out${name} = out${name} + subgroupPartial${name}[i];`)
    .join("\n");
  const biasAdds = cells
    .map(({ row, name }) => `      out${name} = out${name} + bias[row${row}];`)
    .join("\n");
  const writes = batches
    .map((batch) => {
      const rowWrites = rows
        .map((row) => `      output[outputBase${batch} + row${row}] = out${batch}${row};`)
        .join("\n");
      if (batch === 0) {
        return `    let outputBase0 = batch0 * params.outputStride + params.outputOffset;
${rowWrites}`;
      }
      return `    if (batch${batch} < params.batch) {
      let outputBase${batch} = batch${batch} * params.outputStride + params.outputOffset;
${rowWrites}
    }`;
    })
    .join("\n");

  return /* wgsl */ `
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

${partials}

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) subgroupLane: u32,
  @builtin(subgroup_size) subgroupSize: u32,
) {
  let row0 = wid.x * 4u;
${rowLets}
${batchLets}
  let lane = lid.x;
${inputBaseLets}

${chunkBlocks}

${reductions}
  let subgroupIndex = lane / subgroupSize;
  if (subgroupLane == 0u) {
${partialWrites}
  }
  workgroupBarrier();

  if (lane == 0u) {
    let subgroupCount = (64u + subgroupSize - 1u) / subgroupSize;
${outDecls}
    for (var i = 0u; i < subgroupCount; i = i + 1u) {
${outAdds}
    }
    if (params.biasPresent != 0u) {
${biasAdds}
    }
${writes}
  }
}
`;
}

export const MAT_VEC_BATCH_EXACT_ROWS_COLS_512_BATCH3_SUBGROUP_WGSL =
  makeMatVecBatchExactRowsCols512BatchSubgroup(3);
export const MAT_VEC_BATCH_EXACT_ROWS_COLS_512_BATCH4_SUBGROUP_WGSL =
  makeMatVecBatchExactRowsCols512BatchSubgroup(4);

export const MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_BATCH2_SUBGROUP_WGSL = /* wgsl */ `
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

var<workgroup> subgroupPartial00: array<f32, 128>;
var<workgroup> subgroupPartial01: array<f32, 128>;
var<workgroup> subgroupPartial02: array<f32, 128>;
var<workgroup> subgroupPartial03: array<f32, 128>;
var<workgroup> subgroupPartial10: array<f32, 128>;
var<workgroup> subgroupPartial11: array<f32, 128>;
var<workgroup> subgroupPartial12: array<f32, 128>;
var<workgroup> subgroupPartial13: array<f32, 128>;

@compute @workgroup_size(128)
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
  let inputBase0 = batch0 * params.inputStride + params.inputOffset;
  let inputBase1 = batch1 * params.inputStride + params.inputOffset;

  let col0 = lane;
  let x00 = input[inputBase0 + col0];
  var x10 = 0.0;
  if (batch1 < params.batch) {
    x10 = input[inputBase1 + col0];
  }
  let m00 = matrix[row0 * 1024u + col0];
  let m10 = matrix[row1 * 1024u + col0];
  let m20 = matrix[row2 * 1024u + col0];
  let m30 = matrix[row3 * 1024u + col0];
  var sum00 = m00 * x00;
  var sum01 = m10 * x00;
  var sum02 = m20 * x00;
  var sum03 = m30 * x00;
  var sum10 = m00 * x10;
  var sum11 = m10 * x10;
  var sum12 = m20 * x10;
  var sum13 = m30 * x10;

  let col1 = lane + 128u;
  let x01 = input[inputBase0 + col1];
  var x11 = 0.0;
  if (batch1 < params.batch) {
    x11 = input[inputBase1 + col1];
  }
  let m01 = matrix[row0 * 1024u + col1];
  let m11 = matrix[row1 * 1024u + col1];
  let m21 = matrix[row2 * 1024u + col1];
  let m31 = matrix[row3 * 1024u + col1];
  sum00 = sum00 + m01 * x01;
  sum01 = sum01 + m11 * x01;
  sum02 = sum02 + m21 * x01;
  sum03 = sum03 + m31 * x01;
  sum10 = sum10 + m01 * x11;
  sum11 = sum11 + m11 * x11;
  sum12 = sum12 + m21 * x11;
  sum13 = sum13 + m31 * x11;

  let col2 = lane + 256u;
  let x02 = input[inputBase0 + col2];
  var x12 = 0.0;
  if (batch1 < params.batch) {
    x12 = input[inputBase1 + col2];
  }
  let m02 = matrix[row0 * 1024u + col2];
  let m12 = matrix[row1 * 1024u + col2];
  let m22 = matrix[row2 * 1024u + col2];
  let m32 = matrix[row3 * 1024u + col2];
  sum00 = sum00 + m02 * x02;
  sum01 = sum01 + m12 * x02;
  sum02 = sum02 + m22 * x02;
  sum03 = sum03 + m32 * x02;
  sum10 = sum10 + m02 * x12;
  sum11 = sum11 + m12 * x12;
  sum12 = sum12 + m22 * x12;
  sum13 = sum13 + m32 * x12;

  let col3 = lane + 384u;
  let x03 = input[inputBase0 + col3];
  var x13 = 0.0;
  if (batch1 < params.batch) {
    x13 = input[inputBase1 + col3];
  }
  let m03 = matrix[row0 * 1024u + col3];
  let m13 = matrix[row1 * 1024u + col3];
  let m23 = matrix[row2 * 1024u + col3];
  let m33 = matrix[row3 * 1024u + col3];
  sum00 = sum00 + m03 * x03;
  sum01 = sum01 + m13 * x03;
  sum02 = sum02 + m23 * x03;
  sum03 = sum03 + m33 * x03;
  sum10 = sum10 + m03 * x13;
  sum11 = sum11 + m13 * x13;
  sum12 = sum12 + m23 * x13;
  sum13 = sum13 + m33 * x13;

  let col4 = lane + 512u;
  let x04 = input[inputBase0 + col4];
  var x14 = 0.0;
  if (batch1 < params.batch) {
    x14 = input[inputBase1 + col4];
  }
  let m04 = matrix[row0 * 1024u + col4];
  let m14 = matrix[row1 * 1024u + col4];
  let m24 = matrix[row2 * 1024u + col4];
  let m34 = matrix[row3 * 1024u + col4];
  sum00 = sum00 + m04 * x04;
  sum01 = sum01 + m14 * x04;
  sum02 = sum02 + m24 * x04;
  sum03 = sum03 + m34 * x04;
  sum10 = sum10 + m04 * x14;
  sum11 = sum11 + m14 * x14;
  sum12 = sum12 + m24 * x14;
  sum13 = sum13 + m34 * x14;

  let col5 = lane + 640u;
  let x05 = input[inputBase0 + col5];
  var x15 = 0.0;
  if (batch1 < params.batch) {
    x15 = input[inputBase1 + col5];
  }
  let m05 = matrix[row0 * 1024u + col5];
  let m15 = matrix[row1 * 1024u + col5];
  let m25 = matrix[row2 * 1024u + col5];
  let m35 = matrix[row3 * 1024u + col5];
  sum00 = sum00 + m05 * x05;
  sum01 = sum01 + m15 * x05;
  sum02 = sum02 + m25 * x05;
  sum03 = sum03 + m35 * x05;
  sum10 = sum10 + m05 * x15;
  sum11 = sum11 + m15 * x15;
  sum12 = sum12 + m25 * x15;
  sum13 = sum13 + m35 * x15;

  let col6 = lane + 768u;
  let x06 = input[inputBase0 + col6];
  var x16 = 0.0;
  if (batch1 < params.batch) {
    x16 = input[inputBase1 + col6];
  }
  let m06 = matrix[row0 * 1024u + col6];
  let m16 = matrix[row1 * 1024u + col6];
  let m26 = matrix[row2 * 1024u + col6];
  let m36 = matrix[row3 * 1024u + col6];
  sum00 = sum00 + m06 * x06;
  sum01 = sum01 + m16 * x06;
  sum02 = sum02 + m26 * x06;
  sum03 = sum03 + m36 * x06;
  sum10 = sum10 + m06 * x16;
  sum11 = sum11 + m16 * x16;
  sum12 = sum12 + m26 * x16;
  sum13 = sum13 + m36 * x16;

  let col7 = lane + 896u;
  let x07 = input[inputBase0 + col7];
  var x17 = 0.0;
  if (batch1 < params.batch) {
    x17 = input[inputBase1 + col7];
  }
  let m07 = matrix[row0 * 1024u + col7];
  let m17 = matrix[row1 * 1024u + col7];
  let m27 = matrix[row2 * 1024u + col7];
  let m37 = matrix[row3 * 1024u + col7];
  sum00 = sum00 + m07 * x07;
  sum01 = sum01 + m17 * x07;
  sum02 = sum02 + m27 * x07;
  sum03 = sum03 + m37 * x07;
  sum10 = sum10 + m07 * x17;
  sum11 = sum11 + m17 * x17;
  sum12 = sum12 + m27 * x17;
  sum13 = sum13 + m37 * x17;

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
    let subgroupCount = (128u + subgroupSize - 1u) / subgroupSize;
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
    if (params.biasPresent != 0u) {
      out00 = out00 + bias[row0];
      out01 = out01 + bias[row1];
      out02 = out02 + bias[row2];
      out03 = out03 + bias[row3];
      out10 = out10 + bias[row0];
      out11 = out11 + bias[row1];
      out12 = out12 + bias[row2];
      out13 = out13 + bias[row3];
    }
    let outputBase0 = batch0 * params.outputStride + params.outputOffset;
    output[outputBase0 + row0] = out00;
    output[outputBase0 + row1] = out01;
    output[outputBase0 + row2] = out02;
    output[outputBase0 + row3] = out03;
    if (batch1 < params.batch) {
      let outputBase1 = batch1 * params.outputStride + params.outputOffset;
      output[outputBase1 + row0] = out10;
      output[outputBase1 + row1] = out11;
      output[outputBase1 + row2] = out12;
      output[outputBase1 + row3] = out13;
    }
  }
}
`;

function makeMatVecBatchExactRowsCols1326BatchSubgroup(batchCount: number): string {
  const rows = [0, 1, 2, 3];
  const batches = Array.from({ length: batchCount }, (_, batch) => batch);
  const cells = batches.flatMap((batch) =>
    rows.map((row) => ({ batch, row, name: `${batch}${row}` })),
  );
  const partials = cells
    .map(({ name }) => `var<workgroup> subgroupPartial${name}: array<f32, 128>;`)
    .join("\n");
  const rowLets = rows
    .slice(1)
    .map((row) => `  let row${row} = row0 + ${row}u;`)
    .join("\n");
  const batchLets = batches
    .map((batch) =>
      batch === 0
        ? `  let batch0 = wid.y * ${batchCount}u;`
        : `  let batch${batch} = batch0 + ${batch}u;`,
    )
    .join("\n");
  const inputBaseLets = batches
    .map(
      (batch) =>
        `  let inputBase${batch} = batch${batch} * params.inputStride + params.inputOffset;`,
    )
    .join("\n");
  const firstInputs = batches
    .map((batch) =>
      batch === 0
        ? `  let x${batch}0 = input[inputBase${batch} + col0];`
        : `  var x${batch}0 = 0.0;
  if (batch${batch} < params.batch) {
    x${batch}0 = input[inputBase${batch} + col0];
  }`,
    )
    .join("\n");
  const firstSums = cells
    .map(({ batch, row, name }) => `  var sum${name} = m${row}0 * x${batch}0;`)
    .join("\n");
  const chunks = [1, 2, 3, 4]
    .map((chunk) => {
      const inputs = batches
        .map((batch) =>
          batch === 0
            ? `  let x${batch}${chunk} = input[inputBase${batch} + col${chunk}];`
            : `  var x${batch}${chunk} = 0.0;
  if (batch${batch} < params.batch) {
    x${batch}${chunk} = input[inputBase${batch} + col${chunk}];
  }`,
        )
        .join("\n");
      const matrixLoads = rows
        .map((row) => `  let m${row}${chunk} = matrix[row${row} * 1326u + col${chunk}];`)
        .join("\n");
      const adds = cells
        .map(
          ({ batch, row, name }) =>
            `  sum${name} = sum${name} + m${row}${chunk} * x${batch}${chunk};`,
        )
        .join("\n");
      return `  let col${chunk} = lane + ${chunk * 256}u;
${inputs}
${matrixLoads}
${adds}`;
    })
    .join("\n\n");
  const lastInputs = batches
    .map((batch) =>
      batch === 0
        ? `    let x${batch}5 = input[inputBase${batch} + col5];`
        : `    var x${batch}5 = 0.0;
    if (batch${batch} < params.batch) {
      x${batch}5 = input[inputBase${batch} + col5];
    }`,
    )
    .join("\n");
  const lastAdds = cells
    .map(({ batch, row, name }) => `    sum${name} = sum${name} + m${row}5 * x${batch}5;`)
    .join("\n");
  const reductions = cells
    .map(({ name }) => `  let reduced${name} = subgroupAdd(sum${name});`)
    .join("\n");
  const partialWrites = cells
    .map(({ name }) => `    subgroupPartial${name}[subgroupIndex] = reduced${name};`)
    .join("\n");
  const outDecls = cells.map(({ name }) => `    var out${name} = 0.0;`).join("\n");
  const outAdds = cells
    .map(({ name }) => `      out${name} = out${name} + subgroupPartial${name}[i];`)
    .join("\n");
  const biasAdds = cells
    .map(({ row, name }) => `      out${name} = out${name} + bias[row${row}];`)
    .join("\n");
  const writes = batches
    .map((batch) => {
      const rowWrites = rows
        .map((row) => `      output[outputBase${batch} + row${row}] = out${batch}${row};`)
        .join("\n");
      if (batch === 0) {
        return `    let outputBase0 = batch0 * params.outputStride + params.outputOffset;
${rowWrites}`;
      }
      return `    if (batch${batch} < params.batch) {
      let outputBase${batch} = batch${batch} * params.outputStride + params.outputOffset;
${rowWrites}
    }`;
    })
    .join("\n");

  return /* wgsl */ `
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

${partials}

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) subgroupLane: u32,
  @builtin(subgroup_size) subgroupSize: u32,
) {
  let row0 = wid.x * 4u;
${rowLets}
${batchLets}
  let lane = lid.x;
${inputBaseLets}

  let col0 = lane;
${firstInputs}
  let m00 = matrix[row0 * 1326u + col0];
  let m10 = matrix[row1 * 1326u + col0];
  let m20 = matrix[row2 * 1326u + col0];
  let m30 = matrix[row3 * 1326u + col0];
${firstSums}

${chunks}

  let col5 = lane + 1280u;
  if (col5 < 1326u) {
${lastInputs}
    let m05 = matrix[row0 * 1326u + col5];
    let m15 = matrix[row1 * 1326u + col5];
    let m25 = matrix[row2 * 1326u + col5];
    let m35 = matrix[row3 * 1326u + col5];
${lastAdds}
  }

${reductions}
  let subgroupIndex = lane / subgroupSize;
  if (subgroupLane == 0u) {
${partialWrites}
  }
  workgroupBarrier();

  if (lane == 0u) {
    let subgroupCount = (256u + subgroupSize - 1u) / subgroupSize;
${outDecls}
    for (var i = 0u; i < subgroupCount; i = i + 1u) {
${outAdds}
    }
    if (params.biasPresent != 0u) {
${biasAdds}
    }
${writes}
  }
}
`;
}

export const MAT_VEC_BATCH_EXACT_ROWS_COLS_1326_BATCH2_SUBGROUP_WGSL =
  makeMatVecBatchExactRowsCols1326BatchSubgroup(2);
export const MAT_VEC_BATCH_EXACT_ROWS_COLS_1326_BATCH4_SUBGROUP_WGSL =
  makeMatVecBatchExactRowsCols1326BatchSubgroup(4);

export const LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_COLS_512_WGSL = unrollMatVecBatchColumns(
  LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_WGSL,
  512,
  true,
);
export const LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_WGSL = unrollMatVecBatchColumns(
  LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_WGSL,
  1024,
  true,
);

export const LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_COLS_512_SUBGROUP_WGSL = /* wgsl */ `
enable subgroups;

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
  var sum0 = matrix[row0 * 512u + col0] * x0;
  var sum1 = matrix[row1 * 512u + col0] * x0;
  var sum2 = matrix[row2 * 512u + col0] * x0;
  var sum3 = matrix[row3 * 512u + col0] * x0;

  let col1 = lane + 256u;
  let x1 = leaky_relu(input[inputBase + col1]);
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
    let outputBase = batch * params.outputStride;
    output[outputBase + row0] = out0;
    output[outputBase + row1] = out1;
    output[outputBase + row2] = out2;
    output[outputBase + row3] = out3;
  }
}
`;

export const LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_SUBGROUP_WGSL = /* wgsl */ `
enable subgroups;

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
    output[outputBase + row0] = out0;
    output[outputBase + row1] = out1;
    output[outputBase + row2] = out2;
    output[outputBase + row3] = out3;
  }
}
`;

export const LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_BATCH2_SUBGROUP_WGSL = /* wgsl */ `
enable subgroups;

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

var<workgroup> subgroupPartial00: array<f32, 128>;
var<workgroup> subgroupPartial01: array<f32, 128>;
var<workgroup> subgroupPartial02: array<f32, 128>;
var<workgroup> subgroupPartial03: array<f32, 128>;
var<workgroup> subgroupPartial10: array<f32, 128>;
var<workgroup> subgroupPartial11: array<f32, 128>;
var<workgroup> subgroupPartial12: array<f32, 128>;
var<workgroup> subgroupPartial13: array<f32, 128>;

fn leaky_relu(x: f32) -> f32 {
  return select(0.01 * x, x, x >= 0.0);
}

@compute @workgroup_size(128)
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
  let x00 = leaky_relu(input[inputBase0 + col0]);
  var x10 = 0.0;
  if (batch1 < params.batch) {
    x10 = leaky_relu(input[inputBase1 + col0]);
  }
  let m00 = matrix[row0 * 1024u + col0];
  let m10 = matrix[row1 * 1024u + col0];
  let m20 = matrix[row2 * 1024u + col0];
  let m30 = matrix[row3 * 1024u + col0];
  var sum00 = m00 * x00;
  var sum01 = m10 * x00;
  var sum02 = m20 * x00;
  var sum03 = m30 * x00;
  var sum10 = m00 * x10;
  var sum11 = m10 * x10;
  var sum12 = m20 * x10;
  var sum13 = m30 * x10;

  let col1 = lane + 128u;
  let x01 = leaky_relu(input[inputBase0 + col1]);
  var x11 = 0.0;
  if (batch1 < params.batch) {
    x11 = leaky_relu(input[inputBase1 + col1]);
  }
  let m01 = matrix[row0 * 1024u + col1];
  let m11 = matrix[row1 * 1024u + col1];
  let m21 = matrix[row2 * 1024u + col1];
  let m31 = matrix[row3 * 1024u + col1];
  sum00 = sum00 + m01 * x01;
  sum01 = sum01 + m11 * x01;
  sum02 = sum02 + m21 * x01;
  sum03 = sum03 + m31 * x01;
  sum10 = sum10 + m01 * x11;
  sum11 = sum11 + m11 * x11;
  sum12 = sum12 + m21 * x11;
  sum13 = sum13 + m31 * x11;

  let col2 = lane + 256u;
  let x02 = leaky_relu(input[inputBase0 + col2]);
  var x12 = 0.0;
  if (batch1 < params.batch) {
    x12 = leaky_relu(input[inputBase1 + col2]);
  }
  let m02 = matrix[row0 * 1024u + col2];
  let m12 = matrix[row1 * 1024u + col2];
  let m22 = matrix[row2 * 1024u + col2];
  let m32 = matrix[row3 * 1024u + col2];
  sum00 = sum00 + m02 * x02;
  sum01 = sum01 + m12 * x02;
  sum02 = sum02 + m22 * x02;
  sum03 = sum03 + m32 * x02;
  sum10 = sum10 + m02 * x12;
  sum11 = sum11 + m12 * x12;
  sum12 = sum12 + m22 * x12;
  sum13 = sum13 + m32 * x12;

  let col3 = lane + 384u;
  let x03 = leaky_relu(input[inputBase0 + col3]);
  var x13 = 0.0;
  if (batch1 < params.batch) {
    x13 = leaky_relu(input[inputBase1 + col3]);
  }
  let m03 = matrix[row0 * 1024u + col3];
  let m13 = matrix[row1 * 1024u + col3];
  let m23 = matrix[row2 * 1024u + col3];
  let m33 = matrix[row3 * 1024u + col3];
  sum00 = sum00 + m03 * x03;
  sum01 = sum01 + m13 * x03;
  sum02 = sum02 + m23 * x03;
  sum03 = sum03 + m33 * x03;
  sum10 = sum10 + m03 * x13;
  sum11 = sum11 + m13 * x13;
  sum12 = sum12 + m23 * x13;
  sum13 = sum13 + m33 * x13;

  let col4 = lane + 512u;
  let x04 = leaky_relu(input[inputBase0 + col4]);
  var x14 = 0.0;
  if (batch1 < params.batch) {
    x14 = leaky_relu(input[inputBase1 + col4]);
  }
  let m04 = matrix[row0 * 1024u + col4];
  let m14 = matrix[row1 * 1024u + col4];
  let m24 = matrix[row2 * 1024u + col4];
  let m34 = matrix[row3 * 1024u + col4];
  sum00 = sum00 + m04 * x04;
  sum01 = sum01 + m14 * x04;
  sum02 = sum02 + m24 * x04;
  sum03 = sum03 + m34 * x04;
  sum10 = sum10 + m04 * x14;
  sum11 = sum11 + m14 * x14;
  sum12 = sum12 + m24 * x14;
  sum13 = sum13 + m34 * x14;

  let col5 = lane + 640u;
  let x05 = leaky_relu(input[inputBase0 + col5]);
  var x15 = 0.0;
  if (batch1 < params.batch) {
    x15 = leaky_relu(input[inputBase1 + col5]);
  }
  let m05 = matrix[row0 * 1024u + col5];
  let m15 = matrix[row1 * 1024u + col5];
  let m25 = matrix[row2 * 1024u + col5];
  let m35 = matrix[row3 * 1024u + col5];
  sum00 = sum00 + m05 * x05;
  sum01 = sum01 + m15 * x05;
  sum02 = sum02 + m25 * x05;
  sum03 = sum03 + m35 * x05;
  sum10 = sum10 + m05 * x15;
  sum11 = sum11 + m15 * x15;
  sum12 = sum12 + m25 * x15;
  sum13 = sum13 + m35 * x15;

  let col6 = lane + 768u;
  let x06 = leaky_relu(input[inputBase0 + col6]);
  var x16 = 0.0;
  if (batch1 < params.batch) {
    x16 = leaky_relu(input[inputBase1 + col6]);
  }
  let m06 = matrix[row0 * 1024u + col6];
  let m16 = matrix[row1 * 1024u + col6];
  let m26 = matrix[row2 * 1024u + col6];
  let m36 = matrix[row3 * 1024u + col6];
  sum00 = sum00 + m06 * x06;
  sum01 = sum01 + m16 * x06;
  sum02 = sum02 + m26 * x06;
  sum03 = sum03 + m36 * x06;
  sum10 = sum10 + m06 * x16;
  sum11 = sum11 + m16 * x16;
  sum12 = sum12 + m26 * x16;
  sum13 = sum13 + m36 * x16;

  let col7 = lane + 896u;
  let x07 = leaky_relu(input[inputBase0 + col7]);
  var x17 = 0.0;
  if (batch1 < params.batch) {
    x17 = leaky_relu(input[inputBase1 + col7]);
  }
  let m07 = matrix[row0 * 1024u + col7];
  let m17 = matrix[row1 * 1024u + col7];
  let m27 = matrix[row2 * 1024u + col7];
  let m37 = matrix[row3 * 1024u + col7];
  sum00 = sum00 + m07 * x07;
  sum01 = sum01 + m17 * x07;
  sum02 = sum02 + m27 * x07;
  sum03 = sum03 + m37 * x07;
  sum10 = sum10 + m07 * x17;
  sum11 = sum11 + m17 * x17;
  sum12 = sum12 + m27 * x17;
  sum13 = sum13 + m37 * x17;

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
    let subgroupCount = (128u + subgroupSize - 1u) / subgroupSize;
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
    if (params.biasPresent != 0u) {
      out00 = out00 + bias[row0];
      out01 = out01 + bias[row1];
      out02 = out02 + bias[row2];
      out03 = out03 + bias[row3];
      out10 = out10 + bias[row0];
      out11 = out11 + bias[row1];
      out12 = out12 + bias[row2];
      out13 = out13 + bias[row3];
    }
    let outputBase0 = batch0 * params.outputStride;
    output[outputBase0 + row0] = out00;
    output[outputBase0 + row1] = out01;
    output[outputBase0 + row2] = out02;
    output[outputBase0 + row3] = out03;
    if (batch1 < params.batch) {
      let outputBase1 = batch1 * params.outputStride;
      output[outputBase1 + row0] = out10;
      output[outputBase1 + row1] = out11;
      output[outputBase1 + row2] = out12;
      output[outputBase1 + row3] = out13;
    }
  }
}
`;

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

export const LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_WGSL = addLeakyReluResidualOutput(
  LEAKY_RELU_MAT_VEC_BATCH_WGSL,
);
export const LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_EXACT_ROWS_WGSL = addLeakyReluResidualOutput(
  LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_WGSL,
);
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

function makeLeakyReluResidualMatVecBatchExactRowsCols1024BatchSubgroup(
  batchCount: number,
): string {
  const rows = [0, 1, 2, 3];
  const batches = Array.from({ length: batchCount }, (_, batch) => batch);
  const cells = batches.flatMap((batch) =>
    rows.map((row) => ({ batch, row, name: `${batch}${row}` })),
  );
  const partials = cells
    .map(({ name }) => `var<workgroup> subgroupPartial${name}: array<f32, 128>;`)
    .join("\n");
  const rowLets = rows
    .slice(1)
    .map((row) => `  let row${row} = row0 + ${row}u;`)
    .join("\n");
  const batchLets = batches
    .map((batch) =>
      batch === 0
        ? `  let batch0 = wid.y * ${batchCount}u;`
        : `  let batch${batch} = batch0 + ${batch}u;`,
    )
    .join("\n");
  const inputBaseLets = batches
    .map((batch) => `  let inputBase${batch} = batch${batch} * params.inputStride;`)
    .join("\n");
  const firstInputs = batches
    .map((batch) =>
      batch === 0
        ? `  let x${batch}0 = leaky_relu(input[inputBase${batch} + col0]);`
        : `  var x${batch}0 = 0.0;
  if (batch${batch} < params.batch) {
    x${batch}0 = leaky_relu(input[inputBase${batch} + col0]);
  }`,
    )
    .join("\n");
  const firstSums = batches
    .flatMap((batch) => rows.map((row) => `  var sum${batch}${row} = m${row}0 * x${batch}0;`))
    .join("\n");
  const chunks = [1, 2, 3, 4, 5, 6, 7]
    .map((chunk) => {
      const inputs = batches
        .map((batch) =>
          batch === 0
            ? `  let x${batch}${chunk} = leaky_relu(input[inputBase${batch} + col${chunk}]);`
            : `  var x${batch}${chunk} = 0.0;
  if (batch${batch} < params.batch) {
    x${batch}${chunk} = leaky_relu(input[inputBase${batch} + col${chunk}]);
  }`,
        )
        .join("\n");
      const matrixLoads = rows
        .map((row) => `  let m${row}${chunk} = matrix[row${row} * 1024u + col${chunk}];`)
        .join("\n");
      const adds = batches
        .flatMap((batch) =>
          rows.map(
            (row) =>
              `  sum${batch}${row} = sum${batch}${row} + m${row}${chunk} * x${batch}${chunk};`,
          ),
        )
        .join("\n");
      return `  let col${chunk} = lane + ${chunk * 128}u;
${inputs}
${matrixLoads}
${adds}`;
    })
    .join("\n\n");
  const reductions = cells
    .map(({ name }) => `  let reduced${name} = subgroupAdd(sum${name});`)
    .join("\n");
  const partialWrites = cells
    .map(({ name }) => `    subgroupPartial${name}[subgroupIndex] = reduced${name};`)
    .join("\n");
  const outDecls = cells.map(({ name }) => `    var out${name} = 0.0;`).join("\n");
  const outAdds = cells
    .map(({ name }) => `      out${name} = out${name} + subgroupPartial${name}[i];`)
    .join("\n");
  const biasAdds = cells
    .map(({ row, name }) => `      out${name} = out${name} + bias[row${row}];`)
    .join("\n");
  const writes = batches
    .map((batch) => {
      const rowWrites = rows
        .map(
          (row) =>
            `      output[outputBase${batch} + row${row}] = residual[outputBase${batch} + row${row}] + params.alpha * out${batch}${row};`,
        )
        .join("\n");
      if (batch === 0) {
        return `    let outputBase0 = batch0 * params.outputStride;
${rowWrites}`;
      }
      return `    if (batch${batch} < params.batch) {
      let outputBase${batch} = batch${batch} * params.outputStride;
${rowWrites}
    }`;
    })
    .join("\n");

  return /* wgsl */ `
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

${partials}

fn leaky_relu(x: f32) -> f32 {
  return select(0.01 * x, x, x >= 0.0);
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) subgroupLane: u32,
  @builtin(subgroup_size) subgroupSize: u32,
) {
  let row0 = wid.x * 4u;
${rowLets}
${batchLets}
  let lane = lid.x;
${inputBaseLets}

  let col0 = lane;
${firstInputs}
  let m00 = matrix[row0 * 1024u + col0];
  let m10 = matrix[row1 * 1024u + col0];
  let m20 = matrix[row2 * 1024u + col0];
  let m30 = matrix[row3 * 1024u + col0];
${firstSums}

${chunks}

${reductions}
  let subgroupIndex = lane / subgroupSize;
  if (subgroupLane == 0u) {
${partialWrites}
  }
  workgroupBarrier();

  if (lane == 0u) {
    let subgroupCount = (128u + subgroupSize - 1u) / subgroupSize;
${outDecls}
    for (var i = 0u; i < subgroupCount; i = i + 1u) {
${outAdds}
    }
    if (params.biasPresent != 0u) {
${biasAdds}
    }
${writes}
  }
}
`;
}

export const LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_BATCH2_SUBGROUP_WGSL =
  makeLeakyReluResidualMatVecBatchExactRowsCols1024BatchSubgroup(2);
export const LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_BATCH4_SUBGROUP_WGSL =
  makeLeakyReluResidualMatVecBatchExactRowsCols1024BatchSubgroup(4);
