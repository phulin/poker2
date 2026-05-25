import { performance } from "node:perf_hooks";
import {
  LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_SUBGROUP_WGSL,
  LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_WGSL,
  LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_COLS_512_SUBGROUP_WGSL,
  LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_COLS_512_WGSL,
  LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_WGSL,
  LEAKY_RELU_MAT_VEC_BATCH_TILED_GEMM_WGSL,
  LEAKY_RELU_MAT_VEC_BATCH_WGSL,
  LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_BATCH2_SUBGROUP_WGSL,
  LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_SUBGROUP_WGSL,
  LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_WGSL,
  LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_EXACT_ROWS_COLS_512_WGSL,
  LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_EXACT_ROWS_WGSL,
  LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_TILED_GEMM_WGSL,
  LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_WGSL,
  MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_BATCH2_SUBGROUP_WGSL,
  MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_WGSL,
  MAT_VEC_BATCH_EXACT_ROWS_COLS_1326_BATCH2_SUBGROUP_WGSL,
  MAT_VEC_BATCH_EXACT_ROWS_COLS_512_BATCH2_SUBGROUP_WGSL,
  MAT_VEC_BATCH_EXACT_ROWS_COLS_512_BATCH3_SUBGROUP_WGSL,
  MAT_VEC_BATCH_EXACT_ROWS_COLS_512_SUBGROUP_WGSL,
  MAT_VEC_BATCH_EXACT_ROWS_COLS_512_WGSL,
  MAT_VEC_BATCH_EXACT_ROWS_WGSL,
  MAT_VEC_BATCH_SMALL_COLS_WGSL,
  MAT_VEC_BATCH_TILED_GEMM_WGSL,
  MAT_VEC_BATCH_WGSL,
} from "./modelKernels.js";
import {
  makeEmptyStorageBuffer,
  makeStorageBuffer,
  makeUniformBuffer,
  readFloatBuffer,
} from "./gpuBuffers.js";
import { createDawnDevice } from "./gpu.js";
import { NUM_HANDS } from "./hunlEnv.js";

type Kind = "plain" | "leaky" | "residual";

interface Options {
  suite: "single" | "model";
  kind: Kind | "all";
  rows: number;
  cols: number;
  batch: number;
  inputStride?: number;
  outputStride?: number;
  inputOffset: number;
  outputOffset: number;
  bias: boolean;
  warmups: number;
  runs: number;
  dispatches: number;
  variant?: string;
  validate: boolean;
}

interface Variant {
  name: string;
  kind: Kind;
  shader: string;
  dispatchX: (shape: Shape) => number;
  dispatchY: (shape: Shape) => number;
  matrix: "rowMajor" | "tiled";
  supports: (shape: Shape, device: GPUDevice) => boolean;
}

interface Shape {
  rows: number;
  cols: number;
  batch: number;
  inputStride: number;
  outputStride: number;
  inputOffset: number;
  outputOffset: number;
  bias: boolean;
}

const ROW_BLOCK = 4;
const TILED_ROW_TILE = 8;
const TILED_COL_TILE = 32;

function getArg(args: string[], name: string, fallback?: string): string {
  const index = args.indexOf(name);
  const value = index >= 0 ? args[index + 1] : undefined;
  if (value) return value;
  if (fallback !== undefined) return fallback;
  throw new Error(`Missing ${name}`);
}

function hasArg(args: string[], name: string): boolean {
  return args.includes(name);
}

function intArg(args: string[], name: string, fallback: number): number {
  const value = Number.parseInt(getArg(args, name, String(fallback)), 10);
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`${name} must be a non-negative integer`);
  }
  return value;
}

function readOptions(): Options {
  const args = process.argv.slice(2);
  const suite = getArg(args, "--suite", "single") as Options["suite"];
  if (!["single", "model"].includes(suite)) {
    throw new Error("--suite must be single or model");
  }
  const kind = getArg(args, "--kind", "all") as Options["kind"];
  if (!["plain", "leaky", "residual", "all"].includes(kind)) {
    throw new Error("--kind must be plain, leaky, residual, or all");
  }
  const rows = intArg(args, "--rows", 1024);
  const cols = intArg(args, "--cols", 512);
  const batch = intArg(args, "--batch", 64);
  const inputStrideRaw = getArg(args, "--input-stride", "");
  const outputStrideRaw = getArg(args, "--output-stride", "");
  const variant = getArg(args, "--variant", "");
  return {
    suite,
    kind,
    rows,
    cols,
    batch,
    ...(inputStrideRaw ? { inputStride: Number.parseInt(inputStrideRaw, 10) } : {}),
    ...(outputStrideRaw ? { outputStride: Number.parseInt(outputStrideRaw, 10) } : {}),
    inputOffset: intArg(args, "--input-offset", 0),
    outputOffset: intArg(args, "--output-offset", 0),
    bias: !hasArg(args, "--no-bias"),
    warmups: intArg(args, "--warmups", 10),
    runs: intArg(args, "--runs", 30),
    dispatches: Math.max(1, intArg(args, "--dispatches", 50)),
    ...(variant ? { variant } : {}),
    validate: !hasArg(args, "--no-validate"),
  };
}

function modelSuiteShapes(batch: number): Array<{ name: string; kind: Kind; shape: Shape }> {
  return [
    {
      name: "belief_linear_in_1024x512",
      kind: "plain",
      shape: {
        rows: 1024,
        cols: 512,
        batch,
        inputStride: 512,
        outputStride: 1024,
        inputOffset: 0,
        outputOffset: 0,
        bias: false,
      },
    },
    {
      name: "hand_embedding_512x1326",
      kind: "plain",
      shape: {
        rows: 512,
        cols: NUM_HANDS,
        batch,
        inputStride: 2 * NUM_HANDS,
        outputStride: 1024,
        inputOffset: 0,
        outputOffset: 0,
        bias: false,
      },
    },
    {
      name: "hidden_linear_in_1024x512",
      kind: "plain",
      shape: {
        rows: 1024,
        cols: 512,
        batch,
        inputStride: 512,
        outputStride: 1024,
        inputOffset: 0,
        outputOffset: 0,
        bias: false,
      },
    },
    {
      name: "hidden_linear_out_512x1024",
      kind: "leaky",
      shape: {
        rows: 512,
        cols: 1024,
        batch,
        inputStride: 1024,
        outputStride: 512,
        inputOffset: 0,
        outputOffset: 0,
        bias: true,
      },
    },
    {
      name: "residual_linear_out_512x1024",
      kind: "residual",
      shape: {
        rows: 512,
        cols: 1024,
        batch,
        inputStride: 1024,
        outputStride: 512,
        inputOffset: 0,
        outputOffset: 0,
        bias: true,
      },
    },
    {
      name: "value_head_2652x512",
      kind: "plain",
      shape: {
        rows: 2 * NUM_HANDS,
        cols: 512,
        batch,
        inputStride: 512,
        outputStride: 2 * NUM_HANDS,
        inputOffset: 0,
        outputOffset: 0,
        bias: true,
      },
    },
  ];
}

function stats(samples: number[]): {
  meanMs: number;
  medianMs: number;
  minMs: number;
  maxMs: number;
} {
  const sorted = [...samples].sort((a, b) => a - b);
  return {
    meanMs: samples.reduce((sum, sample) => sum + sample, 0) / samples.length,
    medianMs: sorted[Math.floor(sorted.length / 2)]!,
    minMs: sorted[0]!,
    maxMs: sorted[sorted.length - 1]!,
  };
}

function fillDeterministic(length: number, scale: number): Float32Array<ArrayBuffer> {
  const out = new Float32Array(length);
  for (let i = 0; i < length; i += 1) {
    out[i] = Math.fround((((i * 1664525 + 1013904223) >>> 0) % 2001 - 1000) * scale);
  }
  return out;
}

function packMatrixTiled(
  data: Float32Array<ArrayBuffer>,
  rows: number,
  cols: number,
): Float32Array<ArrayBuffer> {
  const rowTiles = Math.ceil(rows / TILED_ROW_TILE);
  const colTiles = Math.ceil(cols / TILED_COL_TILE);
  const out = new Float32Array(rowTiles * colTiles * TILED_COL_TILE * TILED_ROW_TILE);
  for (let rowTile = 0; rowTile < rowTiles; rowTile += 1) {
    for (let colTile = 0; colTile < colTiles; colTile += 1) {
      const tileBase = (rowTile * colTiles + colTile) * TILED_COL_TILE * TILED_ROW_TILE;
      for (let colLocal = 0; colLocal < TILED_COL_TILE; colLocal += 1) {
        const col = colTile * TILED_COL_TILE + colLocal;
        for (let rowLocal = 0; rowLocal < TILED_ROW_TILE; rowLocal += 1) {
          const row = rowTile * TILED_ROW_TILE + rowLocal;
          if (row < rows && col < cols) {
            out[tileBase + colLocal * TILED_ROW_TILE + rowLocal] = data[row * cols + col]!;
          }
        }
      }
    }
  }
  return out;
}

function rowGroups(rows: number): number {
  return Math.ceil(rows / ROW_BLOCK);
}

function hasSubgroups(device: GPUDevice): boolean {
  return device.features.has("subgroups" as GPUFeatureName);
}

function variants(): Variant[] {
  const plainGeneric: Variant = {
    name: "plain.generic",
    kind: "plain",
    shader: MAT_VEC_BATCH_WGSL,
    dispatchX: (s) => rowGroups(s.rows),
    dispatchY: (s) => s.batch,
    matrix: "rowMajor",
    supports: () => true,
  };
  const leakyGeneric: Variant = {
    name: "leaky.generic",
    kind: "leaky",
    shader: LEAKY_RELU_MAT_VEC_BATCH_WGSL,
    dispatchX: (s) => rowGroups(s.rows),
    dispatchY: (s) => s.batch,
    matrix: "rowMajor",
    supports: (s) => s.inputOffset === 0 && s.outputOffset === 0,
  };
  const residualGeneric: Variant = {
    name: "residual.generic",
    kind: "residual",
    shader: LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_WGSL,
    dispatchX: (s) => rowGroups(s.rows),
    dispatchY: (s) => s.batch,
    matrix: "rowMajor",
    supports: (s) => s.inputOffset === 0 && s.outputOffset === 0,
  };
  return [
    plainGeneric,
    {
      ...plainGeneric,
      name: "plain.small_cols",
      shader: MAT_VEC_BATCH_SMALL_COLS_WGSL,
      dispatchX: (s) => Math.ceil((s.rows * s.batch) / 64),
      dispatchY: () => 1,
      supports: (s) => s.cols <= 128 && s.inputOffset === 0 && s.outputOffset === 0 && !s.bias,
    },
    {
      ...plainGeneric,
      name: "plain.exact_rows",
      shader: MAT_VEC_BATCH_EXACT_ROWS_WGSL,
      supports: (s) => s.rows % ROW_BLOCK === 0,
    },
    {
      ...plainGeneric,
      name: "plain.512",
      shader: MAT_VEC_BATCH_EXACT_ROWS_COLS_512_WGSL,
      supports: (s) => s.rows % ROW_BLOCK === 0 && s.cols === 512,
    },
    {
      ...plainGeneric,
      name: "plain.512.subgroup",
      shader: MAT_VEC_BATCH_EXACT_ROWS_COLS_512_SUBGROUP_WGSL,
      supports: (s, d) => hasSubgroups(d) && s.rows % ROW_BLOCK === 0 && s.cols === 512,
    },
    {
      ...plainGeneric,
      name: "plain.512.batch2.subgroup",
      shader: MAT_VEC_BATCH_EXACT_ROWS_COLS_512_BATCH2_SUBGROUP_WGSL,
      dispatchY: (s) => Math.ceil(s.batch / 2),
      supports: (s, d) =>
        hasSubgroups(d) &&
        s.rows % ROW_BLOCK === 0 &&
        s.cols === 512 &&
        s.inputStride === 512 &&
        s.outputStride === s.rows &&
        s.inputOffset === 0 &&
        s.outputOffset === 0,
    },
    {
      ...plainGeneric,
      name: "plain.512.batch3.subgroup",
      shader: MAT_VEC_BATCH_EXACT_ROWS_COLS_512_BATCH3_SUBGROUP_WGSL,
      dispatchY: (s) => Math.ceil(s.batch / 3),
      supports: (s, d) =>
        hasSubgroups(d) &&
        s.rows === 1024 &&
        s.cols === 512 &&
        s.inputStride === 512 &&
        s.outputStride === 1024 &&
        s.inputOffset === 0 &&
        s.outputOffset === 0 &&
        !s.bias,
    },
    {
      ...plainGeneric,
      name: "plain.1024",
      shader: MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_WGSL,
      supports: (s) => s.rows % ROW_BLOCK === 0 && s.cols === 1024,
    },
    {
      ...plainGeneric,
      name: "plain.1024.batch2.subgroup",
      shader: MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_BATCH2_SUBGROUP_WGSL,
      dispatchY: (s) => Math.ceil(s.batch / 2),
      supports: (s, d) =>
        hasSubgroups(d) &&
        s.rows === 1024 &&
        s.cols === 1024 &&
        s.inputStride === 1024 &&
        s.outputStride === 1024 &&
        s.inputOffset === 0 &&
        s.outputOffset === 0 &&
        !s.bias,
    },
    {
      ...plainGeneric,
      name: "plain.1326.batch2.subgroup",
      shader: MAT_VEC_BATCH_EXACT_ROWS_COLS_1326_BATCH2_SUBGROUP_WGSL,
      dispatchY: (s) => Math.ceil(s.batch / 2),
      supports: (s, d) =>
        hasSubgroups(d) &&
        s.rows === 512 &&
        s.cols === NUM_HANDS &&
        s.inputStride === 2 * NUM_HANDS &&
        s.outputStride === 1024 &&
        !s.bias,
    },
    {
      ...plainGeneric,
      name: "plain.tiled_gemm",
      shader: MAT_VEC_BATCH_TILED_GEMM_WGSL,
      dispatchX: (s) => Math.ceil(s.rows / TILED_ROW_TILE),
      dispatchY: (s) => Math.ceil(s.batch / TILED_ROW_TILE),
      matrix: "tiled",
      supports: () => true,
    },
    leakyGeneric,
    {
      ...leakyGeneric,
      name: "leaky.exact_rows",
      shader: LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_WGSL,
      supports: (s) => s.rows % ROW_BLOCK === 0,
    },
    {
      ...leakyGeneric,
      name: "leaky.512",
      shader: LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_COLS_512_WGSL,
      supports: (s) => s.rows % ROW_BLOCK === 0 && s.cols === 512,
    },
    {
      ...leakyGeneric,
      name: "leaky.512.subgroup",
      shader: LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_COLS_512_SUBGROUP_WGSL,
      supports: (s, d) => hasSubgroups(d) && s.rows % ROW_BLOCK === 0 && s.cols === 512,
    },
    {
      ...leakyGeneric,
      name: "leaky.1024",
      shader: LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_WGSL,
      supports: (s) => s.rows % ROW_BLOCK === 0 && s.cols === 1024,
    },
    {
      ...leakyGeneric,
      name: "leaky.1024.subgroup",
      shader: LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_SUBGROUP_WGSL,
      supports: (s, d) => hasSubgroups(d) && s.rows % ROW_BLOCK === 0 && s.cols === 1024,
    },
    {
      ...leakyGeneric,
      name: "leaky.tiled_gemm",
      shader: LEAKY_RELU_MAT_VEC_BATCH_TILED_GEMM_WGSL,
      dispatchX: (s) => Math.ceil(s.rows / TILED_ROW_TILE),
      dispatchY: (s) => Math.ceil(s.batch / TILED_ROW_TILE),
      matrix: "tiled",
      supports: (s) => s.inputOffset === 0 && s.outputOffset === 0,
    },
    residualGeneric,
    {
      ...residualGeneric,
      name: "residual.exact_rows",
      shader: LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_EXACT_ROWS_WGSL,
      supports: (s) => s.rows % ROW_BLOCK === 0,
    },
    {
      ...residualGeneric,
      name: "residual.512",
      shader: LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_EXACT_ROWS_COLS_512_WGSL,
      supports: (s) => s.rows % ROW_BLOCK === 0 && s.cols === 512,
    },
    {
      ...residualGeneric,
      name: "residual.1024",
      shader: LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_WGSL,
      supports: (s) => s.rows % ROW_BLOCK === 0 && s.cols === 1024,
    },
    {
      ...residualGeneric,
      name: "residual.1024.subgroup",
      shader: LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_SUBGROUP_WGSL,
      supports: (s, d) => hasSubgroups(d) && s.rows % ROW_BLOCK === 0 && s.cols === 1024,
    },
    {
      ...residualGeneric,
      name: "residual.1024.batch2.subgroup",
      shader: LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_BATCH2_SUBGROUP_WGSL,
      dispatchY: (s) => Math.ceil(s.batch / 2),
      supports: (s, d) =>
        hasSubgroups(d) &&
        s.rows === 512 &&
        s.cols === 1024 &&
        s.inputStride === 1024 &&
        s.outputStride === 512 &&
        s.bias,
    },
    {
      ...residualGeneric,
      name: "residual.tiled_gemm",
      shader: LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_TILED_GEMM_WGSL,
      dispatchX: (s) => Math.ceil(s.rows / TILED_ROW_TILE),
      dispatchY: (s) => Math.ceil(s.batch / TILED_ROW_TILE),
      matrix: "tiled",
      supports: (s) => s.inputOffset === 0 && s.outputOffset === 0,
    },
  ];
}

function createPipeline(device: GPUDevice, variant: Variant): GPUComputePipeline {
  return device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: variant.shader }),
      entryPoint: "main",
    },
  });
}

function uniformWords(kind: Kind, shape: Shape): Uint32Array<ArrayBuffer> {
  if (kind === "plain") {
    return new Uint32Array([
      shape.rows,
      shape.cols,
      shape.batch,
      shape.inputStride,
      shape.outputStride,
      shape.inputOffset,
      shape.outputOffset,
      shape.bias ? 1 : 0,
    ]);
  }
  if (kind === "leaky") {
    return new Uint32Array([
      shape.rows,
      shape.cols,
      shape.batch,
      shape.inputStride,
      shape.outputStride,
      shape.bias ? 1 : 0,
      0,
      0,
    ]);
  }
  const alphaBits = new Uint32Array(new Float32Array([1.0]).buffer)[0]!;
  return new Uint32Array([
    shape.rows,
    shape.cols,
    shape.batch,
    shape.inputStride,
    shape.outputStride,
    shape.bias ? 1 : 0,
    alphaBits,
    0,
  ]);
}

function bindGroup(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  kind: Kind,
  matrix: GPUBuffer,
  input: GPUBuffer,
  bias: GPUBuffer,
  output: GPUBuffer,
  residual: GPUBuffer,
  uniform: GPUBuffer,
): GPUBindGroup {
  const entries: GPUBindGroupEntry[] = [
    { binding: 0, resource: { buffer: matrix } },
    { binding: 1, resource: { buffer: input } },
    { binding: 2, resource: { buffer: bias } },
    { binding: 3, resource: { buffer: output } },
  ];
  if (kind === "residual") {
    entries.push({ binding: 4, resource: { buffer: residual } });
    entries.push({ binding: 5, resource: { buffer: uniform } });
  } else {
    entries.push({ binding: 4, resource: { buffer: uniform } });
  }
  return device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
}

function encodeDispatches(
  encoder: GPUCommandEncoder,
  pipeline: GPUComputePipeline,
  bindGroupForRun: GPUBindGroup,
  variant: Variant,
  shape: Shape,
  dispatches: number,
): void {
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroupForRun);
  for (let i = 0; i < dispatches; i += 1) {
    pass.dispatchWorkgroups(variant.dispatchX(shape), variant.dispatchY(shape));
  }
  pass.end();
}

async function runOnce(
  device: GPUDevice,
  variant: Variant,
  shape: Shape,
  buffers: {
    rowMajorMatrix: GPUBuffer;
    tiledMatrix: GPUBuffer;
    input: GPUBuffer;
    bias: GPUBuffer;
    residual: GPUBuffer;
  },
): Promise<Float32Array<ArrayBufferLike>> {
  const pipeline = createPipeline(device, variant);
  const output = makeEmptyStorageBuffer(device, shape.batch * shape.outputStride);
  const uniform = makeUniformBuffer(device, uniformWords(variant.kind, shape));
  const group = bindGroup(
    device,
    pipeline,
    variant.kind,
    variant.matrix === "tiled" ? buffers.tiledMatrix : buffers.rowMajorMatrix,
    buffers.input,
    buffers.bias,
    output,
    buffers.residual,
    uniform,
  );
  const encoder = device.createCommandEncoder();
  encodeDispatches(encoder, pipeline, group, variant, shape, 1);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const result = await readFloatBuffer(device, output, shape.batch * shape.outputStride);
  output.destroy();
  uniform.destroy();
  return result;
}

async function timeVariant(
  device: GPUDevice,
  variant: Variant,
  shape: Shape,
  options: Options,
  buffers: {
    rowMajorMatrix: GPUBuffer;
    tiledMatrix: GPUBuffer;
    input: GPUBuffer;
    bias: GPUBuffer;
    residual: GPUBuffer;
  },
): Promise<number[]> {
  const pipeline = createPipeline(device, variant);
  const output = makeEmptyStorageBuffer(device, shape.batch * shape.outputStride);
  const uniform = makeUniformBuffer(device, uniformWords(variant.kind, shape));
  const group = bindGroup(
    device,
    pipeline,
    variant.kind,
    variant.matrix === "tiled" ? buffers.tiledMatrix : buffers.rowMajorMatrix,
    buffers.input,
    buffers.bias,
    output,
    buffers.residual,
    uniform,
  );
  const samples: number[] = [];
  for (let i = 0; i < options.warmups + options.runs; i += 1) {
    const encoder = device.createCommandEncoder();
    encodeDispatches(encoder, pipeline, group, variant, shape, options.dispatches);
    const start = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const elapsed = performance.now() - start;
    if (i >= options.warmups) {
      samples.push(elapsed / options.dispatches);
    }
  }
  output.destroy();
  uniform.destroy();
  return samples;
}

function maxDiff(
  a: Float32Array<ArrayBufferLike>,
  b: Float32Array<ArrayBufferLike>,
): number {
  let out = 0;
  for (let i = 0; i < a.length; i += 1) {
    const diff = Math.abs(a[i]! - b[i]!);
    if (diff > out) out = diff;
  }
  return out;
}

function referenceName(kind: Kind): string {
  return `${kind}.generic`;
}

const device = await createDawnDevice();
const options = readOptions();

async function benchmarkShape(
  name: string,
  kindFilter: Kind | "all",
  shape: Shape,
): Promise<{
  name: string;
  shape: Shape;
  variants: unknown[];
}> {
  const allVariants = variants().filter((variant) => {
    if (kindFilter !== "all" && variant.kind !== kindFilter) return false;
    if (options.variant && variant.name !== options.variant) return false;
    return variant.supports(shape, device);
  });
  const matrixData = fillDeterministic(shape.rows * shape.cols, 0.001);
  const inputData = fillDeterministic(shape.batch * shape.inputStride, 0.002);
  const biasData = fillDeterministic(shape.rows, 0.0001);
  const residualData = fillDeterministic(shape.batch * shape.outputStride, 0.003);
  const buffers = {
    rowMajorMatrix: makeStorageBuffer(device, matrixData),
    tiledMatrix: makeStorageBuffer(device, packMatrixTiled(matrixData, shape.rows, shape.cols)),
    input: makeStorageBuffer(device, inputData),
    bias: makeStorageBuffer(device, biasData),
    residual: makeStorageBuffer(device, residualData),
  };
  try {
    const references = new Map<Kind, Float32Array<ArrayBufferLike>>();
    const results = [];
    for (const variant of allVariants) {
      let drift: number | undefined;
      if (options.validate) {
        const refKey = variant.kind;
        let reference = references.get(refKey);
        if (!reference) {
          const referenceVariant = variants().find((item) => item.name === referenceName(variant.kind));
          if (!referenceVariant) throw new Error(`missing reference for ${variant.kind}`);
          reference = await runOnce(device, referenceVariant, shape, buffers);
          references.set(refKey, reference);
        }
        const candidate = await runOnce(device, variant, shape, buffers);
        drift = maxDiff(reference, candidate);
      }
      const samples = await timeVariant(device, variant, shape, options, buffers);
      results.push({
        name: variant.name,
        kind: variant.kind,
        matrix: variant.matrix,
        dispatch: {
          x: variant.dispatchX(shape),
          y: variant.dispatchY(shape),
          repeated: options.dispatches,
        },
        ...(drift === undefined ? {} : { maxDiff: drift }),
        ...stats(samples),
        samplesMs: samples,
      });
    }
    return { name, shape, variants: results };
  } finally {
    for (const buffer of Object.values(buffers)) {
      buffer.destroy();
    }
  }
}

try {
  const cases =
    options.suite === "model"
      ? modelSuiteShapes(options.batch).filter((item) =>
          options.kind === "all" ? true : item.kind === options.kind,
        )
      : [
          {
            name: "single",
            kind: options.kind,
            shape: {
              rows: options.rows,
              cols: options.cols,
              batch: options.batch,
              inputStride: options.inputStride ?? options.cols,
              outputStride: options.outputStride ?? options.rows,
              inputOffset: options.inputOffset,
              outputOffset: options.outputOffset,
              bias: options.bias,
            },
          },
        ];
  const caseResults = [];
  for (const item of cases) {
    caseResults.push(await benchmarkShape(item.name, item.kind, item.shape));
  }
  console.log(
    JSON.stringify(
      {
        backend: process.env.WEBGPU_BACKEND ?? null,
        suite: options.suite,
        warmups: options.warmups,
        runs: options.runs,
        dispatches: options.dispatches,
        cases: caseResults,
      },
      null,
      2,
    ),
  );
} finally {
  device.destroy();
}
