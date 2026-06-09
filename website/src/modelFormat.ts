import type {
  BetterFfnCurriculumStreet,
  BetterFfnManifest,
  BetterFfnModelSetManifest,
  BetterFfnTensorManifest,
} from "./types.js";

export interface LoadedTensor {
  manifest: BetterFfnTensorManifest;
  data: Float32Array<ArrayBufferLike>;
}

export type TensorMap = Map<string, LoadedTensor>;

export interface ResolvedCfrDefaults {
  iterations: number;
  depth: number;
  cfrAvg: boolean;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isPositiveInteger(value: unknown): value is number {
  return Number.isInteger(value) && Number(value) > 0;
}

function optionalNumber(value: unknown, label: string): void {
  if (value !== undefined && value !== null && typeof value !== "number") {
    throw new Error(`model manifest cfr.${label} must be a number`);
  }
}

function optionalAllInNumber(value: unknown, label: string): void {
  if (value !== undefined && value !== null && typeof value !== "number") {
    throw new Error(`model manifest allIn.${label} must be a number`);
  }
}

function optionalInt16Dtype(value: unknown, label: string): void {
  if (value !== undefined && value !== "int16") {
    throw new Error(`model manifest allIn.${label}.dtype must be int16`);
  }
}

function requiredString(value: unknown, label: string): void {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`model manifest allIn.${label} must be a non-empty string`);
  }
}

function requiredModelSetString(value: unknown, label: string): void {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`model set manifest ${label} must be a non-empty string`);
  }
}

function expectedWeightsByteLength(manifest: BetterFfnManifest): number {
  return manifest.tensors.reduce(
    (max, tensor) => Math.max(max, tensor.byteOffset + tensor.byteLength),
    0,
  );
}

function float16ToFloat32(value: number): number {
  const sign = value & 0x8000 ? -1 : 1;
  const exponent = (value >> 10) & 0x1f;
  const fraction = value & 0x03ff;
  if (exponent === 0) {
    return sign * (fraction === 0 ? 0 : 2 ** -14 * (fraction / 1024));
  }
  if (exponent === 0x1f) {
    return fraction === 0 ? sign * Infinity : NaN;
  }
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function tensorDataFromWeights(
  tensor: BetterFfnTensorManifest,
  weights: ArrayBuffer,
  expectedElements: number,
): Float32Array<ArrayBuffer> | Float32Array<ArrayBufferLike> {
  if (tensor.dtype === "float32") {
    if (tensor.byteOffset % Float32Array.BYTES_PER_ELEMENT !== 0) {
      throw new Error(`unaligned tensor offset for ${tensor.name}`);
    }
    return new Float32Array(weights, tensor.byteOffset, expectedElements);
  }
  if (tensor.dtype === "float16") {
    if (tensor.byteOffset % Uint16Array.BYTES_PER_ELEMENT !== 0) {
      throw new Error(`unaligned fp16 tensor offset for ${tensor.name}`);
    }
    const fp16 = new Uint16Array(weights, tensor.byteOffset, expectedElements);
    const out = new Float32Array(expectedElements);
    for (let i = 0; i < expectedElements; i += 1) {
      out[i] = float16ToFloat32(fp16[i]!);
    }
    return out;
  }
  throw new Error(`unsupported dtype for ${tensor.name}: ${tensor.dtype}`);
}

export function parseBetterFfnManifest(value: unknown): BetterFfnManifest {
  if (!isRecord(value)) {
    throw new Error("model manifest must be an object");
  }
  if (value.schemaVersion !== 1 || value.format !== "p2.better_ffn.webgpu") {
    throw new Error("unsupported BetterFFN manifest schema");
  }
  const manifest = value as unknown as BetterFfnManifest;
  if (manifest.architecture.nonlinearity !== "leaky_relu") {
    throw new Error(`unsupported BetterFFN nonlinearity ${manifest.architecture.nonlinearity}`);
  }
  if (manifest.architecture.normalization !== "rmsnorm") {
    throw new Error(`unsupported BetterFFN normalization ${manifest.architecture.normalization}`);
  }
  if (manifest.architecture.numHands !== 1326) {
    throw new Error(`unsupported hand count ${manifest.architecture.numHands}; expected 1326`);
  }
  if (manifest.architecture.numPlayers !== 2) {
    throw new Error("only two-player BetterFFN checkpoints are supported");
  }
  if (!manifest.architecture.sharedTrunk) {
    throw new Error("BetterFFN checkpoints with shared_trunk=false are not supported");
  }
  manifest.architecture.contextDim ??= 11;
  if (
    !Number.isInteger(manifest.architecture.contextDim) ||
    manifest.architecture.contextDim <= 0
  ) {
    throw new Error(`unsupported contextDim ${manifest.architecture.contextDim}`);
  }
  manifest.architecture.policyRank ??= 64;
  manifest.architecture.policyHandBiasRank ??= 32;
  manifest.architecture.splitPolicyValue ??= false;
  manifest.architecture.boardInteractionDim ??= 0;
  if (
    !Number.isInteger(manifest.architecture.boardInteractionDim) ||
    manifest.architecture.boardInteractionDim < 0
  ) {
    throw new Error(`unsupported boardInteractionDim ${manifest.architecture.boardInteractionDim}`);
  }
  if (manifest.cfr !== undefined) {
    if (!isRecord(manifest.cfr)) {
      throw new Error("model manifest cfr must be an object");
    }
    if (manifest.cfr.iterations !== undefined && !isPositiveInteger(manifest.cfr.iterations)) {
      throw new Error("model manifest cfr.iterations must be a positive integer");
    }
    if (manifest.cfr.depth !== undefined && !isPositiveInteger(manifest.cfr.depth)) {
      throw new Error("model manifest cfr.depth must be a positive integer");
    }
    if (manifest.cfr.cfrAvg !== undefined && typeof manifest.cfr.cfrAvg !== "boolean") {
      throw new Error("model manifest cfr.cfrAvg must be a boolean");
    }
    optionalNumber(manifest.cfr.scheduleProgress, "scheduleProgress");
    optionalNumber(manifest.cfr.dcfrAlpha, "dcfrAlpha");
    optionalNumber(manifest.cfr.dcfrAlphaFinal, "dcfrAlphaFinal");
    optionalNumber(manifest.cfr.dcfrBeta, "dcfrBeta");
    optionalNumber(manifest.cfr.dcfrBetaFinal, "dcfrBetaFinal");
    optionalNumber(manifest.cfr.dcfrGamma, "dcfrGamma");
    optionalNumber(manifest.cfr.dcfrGammaFinal, "dcfrGammaFinal");
  }
  if (manifest.allIn !== undefined) {
    if (!isRecord(manifest.allIn)) {
      throw new Error("model manifest allIn must be an object");
    }
    if (manifest.allIn.enabled !== undefined && typeof manifest.allIn.enabled !== "boolean") {
      throw new Error("model manifest allIn.enabled must be a boolean");
    }
    optionalAllInNumber(manifest.allIn.scale, "scale");
    if (manifest.allIn.preflop !== undefined) {
      if (!isRecord(manifest.allIn.preflop)) {
        throw new Error("model manifest allIn.preflop must be an object");
      }
      requiredString(manifest.allIn.preflop.file, "preflop.file");
      optionalInt16Dtype(manifest.allIn.preflop.dtype, "preflop");
      optionalAllInNumber(manifest.allIn.preflop.scale, "preflop.scale");
    }
    if (manifest.allIn.flop !== undefined) {
      if (!isRecord(manifest.allIn.flop)) {
        throw new Error("model manifest allIn.flop must be an object");
      }
      requiredString(manifest.allIn.flop.actualToCanonFile, "flop.actualToCanonFile");
      requiredString(manifest.allIn.flop.actualPermFile, "flop.actualPermFile");
      requiredString(manifest.allIn.flop.comboPermsFile, "flop.comboPermsFile");
      requiredString(manifest.allIn.flop.tablePathTemplate, "flop.tablePathTemplate");
      optionalInt16Dtype(manifest.allIn.flop.dtype, "flop");
      optionalAllInNumber(manifest.allIn.flop.scale, "flop.scale");
    }
    if (manifest.allIn.turn !== undefined) {
      if (!isRecord(manifest.allIn.turn)) {
        throw new Error("model manifest allIn.turn must be an object");
      }
      requiredString(manifest.allIn.turn.tablePathTemplate, "turn.tablePathTemplate");
      optionalInt16Dtype(manifest.allIn.turn.dtype, "turn");
      optionalAllInNumber(manifest.allIn.turn.scale, "turn.scale");
    }
  }
  manifest.weights.storageDtype ??= "float32";
  if (manifest.weights.storageDtype !== "float16" && manifest.weights.storageDtype !== "float32") {
    throw new Error(`unsupported weights.storageDtype ${manifest.weights.storageDtype}`);
  }
  if (manifest.weights.compression !== undefined) {
    if (!isRecord(manifest.weights.compression)) {
      throw new Error("model manifest weights.compression must be an object");
    }
    if (manifest.weights.compression.format !== "gzip") {
      throw new Error(`unsupported weights compression ${manifest.weights.compression.format}`);
    }
    if (!isPositiveInteger(manifest.weights.compression.byteLength)) {
      throw new Error("model manifest weights.compression.byteLength must be positive");
    }
  }
  return manifest;
}

export function isBetterFfnModelSetManifest(value: unknown): value is BetterFfnModelSetManifest {
  return (
    isRecord(value) &&
    value.schemaVersion === 1 &&
    value.format === "p2.better_ffn.curriculum.webgpu"
  );
}

export function parseBetterFfnModelSetManifest(value: unknown): BetterFfnModelSetManifest {
  if (!isRecord(value)) {
    throw new Error("model set manifest must be an object");
  }
  if (value.schemaVersion !== 1 || value.format !== "p2.better_ffn.curriculum.webgpu") {
    throw new Error("unsupported BetterFFN model set manifest schema");
  }
  const manifest = value as unknown as BetterFfnModelSetManifest;
  const streetNames: BetterFfnCurriculumStreet[] = ["flop", "turn", "river"];
  if (!streetNames.includes(manifest.defaultStage)) {
    throw new Error("model set manifest defaultStage must be flop, turn, or river");
  }
  if (!isRecord(manifest.streets)) {
    throw new Error("model set manifest streets must be an object");
  }
  for (const street of streetNames) {
    const entry = manifest.streets[street];
    if (!isRecord(entry)) {
      throw new Error(`model set manifest streets.${street} must be an object`);
    }
    requiredModelSetString(entry.label, `streets.${street}.label`);
    requiredModelSetString(entry.manifest, `streets.${street}.manifest`);
  }
  return manifest;
}

export function parseBetterFfnManifestOrModelSet(
  value: unknown,
): BetterFfnManifest | BetterFfnModelSetManifest {
  return isBetterFfnModelSetManifest(value)
    ? parseBetterFfnModelSetManifest(value)
    : parseBetterFfnManifest(value);
}

export function resolveCfrDefaults(manifest: BetterFfnManifest): ResolvedCfrDefaults {
  return {
    iterations: isPositiveInteger(manifest.cfr?.iterations) ? manifest.cfr.iterations : 600,
    depth: isPositiveInteger(manifest.cfr?.depth) ? manifest.cfr.depth : 5,
    cfrAvg: typeof manifest.cfr?.cfrAvg === "boolean" ? manifest.cfr.cfrAvg : true,
  };
}

export function tensorsFromWeights(manifest: BetterFfnManifest, weights: ArrayBuffer): TensorMap {
  const expectedByteLength = expectedWeightsByteLength(manifest);
  if (weights.byteLength !== expectedByteLength) {
    throw new Error(
      `decoded weights have ${weights.byteLength} bytes, expected ${expectedByteLength}`,
    );
  }
  const out: TensorMap = new Map();
  for (const tensor of manifest.tensors) {
    const end = tensor.byteOffset + tensor.byteLength;
    if (end > weights.byteLength) {
      throw new Error(`tensor ${tensor.name} extends past decoded weights`);
    }
    const expectedElements = tensor.shape.reduce((acc, dim) => acc * dim, 1);
    const bytesPerElement =
      tensor.dtype === "float32"
        ? Float32Array.BYTES_PER_ELEMENT
        : tensor.dtype === "float16"
          ? Uint16Array.BYTES_PER_ELEMENT
          : undefined;
    if (bytesPerElement === undefined) {
      throw new Error(`unsupported dtype for ${tensor.name}: ${tensor.dtype}`);
    }
    if (expectedElements * bytesPerElement !== tensor.byteLength) {
      throw new Error(`tensor ${tensor.name} shape does not match byteLength`);
    }
    out.set(tensor.name, {
      manifest: tensor,
      data: tensorDataFromWeights(tensor, weights, expectedElements),
    });
  }
  return out;
}

export function requireTensor(
  tensors: TensorMap,
  name: string,
  shape?: readonly number[],
): LoadedTensor {
  const tensor = tensors.get(name);
  if (!tensor) {
    throw new Error(`model tensor ${name} is missing`);
  }
  if (shape) {
    const actual = tensor.manifest.shape;
    const same =
      actual.length === shape.length && actual.every((value, index) => value === shape[index]);
    if (!same) {
      throw new Error(
        `model tensor ${name} has shape [${actual.join(",")}], expected [${shape.join(",")}]`,
      );
    }
  }
  return tensor;
}

export function makeActionLabels(betBins: readonly number[]): string[] {
  const formatBetBin = (value: number): string =>
    Number.isInteger(value) ? value.toFixed(0) : String(value);
  return [
    "fold",
    "check_call",
    ...betBins.map((value) => `bet_${formatBetBin(value)}x_pot`),
    "all_in",
  ];
}
