import type { BetterFfnManifest, BetterFfnTensorManifest } from "./types.js";

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

export function parseBetterFfnManifest(value: unknown): BetterFfnManifest {
  if (!isRecord(value)) {
    throw new Error("model manifest must be an object");
  }
  if (value.schemaVersion !== 1 || value.format !== "p2.better_ffn.webgpu") {
    throw new Error("unsupported BetterFFN manifest schema");
  }
  const manifest = value as unknown as BetterFfnManifest;
  if (manifest.architecture.nonlinearity !== "leaky_relu") {
    throw new Error(
      `unsupported BetterFFN nonlinearity ${manifest.architecture.nonlinearity}`,
    );
  }
  if (manifest.architecture.normalization !== "rmsnorm") {
    throw new Error(
      `unsupported BetterFFN normalization ${manifest.architecture.normalization}`,
    );
  }
  if (manifest.architecture.numHands !== 1326) {
    throw new Error(
      `unsupported hand count ${manifest.architecture.numHands}; expected 1326`,
    );
  }
  if (manifest.architecture.numPlayers !== 2) {
    throw new Error("only two-player BetterFFN checkpoints are supported");
  }
  if (!manifest.architecture.sharedTrunk) {
    throw new Error("BetterFFN checkpoints with shared_trunk=false are not supported");
  }
  manifest.architecture.boardInteractionDim ??= 0;
  if (
    !Number.isInteger(manifest.architecture.boardInteractionDim) ||
    manifest.architecture.boardInteractionDim < 0
  ) {
    throw new Error(
      `unsupported boardInteractionDim ${manifest.architecture.boardInteractionDim}`,
    );
  }
  if (manifest.cfr !== undefined) {
    if (!isRecord(manifest.cfr)) {
      throw new Error("model manifest cfr must be an object");
    }
    if (
      manifest.cfr.iterations !== undefined &&
      !isPositiveInteger(manifest.cfr.iterations)
    ) {
      throw new Error("model manifest cfr.iterations must be a positive integer");
    }
    if (manifest.cfr.depth !== undefined && !isPositiveInteger(manifest.cfr.depth)) {
      throw new Error("model manifest cfr.depth must be a positive integer");
    }
    if (
      manifest.cfr.cfrAvg !== undefined &&
      typeof manifest.cfr.cfrAvg !== "boolean"
    ) {
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
  return manifest;
}

export function resolveCfrDefaults(manifest: BetterFfnManifest): ResolvedCfrDefaults {
  return {
    iterations: isPositiveInteger(manifest.cfr?.iterations)
      ? manifest.cfr.iterations
      : 16,
    depth: isPositiveInteger(manifest.cfr?.depth) ? manifest.cfr.depth : 1,
    cfrAvg:
      typeof manifest.cfr?.cfrAvg === "boolean" ? manifest.cfr.cfrAvg : true,
  };
}

export function tensorsFromWeights(
  manifest: BetterFfnManifest,
  weights: ArrayBuffer,
): TensorMap {
  if (weights.byteLength !== manifest.weights.byteLength) {
    throw new Error(
      `weights.bin has ${weights.byteLength} bytes, expected ${manifest.weights.byteLength}`,
    );
  }
  const out: TensorMap = new Map();
  for (const tensor of manifest.tensors) {
    if (tensor.dtype !== "float32") {
      throw new Error(`unsupported dtype for ${tensor.name}: ${tensor.dtype}`);
    }
    if (tensor.byteOffset % Float32Array.BYTES_PER_ELEMENT !== 0) {
      throw new Error(`unaligned tensor offset for ${tensor.name}`);
    }
    const end = tensor.byteOffset + tensor.byteLength;
    if (end > weights.byteLength) {
      throw new Error(`tensor ${tensor.name} extends past weights.bin`);
    }
    const expectedElements = tensor.shape.reduce((acc, dim) => acc * dim, 1);
    if (expectedElements * Float32Array.BYTES_PER_ELEMENT !== tensor.byteLength) {
      throw new Error(`tensor ${tensor.name} shape does not match byteLength`);
    }
    out.set(tensor.name, {
      manifest: tensor,
      data: new Float32Array(weights, tensor.byteOffset, expectedElements),
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
      actual.length === shape.length &&
      actual.every((value, index) => value === shape[index]);
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
