import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { gunzipSync } from "node:zlib";
import { createManifestAllInTableProvider } from "./allInTables.js";
import { BetterFfnWebGpuModel } from "./betterFfnWebGpuModel.js";
import {
  isBetterFfnModelSetManifest,
  parseBetterFfnManifest,
  parseBetterFfnManifestOrModelSet,
} from "./modelFormat.js";
import { BetterFfnModelRegistry, type BetterFfnRuntime } from "./modelRegistry.js";
import type { BetterFfnManifest } from "./types.js";

export async function loadNodeModel(
  device: GPUDevice,
  manifestPath: string,
  weightsPath?: string,
): Promise<BetterFfnWebGpuModel> {
  const manifestText = await readFile(manifestPath, "utf8");
  const manifest = parseBetterFfnManifest(JSON.parse(manifestText));
  return await loadNodeModelFromManifest(device, manifestPath, manifest, weightsPath);
}

export async function loadNodeRuntime(
  device: GPUDevice,
  manifestPath: string,
  weightsPath?: string,
): Promise<BetterFfnRuntime> {
  const manifestText = await readFile(manifestPath, "utf8");
  const parsed = parseBetterFfnManifestOrModelSet(JSON.parse(manifestText));
  if (!isBetterFfnModelSetManifest(parsed)) {
    return await loadNodeModelFromManifest(device, manifestPath, parsed, weightsPath);
  }
  if (weightsPath) {
    throw new Error("--weights is not supported for curriculum model sets");
  }

  const stages = {} as ConstructorParameters<typeof BetterFfnModelRegistry>[0];
  for (const street of ["flop", "turn", "river"] as const) {
    const stagePath = resolve(dirname(manifestPath), parsed.streets[street].manifest);
    stages[street] = await loadNodeModel(device, stagePath);
  }
  return new BetterFfnModelRegistry(stages, parsed.defaultStage);
}

async function loadNodeModelFromManifest(
  device: GPUDevice,
  manifestPath: string,
  manifest: BetterFfnManifest,
  weightsPath?: string,
): Promise<BetterFfnWebGpuModel> {
  const resolvedWeightsPath = weightsPath ?? resolve(dirname(manifestPath), manifest.weights.file);
  const weightsBuffer = await readFile(resolvedWeightsPath);
  const payload = weightsBuffer.buffer.slice(
    weightsBuffer.byteOffset,
    weightsBuffer.byteOffset + weightsBuffer.byteLength,
  );
  let weights = payload;
  if (manifest.weights.compression?.format === "gzip") {
    const decoded = gunzipSync(new Uint8Array(payload));
    weights = decoded.buffer.slice(decoded.byteOffset, decoded.byteOffset + decoded.byteLength);
  }
  const model = BetterFfnWebGpuModel.fromBuffers(device, manifest, weights);
  model.allInTableProvider = createManifestAllInTableProvider(
    manifest,
    pathToFileURL(manifestPath).toString(),
    async (url) => {
      const data = await readFile(fileURLToPath(url));
      return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
    },
    device,
  );
  return model;
}
