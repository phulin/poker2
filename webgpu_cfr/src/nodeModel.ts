import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { gunzipSync } from "node:zlib";
import { createManifestAllInTableProvider } from "./allInTables.js";
import { BetterFfnWebGpuModel } from "./betterFfnWebGpuModel.js";
import { parseBetterFfnManifest } from "./modelFormat.js";

export async function loadNodeModel(
  device: GPUDevice,
  manifestPath: string,
  weightsPath?: string,
): Promise<BetterFfnWebGpuModel> {
  const manifestText = await readFile(manifestPath, "utf8");
  const manifest = parseBetterFfnManifest(JSON.parse(manifestText));
  const resolvedWeightsPath =
    weightsPath ?? resolve(dirname(manifestPath), manifest.weights.file);
  const weightsBuffer = await readFile(resolvedWeightsPath);
  const payload = weightsBuffer.buffer.slice(
    weightsBuffer.byteOffset,
    weightsBuffer.byteOffset + weightsBuffer.byteLength,
  );
  let weights = payload;
  if (manifest.weights.compression?.format === "gzip") {
    const decoded = gunzipSync(new Uint8Array(payload));
    weights = decoded.buffer.slice(
      decoded.byteOffset,
      decoded.byteOffset + decoded.byteLength,
    );
  }
  const model = BetterFfnWebGpuModel.fromBuffers(device, manifest, weights);
  model.allInTableProvider = createManifestAllInTableProvider(
    manifest,
    pathToFileURL(manifestPath).toString(),
    async (url) => {
      const data = await readFile(fileURLToPath(url));
      return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
    },
  );
  return model;
}
