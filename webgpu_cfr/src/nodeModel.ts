import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { gunzipSync } from "node:zlib";
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
  return BetterFfnWebGpuModel.fromBuffers(device, manifest, weights);
}
