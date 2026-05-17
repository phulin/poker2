import { readFile } from "node:fs/promises";
import { BetterFfnWebGpuModel } from "./betterFfnWebGpuModel.js";
import { parseBetterFfnManifest } from "./modelFormat.js";

export async function loadNodeModel(
  device: GPUDevice,
  manifestPath: string,
  weightsPath: string,
): Promise<BetterFfnWebGpuModel> {
  const [manifestText, weightsBuffer] = await Promise.all([
    readFile(manifestPath, "utf8"),
    readFile(weightsPath),
  ]);
  const manifest = parseBetterFfnManifest(JSON.parse(manifestText));
  const weights = weightsBuffer.buffer.slice(
    weightsBuffer.byteOffset,
    weightsBuffer.byteOffset + weightsBuffer.byteLength,
  );
  return BetterFfnWebGpuModel.fromBuffers(device, manifest, weights);
}
