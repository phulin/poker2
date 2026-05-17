import { BetterFfnWebGpuModel } from "./betterFfnWebGpuModel.js";
import {
  BrowserCfrEvaluator,
  createBrowserCfrEvaluator,
} from "./browserEvaluator.js";
import { parseBetterFfnManifest } from "./modelFormat.js";
import type { BetterFfnManifest } from "./types.js";

export async function createBrowserDevice(): Promise<GPUDevice> {
  if (!navigator.gpu) {
    throw new Error("WebGPU is unavailable in this browser");
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("WebGPU did not return an adapter");
  }
  return await adapter.requestDevice();
}

export async function loadBrowserModel(
  manifestUrl: string,
  weightsUrl: string,
  device?: GPUDevice,
): Promise<BetterFfnWebGpuModel> {
  const [manifestResponse, weightsResponse] = await Promise.all([
    fetch(manifestUrl),
    fetch(weightsUrl),
  ]);
  if (!manifestResponse.ok) {
    throw new Error(`failed to fetch ${manifestUrl}: ${manifestResponse.status}`);
  }
  if (!weightsResponse.ok) {
    throw new Error(`failed to fetch ${weightsUrl}: ${weightsResponse.status}`);
  }
  const manifest = parseBetterFfnManifest(await manifestResponse.json());
  const weights = await weightsResponse.arrayBuffer();
  return BetterFfnWebGpuModel.fromBuffers(
    device ?? (await createBrowserDevice()),
    manifest,
    weights,
  );
}

export function loadBrowserModelFromBuffers(
  device: GPUDevice,
  manifest: BetterFfnManifest | unknown,
  weights: ArrayBuffer,
): BetterFfnWebGpuModel {
  return BetterFfnWebGpuModel.fromBuffers(device, manifest, weights);
}

export { BetterFfnWebGpuModel, BrowserCfrEvaluator, createBrowserCfrEvaluator };
