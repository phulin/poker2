import { createManifestAllInTableProvider } from "./allInTables.js";
import { BetterFfnWebGpuModel } from "./betterFfnWebGpuModel.js";
import { BrowserCfrEvaluator, createBrowserCfrEvaluator } from "./browserEvaluator.js";
import {
  decodeModelWeights,
  isLoadedModelSetBytes,
  loadModelBytesWithCache,
  loadRuntimeBytesWithCache,
  type ModelCacheProgress,
} from "./modelCache.js";
import { BetterFfnModelRegistry, type BetterFfnRuntime } from "./modelRegistry.js";
import { parseBetterFfnManifest } from "./modelFormat.js";
import type { BetterFfnManifest } from "./types.js";

function currentOrigin(): string {
  if (typeof location === "undefined") return "this origin";
  return location.origin === "null" ? location.href : location.origin;
}

function absoluteBrowserUrl(url: string): string {
  if (typeof globalThis.location === "undefined") {
    return new URL(url, "http://localhost/").toString();
  }
  return new URL(url, globalThis.location.href).toString();
}

function attachManifestAllInProvider(
  model: BetterFfnWebGpuModel,
  manifestUrl: string,
  device: GPUDevice,
): BetterFfnWebGpuModel {
  model.allInTableProvider = createManifestAllInTableProvider(
    model.manifest,
    absoluteBrowserUrl(manifestUrl),
    undefined,
    device,
  );
  return model;
}

function isLoopbackHost(hostname: string): boolean {
  return (
    hostname === "localhost" ||
    hostname === "127.0.0.1" ||
    hostname === "::1" ||
    hostname === "[::1]"
  );
}

function webGpuUnavailableMessage(): string {
  if (typeof isSecureContext === "boolean" && !isSecureContext) {
    return `WebGPU requires a secure context. This page is running from ${currentOrigin()}; open it through http://localhost, 127.0.0.1, or HTTPS.`;
  }

  if (
    typeof location !== "undefined" &&
    location.protocol !== "https:" &&
    location.protocol !== "file:" &&
    !isLoopbackHost(location.hostname)
  ) {
    return `WebGPU may be hidden on non-secure origins. This page is running from ${currentOrigin()}; open it through http://localhost, 127.0.0.1, or HTTPS.`;
  }

  return "WebGPU is not exposed by this browser. Use a browser with WebGPU enabled and hardware acceleration available.";
}

export interface BrowserDeviceOptions {
  onError?: (message: string) => void;
}

function webGpuErrorKind(error: GPUError): string {
  if (typeof GPUValidationError !== "undefined" && error instanceof GPUValidationError) {
    return "validation";
  }
  if (typeof GPUOutOfMemoryError !== "undefined" && error instanceof GPUOutOfMemoryError) {
    return "out-of-memory";
  }
  if (typeof GPUInternalError !== "undefined" && error instanceof GPUInternalError) {
    return "internal";
  }
  return "unknown";
}

export async function createBrowserDevice(options: BrowserDeviceOptions = {}): Promise<GPUDevice> {
  if (typeof navigator === "undefined" || !navigator.gpu) {
    throw new Error(webGpuUnavailableMessage());
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error(
      "WebGPU is exposed, but the browser did not return an adapter. Check that hardware acceleration and the GPU backend are enabled.",
    );
  }
  try {
    const requiredFeatures: GPUFeatureName[] = [];
    if (adapter.features.has("subgroups" as GPUFeatureName)) {
      requiredFeatures.push("subgroups" as GPUFeatureName);
    }
    const requiredLimits: Record<string, number> = {};
    if (adapter.limits.maxStorageBuffersPerShaderStage >= 10) {
      requiredLimits.maxStorageBuffersPerShaderStage = 10;
    }
    const device = await adapter.requestDevice({
      requiredFeatures,
      ...(Object.keys(requiredLimits).length > 0 ? { requiredLimits } : {}),
    });
    if (options.onError) {
      device.addEventListener("uncapturederror", (event) => {
        options.onError!(
          `Uncaptured WebGPU ${webGpuErrorKind(event.error)} error: ${event.error.message}`,
        );
      });
      void device.lost.then((info) => {
        const detail = info.message ? `: ${info.message}` : "";
        options.onError!(`WebGPU device lost: ${info.reason || "unknown"}${detail}`);
      });
    }
    return device;
  } catch (error) {
    const detail = error instanceof Error ? `: ${error.message}` : "";
    throw new Error(`WebGPU adapter was found, but requestDevice failed${detail}`);
  }
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
  const weights = await decodeModelWeights(await weightsResponse.arrayBuffer(), manifest);
  const modelDevice = device ?? (await createBrowserDevice());
  const model = BetterFfnWebGpuModel.fromBuffers(modelDevice, manifest, weights);
  return attachManifestAllInProvider(model, manifestUrl, modelDevice);
}

export async function loadBrowserModelCached(
  manifestUrl: string,
  options: {
    weightsUrl?: string;
    device?: GPUDevice;
    onProgress?: (progress: ModelCacheProgress) => void;
  } = {},
): Promise<BetterFfnWebGpuModel> {
  const cacheOptions: {
    weightsUrl?: string;
    onProgress?: (progress: ModelCacheProgress) => void;
  } = {};
  if (options.weightsUrl) cacheOptions.weightsUrl = options.weightsUrl;
  if (options.onProgress) cacheOptions.onProgress = options.onProgress;
  const loaded = await loadModelBytesWithCache(manifestUrl, cacheOptions);
  const modelDevice = options.device ?? (await createBrowserDevice());
  const model = BetterFfnWebGpuModel.fromBuffers(modelDevice, loaded.manifest, loaded.weights);
  return attachManifestAllInProvider(model, loaded.manifestUrl, modelDevice);
}

export async function loadBrowserRuntimeCached(
  manifestUrl: string,
  options: {
    device?: GPUDevice;
    onProgress?: (progress: ModelCacheProgress) => void;
  } = {},
): Promise<BetterFfnRuntime> {
  const cacheOptions: { onProgress?: (progress: ModelCacheProgress) => void } = {};
  if (options.onProgress) cacheOptions.onProgress = options.onProgress;
  const loaded = await loadRuntimeBytesWithCache(manifestUrl, cacheOptions);
  const modelDevice = options.device ?? (await createBrowserDevice());
  if (!isLoadedModelSetBytes(loaded)) {
    const model = BetterFfnWebGpuModel.fromBuffers(modelDevice, loaded.manifest, loaded.weights);
    return attachManifestAllInProvider(model, loaded.manifestUrl, modelDevice);
  }

  const stages = {} as ConstructorParameters<typeof BetterFfnModelRegistry>[0];
  for (const street of ["flop", "turn", "river"] as const) {
    const stage = loaded.stages[street];
    stages[street] = attachManifestAllInProvider(
      BetterFfnWebGpuModel.fromBuffers(modelDevice, stage.manifest, stage.weights),
      stage.manifestUrl,
      modelDevice,
    );
  }
  return new BetterFfnModelRegistry(stages, loaded.manifest.defaultStage);
}

export function loadBrowserModelFromBuffers(
  device: GPUDevice,
  manifest: BetterFfnManifest | unknown,
  weights: ArrayBuffer,
): BetterFfnWebGpuModel {
  const model = BetterFfnWebGpuModel.fromBuffers(device, manifest, weights);
  if (model.manifest.allIn) {
    const baseUrl =
      typeof globalThis.location === "undefined" ? "http://localhost/" : globalThis.location.href;
    model.allInTableProvider = createManifestAllInTableProvider(
      model.manifest,
      baseUrl,
      undefined,
      device,
    );
  }
  return model;
}

export {
  BetterFfnWebGpuModel,
  BrowserCfrEvaluator,
  createBrowserCfrEvaluator,
  loadModelBytesWithCache,
  loadRuntimeBytesWithCache,
};
