import { BetterFfnWebGpuModel } from "./betterFfnWebGpuModel.js";
import {
  BrowserCfrEvaluator,
  createBrowserCfrEvaluator,
} from "./browserEvaluator.js";
import { parseBetterFfnManifest } from "./modelFormat.js";
import {
  decodeModelWeights,
  loadModelBytesWithCache,
  type ModelCacheProgress,
} from "./modelCache.js";
import { createManifestAllInTableProvider } from "./allInTables.js";
import type { BetterFfnManifest } from "./types.js";

function currentOrigin(): string {
  if (typeof location === "undefined") return "this origin";
  return location.origin === "null" ? location.href : location.origin;
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

export async function createBrowserDevice(
  options: BrowserDeviceOptions = {},
): Promise<GPUDevice> {
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
  const weights = await decodeModelWeights(
    await weightsResponse.arrayBuffer(),
    manifest,
  );
  const model = BetterFfnWebGpuModel.fromBuffers(
    device ?? (await createBrowserDevice()),
    manifest,
    weights,
  );
  model.allInTableProvider = createManifestAllInTableProvider(manifest, manifestUrl);
  return model;
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
  const model = BetterFfnWebGpuModel.fromBuffers(
    options.device ?? (await createBrowserDevice()),
    loaded.manifest,
    loaded.weights,
  );
  model.allInTableProvider = createManifestAllInTableProvider(
    loaded.manifest,
    manifestUrl,
  );
  return model;
}

export function loadBrowserModelFromBuffers(
  device: GPUDevice,
  manifest: BetterFfnManifest | unknown,
  weights: ArrayBuffer,
): BetterFfnWebGpuModel {
  const model = BetterFfnWebGpuModel.fromBuffers(device, manifest, weights);
  if (model.manifest.allIn) {
    const baseUrl =
      typeof globalThis.location === "undefined"
        ? "http://localhost/"
        : globalThis.location.href;
    model.allInTableProvider = createManifestAllInTableProvider(model.manifest, baseUrl);
  }
  return model;
}

export {
  BetterFfnWebGpuModel,
  BrowserCfrEvaluator,
  createBrowserCfrEvaluator,
  loadModelBytesWithCache,
};
