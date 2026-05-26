import { createManifestAllInTableProvider } from "./allInTables.js";
import {
  BetterFfnWebGpuModel,
  createBrowserCfrEvaluator,
  createBrowserDevice,
} from "./browser.js";
import { loadModelBytesWithCache } from "./modelCache.js";
import type { BrowserCfrEvaluator } from "./browserEvaluator.js";
import type {
  SolverWorkerRequest,
  SolverWorkerResponse,
} from "./solverWorkerMessages.js";

const workerScope = globalThis as unknown as {
  addEventListener: (
    type: "message",
    listener: (event: MessageEvent<SolverWorkerRequest>) => void,
  ) => void;
  location: Location;
  postMessage: (message: SolverWorkerResponse) => void;
  close: () => void;
};

let device: GPUDevice | undefined;
let model: BetterFfnWebGpuModel | undefined;
let evaluator: BrowserCfrEvaluator | undefined;
let initPromise: Promise<void> | undefined;

function post(message: SolverWorkerResponse): void {
  workerScope.postMessage(message);
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

async function initRuntime(manifestUrl: string): Promise<void> {
  if (initPromise) return await initPromise;
  initPromise = (async () => {
    const absoluteManifestUrl = new URL(manifestUrl, workerScope.location.href).toString();
    post({
      type: "model-progress",
      progress: { phase: "manifest", message: "Requesting WebGPU device" },
    });
    const nextDevice = await createBrowserDevice({
      onError: (message) => post({ type: "webgpu-error", message }),
    });
    const loaded = await loadModelBytesWithCache(manifestUrl, {
      onProgress: (progress) => post({ type: "model-progress", progress }),
    });
    post({
      type: "model-progress",
      progress: { phase: "manifest", message: "Creating WebGPU model" },
    });
    const nextModel = BetterFfnWebGpuModel.fromBuffers(
      nextDevice,
      loaded.manifest,
      loaded.weights,
    );
    nextModel.allInTableProvider = createManifestAllInTableProvider(
      nextModel.manifest,
      absoluteManifestUrl,
      undefined,
      nextDevice,
    );
    const nextEvaluator = createBrowserCfrEvaluator(nextDevice, nextModel);
    device = nextDevice;
    model = nextModel;
    evaluator = nextEvaluator;
    post({
      type: "ready",
      runtime: {
        manifest: nextModel.manifest,
        cached: loaded.cached,
        usingSubgroups: nextDevice.features.has("subgroups" as GPUFeatureName),
      },
    });
    post({
      type: "model-progress",
      progress: {
        phase: loaded.cached ? "cache-hit" : "stored",
        message: loaded.cached
          ? "Model loaded from IndexedDB"
          : "Model loaded and cached",
      },
    });
  })();
  return await initPromise;
}

async function solve(id: number, request: SolverWorkerRequest & { type: "solve" }): Promise<void> {
  try {
    if (!evaluator) throw new Error("solver worker is not ready");
    const result = await evaluator.evaluateSpot({
      ...request.request,
      onProgress: (progress) => post({ type: "solve-progress", id, progress }),
    });
    post({ type: "solve-result", id, result });
  } catch (error) {
    post({ type: "solve-error", id, message: errorMessage(error) });
  }
}

function disposeRuntime(): void {
  evaluator?.dispose();
  model?.dispose();
  device?.destroy();
  evaluator = undefined;
  model = undefined;
  device = undefined;
  initPromise = undefined;
}

workerScope.addEventListener("message", (event) => {
  const message = event.data;
  if (message.type === "init") {
    void initRuntime(message.manifestUrl).catch((error) => {
      post({ type: "error", message: errorMessage(error) });
    });
  } else if (message.type === "solve") {
    void solve(message.id, message);
  } else {
    disposeRuntime();
    workerScope.close();
  }
});
