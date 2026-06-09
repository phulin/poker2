import { createManifestAllInTableProvider } from "./allInTables.js";
import { createBrowserCfrEvaluator, createBrowserDevice } from "./browser.js";
import { BetterFfnWebGpuModel } from "./betterFfnWebGpuModel.js";
import type { BrowserCfrEvaluator } from "./browserEvaluator.js";
import { isLoadedModelSetBytes, loadRuntimeBytesWithCache } from "./modelCache.js";
import {
  BetterFfnModelRegistry,
  isBetterFfnModelRegistry,
  rootModelForRuntime,
  type BetterFfnRuntime,
} from "./modelRegistry.js";
import type { SolverWorkerRequest, SolverWorkerResponse } from "./solverWorkerMessages.js";

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
let runtime: BetterFfnRuntime | undefined;
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
    post({
      type: "model-progress",
      progress: { phase: "manifest", message: "Requesting WebGPU device" },
    });
    const nextDevice = await createBrowserDevice({
      onError: (message) => post({ type: "webgpu-error", message }),
    });
    const loaded = await loadRuntimeBytesWithCache(manifestUrl, {
      onProgress: (progress) => post({ type: "model-progress", progress }),
    });
    post({
      type: "model-progress",
      progress: { phase: "manifest", message: "Creating WebGPU runtime" },
    });
    const attachAllInProvider = (
      model: BetterFfnWebGpuModel,
      stageManifestUrl: string,
    ): BetterFfnWebGpuModel => {
      model.allInTableProvider = createManifestAllInTableProvider(
        model.manifest,
        stageManifestUrl,
        undefined,
        nextDevice,
      );
      return model;
    };
    const nextRuntime: BetterFfnRuntime = isLoadedModelSetBytes(loaded)
      ? new BetterFfnModelRegistry(
          {
            flop: attachAllInProvider(
              BetterFfnWebGpuModel.fromBuffers(
                nextDevice,
                loaded.stages.flop.manifest,
                loaded.stages.flop.weights,
              ),
              loaded.stages.flop.manifestUrl,
            ),
            turn: attachAllInProvider(
              BetterFfnWebGpuModel.fromBuffers(
                nextDevice,
                loaded.stages.turn.manifest,
                loaded.stages.turn.weights,
              ),
              loaded.stages.turn.manifestUrl,
            ),
            river: attachAllInProvider(
              BetterFfnWebGpuModel.fromBuffers(
                nextDevice,
                loaded.stages.river.manifest,
                loaded.stages.river.weights,
              ),
              loaded.stages.river.manifestUrl,
            ),
          },
          loaded.manifest.defaultStage,
        )
      : attachAllInProvider(
          BetterFfnWebGpuModel.fromBuffers(nextDevice, loaded.manifest, loaded.weights),
          loaded.manifestUrl,
        );
    const nextModel = rootModelForRuntime(nextRuntime);
    const nextEvaluator = createBrowserCfrEvaluator(nextDevice, nextRuntime);
    device = nextDevice;
    runtime = nextRuntime;
    evaluator = nextEvaluator;
    post({
      type: "ready",
      runtime: {
        manifest: nextModel.manifest,
        ...(isLoadedModelSetBytes(loaded) ? { modelSet: loaded.manifest } : {}),
        cached: loaded.cached,
        usingSubgroups: nextDevice.features.has("subgroups" as GPUFeatureName),
      },
    });
    post({
      type: "model-progress",
      progress: {
        phase: loaded.cached ? "cache-hit" : "stored",
        message: loaded.cached ? "Model loaded from IndexedDB" : "Model loaded and cached",
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

async function prefetchAllInTable(board: readonly number[]): Promise<void> {
  try {
    await initPromise;
    await rootModelForRuntime(runtime!)?.allInTableProvider?.prefetchForBoard?.(board);
  } catch {
    // Prefetch is opportunistic; the solve path will surface any required fetch errors.
  }
}

function disposeRuntime(): void {
  evaluator?.dispose();
  if (runtime) {
    if (isBetterFfnModelRegistry(runtime)) runtime.dispose();
    else runtime.dispose();
  }
  device?.destroy();
  evaluator = undefined;
  runtime = undefined;
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
  } else if (message.type === "prefetch-allin") {
    void prefetchAllInTable(message.board);
  } else {
    disposeRuntime();
    workerScope.close();
  }
});
