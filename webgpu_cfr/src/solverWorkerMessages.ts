import type { ModelCacheProgress } from "./modelCache.js";
import type {
  BetterFfnManifest,
  BrowserEvaluationResult,
  EvaluateSpotRequest,
  SolveProgress,
} from "./types.js";

export interface SolverRuntimeInfo {
  manifest: BetterFfnManifest;
  cached: boolean;
  usingSubgroups: boolean;
}

export type SolverEvaluateSpotRequest = Omit<EvaluateSpotRequest, "onProgress">;

export type SolverWorkerRequest =
  | {
      type: "init";
      manifestUrl: string;
    }
  | {
      type: "solve";
      id: number;
      request: SolverEvaluateSpotRequest;
    }
  | {
      type: "dispose";
    };

export type SolverWorkerResponse =
  | {
      type: "model-progress";
      progress: ModelCacheProgress;
    }
  | {
      type: "ready";
      runtime: SolverRuntimeInfo;
    }
  | {
      type: "error";
      message: string;
    }
  | {
      type: "webgpu-error";
      message: string;
    }
  | {
      type: "solve-progress";
      id: number;
      progress: SolveProgress;
    }
  | {
      type: "solve-result";
      id: number;
      result: BrowserEvaluationResult;
    }
  | {
      type: "solve-error";
      id: number;
      message: string;
    };
