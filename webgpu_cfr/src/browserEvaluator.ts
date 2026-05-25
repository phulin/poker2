import { SparseCfrResolver, type SparseResolveOptions } from "./sparseResolver.js";
import { PublicHunlEnv } from "./hunlEnv.js";
import { buildPublicBeliefs, normalizeBeliefs } from "./beliefs.js";
import { resolveCfrDefaults } from "./modelFormat.js";
import type { BetterFfnWebGpuModel } from "./betterFfnWebGpuModel.js";
import type {
  BrowserEvaluationResult,
  EvaluateSpotRequest,
  SolveProgress,
} from "./types.js";

export class BrowserCfrEvaluator {
  readonly device: GPUDevice;
  readonly model: BetterFfnWebGpuModel;
  private readonly sparseCfr: SparseCfrResolver;

  constructor(device: GPUDevice, model: BetterFfnWebGpuModel) {
    this.device = device;
    this.model = model;
    this.sparseCfr = new SparseCfrResolver(model);
  }

  async evaluateSpot(request: EvaluateSpotRequest): Promise<BrowserEvaluationResult> {
    const cfrDefaults = resolveCfrDefaults(this.model.manifest);
    const iterations = request.iterations ?? cfrDefaults.iterations;
    if (!Number.isInteger(iterations) || iterations <= 0) {
      throw new Error("iterations must be a positive integer");
    }
    const depth = request.depth ?? cfrDefaults.depth;
    if (!Number.isInteger(depth) || depth <= 0) {
      throw new Error("depth must be a positive integer");
    }
    return await this.evaluateSpotSparse(
      request,
      depth,
      iterations,
      cfrDefaults.cfrAvg,
    );
  }

  dispose(): void {
    this.sparseCfr.dispose();
  }

  private async evaluateSpotSparse(
    request: EvaluateSpotRequest,
    depth: number,
    iterations: number,
    defaultCfrAvg: boolean,
  ): Promise<BrowserEvaluationResult> {
    const cfrAvg = request.cfrAvg ?? defaultCfrAvg;
    const numActions = this.model.manifest.architecture.numActions;
    const env = PublicHunlEnv.fromManifest(
      this.model.manifest,
      request.initialState,
    );
    const knownCards: {
      publicCards?: readonly number[];
      heroPlayer?: 0 | 1;
      heroHand?: readonly [number, number];
    } = {};
    if (request.publicCards) knownCards.publicCards = request.publicCards;
    if (request.heroPlayer !== undefined) knownCards.heroPlayer = request.heroPlayer;
    if (request.heroHand) knownCards.heroHand = request.heroHand;
    env.configureKnownCards(knownCards);

    let beliefs = this.initialBeliefs(request);
    const solveCount = request.spot.length + 1;
    const totalIterations = solveCount * iterations;
    const emitProgress = (
      solveIndex: number,
      phase: SolveProgress["phase"],
      iteration: number,
    ): void => {
      const completedIterations = solveIndex * iterations + iteration;
      request.onProgress?.({
        completedIterations,
        totalIterations,
        percent: Math.min(100, (completedIterations / totalIterations) * 100),
        phase,
        solveIndex,
        solveCount,
        iteration,
        iterations,
      });
    };

    for (let solveIndex = 0; solveIndex < request.spot.length; solveIndex += 1) {
      const action = request.spot[solveIndex]!;
      this.assertAction(action, numActions);
      this.assertLegalAction(env, action);
      const solved = await this.sparseCfr.solve(env, beliefs, {
        depth,
        iterations,
        cfrAvg,
        selectedAction: action,
        readPolicy: false,
        readActionProbs: false,
        readBeliefs: true,
        onProgress: ({ iteration }) => emitProgress(solveIndex, "prefix", iteration),
      });
      if (!solved.beliefsAfter) {
        throw new Error("internal error: sparse prefix solve did not produce beliefs");
      }
      env.stepBin(action);
      beliefs = solved.beliefsAfter;
    }

    const finalSolveIndex = request.spot.length;
    const finalOptions: SparseResolveOptions = {
      depth,
      iterations,
      cfrAvg,
      onProgress: ({ iteration }) => emitProgress(finalSolveIndex, "final", iteration),
    };
    if (request.readPolicy !== undefined) finalOptions.readPolicy = request.readPolicy;
    if (request.readActionProbs !== undefined) {
      finalOptions.readActionProbs = request.readActionProbs;
    }
    const final = await this.sparseCfr.solve(env, beliefs, finalOptions);
    const legal = env.legalBinsAmountAndMask();
    return {
      beliefsAtSpot: request.readBeliefs === false ? new Float32Array(0) : beliefs,
      actionProbs: final.actionProbs,
      policy: final.policy,
      actionLabels: [...this.model.actionLabels],
      legalMask: legal.mask,
      actor: env.toAct,
    };
  }

  private assertAction(action: number, numActions: number): void {
    if (!Number.isInteger(action) || action < 0 || action >= numActions) {
      throw new Error(`action ${action} is outside [0, ${numActions})`);
    }
  }

  private initialBeliefs(
    request: EvaluateSpotRequest,
  ): Float32Array<ArrayBufferLike> {
    if (request.initialBeliefs) {
      return normalizeBeliefs(request.initialBeliefs);
    }
    return request.publicCards
      ? buildPublicBeliefs({ publicCards: request.publicCards })
      : buildPublicBeliefs();
  }

  private assertLegalAction(env: PublicHunlEnv, action: number): void {
    const legal = env.legalBinsAmountAndMask();
    if (!legal.mask[action]) {
      throw new Error(`action ${action} is not legal for the current public state`);
    }
  }
}

export function createBrowserCfrEvaluator(
  device: GPUDevice,
  model: BetterFfnWebGpuModel,
): BrowserCfrEvaluator {
  return new BrowserCfrEvaluator(device, model);
}
