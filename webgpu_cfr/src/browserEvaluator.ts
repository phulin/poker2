import { GpuCfrEvaluator } from "./evaluator.js";
import { initialUniformBeliefs, NUM_HANDS, PublicHunlEnv } from "./hunlEnv.js";
import type { BetterFfnWebGpuModel } from "./betterFfnWebGpuModel.js";
import type {
  BrowserEvaluationResult,
  EvaluateSpotRequest,
  LocalCfrProblem,
} from "./types.js";

export class BrowserCfrEvaluator {
  readonly device: GPUDevice;
  readonly model: BetterFfnWebGpuModel;
  private readonly cfr: GpuCfrEvaluator;

  constructor(device: GPUDevice, model: BetterFfnWebGpuModel) {
    this.device = device;
    this.model = model;
    this.cfr = new GpuCfrEvaluator(device);
  }

  async evaluateSpot(request: EvaluateSpotRequest): Promise<BrowserEvaluationResult> {
    if (!Number.isInteger(request.iterations) || request.iterations <= 0) {
      throw new Error("iterations must be a positive integer");
    }
    const numActions = this.model.manifest.architecture.numActions;
    const env = PublicHunlEnv.fromManifest(
      this.model.manifest,
      request.initialState,
    );
    let beliefs: Float32Array<ArrayBufferLike> = initialUniformBeliefs();
    let finalPolicy: Float32Array<ArrayBufferLike> | undefined;
    let finalActionProbs: Float32Array<ArrayBufferLike> | undefined;

    for (const action of request.spot) {
      this.assertAction(action, numActions);
      const problem = await this.buildProblem(env, beliefs);
      problem.action = action;
      const solved = await this.cfr.solve(
        problem,
        beliefs,
        NUM_HANDS,
        numActions,
        request.iterations,
      );
      if (!solved.beliefsAfter) {
        throw new Error("internal error: prefix solve did not update beliefs");
      }
      beliefs = solved.beliefsAfter;
      env.stepBin(action);
    }

    const finalProblem = await this.buildProblem(env, beliefs);
    const final = await this.cfr.solve(
      finalProblem,
      beliefs,
      NUM_HANDS,
      numActions,
      request.iterations,
    );
    finalPolicy = final.policy;
    finalActionProbs = final.actionProbs;

    return {
      beliefsAtSpot: beliefs,
      actionProbs: finalActionProbs,
      policy: finalPolicy,
      actionLabels: [...this.model.actionLabels],
    };
  }

  private async buildProblem(
    env: PublicHunlEnv,
    beliefs: Float32Array<ArrayBufferLike>,
  ): Promise<LocalCfrProblem> {
    const actor = env.toAct;
    const numActions = this.model.manifest.architecture.numActions;
    const legal = env.legalBinsAmountAndMask();
    if (legal.mask.length !== numActions) {
      throw new Error(
        `environment produced ${legal.mask.length} actions, model expects ${numActions}`,
      );
    }
    const childValues = new Float32Array(numActions * 2 * NUM_HANDS);

    for (let action = 0; action < numActions; action += 1) {
      if (!legal.mask[action]) {
        continue;
      }
      const child = env.clone();
      const step = child.stepBin(action, legal);
      const offset = action * 2 * NUM_HANDS;
      if (child.done) {
        childValues.fill(step.reward, offset, offset + NUM_HANDS);
        childValues.fill(-step.reward, offset + NUM_HANDS, offset + 2 * NUM_HANDS);
      } else {
        const values = await this.model.predictHandValues(child, beliefs);
        childValues.set(values, offset);
      }
    }

    return {
      actor,
      legalMask: legal.mask,
      childValues,
    };
  }

  private assertAction(action: number, numActions: number): void {
    if (!Number.isInteger(action) || action < 0 || action >= numActions) {
      throw new Error(`action ${action} is outside [0, ${numActions})`);
    }
  }
}

export function createBrowserCfrEvaluator(
  device: GPUDevice,
  model: BetterFfnWebGpuModel,
): BrowserCfrEvaluator {
  return new BrowserCfrEvaluator(device, model);
}
