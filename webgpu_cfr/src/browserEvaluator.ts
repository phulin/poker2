import { GpuCfrEvaluator } from "./evaluator.js";
import { makeStorageBuffer } from "./gpuBuffers.js";
import { initialUniformBeliefs, NUM_HANDS, PublicHunlEnv } from "./hunlEnv.js";
import type { BetterFfnWebGpuModel } from "./betterFfnWebGpuModel.js";
import type {
  BrowserEvaluationResult,
  EvaluateSpotRequest,
  LocalCfrProblem,
} from "./types.js";

interface GpuLocalCfrProblem {
  actor: 0 | 1;
  action?: number;
  legalMask: number[];
  childValuesBuffer: GPUBuffer;
  dispose: () => void;
}

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
      const solved = await this.solveCurrentProblem(
        env,
        beliefs,
        action,
        request.iterations,
        { readPolicy: false, readActionProbs: false },
      );
      if (!solved.beliefsAfter) {
        throw new Error("internal error: prefix solve did not update beliefs");
      }
      beliefs = solved.beliefsAfter;
      env.stepBin(action);
    }

    const final = await this.solveCurrentProblem(
      env,
      beliefs,
      undefined,
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

  private async solveCurrentProblem(
    env: PublicHunlEnv,
    beliefs: Float32Array<ArrayBufferLike>,
    selectedAction: number | undefined,
    iterations: number,
    readOptions?: { readPolicy?: boolean; readActionProbs?: boolean },
  ) {
    if (this.model.device === this.device) {
      const problem = await this.buildGpuProblem(env, beliefs);
      if (selectedAction !== undefined) {
        problem.action = selectedAction;
      }
      try {
        return await this.cfr.solveGpuChildValues(
          problem,
          beliefs,
          NUM_HANDS,
          this.model.manifest.architecture.numActions,
          iterations,
          problem.childValuesBuffer,
          readOptions,
        );
      } finally {
        problem.dispose();
      }
    }

    const problem = await this.buildProblem(env, beliefs);
    if (selectedAction !== undefined) {
      problem.action = selectedAction;
    }
    return await this.cfr.solve(
      problem,
      beliefs,
      NUM_HANDS,
      this.model.manifest.architecture.numActions,
      iterations,
      readOptions,
    );
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
    const modelChildren: Array<{ action: number; env: PublicHunlEnv }> = [];

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
        modelChildren.push({ action, env: child });
      }
    }

    if (modelChildren.length > 0) {
      const values = await this.model.predictBatchHandValues(
        modelChildren.map((child) => child.env),
        beliefs,
      );
      const childValueSize = 2 * NUM_HANDS;
      for (let i = 0; i < modelChildren.length; i += 1) {
        childValues.set(
          values.subarray(i * childValueSize, (i + 1) * childValueSize),
          modelChildren[i]!.action * childValueSize,
        );
      }
    }

    return {
      actor,
      legalMask: legal.mask,
      childValues,
    };
  }

  private async buildGpuProblem(
    env: PublicHunlEnv,
    beliefs: Float32Array<ArrayBufferLike>,
  ): Promise<GpuLocalCfrProblem> {
    const actor = env.toAct;
    const numActions = this.model.manifest.architecture.numActions;
    const legal = env.legalBinsAmountAndMask();
    if (legal.mask.length !== numActions) {
      throw new Error(
        `environment produced ${legal.mask.length} actions, model expects ${numActions}`,
      );
    }

    const childValueSize = 2 * NUM_HANDS;
    const childValues = new Float32Array(numActions * childValueSize);
    const modelChildren: Array<{ action: number; env: PublicHunlEnv }> = [];

    for (let action = 0; action < numActions; action += 1) {
      if (!legal.mask[action]) continue;
      const child = env.clone();
      const step = child.stepBin(action, legal);
      const offset = action * childValueSize;
      if (child.done) {
        childValues.fill(step.reward, offset, offset + NUM_HANDS);
        childValues.fill(-step.reward, offset + NUM_HANDS, offset + childValueSize);
      } else {
        modelChildren.push({ action, env: child });
      }
    }

    const childValuesBuffer = makeStorageBuffer(this.device, childValues);
    let modelValuesDispose: (() => void) | undefined;

    if (modelChildren.length > 0) {
      const values = await this.model.predictBatchHandValuesGpu(
        modelChildren.map((child) => child.env),
        beliefs,
      );
      modelValuesDispose = values.dispose;
      const bytesPerChild = childValueSize * Float32Array.BYTES_PER_ELEMENT;
      const encoder = this.device.createCommandEncoder();
      for (let i = 0; i < modelChildren.length; i += 1) {
        encoder.copyBufferToBuffer(
          values.buffer,
          i * bytesPerChild,
          childValuesBuffer,
          modelChildren[i]!.action * bytesPerChild,
          bytesPerChild,
        );
      }
      this.device.queue.submit([encoder.finish()]);
    }

    return {
      actor,
      legalMask: legal.mask,
      childValuesBuffer,
      dispose: () => {
        modelValuesDispose?.();
        childValuesBuffer.destroy();
      },
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
