import { GpuCfrEvaluator } from "./evaluator.js";
import {
  makeEmptyStorageBuffer,
  makeStorageBuffer,
  readFloatBuffer,
  readUintBuffer,
} from "./gpuBuffers.js";
import { GpuPokerState } from "./gpuPokerState.js";
import { STATE_STRIDE } from "./pokerStateKernels.js";
import {
  initialUniformBeliefs,
  NUM_HANDS,
  PublicHunlEnv,
  showdownTerminalValues,
} from "./hunlEnv.js";
import { buildHeroOnlyBeliefs, normalizeBeliefs } from "./beliefs.js";
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

interface GpuStateLocalCfrProblem {
  actor: 0 | 1;
  action?: number;
  legalMaskBuffer: GPUBuffer;
  childValuesBuffer: GPUBuffer;
  dispose: () => void;
}

interface ChildValueBuffer {
  buffer: GPUBuffer;
  key: number;
}

interface CollectedChildStates {
  actor: 0 | 1;
  legalMask: number[];
  terminalValues: Float32Array<ArrayBuffer>;
  modelChildren: Array<{ action: number; env: PublicHunlEnv }>;
}

export class BrowserCfrEvaluator {
  readonly device: GPUDevice;
  readonly model: BetterFfnWebGpuModel;
  private readonly cfr: GpuCfrEvaluator;
  private readonly childValuePool = new Map<number, GPUBuffer[]>();

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
    const gpuState = new GpuPokerState(this.device, this.model.manifest, {
      ...(request.initialState ? { initialState: request.initialState } : {}),
      ...(request.publicCards ? { publicCards: request.publicCards } : {}),
      ...(request.heroPlayer !== undefined ? { heroPlayer: request.heroPlayer } : {}),
      ...(request.heroHand ? { heroHand: request.heroHand } : {}),
    });
    const knownCards: {
      publicCards?: readonly number[];
      heroPlayer?: 0 | 1;
      heroHand?: readonly [number, number];
    } = {};
    if (request.publicCards) knownCards.publicCards = request.publicCards;
    if (request.heroPlayer !== undefined) knownCards.heroPlayer = request.heroPlayer;
    if (request.heroHand) knownCards.heroHand = request.heroHand;
    env.configureKnownCards(knownCards);
    const initialBeliefs = this.initialBeliefs(request);
    const initialBeliefsBuffer = makeStorageBuffer(this.device, initialBeliefs);
    let beliefsBuffer = initialBeliefsBuffer;
    const releaseBeliefBuffers: Array<() => void> = [
      () => initialBeliefsBuffer.destroy(),
    ];
    let finalPolicy: Float32Array<ArrayBufferLike> | undefined;
    let finalActionProbs: Float32Array<ArrayBufferLike> | undefined;
    let beliefsAtSpot: Float32Array<ArrayBufferLike> | undefined;

    try {
      for (const action of request.spot) {
        this.assertAction(action, numActions);
        this.assertLegalAction(env, action);
        const solved = await this.solveCurrentGpuState(
          gpuState,
          env,
          env.toAct,
          beliefsBuffer,
          action,
          request.iterations,
          {
            readPolicy: false,
            readActionProbs: false,
            readBeliefs: false,
            returnGpuBeliefs: true,
          },
        );
        if (!solved.beliefsAfterBuffer || !solved.releaseBeliefsAfterBuffer) {
          throw new Error("internal error: prefix solve did not produce GPU beliefs");
        }
        beliefsBuffer = solved.beliefsAfterBuffer;
        releaseBeliefBuffers.push(solved.releaseBeliefsAfterBuffer);
        gpuState.step(action);
        env.stepBin(action);
      }

      const final = await this.solveCurrentGpuState(
        gpuState,
        env,
        env.toAct,
        beliefsBuffer,
        undefined,
        request.iterations,
      );
      finalPolicy = final.policy;
      finalActionProbs = final.actionProbs;
      beliefsAtSpot =
        request.spot.length === 0
          ? initialBeliefs
          : await readFloatBuffer(this.device, beliefsBuffer, 2 * NUM_HANDS);
      const legal = await gpuState.legalMask();

      return {
        beliefsAtSpot,
        actionProbs: finalActionProbs,
        policy: finalPolicy,
        actionLabels: [...this.model.actionLabels],
        legalMask: Array.from(legal),
        actor: env.toAct,
      };
    } finally {
      for (let i = releaseBeliefBuffers.length - 1; i >= 0; i -= 1) {
        releaseBeliefBuffers[i]!();
      }
      gpuState.dispose();
    }
  }

  dispose(): void {
    this.cfr.dispose();
    for (const buffers of this.childValuePool.values()) {
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
    this.childValuePool.clear();
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

  private async solveCurrentGpuState(
    state: GpuPokerState,
    env: PublicHunlEnv,
    actor: 0 | 1,
    beliefs: Float32Array<ArrayBufferLike> | GPUBuffer,
    selectedAction: number | undefined,
    iterations: number,
    readOptions?: {
      readPolicy?: boolean;
      readActionProbs?: boolean;
      readBeliefs?: boolean;
      returnGpuBeliefs?: boolean;
    },
  ) {
    const problem = await this.buildGpuStateProblem(state, env, actor, beliefs);
    if (selectedAction !== undefined) {
      problem.action = selectedAction;
    }
    let disposeWithGpuBeliefs = false;
    try {
      const cfrProblem: { actor: 0 | 1; action?: number } = {
        actor: problem.actor,
      };
      if (problem.action !== undefined) {
        cfrProblem.action = problem.action;
      }
      const solved = await this.cfr.solveGpuBuffers(
        cfrProblem,
        beliefs,
        NUM_HANDS,
        this.model.manifest.architecture.numActions,
        iterations,
        problem.childValuesBuffer,
        problem.legalMaskBuffer,
        readOptions,
      );
      if (solved.releaseBeliefsAfterBuffer) {
        const releaseBeliefsAfterBuffer = solved.releaseBeliefsAfterBuffer;
        solved.releaseBeliefsAfterBuffer = () => {
          releaseBeliefsAfterBuffer();
          problem.dispose();
        };
        disposeWithGpuBeliefs = true;
      }
      return solved;
    } finally {
      if (!disposeWithGpuBeliefs) {
        problem.dispose();
      }
    }
  }

  private async buildProblem(
    env: PublicHunlEnv,
    beliefs: Float32Array<ArrayBufferLike>,
  ): Promise<LocalCfrProblem> {
    const collected = this.collectChildStates(env, beliefs);
    const childValues = collected.terminalValues;

    if (collected.modelChildren.length > 0) {
      const values = await this.model.predictBatchHandValues(
        collected.modelChildren.map((child) => child.env),
        beliefs,
      );
      const childValueSize = 2 * NUM_HANDS;
      for (let i = 0; i < collected.modelChildren.length; i += 1) {
        childValues.set(
          values.subarray(i * childValueSize, (i + 1) * childValueSize),
          collected.modelChildren[i]!.action * childValueSize,
        );
      }
    }

    return {
      actor: collected.actor,
      legalMask: collected.legalMask,
      childValues,
    };
  }

  private async buildGpuProblem(
    env: PublicHunlEnv,
    beliefs: Float32Array<ArrayBufferLike>,
  ): Promise<GpuLocalCfrProblem> {
    const collected = this.collectChildStates(env, beliefs);
    const childValueSize = 2 * NUM_HANDS;
    const childValues = this.acquireChildValueBuffer(collected.terminalValues);
    let modelValuesDispose: (() => void) | undefined;

    if (collected.modelChildren.length > 0) {
      const values = await this.model.predictBatchHandValuesGpu(
        collected.modelChildren.map((child) => child.env),
        beliefs,
      );
      modelValuesDispose = values.dispose;
      const bytesPerChild = childValueSize * Float32Array.BYTES_PER_ELEMENT;
      const encoder = this.device.createCommandEncoder();
      for (let i = 0; i < collected.modelChildren.length; i += 1) {
        encoder.copyBufferToBuffer(
          values.buffer,
          i * bytesPerChild,
          childValues.buffer,
          collected.modelChildren[i]!.action * bytesPerChild,
          bytesPerChild,
        );
      }
      this.device.queue.submit([encoder.finish()]);
    }

    return {
      actor: collected.actor,
      legalMask: collected.legalMask,
      childValuesBuffer: childValues.buffer,
      dispose: () => {
        modelValuesDispose?.();
        this.releaseChildValueBuffer(childValues);
      },
    };
  }

  private async buildGpuStateProblem(
    state: GpuPokerState,
    env: PublicHunlEnv,
    actor: 0 | 1,
    beliefs: Float32Array<ArrayBufferLike> | GPUBuffer,
  ): Promise<GpuStateLocalCfrProblem> {
    let modelValues: Awaited<ReturnType<BetterFfnWebGpuModel["predictBatchHandValuesGpuStates"]>> | undefined;
    let modelStates: GPUBuffer | undefined;
    const beliefBuffer =
      beliefs instanceof Float32Array ? makeStorageBuffer(this.device, beliefs) : beliefs;
    const ownsBeliefBuffer = beliefs instanceof Float32Array;
    const preModelEncoder = this.device.createCommandEncoder();
    const children = state.buildChildren(preModelEncoder);
    modelStates = this.device.createBuffer({
      size: this.model.manifest.architecture.numActions * STATE_STRIDE * Float32Array.BYTES_PER_ELEMENT,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });
    const modelActionMap = makeEmptyStorageBuffer(
      this.device,
      this.model.manifest.architecture.numActions,
    );
    const modelCountBuffer = makeStorageBuffer(this.device, new Uint32Array([0]));
    state.encodeCompactModelStates(
      preModelEncoder,
      children,
      modelStates,
      modelActionMap,
      modelCountBuffer,
    );
    state.encodeTerminalValues(preModelEncoder, children, beliefBuffer);
    this.device.queue.submit([preModelEncoder.finish()]);

    const modelActionCount = (await readUintBuffer(this.device, modelCountBuffer, 1))[0] ?? 0;
    if (modelActionCount > 0) {
      modelValues = await this.model.predictBatchHandValuesGpuStates(
        modelStates!,
        modelActionCount,
        beliefs,
      );
      const valueEncoder = this.device.createCommandEncoder();
      state.encodeScatterModelValues(
        valueEncoder,
        modelValues.buffer,
        modelActionMap,
        modelActionCount,
        children.childValues,
      );
      this.device.queue.submit([valueEncoder.finish()]);
    }

    return {
      actor,
      legalMaskBuffer: children.legalMask,
      childValuesBuffer: children.childValues,
      dispose: () => {
        modelValues?.dispose();
        modelStates?.destroy();
        modelActionMap.destroy();
        modelCountBuffer.destroy();
        if (ownsBeliefBuffer) {
          beliefBuffer.destroy();
        }
        children.dispose();
      },
    };
  }

  private collectChildStates(
    env: PublicHunlEnv,
    beliefs: Float32Array<ArrayBufferLike>,
  ): CollectedChildStates {
    const actor = env.toAct;
    const numActions = this.model.manifest.architecture.numActions;
    const legal = env.legalBinsAmountAndMask();
    if (legal.mask.length !== numActions) {
      throw new Error(
        `environment produced ${legal.mask.length} actions, model expects ${numActions}`,
      );
    }

    const childValueSize = 2 * NUM_HANDS;
    const terminalValues = new Float32Array(numActions * childValueSize);
    const modelChildren: Array<{ action: number; env: PublicHunlEnv }> = [];

    for (let action = 0; action < numActions; action += 1) {
      if (!legal.mask[action]) continue;
      const child = env.clone();
      const step = child.stepBin(action, legal);
      const offset = action * childValueSize;
      if (child.done) {
        if (child.hasFolded[0] || child.hasFolded[1]) {
          terminalValues.fill(step.reward, offset, offset + NUM_HANDS);
          terminalValues.fill(
            -step.reward,
            offset + NUM_HANDS,
            offset + childValueSize,
          );
        } else {
          terminalValues.set(
            showdownTerminalValues(child, beliefs),
            offset,
          );
        }
      } else {
        modelChildren.push({ action, env: child });
      }
    }

    return {
      actor,
      legalMask: legal.mask,
      terminalValues,
      modelChildren,
    };
  }

  private acquireChildValueBuffer(data: Float32Array<ArrayBuffer>): ChildValueBuffer {
    const key = Math.max(4, Math.ceil(data.byteLength / 4) * 4);
    const buffer =
      this.childValuePool.get(key)?.pop() ??
      this.device.createBuffer({
        size: key,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
      });
    this.device.queue.writeBuffer(buffer, 0, data);
    return { buffer, key };
  }

  private releaseChildValueBuffer(childValues: ChildValueBuffer): void {
    const buffers = this.childValuePool.get(childValues.key);
    if (buffers) {
      buffers.push(childValues.buffer);
    } else {
      this.childValuePool.set(childValues.key, [childValues.buffer]);
    }
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
    if (request.heroHand) {
      const options: {
        heroPlayer: 0 | 1;
        heroHand: readonly [number, number];
        publicCards?: readonly number[];
      } = {
        heroPlayer: request.heroPlayer ?? 0,
        heroHand: request.heroHand,
      };
      if (request.publicCards) options.publicCards = request.publicCards;
      return buildHeroOnlyBeliefs(options);
    }
    return initialUniformBeliefs();
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
