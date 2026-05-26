import { makeEmptyStorageBuffer, makeStorageBuffer, readUintBuffer } from "./gpuBuffers.js";
import { createComputePipeline, dispatchCompute } from "./gpuPipeline.js";
import { PublicHunlEnv } from "./hunlEnv.js";
import {
  STATE_ACTED_SINCE_RESET,
  STATE_ACTIONS_LAST_ROUND,
  STATE_ACTIONS_THIS_ROUND,
  STATE_BOARD0,
  STATE_BUTTON,
  STATE_COMMITTED0,
  STATE_COMMITTED1,
  STATE_DECK_POS,
  STATE_DECK0,
  STATE_DONE,
  STATE_HAS_FOLDED0,
  STATE_HAS_FOLDED1,
  STATE_IS_ALLIN0,
  STATE_IS_ALLIN1,
  STATE_LAST_TO_ACT,
  STATE_MIN_RAISE,
  STATE_POT,
  STATE_SCALE,
  STATE_STACK0,
  STATE_STACK1,
  STATE_STARTING0,
  STATE_STARTING1,
  STATE_STREET,
  STATE_STRIDE,
  STATE_TO_ACT,
  STATE_WINNER,
} from "./pokerStateKernels/layout.js";
import {
  POKER_BUILD_CHILD_STATES_WGSL,
  POKER_COMPACT_MODEL_STATES_WGSL,
  POKER_LEGAL_WGSL,
  POKER_SCATTER_MODEL_VALUES_WGSL,
  POKER_STEP_WGSL,
} from "./pokerStateKernels/stateMachine.js";
import {
  POKER_TERMINAL_RANKS_WGSL,
  POKER_TERMINAL_VALUES_WGSL,
} from "./pokerStateKernels/terminal.js";
import type { BetterFfnManifest, BrowserCfrInitialState, PlayerIndex } from "./types.js";

export interface GpuChildStateBatch {
  states: GPUBuffer;
  legalMask: GPUBuffer;
  terminalMask: GPUBuffer;
  terminalRanks: GPUBuffer;
  childValues: GPUBuffer;
  numActions: number;
  dispose: () => void;
}

interface GpuPokerStateSharedResources {
  handCards: GPUBuffer;
  legalPipeline: GPUComputePipeline;
  stepPipeline: GPUComputePipeline;
  buildChildrenPipeline: GPUComputePipeline;
  compactModelStatesPipeline: GPUComputePipeline;
  scatterModelValuesPipeline: GPUComputePipeline;
  terminalRanksPipeline: GPUComputePipeline;
  terminalValuesPipeline: GPUComputePipeline;
}

interface PooledStorageBuffer {
  buffer: GPUBuffer;
  key: number;
}

const sharedResourcesByDevice = new WeakMap<GPUDevice, GpuPokerStateSharedResources>();

export class GpuPokerState {
  readonly device: GPUDevice;
  readonly numActions: number;
  readonly state: GPUBuffer;
  private readonly betBins: GPUBuffer;
  private readonly handCards: GPUBuffer;
  private readonly legalPipeline: GPUComputePipeline;
  private readonly stepPipeline: GPUComputePipeline;
  private readonly buildChildrenPipeline: GPUComputePipeline;
  private readonly compactModelStatesPipeline: GPUComputePipeline;
  private readonly scatterModelValuesPipeline: GPUComputePipeline;
  private readonly terminalRanksPipeline: GPUComputePipeline;
  private readonly terminalValuesPipeline: GPUComputePipeline;
  private readonly paramsBuffer: GPUBuffer;
  private readonly childBatchPool = new Map<number, GPUBuffer[]>();
  private readonly numBetBins: number;
  private readonly flopShowdown: boolean;

  constructor(
    device: GPUDevice,
    manifest: BetterFfnManifest,
    options: {
      initialState?: BrowserCfrInitialState;
      publicCards?: readonly number[];
      heroPlayer?: PlayerIndex;
      heroHand?: readonly [number, number];
    } = {},
  ) {
    this.device = device;
    this.numActions = manifest.architecture.numActions;
    this.numBetBins = (options.initialState?.betBins ?? manifest.env.betBins).length;
    this.flopShowdown = options.initialState?.flopShowdown ?? manifest.env.flopShowdown;
    const env = PublicHunlEnv.fromManifest(manifest, options.initialState);
    env.configureKnownCards({
      ...(options.publicCards ? { publicCards: options.publicCards } : {}),
      ...(options.heroPlayer !== undefined ? { heroPlayer: options.heroPlayer } : {}),
      ...(options.heroHand ? { heroHand: options.heroHand } : {}),
    });
    this.state = makeStorageBuffer(device, stateDataFromEnv(env));
    this.betBins = makeStorageBuffer(
      device,
      new Float32Array(options.initialState?.betBins ?? manifest.env.betBins),
    );
    const shared = sharedResources(device);
    this.handCards = shared.handCards;
    this.paramsBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.legalPipeline = shared.legalPipeline;
    this.stepPipeline = shared.stepPipeline;
    this.buildChildrenPipeline = shared.buildChildrenPipeline;
    this.compactModelStatesPipeline = shared.compactModelStatesPipeline;
    this.scatterModelValuesPipeline = shared.scatterModelValuesPipeline;
    this.terminalRanksPipeline = shared.terminalRanksPipeline;
    this.terminalValuesPipeline = shared.terminalValuesPipeline;
  }

  async legalMask(): Promise<Uint32Array<ArrayBufferLike>> {
    const mask = makeEmptyStorageBuffer(this.device, this.numActions);
    try {
      this.writeParams(0);
      const encoder = this.device.createCommandEncoder();
      this.encode(encoder, this.legalPipeline, Math.ceil(this.numActions / 64), [
        { binding: 0, resource: { buffer: this.state } },
        { binding: 1, resource: { buffer: this.betBins } },
        { binding: 2, resource: { buffer: mask } },
        { binding: 3, resource: { buffer: this.paramsBuffer } },
      ]);
      this.device.queue.submit([encoder.finish()]);
      return await readUintBuffer(this.device, mask, this.numActions);
    } finally {
      mask.destroy();
    }
  }

  step(action: number): void {
    this.writeParams(action);
    const encoder = this.device.createCommandEncoder();
    this.encode(encoder, this.stepPipeline, 1, [
      { binding: 0, resource: { buffer: this.state } },
      { binding: 1, resource: { buffer: this.betBins } },
      { binding: 2, resource: { buffer: this.paramsBuffer } },
    ]);
    this.device.queue.submit([encoder.finish()]);
  }

  buildChildren(encoder?: GPUCommandEncoder): GpuChildStateBatch {
    this.writeParams(0);
    const states = this.acquireChildBatchStorage(this.numActions * STATE_STRIDE);
    const legalMask = this.acquireChildBatchStorage(this.numActions);
    const terminalMask = this.acquireChildBatchStorage(this.numActions);
    const terminalRanks = this.acquireChildBatchStorage(this.numActions * 1326);
    const childValues = this.acquireChildBatchStorage(this.numActions * 2 * 1326);
    const commandEncoder = encoder ?? this.device.createCommandEncoder();
    this.encode(commandEncoder, this.buildChildrenPipeline, Math.ceil(this.numActions / 64), [
      { binding: 0, resource: { buffer: this.state } },
      { binding: 1, resource: { buffer: states.buffer } },
      { binding: 2, resource: { buffer: this.betBins } },
      { binding: 3, resource: { buffer: legalMask.buffer } },
      { binding: 4, resource: { buffer: terminalMask.buffer } },
      { binding: 5, resource: { buffer: this.paramsBuffer } },
    ]);
    if (!encoder) {
      this.device.queue.submit([commandEncoder.finish()]);
    }
    return {
      states: states.buffer,
      legalMask: legalMask.buffer,
      terminalMask: terminalMask.buffer,
      terminalRanks: terminalRanks.buffer,
      childValues: childValues.buffer,
      numActions: this.numActions,
      dispose: () => {
        this.releaseChildBatchStorage(states);
        this.releaseChildBatchStorage(legalMask);
        this.releaseChildBatchStorage(terminalMask);
        this.releaseChildBatchStorage(terminalRanks);
        this.releaseChildBatchStorage(childValues);
      },
    };
  }

  writeTerminalValues(batch: GpuChildStateBatch, beliefs: GPUBuffer): void {
    this.writeParams(0);
    const encoder = this.device.createCommandEncoder();
    this.encodeTerminalValues(encoder, batch, beliefs);
    this.device.queue.submit([encoder.finish()]);
  }

  encodeTerminalValues(
    encoder: GPUCommandEncoder,
    batch: GpuChildStateBatch,
    beliefs: GPUBuffer,
  ): void {
    this.writeParams(0);
    this.encode3d(encoder, this.terminalRanksPipeline, 21, 1, this.numActions, [
      { binding: 0, resource: { buffer: batch.states } },
      { binding: 1, resource: { buffer: batch.legalMask } },
      { binding: 2, resource: { buffer: batch.terminalMask } },
      { binding: 3, resource: { buffer: this.handCards } },
      { binding: 4, resource: { buffer: batch.terminalRanks } },
      { binding: 5, resource: { buffer: this.paramsBuffer } },
    ]);
    this.encode3d(encoder, this.terminalValuesPipeline, 21, 2, this.numActions, [
      { binding: 0, resource: { buffer: batch.states } },
      { binding: 1, resource: { buffer: batch.legalMask } },
      { binding: 2, resource: { buffer: batch.terminalMask } },
      { binding: 3, resource: { buffer: beliefs } },
      { binding: 4, resource: { buffer: this.handCards } },
      { binding: 5, resource: { buffer: batch.terminalRanks } },
      { binding: 6, resource: { buffer: batch.childValues } },
      { binding: 7, resource: { buffer: this.paramsBuffer } },
    ]);
  }

  encodeCompactModelStates(
    encoder: GPUCommandEncoder,
    batch: GpuChildStateBatch,
    modelStates: GPUBuffer,
    modelActionMap: GPUBuffer,
    modelCount: GPUBuffer,
  ): void {
    this.writeParams(0);
    this.encode(encoder, this.compactModelStatesPipeline, Math.ceil(this.numActions / 64), [
      { binding: 0, resource: { buffer: batch.states } },
      { binding: 1, resource: { buffer: batch.legalMask } },
      { binding: 2, resource: { buffer: batch.terminalMask } },
      { binding: 3, resource: { buffer: modelStates } },
      { binding: 4, resource: { buffer: modelActionMap } },
      { binding: 5, resource: { buffer: modelCount } },
      { binding: 6, resource: { buffer: this.paramsBuffer } },
    ]);
  }

  encodeScatterModelValues(
    encoder: GPUCommandEncoder,
    modelValues: GPUBuffer,
    modelActionMap: GPUBuffer,
    modelActionCount: number,
    childValues: GPUBuffer,
  ): void {
    this.device.queue.writeBuffer(
      this.paramsBuffer,
      0,
      new Uint32Array([Math.max(0, Math.trunc(modelActionCount)), 0, 0, 0]),
    );
    this.encode3d(encoder, this.scatterModelValuesPipeline, 21, 2, modelActionCount, [
      { binding: 0, resource: { buffer: modelValues } },
      { binding: 1, resource: { buffer: modelActionMap } },
      { binding: 2, resource: { buffer: childValues } },
      { binding: 3, resource: { buffer: this.paramsBuffer } },
    ]);
  }

  dispose(): void {
    this.state.destroy();
    this.betBins.destroy();
    this.paramsBuffer.destroy();
    for (const buffers of this.childBatchPool.values()) {
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
    this.childBatchPool.clear();
  }

  private writeParams(action: number): void {
    this.device.queue.writeBuffer(
      this.paramsBuffer,
      0,
      new Uint32Array([
        this.numActions,
        Math.max(0, Math.trunc(action)),
        this.numBetBins,
        this.flopShowdown ? 1 : 0,
      ]),
    );
  }

  private acquireChildBatchStorage(elements: number): PooledStorageBuffer {
    const key = Math.max(4, Math.ceil((elements * Float32Array.BYTES_PER_ELEMENT) / 4) * 4);
    const buffer =
      this.childBatchPool.get(key)?.pop() ??
      this.device.createBuffer({
        size: key,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
    return { buffer, key };
  }

  private releaseChildBatchStorage(storage: PooledStorageBuffer): void {
    const buffers = this.childBatchPool.get(storage.key);
    if (buffers) {
      buffers.push(storage.buffer);
    } else {
      this.childBatchPool.set(storage.key, [storage.buffer]);
    }
  }

  private encode(
    encoder: GPUCommandEncoder,
    pipeline: GPUComputePipeline,
    workgroups: number,
    entries: GPUBindGroupEntry[],
  ): void {
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries,
    });
    dispatchCompute(encoder, pipeline, bindGroup, workgroups);
  }

  private encode3d(
    encoder: GPUCommandEncoder,
    pipeline: GPUComputePipeline,
    x: number,
    y: number,
    z: number,
    entries: GPUBindGroupEntry[],
  ): void {
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries,
    });
    dispatchCompute(encoder, pipeline, bindGroup, x, y, z);
  }
}

function sharedResources(device: GPUDevice): GpuPokerStateSharedResources {
  let resources = sharedResourcesByDevice.get(device);
  if (!resources) {
    resources = {
      handCards: makeStorageBuffer(device, buildHandCards()),
      legalPipeline: pipeline(device, POKER_LEGAL_WGSL, "poker-state-legal"),
      stepPipeline: pipeline(device, POKER_STEP_WGSL, "poker-state-step"),
      buildChildrenPipeline: pipeline(
        device,
        POKER_BUILD_CHILD_STATES_WGSL,
        "poker-state-build-children",
      ),
      compactModelStatesPipeline: pipeline(
        device,
        POKER_COMPACT_MODEL_STATES_WGSL,
        "poker-state-compact-model-states",
      ),
      scatterModelValuesPipeline: pipeline(
        device,
        POKER_SCATTER_MODEL_VALUES_WGSL,
        "poker-state-scatter-model-values",
      ),
      terminalRanksPipeline: pipeline(
        device,
        POKER_TERMINAL_RANKS_WGSL,
        "poker-state-terminal-ranks",
      ),
      terminalValuesPipeline: pipeline(
        device,
        POKER_TERMINAL_VALUES_WGSL,
        "poker-state-terminal-values",
      ),
    };
    sharedResourcesByDevice.set(device, resources);
  }
  return resources;
}

function pipeline(device: GPUDevice, source: string, label: string): GPUComputePipeline {
  return createComputePipeline(device, source, label);
}

function buildHandCards(): Uint32Array<ArrayBuffer> {
  const data = new Uint32Array(1326 * 2);
  let hand = 0;
  for (let c0 = 0; c0 < 51; c0 += 1) {
    for (let c1 = c0 + 1; c1 < 52; c1 += 1) {
      data[2 * hand] = c0;
      data[2 * hand + 1] = c1;
      hand += 1;
    }
  }
  return data;
}

function stateDataFromEnv(env: PublicHunlEnv): Float32Array<ArrayBuffer> {
  const data = new Float32Array(STATE_STRIDE);
  data[STATE_BUTTON] = env.button;
  data[STATE_STREET] = env.street;
  data[STATE_TO_ACT] = env.toAct;
  data[STATE_LAST_TO_ACT] = env.lastToAct;
  data[STATE_POT] = env.pot;
  data[STATE_MIN_RAISE] = env.minRaise;
  data[STATE_ACTIONS_THIS_ROUND] = env.actionsThisRound;
  data[STATE_ACTIONS_LAST_ROUND] = env.actionsLastRound;
  data[STATE_ACTED_SINCE_RESET] = env.actedSinceReset ? 1 : 0;
  data[STATE_STACK0] = env.stacks[0];
  data[STATE_STACK1] = env.stacks[1];
  data[STATE_COMMITTED0] = env.committed[0];
  data[STATE_COMMITTED1] = env.committed[1];
  data[STATE_STARTING0] = env.startingStacks[0];
  data[STATE_STARTING1] = env.startingStacks[1];
  data[STATE_SCALE] = env.scale;
  data[STATE_IS_ALLIN0] = env.isAllin[0] ? 1 : 0;
  data[STATE_IS_ALLIN1] = env.isAllin[1] ? 1 : 0;
  data[STATE_HAS_FOLDED0] = env.hasFolded[0] ? 1 : 0;
  data[STATE_HAS_FOLDED1] = env.hasFolded[1] ? 1 : 0;
  data[STATE_DONE] = env.done ? 1 : 0;
  data[STATE_WINNER] = env.winner;
  data[STATE_DECK_POS] = env.deckPos;
  for (let i = 0; i < 5; i += 1) {
    data[STATE_BOARD0 + i] = env.boardIndices[i] ?? -1;
  }
  for (let i = 0; i < 9; i += 1) {
    data[STATE_DECK0 + i] = env.deck[i] ?? -1;
  }
  data[STATE_DECK0 + 9] = env.bb;
  return data;
}
