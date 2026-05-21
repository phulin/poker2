import {
  ADD3_WGSL,
  LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_WGSL,
  LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_BATCH2_SUBGROUP_WGSL,
  LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_SUBGROUP_WGSL,
  LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_COLS_512_WGSL,
  LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_COLS_512_SUBGROUP_WGSL,
  LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_WGSL,
  LEAKY_RELU_MAT_VEC_BATCH_WGSL,
  LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_WGSL,
  LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_BATCH2_SUBGROUP_WGSL,
  LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_SUBGROUP_WGSL,
  LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_EXACT_ROWS_COLS_512_WGSL,
  LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_EXACT_ROWS_WGSL,
  LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_WGSL,
  MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_WGSL,
  MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_BATCH2_SUBGROUP_WGSL,
  MAT_VEC_BATCH_EXACT_ROWS_COLS_1326_BATCH2_SUBGROUP_WGSL,
  MAT_VEC_BATCH_EXACT_ROWS_COLS_512_BATCH2_SUBGROUP_WGSL,
  MAT_VEC_BATCH_EXACT_ROWS_COLS_512_SUBGROUP_WGSL,
  MAT_VEC_BATCH_EXACT_ROWS_COLS_512_WGSL,
  MAT_VEC_BATCH_EXACT_ROWS_WGSL,
  MAT_VEC_BATCH_SMALL_COLS_WGSL,
  MAT_VEC_BATCH_WGSL,
  MAT_VEC_WGSL,
  PLAYER_BOARD_HADAMARD_WGSL,
  REPEAT_ROWS_WGSL,
  RMS_NORM_BATCH_WGSL,
  RMS_NORM_WGSL,
  SCALED_RESIDUAL_ADD_WGSL,
  ZERO_SUM_BATCH_WGSL,
} from "./modelKernels.js";
import { POKER_ENCODE_FEATURES_WGSL } from "./pokerStateKernels.js";
import {
  makeStorageBuffer,
  readFloatBuffer,
} from "./gpuBuffers.js";
import {
  parseBetterFfnManifest,
  requireTensor,
  tensorsFromWeights,
  type TensorMap,
} from "./modelFormat.js";
import {
  encodeBetterFeatures,
  handCombos,
  NUM_HANDS,
  type PublicHunlEnv,
} from "./hunlEnv.js";
import type { BetterFfnManifest } from "./types.js";

interface GpuTensor {
  data: Float32Array<ArrayBufferLike>;
  shape: number[];
  buffer: GPUBuffer;
}

interface TempBuffer {
  buffer: GPUBuffer;
  key: number;
  kind: "storage" | "uniform";
}

interface PredictOptions {
  includePolicy?: boolean;
}

const BATCH_ROW_BLOCK = 4;

export interface BetterFfnPrediction {
  handValues: Float32Array<ArrayBufferLike>;
  policyLogits?: Float32Array<ArrayBufferLike>;
}

export interface GpuHandValuePrediction {
  buffer: GPUBuffer;
  batch: number;
  valuesPerSample: number;
  dispose: () => void;
}

export interface PreparedBatchFeatures {
  batch: number;
  baseEmbedding: GPUBuffer;
  contextFeatures: GPUBuffer;
  boardRankLow?: GPUBuffer;
  boardSuitLow?: GPUBuffer;
  dispose: () => void;
}

export class BetterFfnWebGpuModel {
  readonly device: GPUDevice;
  readonly manifest: BetterFfnManifest;
  readonly actionLabels: string[];
  private readonly tensors = new Map<string, GpuTensor>();
  private readonly dummyBias: GPUBuffer;
  private readonly handEmbeddingT: GPUBuffer;
  private readonly rankPairOneHotT?: GPUBuffer;
  private readonly suitPairOneHotT?: GPUBuffer;
  private readonly rankPairLowT?: GPUBuffer;
  private readonly suitPairLowT?: GPUBuffer;
  private readonly matVecPipeline: GPUComputePipeline;
  private readonly matVecBatchPipeline: GPUComputePipeline;
  private readonly matVecBatchSmallColsPipeline: GPUComputePipeline;
  private readonly matVecBatchExactRowsPipeline: GPUComputePipeline;
  private readonly matVecBatchExactRowsCols512Pipeline: GPUComputePipeline;
  private readonly matVecBatchExactRowsCols512Batch2SubgroupPipeline:
    | GPUComputePipeline
    | undefined;
  private readonly matVecBatchExactRowsCols512SubgroupPipeline:
    | GPUComputePipeline
    | undefined;
  private readonly matVecBatchExactRowsCols1024Pipeline: GPUComputePipeline;
  private readonly matVecBatchExactRowsCols1024Batch2SubgroupPipeline:
    | GPUComputePipeline
    | undefined;
  private readonly matVecBatchExactRowsCols1326Batch2SubgroupPipeline:
    | GPUComputePipeline
    | undefined;
  private readonly leakyReluMatVecBatchPipeline: GPUComputePipeline;
  private readonly leakyReluMatVecBatchExactRowsPipeline: GPUComputePipeline;
  private readonly leakyReluMatVecBatchExactRowsCols512Pipeline: GPUComputePipeline;
  private readonly leakyReluMatVecBatchExactRowsCols512SubgroupPipeline:
    | GPUComputePipeline
    | undefined;
  private readonly leakyReluMatVecBatchExactRowsCols1024Pipeline: GPUComputePipeline;
  private readonly leakyReluMatVecBatchExactRowsCols1024Batch2SubgroupPipeline:
    | GPUComputePipeline
    | undefined;
  private readonly leakyReluMatVecBatchExactRowsCols1024SubgroupPipeline:
    | GPUComputePipeline
    | undefined;
  private readonly leakyReluResidualMatVecBatchPipeline: GPUComputePipeline;
  private readonly leakyReluResidualMatVecBatchExactRowsPipeline: GPUComputePipeline;
  private readonly leakyReluResidualMatVecBatchExactRowsCols512Pipeline: GPUComputePipeline;
  private readonly leakyReluResidualMatVecBatchExactRowsCols1024Pipeline: GPUComputePipeline;
  private readonly leakyReluResidualMatVecBatchExactRowsCols1024Batch2SubgroupPipeline:
    | GPUComputePipeline
    | undefined;
  private readonly leakyReluResidualMatVecBatchExactRowsCols1024SubgroupPipeline:
    | GPUComputePipeline
    | undefined;
  private readonly playerBoardHadamardPipeline: GPUComputePipeline;
  private readonly rmsNormPipeline: GPUComputePipeline;
  private readonly rmsNormBatchPipeline: GPUComputePipeline;
  private readonly scaledResidualAddPipeline: GPUComputePipeline;
  private readonly add3Pipeline: GPUComputePipeline;
  private readonly repeatRowsPipeline: GPUComputePipeline;
  private readonly zeroSumBatchPipeline: GPUComputePipeline;
  private readonly stateFeaturePipeline: GPUComputePipeline;
  private readonly storagePool = new Map<number, GPUBuffer[]>();
  private readonly uniformPool = new Map<number, GPUBuffer[]>();
  private recordingEncoder: GPUCommandEncoder | undefined;

  constructor(
    device: GPUDevice,
    manifestInput: BetterFfnManifest | unknown,
    weights: ArrayBuffer,
  ) {
    this.device = device;
    this.manifest = parseBetterFfnManifest(manifestInput);
    this.actionLabels = [...this.manifest.actionLabels];
    const loaded = tensorsFromWeights(this.manifest, weights);
    this.validateRequiredTensors(loaded);
    for (const [name, tensor] of loaded) {
      const buffer = makeStorageBuffer(device, tensor.data);
      this.tensors.set(name, {
        data: tensor.data,
        shape: tensor.manifest.shape,
        buffer,
      });
    }
    this.dummyBias = makeStorageBuffer(device, new Float32Array([0]));
    this.handEmbeddingT = makeStorageBuffer(device, this.buildHandEmbeddingT());
    if (this.manifest.architecture.boardInteractionDim > 0) {
      this.rankPairOneHotT = makeStorageBuffer(device, this.buildPairOneHotT(13, 91));
      this.suitPairOneHotT = makeStorageBuffer(device, this.buildPairOneHotT(4, 10));
      this.rankPairLowT = makeStorageBuffer(
        device,
        this.transposeTensor("rank_pair_low_embedding.weight"),
      );
      this.suitPairLowT = makeStorageBuffer(
        device,
        this.transposeTensor("suit_pair_low_embedding.weight"),
      );
    }
    this.matVecPipeline = this.pipeline(MAT_VEC_WGSL, "better-ffn-mat-vec");
    this.matVecBatchPipeline = this.pipeline(
      MAT_VEC_BATCH_WGSL,
      "better-ffn-mat-vec-batch",
    );
    this.matVecBatchSmallColsPipeline = this.pipeline(
      MAT_VEC_BATCH_SMALL_COLS_WGSL,
      "better-ffn-mat-vec-batch-small-cols",
    );
    this.matVecBatchExactRowsPipeline = this.pipeline(
      MAT_VEC_BATCH_EXACT_ROWS_WGSL,
      "better-ffn-mat-vec-batch-exact-rows",
    );
    this.matVecBatchExactRowsCols512Pipeline = this.pipeline(
      MAT_VEC_BATCH_EXACT_ROWS_COLS_512_WGSL,
      "better-ffn-mat-vec-batch-exact-rows-cols-512",
    );
    this.matVecBatchExactRowsCols512Batch2SubgroupPipeline =
      device.features.has("subgroups" as GPUFeatureName)
        ? this.pipeline(
            MAT_VEC_BATCH_EXACT_ROWS_COLS_512_BATCH2_SUBGROUP_WGSL,
            "better-ffn-mat-vec-batch-exact-rows-cols-512-batch2-subgroup",
          )
        : undefined;
    this.matVecBatchExactRowsCols512SubgroupPipeline =
      device.features.has("subgroups" as GPUFeatureName)
        ? this.pipeline(
            MAT_VEC_BATCH_EXACT_ROWS_COLS_512_SUBGROUP_WGSL,
            "better-ffn-mat-vec-batch-exact-rows-cols-512-subgroup",
          )
        : undefined;
    this.matVecBatchExactRowsCols1024Pipeline = this.pipeline(
      MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_WGSL,
      "better-ffn-mat-vec-batch-exact-rows-cols-1024",
    );
    this.matVecBatchExactRowsCols1024Batch2SubgroupPipeline =
      device.features.has("subgroups" as GPUFeatureName)
        ? this.pipeline(
            MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_BATCH2_SUBGROUP_WGSL,
            "better-ffn-mat-vec-batch-exact-rows-cols-1024-batch2-subgroup",
          )
        : undefined;
    this.matVecBatchExactRowsCols1326Batch2SubgroupPipeline =
      device.features.has("subgroups" as GPUFeatureName)
        ? this.pipeline(
            MAT_VEC_BATCH_EXACT_ROWS_COLS_1326_BATCH2_SUBGROUP_WGSL,
            "better-ffn-mat-vec-batch-exact-rows-cols-1326-batch2-subgroup",
          )
        : undefined;
    this.leakyReluMatVecBatchPipeline = this.pipeline(
      LEAKY_RELU_MAT_VEC_BATCH_WGSL,
      "better-ffn-leaky-relu-mat-vec-batch",
    );
    this.leakyReluMatVecBatchExactRowsPipeline = this.pipeline(
      LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_WGSL,
      "better-ffn-leaky-relu-mat-vec-batch-exact-rows",
    );
    this.leakyReluMatVecBatchExactRowsCols512Pipeline = this.pipeline(
      LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_COLS_512_WGSL,
      "better-ffn-leaky-relu-mat-vec-batch-exact-rows-cols-512",
    );
    this.leakyReluMatVecBatchExactRowsCols512SubgroupPipeline =
      device.features.has("subgroups" as GPUFeatureName)
        ? this.pipeline(
            LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_COLS_512_SUBGROUP_WGSL,
            "better-ffn-leaky-relu-mat-vec-batch-exact-rows-cols-512-subgroup",
          )
        : undefined;
    this.leakyReluMatVecBatchExactRowsCols1024Pipeline = this.pipeline(
      LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_WGSL,
      "better-ffn-leaky-relu-mat-vec-batch-exact-rows-cols-1024",
    );
    this.leakyReluMatVecBatchExactRowsCols1024Batch2SubgroupPipeline =
      device.features.has("subgroups" as GPUFeatureName)
        ? this.pipeline(
            LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_BATCH2_SUBGROUP_WGSL,
            "better-ffn-leaky-relu-mat-vec-batch-exact-rows-cols-1024-batch2-subgroup",
          )
        : undefined;
    this.leakyReluMatVecBatchExactRowsCols1024SubgroupPipeline =
      device.features.has("subgroups" as GPUFeatureName)
        ? this.pipeline(
            LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_SUBGROUP_WGSL,
            "better-ffn-leaky-relu-mat-vec-batch-exact-rows-cols-1024-subgroup",
          )
        : undefined;
    this.leakyReluResidualMatVecBatchPipeline = this.pipeline(
      LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_WGSL,
      "better-ffn-leaky-relu-residual-mat-vec-batch",
    );
    this.leakyReluResidualMatVecBatchExactRowsPipeline = this.pipeline(
      LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_EXACT_ROWS_WGSL,
      "better-ffn-leaky-relu-residual-mat-vec-batch-exact-rows",
    );
    this.leakyReluResidualMatVecBatchExactRowsCols512Pipeline = this.pipeline(
      LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_EXACT_ROWS_COLS_512_WGSL,
      "better-ffn-leaky-relu-residual-mat-vec-batch-exact-rows-cols-512",
    );
    this.leakyReluResidualMatVecBatchExactRowsCols1024Pipeline = this.pipeline(
      LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_WGSL,
      "better-ffn-leaky-relu-residual-mat-vec-batch-exact-rows-cols-1024",
    );
    this.leakyReluResidualMatVecBatchExactRowsCols1024Batch2SubgroupPipeline =
      device.features.has("subgroups" as GPUFeatureName)
        ? this.pipeline(
            LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_BATCH2_SUBGROUP_WGSL,
            "better-ffn-leaky-relu-residual-mat-vec-batch-exact-rows-cols-1024-batch2-subgroup",
          )
        : undefined;
    this.leakyReluResidualMatVecBatchExactRowsCols1024SubgroupPipeline =
      device.features.has("subgroups" as GPUFeatureName)
        ? this.pipeline(
            LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_SUBGROUP_WGSL,
            "better-ffn-leaky-relu-residual-mat-vec-batch-exact-rows-cols-1024-subgroup",
          )
        : undefined;
    this.playerBoardHadamardPipeline = this.pipeline(
      PLAYER_BOARD_HADAMARD_WGSL,
      "better-ffn-player-board-hadamard",
    );
    this.rmsNormPipeline = this.pipeline(
      RMS_NORM_WGSL,
      "better-ffn-rms-norm",
    );
    this.rmsNormBatchPipeline = this.pipeline(
      RMS_NORM_BATCH_WGSL,
      "better-ffn-rms-norm-batch",
    );
    this.scaledResidualAddPipeline = this.pipeline(
      SCALED_RESIDUAL_ADD_WGSL,
      "better-ffn-scaled-residual-add",
    );
    this.add3Pipeline = this.pipeline(ADD3_WGSL, "better-ffn-add3");
    this.repeatRowsPipeline = this.pipeline(
      REPEAT_ROWS_WGSL,
      "better-ffn-repeat-rows",
    );
    this.zeroSumBatchPipeline = this.pipeline(
      ZERO_SUM_BATCH_WGSL,
      "better-ffn-zero-sum-batch",
    );
    this.stateFeaturePipeline = this.pipeline(
      POKER_ENCODE_FEATURES_WGSL,
      "better-ffn-state-features",
    );
  }

  static fromBuffers(
    device: GPUDevice,
    manifest: BetterFfnManifest | unknown,
    weights: ArrayBuffer,
  ): BetterFfnWebGpuModel {
    return new BetterFfnWebGpuModel(device, manifest, weights);
  }

  async predictHandValues(
    env: PublicHunlEnv,
    beliefs: Float32Array<ArrayBufferLike>,
  ): Promise<Float32Array<ArrayBufferLike>> {
    return (await this.predict(env, beliefs, { includePolicy: false })).handValues;
  }

  async predictBatchHandValues(
    envs: readonly PublicHunlEnv[],
    beliefs: Float32Array<ArrayBufferLike>,
  ): Promise<Float32Array<ArrayBufferLike>> {
    return (await this.predictBatch(envs, beliefs, { includePolicy: false }))
      .handValues;
  }

  async predictBatchHandValuesGpu(
    envs: readonly PublicHunlEnv[],
    beliefs: Float32Array<ArrayBufferLike> | GPUBuffer,
    prepared?: PreparedBatchFeatures,
    beforeSubmit?: (encoder: GPUCommandEncoder, handValues: GPUBuffer) => void,
  ): Promise<GpuHandValuePrediction> {
    const prediction = this.enqueuePredictBatch(envs, beliefs, {
      includePolicy: false,
    }, prepared, beforeSubmit);
    return {
      buffer: prediction.handValuesBuffer,
      batch: prediction.batch,
      valuesPerSample: 2 * NUM_HANDS,
      dispose: prediction.dispose,
    };
  }

  prepareBatchFeatures(envs: readonly PublicHunlEnv[]): PreparedBatchFeatures {
    if (envs.length === 0) {
      const empty = this.device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      return {
        batch: 0,
        baseEmbedding: empty,
        contextFeatures: empty,
        dispose: () => empty.destroy(),
      };
    }

    const batch = envs.length;
    const hiddenDim = this.manifest.architecture.hiddenDim;
    const interactionDim = this.manifest.architecture.boardInteractionDim;
    const temps: TempBuffer[] = [];
    const persistent: TempBuffer[] = [];
    const detach = (buffer: GPUBuffer): void => {
      const index = temps.findIndex((temp) => temp.buffer === buffer);
      if (index < 0) return;
      persistent.push(temps[index]!);
      temps.splice(index, 1);
    };
    const storage = (
      data: Float32Array<ArrayBufferLike> | Uint32Array<ArrayBufferLike>,
    ): GPUBuffer => {
      const temp = this.acquireStorage(data.length);
      this.device.queue.writeBuffer(
        temp.buffer,
        0,
        data as Float32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>,
      );
      temps.push(temp);
      return temp.buffer;
    };
    const empty = (elements: number): GPUBuffer => {
      const temp = this.acquireStorage(elements);
      temps.push(temp);
      return temp.buffer;
    };
    const uniform = (
      data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    ): GPUBuffer => {
      const temp = this.acquireUniform(data.byteLength);
      this.device.queue.writeBuffer(temp.buffer, 0, data);
      temps.push(temp);
      return temp.buffer;
    };

    const encoder = this.device.createCommandEncoder();
    this.recordingEncoder = encoder;
    try {
      const encodedFeatures = envs.map((env) => encodeBetterFeatures(env));
      const context = new Float32Array(batch * 11);
      const base = new Float32Array(batch * hiddenDim);
      for (let i = 0; i < batch; i += 1) {
        const features = encodedFeatures[i]!;
        context.set(features.context, i * 11);
        base.set(this.baseEmbedding(features.street, features.board), i * hiddenDim);
      }
      const contextFeatures = this.leakyReluBlockBatch(
        "context_encoder",
        storage(context),
        batch,
        11,
        hiddenDim,
        hiddenDim,
        empty,
        uniform,
      );
      const baseEmbedding = storage(base);

      let boardRankLow: GPUBuffer | undefined;
      let boardSuitLow: GPUBuffer | undefined;
      if (interactionDim > 0) {
        const rankCountsCpu = new Float32Array(batch * 13);
        const suitCountsCpu = new Float32Array(batch * 4);
        for (let row = 0; row < encodedFeatures.length; row += 1) {
          const board = encodedFeatures[row]!.board;
          for (let i = 0; i < 5; i += 1) {
            const card = board[i] ?? -1;
            if (card >= 0) {
              const rankOffset = row * 13 + (card % 13);
              const suitOffset = row * 4 + Math.floor(card / 13);
              rankCountsCpu[rankOffset] = rankCountsCpu[rankOffset]! + 1;
              suitCountsCpu[suitOffset] = suitCountsCpu[suitOffset]! + 1;
            }
          }
        }
        boardRankLow = empty(batch * interactionDim);
        this.matVecBatch(
          this.tensor("board_rank_low.weight").buffer,
          storage(rankCountsCpu),
          this.dummyBias,
          boardRankLow,
          interactionDim,
          13,
          batch,
          13,
          interactionDim,
          0,
          0,
          false,
          uniform,
        );
        boardSuitLow = empty(batch * interactionDim);
        this.matVecBatch(
          this.tensor("board_suit_low.weight").buffer,
          storage(suitCountsCpu),
          this.dummyBias,
          boardSuitLow,
          interactionDim,
          4,
          batch,
          4,
          interactionDim,
          0,
          0,
          false,
          uniform,
        );
      }

      this.recordingEncoder = undefined;
      this.device.queue.submit([encoder.finish()]);
      detach(baseEmbedding);
      detach(contextFeatures);
      if (boardRankLow) detach(boardRankLow);
      if (boardSuitLow) detach(boardSuitLow);
      let disposed = false;
      return {
        batch,
        baseEmbedding,
        contextFeatures,
        ...(boardRankLow ? { boardRankLow } : {}),
        ...(boardSuitLow ? { boardSuitLow } : {}),
        dispose: () => {
          if (disposed) return;
          disposed = true;
          for (const temp of persistent) {
            temp.buffer.destroy();
          }
          for (const temp of temps) {
            this.releaseTemp(temp);
          }
        },
      };
    } catch (error) {
      this.recordingEncoder = undefined;
      for (const temp of persistent) {
        temp.buffer.destroy();
      }
      for (const temp of temps) {
        this.releaseTemp(temp);
      }
      throw error;
    }
  }

  async predictBatchHandValuesGpuStates(
    states: GPUBuffer,
    batch: number,
    beliefs: Float32Array<ArrayBufferLike> | GPUBuffer,
  ): Promise<GpuHandValuePrediction> {
    const prediction = this.enqueuePredictBatchFromStateBuffer(states, batch, beliefs, {
      includePolicy: false,
    });
    return {
      buffer: prediction.handValuesBuffer,
      batch: prediction.batch,
      valuesPerSample: 2 * NUM_HANDS,
      dispose: prediction.dispose,
    };
  }

  async predict(
    env: PublicHunlEnv,
    beliefs: Float32Array<ArrayBufferLike>,
    options: PredictOptions = {},
  ): Promise<BetterFfnPrediction> {
    const result = await this.predictBatch([env], beliefs, options);
    const handValues = result.handValues.slice(0, 2 * NUM_HANDS);
    const prediction: BetterFfnPrediction = { handValues };
    if (result.policyLogits) {
      prediction.policyLogits = result.policyLogits.slice(
        0,
        this.manifest.architecture.numActions * NUM_HANDS,
      );
    }
    return prediction;
  }

  async predictBatch(
    envs: readonly PublicHunlEnv[],
    beliefs: Float32Array<ArrayBufferLike>,
    options: PredictOptions = {},
  ): Promise<BetterFfnPrediction> {
    const gpuPrediction = this.enqueuePredictBatch(envs, beliefs, options);
    try {
      const handValues = await readFloatBuffer(
        this.device,
        gpuPrediction.handValuesBuffer,
        gpuPrediction.batch * 2 * NUM_HANDS,
      );
      const prediction: BetterFfnPrediction = { handValues };
      if (gpuPrediction.policyLogitsBuffer) {
        prediction.policyLogits = await readFloatBuffer(
          this.device,
          gpuPrediction.policyLogitsBuffer,
          gpuPrediction.batch * this.manifest.architecture.numActions * NUM_HANDS,
        );
      }
      return prediction;
    } finally {
      gpuPrediction.dispose();
    }
  }

  private enqueuePredictBatch(
    envs: readonly PublicHunlEnv[],
    beliefs: Float32Array<ArrayBufferLike> | GPUBuffer,
    options: PredictOptions = {},
    prepared?: PreparedBatchFeatures,
    beforeSubmit?: (encoder: GPUCommandEncoder, handValues: GPUBuffer) => void,
  ): {
    handValuesBuffer: GPUBuffer;
    policyLogitsBuffer?: GPUBuffer;
    batch: number;
    dispose: () => void;
  } {
    if (envs.length === 0) {
      const empty = this.acquireStorage(1);
      return {
        handValuesBuffer: empty.buffer,
        batch: 0,
        dispose: () => this.releaseTemp(empty),
      };
    }
    const hiddenDim = this.manifest.architecture.hiddenDim;
    const ffnDim = this.manifest.architecture.ffnDim;
    const rangeHiddenDim = this.manifest.architecture.rangeHiddenDim;
    const numPlayers = this.manifest.architecture.numPlayers;
    const numActions = this.manifest.architecture.numActions;
    const batch = envs.length;
    if (prepared && prepared.batch !== batch) {
      throw new Error(`prepared features batch ${prepared.batch} does not match ${batch}`);
    }
    const singleBeliefSize = numPlayers * NUM_HANDS;
    const batchBeliefSize = batch * singleBeliefSize;
    if (
      beliefs instanceof Float32Array &&
      beliefs.length !== singleBeliefSize &&
      beliefs.length !== batchBeliefSize
    ) {
      throw new Error(
        `belief vector has ${beliefs.length} entries, expected ${singleBeliefSize} or ${batchBeliefSize}`,
      );
    }

    const temps: TempBuffer[] = [];
    const storage = (
      data: Float32Array<ArrayBufferLike> | Uint32Array<ArrayBufferLike>,
    ): GPUBuffer => {
      const temp = this.acquireStorage(data.length);
      this.device.queue.writeBuffer(
        temp.buffer,
        0,
        data as Float32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>,
      );
      temps.push(temp);
      return temp.buffer;
    };
    const empty = (elements: number): GPUBuffer => {
      const temp = this.acquireStorage(elements);
      temps.push(temp);
      return temp.buffer;
    };
    const uniform = (
      data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    ): GPUBuffer => {
      const temp = this.acquireUniform(data.byteLength);
      this.device.queue.writeBuffer(temp.buffer, 0, data);
      temps.push(temp);
      return temp.buffer;
    };

    const encoder = this.device.createCommandEncoder();
    this.recordingEncoder = encoder;
    let submitted = false;

    try {
      const submitPredictionCommands = (): void => {
        if (submitted) return;
        this.recordingEncoder = undefined;
        this.device.queue.submit([encoder.finish()]);
        submitted = true;
      };
      let beliefBuffer: GPUBuffer;
      if (beliefs instanceof Float32Array) {
        const batchedBeliefs =
          beliefs.length === batchBeliefSize
            ? beliefs
            : new Float32Array(batchBeliefSize);
        if (beliefs.length === singleBeliefSize) {
          for (let i = 0; i < batch; i += 1) {
            batchedBeliefs.set(beliefs, i * singleBeliefSize);
          }
        }
        beliefBuffer = storage(batchedBeliefs);
      } else {
        beliefBuffer = beliefs;
      }
      const perPlayerBelief = empty(batch * numPlayers * hiddenDim);
      for (let player = 0; player < numPlayers; player += 1) {
        this.matVecBatch(
          this.handEmbeddingT,
          beliefBuffer,
          this.dummyBias,
          perPlayerBelief,
          hiddenDim,
          NUM_HANDS,
          batch,
          numPlayers * NUM_HANDS,
          numPlayers * hiddenDim,
          player * NUM_HANDS,
          player * hiddenDim,
          false,
          uniform,
        );
      }

      const beliefFeatures = this.leakyReluBlockBatch(
        "belief_proj",
        perPlayerBelief,
        batch,
        numPlayers * hiddenDim,
        rangeHiddenDim === 0 ? ffnDim : numPlayers * rangeHiddenDim,
        hiddenDim,
        empty,
        uniform,
      );

      const encodedFeatures = prepared
        ? undefined
        : envs.map((env) => encodeBetterFeatures(env));
      const interactionFeatures = this.buildBeliefBoardInteractionGpu(
        beliefBuffer,
        encodedFeatures?.map((features) => features.board),
        batch,
        numPlayers,
        hiddenDim,
        empty,
        storage,
        uniform,
        prepared,
      );
      const contextFeatures =
        prepared?.contextFeatures ??
        this.contextFeaturesForBatch(
          encodedFeatures!,
          batch,
          hiddenDim,
          empty,
          storage,
          uniform,
        );
      const baseEmbedding =
        prepared?.baseEmbedding ??
        this.baseEmbeddingForBatch(encodedFeatures!, batch, hiddenDim, storage);
      let x = empty(batch * hiddenDim);
      if (interactionFeatures) {
        const interactedBase = empty(batch * hiddenDim);
        this.add3(
          baseEmbedding,
          interactionFeatures.rank,
          interactionFeatures.suit,
          interactedBase,
          batch * hiddenDim,
          uniform,
        );
        this.add3(
          interactedBase,
          contextFeatures,
          beliefFeatures,
          x,
          batch * hiddenDim,
          uniform,
        );
      } else {
        this.add3(
          baseEmbedding,
          contextFeatures,
          beliefFeatures,
          x,
          batch * hiddenDim,
          uniform,
        );
      }

      const alpha = 1 / Math.sqrt(
        this.manifest.architecture.numHiddenLayers +
          this.manifest.architecture.numValueLayers,
      );
      for (let i = 0; i < this.manifest.architecture.numHiddenLayers; i += 1) {
        const out = this.leakyReluResidualBlockBatch(
          `trunk.${i}.inner`,
          x,
          batch,
          hiddenDim,
          ffnDim,
          alpha,
          empty,
          uniform,
        );
        x = out;
      }

      let valueInput = x;
      const valueRawBuffer = this.headBatch(
        "hand_value_head",
        valueInput,
        batch,
        hiddenDim,
        ffnDim,
        this.manifest.architecture.numValueLayers,
        numPlayers * NUM_HANDS,
        alpha,
        empty,
        uniform,
      );

      let policyBuffer: GPUBuffer | undefined;
      if (options.includePolicy) {
        let policyInput = x;
        const policyAlpha = alpha;
        policyBuffer = this.headBatch(
          "policy_head",
          policyInput,
          batch,
          hiddenDim,
          ffnDim,
          this.manifest.architecture.numPolicyLayers,
          numActions * NUM_HANDS,
          policyAlpha,
          empty,
          uniform,
        );
      }

      if (this.manifest.architecture.enforceZeroSum) {
        this.zeroSumBatch(valueRawBuffer, beliefBuffer, batch, uniform);
      }
      beforeSubmit?.(encoder, valueRawBuffer);
      submitPredictionCommands();

      let disposed = false;
      return {
        handValuesBuffer: valueRawBuffer,
        ...(policyBuffer ? { policyLogitsBuffer: policyBuffer } : {}),
        batch,
        dispose: () => {
          if (disposed) return;
          disposed = true;
          for (const temp of temps) {
            this.releaseTemp(temp);
          }
        },
      };
    } catch (error) {
      this.recordingEncoder = undefined;
      for (const temp of temps) {
        this.releaseTemp(temp);
      }
      throw error;
    }
  }

  private enqueuePredictBatchFromStateBuffer(
    states: GPUBuffer,
    batch: number,
    beliefs: Float32Array<ArrayBufferLike> | GPUBuffer,
    options: PredictOptions = {},
  ): {
    handValuesBuffer: GPUBuffer;
    policyLogitsBuffer?: GPUBuffer;
    batch: number;
    dispose: () => void;
  } {
    if (batch === 0) {
      const empty = this.acquireStorage(1);
      return {
        handValuesBuffer: empty.buffer,
        batch: 0,
        dispose: () => this.releaseTemp(empty),
      };
    }
    const hiddenDim = this.manifest.architecture.hiddenDim;
    const ffnDim = this.manifest.architecture.ffnDim;
    const rangeHiddenDim = this.manifest.architecture.rangeHiddenDim;
    const numPlayers = this.manifest.architecture.numPlayers;
    const numActions = this.manifest.architecture.numActions;
    if (beliefs instanceof Float32Array && beliefs.length !== numPlayers * NUM_HANDS) {
      throw new Error(
        `belief vector has ${beliefs.length} entries, expected ${numPlayers * NUM_HANDS}`,
      );
    }

    const temps: TempBuffer[] = [];
    const storage = (
      data: Float32Array<ArrayBufferLike> | Uint32Array<ArrayBufferLike>,
    ): GPUBuffer => {
      const temp = this.acquireStorage(data.length);
      this.device.queue.writeBuffer(
        temp.buffer,
        0,
        data as Float32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>,
      );
      temps.push(temp);
      return temp.buffer;
    };
    const empty = (elements: number): GPUBuffer => {
      const temp = this.acquireStorage(elements);
      temps.push(temp);
      return temp.buffer;
    };
    const uniform = (
      data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    ): GPUBuffer => {
      const temp = this.acquireUniform(data.byteLength);
      this.device.queue.writeBuffer(temp.buffer, 0, data);
      temps.push(temp);
      return temp.buffer;
    };

    const encoder = this.device.createCommandEncoder();
    this.recordingEncoder = encoder;
    let submitted = false;

    try {
      const submitPredictionCommands = (): void => {
        if (submitted) return;
        this.recordingEncoder = undefined;
        this.device.queue.submit([encoder.finish()]);
        submitted = true;
      };

      const beliefBuffer = beliefs instanceof Float32Array ? storage(beliefs) : beliefs;
      const context = empty(batch * 11);
      const baseEmbedding = empty(batch * hiddenDim);
      const rankCounts = empty(batch * 13);
      const suitCounts = empty(batch * 4);
      this.stateFeatures(
        states,
        context,
        baseEmbedding,
        rankCounts,
        suitCounts,
        batch,
        hiddenDim,
        uniform,
      );

      const perPlayerBelief = empty(batch * numPlayers * hiddenDim);
      for (let player = 0; player < numPlayers; player += 1) {
        this.matVecBatch(
          this.handEmbeddingT,
          beliefBuffer,
          this.dummyBias,
          perPlayerBelief,
          hiddenDim,
          NUM_HANDS,
          1,
          0,
          numPlayers * hiddenDim,
          player * NUM_HANDS,
          player * hiddenDim,
          false,
          uniform,
        );
      }

      const singleBeliefFeatures = this.leakyReluBlockBatch(
        "belief_proj",
        perPlayerBelief,
        1,
        numPlayers * hiddenDim,
        rangeHiddenDim === 0 ? ffnDim : numPlayers * rangeHiddenDim,
        hiddenDim,
        empty,
        uniform,
      );
      const beliefFeatures =
        batch === 1
          ? singleBeliefFeatures
          : this.repeatRows(singleBeliefFeatures, batch, hiddenDim, empty, uniform);

      const interactionFeatures = this.buildBeliefBoardInteractionGpu(
        beliefBuffer,
        undefined,
        batch,
        numPlayers,
        hiddenDim,
        empty,
        storage,
        uniform,
        { rankCounts, suitCounts, sharedBeliefs: true },
      );
      const contextFeatures = this.leakyReluBlockBatch(
        "context_encoder",
        context,
        batch,
        11,
        hiddenDim,
        hiddenDim,
        empty,
        uniform,
      );

      let x = empty(batch * hiddenDim);
      if (interactionFeatures) {
        const interactedBase = empty(batch * hiddenDim);
        this.add3(
          baseEmbedding,
          interactionFeatures.rank,
          interactionFeatures.suit,
          interactedBase,
          batch * hiddenDim,
          uniform,
        );
        this.add3(
          interactedBase,
          contextFeatures,
          beliefFeatures,
          x,
          batch * hiddenDim,
          uniform,
        );
      } else {
        this.add3(
          baseEmbedding,
          contextFeatures,
          beliefFeatures,
          x,
          batch * hiddenDim,
          uniform,
        );
      }

      const alpha = 1 / Math.sqrt(
        this.manifest.architecture.numHiddenLayers +
          this.manifest.architecture.numValueLayers,
      );
      for (let i = 0; i < this.manifest.architecture.numHiddenLayers; i += 1) {
        const out = this.leakyReluResidualBlockBatch(
          `trunk.${i}.inner`,
          x,
          batch,
          hiddenDim,
          ffnDim,
          alpha,
          empty,
          uniform,
        );
        x = out;
      }

      let valueInput = x;
      const valueRawBuffer = this.headBatch(
        "hand_value_head",
        valueInput,
        batch,
        hiddenDim,
        ffnDim,
        this.manifest.architecture.numValueLayers,
        numPlayers * NUM_HANDS,
        alpha,
        empty,
        uniform,
      );

      let policyBuffer: GPUBuffer | undefined;
      if (options.includePolicy) {
        policyBuffer = this.headBatch(
          "policy_head",
          x,
          batch,
          hiddenDim,
          ffnDim,
          this.manifest.architecture.numPolicyLayers,
          numActions * NUM_HANDS,
          alpha,
          empty,
          uniform,
        );
      }

      if (this.manifest.architecture.enforceZeroSum) {
        this.zeroSumBatch(valueRawBuffer, beliefBuffer, batch, uniform, 0);
      }
      submitPredictionCommands();

      let disposed = false;
      return {
        handValuesBuffer: valueRawBuffer,
        ...(policyBuffer ? { policyLogitsBuffer: policyBuffer } : {}),
        batch,
        dispose: () => {
          if (disposed) return;
          disposed = true;
          for (const temp of temps) {
            this.releaseTemp(temp);
          }
        },
      };
    } catch (error) {
      this.recordingEncoder = undefined;
      for (const temp of temps) {
        this.releaseTemp(temp);
      }
      throw error;
    }
  }

  dispose(): void {
    for (const tensor of this.tensors.values()) {
      tensor.buffer.destroy();
    }
    this.dummyBias.destroy();
    this.handEmbeddingT.destroy();
    this.rankPairOneHotT?.destroy();
    this.suitPairOneHotT?.destroy();
    this.rankPairLowT?.destroy();
    this.suitPairLowT?.destroy();
    for (const pool of [this.storagePool, this.uniformPool]) {
      for (const buffers of pool.values()) {
        for (const buffer of buffers) {
          buffer.destroy();
        }
      }
      pool.clear();
    }
  }

  private validateRequiredTensors(tensors: TensorMap): void {
    const hidden = this.manifest.architecture.hiddenDim;
    const ffn = this.manifest.architecture.ffnDim;
    const rangeHidden = this.manifest.architecture.rangeHiddenDim;
    const actions = this.manifest.architecture.numActions;
    requireTensor(tensors, "street_embedding.weight", [5, hidden]);
    requireTensor(tensors, "rank_embedding.weight", [14, hidden]);
    requireTensor(tensors, "suit_embedding.weight", [5, hidden]);
    const boardInteraction = this.manifest.architecture.boardInteractionDim;
    if (boardInteraction > 0) {
      requireTensor(tensors, "rank_pair_low_embedding.weight", [91, boardInteraction]);
      requireTensor(tensors, "board_rank_low.weight", [boardInteraction, 13]);
      requireTensor(tensors, "rank_board_interaction_out.weight", [
        hidden,
        2 * boardInteraction,
      ]);
      requireTensor(tensors, "suit_pair_low_embedding.weight", [10, boardInteraction]);
      requireTensor(tensors, "board_suit_low.weight", [boardInteraction, 4]);
      requireTensor(tensors, "suit_board_interaction_out.weight", [
        hidden,
        2 * boardInteraction,
      ]);
    }
    this.requireLinearBlock(
      tensors,
      "belief_proj",
      2 * hidden,
      rangeHidden === 0 ? ffn : 2 * rangeHidden,
      hidden,
    );
    this.requireLinearBlock(tensors, "context_encoder", 11, hidden, hidden);
    for (let i = 0; i < this.manifest.architecture.numHiddenLayers; i += 1) {
      this.requireLinearBlock(tensors, `trunk.${i}.inner`, hidden, ffn, hidden);
    }
    this.requireHead(
      tensors,
      "hand_value_head",
      this.manifest.architecture.numValueLayers,
      hidden,
      ffn,
      2 * NUM_HANDS,
    );
    this.requireHead(
      tensors,
      "policy_head",
      this.manifest.architecture.numPolicyLayers,
      hidden,
      ffn,
      actions * NUM_HANDS,
    );
  }

  private requireHead(
    tensors: TensorMap,
    head: "hand_value_head" | "policy_head",
    numLayers: number,
    hiddenDim: number,
    ffnDim: number,
    outDim: number,
  ): void {
    const directOutputPrefix = `${head}.${numLayers}`;
    if (tensors.has(`${directOutputPrefix}.linear_out.weight`)) {
      for (let i = 0; i < numLayers; i += 1) {
        this.requireLinearBlock(tensors, `${head}.${i}.inner`, hiddenDim, ffnDim, hiddenDim);
      }
      this.requireOutputProjection(tensors, directOutputPrefix, hiddenDim, outDim);
      return;
    }

    for (let i = 0; i < numLayers - 1; i += 1) {
      this.requireLinearBlock(tensors, `${head}.${i}.inner`, hiddenDim, ffnDim, hiddenDim);
    }
    this.requireLinearBlock(
      tensors,
      `${head}.${numLayers - 1}`,
      hiddenDim,
      ffnDim,
      outDim,
    );
  }

  private requireLinearBlock(
    tensors: TensorMap,
    prefix: string,
    inDim: number,
    hiddenDim: number,
    outDim: number,
  ): void {
    requireTensor(tensors, `${prefix}.norm.weight`, [inDim]);
    requireTensor(tensors, `${prefix}.linear_in.weight`, [hiddenDim, inDim]);
    requireTensor(tensors, `${prefix}.linear_out.weight`, [outDim, hiddenDim]);
    requireTensor(tensors, `${prefix}.linear_out.bias`, [outDim]);
  }

  private requireOutputProjection(
    tensors: TensorMap,
    prefix: string,
    inDim: number,
    outDim: number,
  ): void {
    requireTensor(tensors, `${prefix}.norm.weight`, [inDim]);
    requireTensor(tensors, `${prefix}.linear_out.weight`, [outDim, inDim]);
    requireTensor(tensors, `${prefix}.linear_out.bias`, [outDim]);
  }

  private buildHandEmbeddingT(): Float32Array<ArrayBuffer> {
    const hidden = this.manifest.architecture.hiddenDim;
    const rank = this.tensor("rank_embedding.weight");
    const suit = this.tensor("suit_embedding.weight");
    const combos = handCombos();
    const out = new Float32Array(hidden * NUM_HANDS);
    for (let hand = 0; hand < combos.length; hand += 1) {
      const [c0, c1] = combos[hand]!;
      const r0 = c0 % 13;
      const s0 = Math.floor(c0 / 13);
      const r1 = c1 % 13;
      const s1 = Math.floor(c1 / 13);
      for (let d = 0; d < hidden; d += 1) {
        out[d * NUM_HANDS + hand] =
          rank.data[r0 * hidden + d]! +
          suit.data[s0 * hidden + d]! +
          rank.data[r1 * hidden + d]! +
          suit.data[s1 * hidden + d]!;
      }
    }
    return out;
  }

  private buildBeliefBoardInteractionGpu(
    beliefBuffer: GPUBuffer,
    boards: readonly (readonly number[])[] | undefined,
    batch: number,
    numPlayers: number,
    hidden: number,
    empty: (elements: number) => GPUBuffer,
    storage: (
      data: Float32Array<ArrayBufferLike> | Uint32Array<ArrayBufferLike>,
    ) => GPUBuffer,
    uniform: (
      data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    ) => GPUBuffer,
    precomputed?: {
      rankCounts?: GPUBuffer;
      suitCounts?: GPUBuffer;
      sharedBeliefs?: boolean;
      boardRankLow?: GPUBuffer;
      boardSuitLow?: GPUBuffer;
    },
  ): { rank: GPUBuffer; suit: GPUBuffer } | undefined {
    const interactionDim = this.manifest.architecture.boardInteractionDim;
    if (interactionDim <= 0) return undefined;
    if (
      !this.rankPairOneHotT ||
      !this.suitPairOneHotT ||
      !this.rankPairLowT ||
      !this.suitPairLowT
    ) {
      throw new Error("board interaction buffers are not initialized");
    }

    const rankMass = empty(batch * numPlayers * 91);
    const suitMass = empty(batch * numPlayers * 10);
    for (let player = 0; player < numPlayers; player += 1) {
      this.matVecBatch(
        this.rankPairOneHotT,
        beliefBuffer,
        this.dummyBias,
        rankMass,
        91,
        NUM_HANDS,
        batch,
        precomputed?.sharedBeliefs ? 0 : numPlayers * NUM_HANDS,
        numPlayers * 91,
        player * NUM_HANDS,
        player * 91,
        false,
        uniform,
      );
      this.matVecBatch(
        this.suitPairOneHotT,
        beliefBuffer,
        this.dummyBias,
        suitMass,
        10,
        NUM_HANDS,
        batch,
        precomputed?.sharedBeliefs ? 0 : numPlayers * NUM_HANDS,
        numPlayers * 10,
        player * NUM_HANDS,
        player * 10,
        false,
        uniform,
      );
    }

    const rankPairLow = empty(batch * numPlayers * interactionDim);
    const suitPairLow = empty(batch * numPlayers * interactionDim);
    for (let player = 0; player < numPlayers; player += 1) {
      this.matVecBatch(
        this.rankPairLowT,
        rankMass,
        this.dummyBias,
        rankPairLow,
        interactionDim,
        91,
        batch,
        numPlayers * 91,
        numPlayers * interactionDim,
        player * 91,
        player * interactionDim,
        false,
        uniform,
      );
      this.matVecBatch(
        this.suitPairLowT,
        suitMass,
        this.dummyBias,
        suitPairLow,
        interactionDim,
        10,
        batch,
        numPlayers * 10,
        numPlayers * interactionDim,
        player * 10,
        player * interactionDim,
        false,
        uniform,
      );
    }

    let rankCounts = precomputed?.rankCounts;
    let suitCounts = precomputed?.suitCounts;
    let boardRankLow = precomputed?.boardRankLow;
    let boardSuitLow = precomputed?.boardSuitLow;
    if ((!boardRankLow || !boardSuitLow) && (!rankCounts || !suitCounts)) {
      if (!boards) {
        throw new Error("board arrays are required when board counts are not precomputed");
      }
      const rankCountsCpu = new Float32Array(batch * 13);
      const suitCountsCpu = new Float32Array(batch * 4);
      for (let row = 0; row < boards.length; row += 1) {
        const board = boards[row]!;
        for (let i = 0; i < 5; i += 1) {
          const card = board[i] ?? -1;
          if (card >= 0) {
            const rankOffset = row * 13 + (card % 13);
            const suitOffset = row * 4 + Math.floor(card / 13);
            rankCountsCpu[rankOffset] = rankCountsCpu[rankOffset]! + 1;
            suitCountsCpu[suitOffset] = suitCountsCpu[suitOffset]! + 1;
          }
        }
      }
      rankCounts = storage(rankCountsCpu);
      suitCounts = storage(suitCountsCpu);
    }
    if (!boardRankLow) {
      boardRankLow = empty(batch * interactionDim);
      this.matVecBatch(
        this.tensor("board_rank_low.weight").buffer,
        rankCounts!,
        this.dummyBias,
        boardRankLow,
        interactionDim,
        13,
        batch,
        13,
        interactionDim,
        0,
        0,
        false,
        uniform,
      );
    }
    if (!boardSuitLow) {
      boardSuitLow = empty(batch * interactionDim);
      this.matVecBatch(
        this.tensor("board_suit_low.weight").buffer,
        suitCounts!,
        this.dummyBias,
        boardSuitLow,
        interactionDim,
        4,
        batch,
        4,
        interactionDim,
        0,
        0,
        false,
        uniform,
      );
    }

    const rankGated = empty(batch * numPlayers * interactionDim);
    this.playerBoardHadamard(
      rankPairLow,
      boardRankLow,
      rankGated,
      batch,
      numPlayers,
      interactionDim,
      uniform,
    );
    const suitGated = empty(batch * numPlayers * interactionDim);
    this.playerBoardHadamard(
      suitPairLow,
      boardSuitLow,
      suitGated,
      batch,
      numPlayers,
      interactionDim,
      uniform,
    );

    const rank = empty(batch * hidden);
    this.matVecBatch(
      this.tensor("rank_board_interaction_out.weight").buffer,
      rankGated,
      this.dummyBias,
      rank,
      hidden,
      numPlayers * interactionDim,
      batch,
      numPlayers * interactionDim,
      hidden,
      0,
      0,
      false,
      uniform,
    );
    const suit = empty(batch * hidden);
    this.matVecBatch(
      this.tensor("suit_board_interaction_out.weight").buffer,
      suitGated,
      this.dummyBias,
      suit,
      hidden,
      numPlayers * interactionDim,
      batch,
      numPlayers * interactionDim,
      hidden,
      0,
      0,
      false,
      uniform,
    );
    return { rank, suit };
  }

  private contextFeaturesForBatch(
    encodedFeatures: readonly ReturnType<typeof encodeBetterFeatures>[],
    batch: number,
    hiddenDim: number,
    empty: (elements: number) => GPUBuffer,
    storage: (
      data: Float32Array<ArrayBufferLike> | Uint32Array<ArrayBufferLike>,
    ) => GPUBuffer,
    uniform: (
      data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    ) => GPUBuffer,
  ): GPUBuffer {
    const context = new Float32Array(batch * 11);
    for (let i = 0; i < batch; i += 1) {
      context.set(encodedFeatures[i]!.context, i * 11);
    }
    return this.leakyReluBlockBatch(
      "context_encoder",
      storage(context),
      batch,
      11,
      hiddenDim,
      hiddenDim,
      empty,
      uniform,
    );
  }

  private baseEmbeddingForBatch(
    encodedFeatures: readonly ReturnType<typeof encodeBetterFeatures>[],
    batch: number,
    hiddenDim: number,
    storage: (
      data: Float32Array<ArrayBufferLike> | Uint32Array<ArrayBufferLike>,
    ) => GPUBuffer,
  ): GPUBuffer {
    const base = new Float32Array(batch * hiddenDim);
    for (let i = 0; i < batch; i += 1) {
      const features = encodedFeatures[i]!;
      base.set(this.baseEmbedding(features.street, features.board), i * hiddenDim);
    }
    return storage(base);
  }

  private unorderedPairIndex(first: number, second: number, numItems: number): number {
    const lo = Math.min(first, second);
    const hi = Math.max(first, second);
    return lo * numItems - Math.floor((lo * (lo - 1)) / 2) + (hi - lo);
  }

  private buildPairOneHotT(
    numItems: number,
    numPairs: number,
  ): Float32Array<ArrayBuffer> {
    const combos = handCombos();
    const out = new Float32Array(numPairs * NUM_HANDS);
    for (let hand = 0; hand < combos.length; hand += 1) {
      const [c0, c1] = combos[hand]!;
      const first = numItems === 13 ? c0 % 13 : Math.floor(c0 / 13);
      const second = numItems === 13 ? c1 % 13 : Math.floor(c1 / 13);
      const pair = this.unorderedPairIndex(first, second, numItems);
      out[pair * NUM_HANDS + hand] = 1;
    }
    return out;
  }

  private transposeTensor(name: string): Float32Array<ArrayBuffer> {
    const tensor = this.tensor(name);
    if (tensor.shape.length !== 2) {
      throw new Error(`model tensor ${name} is not rank-2`);
    }
    const [rows, cols] = tensor.shape as [number, number];
    const out = new Float32Array(rows * cols);
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        out[col * rows + row] = tensor.data[row * cols + col]!;
      }
    }
    return out;
  }

  private baseEmbedding(
    street: number,
    board: readonly number[],
  ): Float32Array<ArrayBuffer> {
    const hidden = this.manifest.architecture.hiddenDim;
    const streetEmbedding = this.tensor("street_embedding.weight");
    const rank = this.tensor("rank_embedding.weight");
    const suit = this.tensor("suit_embedding.weight");
    const out = new Float32Array(hidden);
    const streetIndex = Math.max(0, Math.min(4, Math.trunc(street)));
    for (let d = 0; d < hidden; d += 1) {
      out[d] = streetEmbedding.data[streetIndex * hidden + d]!;
    }
    for (let i = 0; i < 5; i += 1) {
      const card = board[i] ?? -1;
      const rankIndex = card >= 0 ? card % 13 : 13;
      const suitIndex = card >= 0 ? Math.floor(card / 13) : 4;
      for (let d = 0; d < hidden; d += 1) {
        out[d] =
          out[d]! +
          rank.data[rankIndex * hidden + d]! +
          suit.data[suitIndex * hidden + d]!;
      }
    }
    return out;
  }

  private headBatch(
    head: "hand_value_head" | "policy_head",
    input: GPUBuffer,
    batch: number,
    hiddenDim: number,
    ffnDim: number,
    numLayers: number,
    outDim: number,
    alpha: number,
    empty: (elements: number) => GPUBuffer,
    uniform: (
      data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    ) => GPUBuffer,
  ): GPUBuffer {
    const directOutputPrefix = `${head}.${numLayers}`;
    if (this.tensors.has(`${directOutputPrefix}.linear_out.weight`)) {
      let headInput = input;
      for (let i = 0; i < numLayers; i += 1) {
        const out = this.leakyReluResidualBlockBatch(
          `${head}.${i}.inner`,
          headInput,
          batch,
          hiddenDim,
          ffnDim,
          alpha,
          empty,
          uniform,
        );
        headInput = out;
      }
      return this.outputProjectionBatch(
        directOutputPrefix,
        headInput,
        batch,
        hiddenDim,
        outDim,
        empty,
        uniform,
      );
    }

    let headInput = input;
    for (let i = 0; i < numLayers - 1; i += 1) {
      const out = this.leakyReluResidualBlockBatch(
        `${head}.${i}.inner`,
        headInput,
        batch,
        hiddenDim,
        ffnDim,
        alpha,
        empty,
        uniform,
      );
      headInput = out;
    }
    return this.leakyReluBlockBatch(
      `${head}.${numLayers - 1}`,
      headInput,
      batch,
      hiddenDim,
      ffnDim,
      outDim,
      empty,
      uniform,
    );
  }

  private outputProjectionBatch(
    prefix: string,
    input: GPUBuffer,
    batch: number,
    inDim: number,
    outDim: number,
    empty: (elements: number) => GPUBuffer,
    uniform: (
      data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    ) => GPUBuffer,
  ): GPUBuffer {
    const normed = empty(batch * inDim);
    this.rmsNormBatch(prefix, input, normed, batch, inDim, inDim, inDim, uniform);
    const out = empty(batch * outDim);
    this.matVecBatch(
      this.tensor(`${prefix}.linear_out.weight`).buffer,
      normed,
      this.tensor(`${prefix}.linear_out.bias`).buffer,
      out,
      outDim,
      inDim,
      batch,
      inDim,
      outDim,
      0,
      0,
      true,
      uniform,
    );
    return out;
  }

  private leakyReluBlockBatch(
    prefix: string,
    input: GPUBuffer,
    batch: number,
    inDim: number,
    hiddenDim: number,
    outDim: number,
    empty: (elements: number) => GPUBuffer,
    uniform: (
      data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    ) => GPUBuffer,
  ): GPUBuffer {
    const normed = empty(batch * inDim);
    this.rmsNormBatch(prefix, input, normed, batch, inDim, inDim, inDim, uniform);
    const linear = empty(batch * hiddenDim);
    const linearInName = `${prefix}.linear_in.weight`;
    this.matVecBatch(
      this.tensor(linearInName).buffer,
      normed,
      this.dummyBias,
      linear,
      hiddenDim,
      inDim,
      batch,
      inDim,
      hiddenDim,
      0,
      0,
      false,
      uniform,
    );
    const out = empty(batch * outDim);
    const linearOutName = `${prefix}.linear_out.weight`;
    this.leakyReluMatVecBatch(
      this.tensor(linearOutName).buffer,
      linear,
      this.tensor(`${prefix}.linear_out.bias`).buffer,
      out,
      outDim,
      hiddenDim,
      batch,
      hiddenDim,
      outDim,
      true,
      uniform,
    );
    return out;
  }

  private leakyReluResidualBlockBatch(
    prefix: string,
    input: GPUBuffer,
    batch: number,
    inDim: number,
    hiddenDim: number,
    alpha: number,
    empty: (elements: number) => GPUBuffer,
    uniform: (
      data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    ) => GPUBuffer,
  ): GPUBuffer {
    const normed = empty(batch * inDim);
    this.rmsNormBatch(prefix, input, normed, batch, inDim, inDim, inDim, uniform);
    const linear = empty(batch * hiddenDim);
    const linearInName = `${prefix}.linear_in.weight`;
    this.matVecBatch(
      this.tensor(linearInName).buffer,
      normed,
      this.dummyBias,
      linear,
      hiddenDim,
      inDim,
      batch,
      inDim,
      hiddenDim,
      0,
      0,
      false,
      uniform,
    );
    const out = empty(batch * inDim);
    const linearOutName = `${prefix}.linear_out.weight`;
    this.leakyReluResidualMatVecBatch(
      this.tensor(linearOutName).buffer,
      linear,
      this.tensor(`${prefix}.linear_out.bias`).buffer,
      input,
      out,
      inDim,
      hiddenDim,
      batch,
      hiddenDim,
      inDim,
      true,
      alpha,
      uniform,
    );
    return out;
  }

  private rmsNorm(
    prefix: string,
    input: GPUBuffer,
    output: GPUBuffer,
    dim: number,
    uniform: (
      data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    ) => GPUBuffer,
  ): void {
    this.submit(this.rmsNormPipeline, 1, [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: this.tensor(`${prefix}.norm.weight`).buffer } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: uniform(new Uint32Array([dim, 0, 0, 0])) } },
    ]);
  }

  private rmsNormBatch(
    prefix: string,
    input: GPUBuffer,
    output: GPUBuffer,
    batch: number,
    dim: number,
    inputStride: number,
    outputStride: number,
    uniform: (
      data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    ) => GPUBuffer,
  ): void {
    this.submit(this.rmsNormBatchPipeline, batch, [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: this.tensor(`${prefix}.norm.weight`).buffer } },
      { binding: 2, resource: { buffer: output } },
      {
        binding: 3,
        resource: {
          buffer: uniform(new Uint32Array([dim, batch, inputStride, outputStride])),
        },
      },
    ]);
  }

  private matVec(
    matrix: GPUBuffer,
    input: GPUBuffer,
    bias: GPUBuffer,
    output: GPUBuffer,
    rows: number,
    cols: number,
    inputOffset: number,
    outputOffset: number,
    biasPresent: boolean,
    uniform: (
      data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    ) => GPUBuffer,
  ): void {
    this.submit(this.matVecPipeline, rows, [
      { binding: 0, resource: { buffer: matrix } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: bias } },
      { binding: 3, resource: { buffer: output } },
      {
        binding: 4,
        resource: {
          buffer: uniform(
            new Uint32Array([
              rows,
              cols,
              inputOffset,
              outputOffset,
              biasPresent ? 1 : 0,
              0,
              0,
              0,
            ]),
          ),
        },
      },
    ]);
  }

  private matVecBatch(
    matrix: GPUBuffer,
    input: GPUBuffer,
    bias: GPUBuffer,
    output: GPUBuffer,
    rows: number,
    cols: number,
    batch: number,
    inputStride: number,
    outputStride: number,
    inputOffset: number,
    outputOffset: number,
    biasPresent: boolean,
    uniform: (
      data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    ) => GPUBuffer,
  ): void {
    const useBatch2LinearIn =
      rows === 1024 &&
      cols === 512 &&
      inputStride === 512 &&
      outputStride === 1024 &&
      inputOffset === 0 &&
      outputOffset === 0 &&
      !biasPresent &&
      this.matVecBatchExactRowsCols512Batch2SubgroupPipeline;
    const useHandEmbeddingBatch2 =
      rows === 512 &&
      cols === NUM_HANDS &&
      inputStride === 2 * NUM_HANDS &&
      outputStride === 1024 &&
      !biasPresent &&
      this.matVecBatchExactRowsCols1326Batch2SubgroupPipeline;
    const useBatch2Cols1024 =
      rows === 1024 &&
      cols === 1024 &&
      inputStride === 1024 &&
      outputStride === 1024 &&
      inputOffset === 0 &&
      outputOffset === 0 &&
      !biasPresent &&
      this.matVecBatchExactRowsCols1024Batch2SubgroupPipeline;
    const useSmallCols =
      cols <= 128 &&
      inputOffset === 0 &&
      outputOffset === 0 &&
      !biasPresent;
    const pipeline = useSmallCols
      ? this.matVecBatchSmallColsPipeline
      : useBatch2LinearIn
      ? this.matVecBatchExactRowsCols512Batch2SubgroupPipeline!
      : useHandEmbeddingBatch2
        ? this.matVecBatchExactRowsCols1326Batch2SubgroupPipeline!
      : useBatch2Cols1024
        ? this.matVecBatchExactRowsCols1024Batch2SubgroupPipeline!
      : rows % BATCH_ROW_BLOCK === 0
        ? cols === 512
          ? (this.matVecBatchExactRowsCols512SubgroupPipeline ??
            this.matVecBatchExactRowsCols512Pipeline)
          : cols === 1024
            ? this.matVecBatchExactRowsCols1024Pipeline
            : this.matVecBatchExactRowsPipeline
        : this.matVecBatchPipeline;
    this.submit2d(
      pipeline,
      useSmallCols ? Math.ceil((rows * batch) / 64) : this.batchRowGroups(rows),
      useBatch2LinearIn || useHandEmbeddingBatch2 || useBatch2Cols1024
        ? Math.ceil(batch / 2)
        : useSmallCols
          ? 1
          : batch,
      [
      { binding: 0, resource: { buffer: matrix } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: bias } },
      { binding: 3, resource: { buffer: output } },
      {
        binding: 4,
        resource: {
          buffer: uniform(
            new Uint32Array([
              rows,
              cols,
              batch,
              inputStride,
              outputStride,
              inputOffset,
              outputOffset,
              biasPresent ? 1 : 0,
            ]),
          ),
        },
      },
      ],
    );
  }

  private leakyReluMatVecBatch(
    matrix: GPUBuffer,
    input: GPUBuffer,
    bias: GPUBuffer,
    output: GPUBuffer,
    rows: number,
    cols: number,
    batch: number,
    inputStride: number,
    outputStride: number,
    biasPresent: boolean,
    uniform: (
      data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    ) => GPUBuffer,
  ): void {
    const useBatch2ValueHead =
      rows === 2 * NUM_HANDS &&
      cols === 1024 &&
      inputStride === 1024 &&
      outputStride === 2 * NUM_HANDS &&
      biasPresent &&
      this.leakyReluMatVecBatchExactRowsCols1024Batch2SubgroupPipeline;
    const pipeline = useBatch2ValueHead
      ? this.leakyReluMatVecBatchExactRowsCols1024Batch2SubgroupPipeline!
      : rows % BATCH_ROW_BLOCK === 0
        ? cols === 512
          ? (this.leakyReluMatVecBatchExactRowsCols512SubgroupPipeline ??
            this.leakyReluMatVecBatchExactRowsCols512Pipeline)
          : cols === 1024
            ? (this.leakyReluMatVecBatchExactRowsCols1024SubgroupPipeline ??
              this.leakyReluMatVecBatchExactRowsCols1024Pipeline)
            : this.leakyReluMatVecBatchExactRowsPipeline
        : this.leakyReluMatVecBatchPipeline;
    this.submit2d(
      pipeline,
      this.batchRowGroups(rows),
      useBatch2ValueHead ? Math.ceil(batch / 2) : batch,
      [
      { binding: 0, resource: { buffer: matrix } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: bias } },
      { binding: 3, resource: { buffer: output } },
      {
        binding: 4,
        resource: {
          buffer: uniform(
            new Uint32Array([
              rows,
              cols,
              batch,
              inputStride,
              outputStride,
              biasPresent ? 1 : 0,
              0,
              0,
            ]),
          ),
        },
      },
      ],
    );
  }

  private leakyReluResidualMatVecBatch(
    matrix: GPUBuffer,
    input: GPUBuffer,
    bias: GPUBuffer,
    residual: GPUBuffer,
    output: GPUBuffer,
    rows: number,
    cols: number,
    batch: number,
    inputStride: number,
    outputStride: number,
    biasPresent: boolean,
    alpha: number,
    uniform: (
      data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    ) => GPUBuffer,
  ): void {
    const useBatch2Residual =
      rows === 512 &&
      cols === 1024 &&
      inputStride === 1024 &&
      outputStride === 512 &&
      biasPresent &&
      this.leakyReluResidualMatVecBatchExactRowsCols1024Batch2SubgroupPipeline;
    const pipeline = useBatch2Residual
      ? this.leakyReluResidualMatVecBatchExactRowsCols1024Batch2SubgroupPipeline!
      : rows % BATCH_ROW_BLOCK === 0
        ? cols === 512
          ? this.leakyReluResidualMatVecBatchExactRowsCols512Pipeline
          : cols === 1024
            ? (this.leakyReluResidualMatVecBatchExactRowsCols1024SubgroupPipeline ??
              this.leakyReluResidualMatVecBatchExactRowsCols1024Pipeline)
            : this.leakyReluResidualMatVecBatchExactRowsPipeline
        : this.leakyReluResidualMatVecBatchPipeline;
    const alphaBits = new Uint32Array(new Float32Array([alpha]).buffer)[0]!;
    this.submit2d(
      pipeline,
      this.batchRowGroups(rows),
      useBatch2Residual ? Math.ceil(batch / 2) : batch,
      [
        { binding: 0, resource: { buffer: matrix } },
        { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: bias } },
        { binding: 3, resource: { buffer: output } },
        { binding: 4, resource: { buffer: residual } },
        {
          binding: 5,
          resource: {
            buffer: uniform(
              new Uint32Array([
                rows,
                cols,
                batch,
                inputStride,
                outputStride,
                biasPresent ? 1 : 0,
                alphaBits,
                0,
              ]),
            ),
          },
        },
      ],
    );
  }

  private batchRowGroups(rows: number): number {
    return Math.ceil(rows / BATCH_ROW_BLOCK);
  }

  private add3(
    a: GPUBuffer,
    b: GPUBuffer,
    c: GPUBuffer,
    output: GPUBuffer,
    dim: number,
    uniform: (
      data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    ) => GPUBuffer,
  ): void {
    this.submit(this.add3Pipeline, Math.ceil(dim / 64), [
      { binding: 0, resource: { buffer: a } },
      { binding: 1, resource: { buffer: b } },
      { binding: 2, resource: { buffer: c } },
      { binding: 3, resource: { buffer: output } },
      { binding: 4, resource: { buffer: uniform(new Uint32Array([dim, 0, 0, 0])) } },
    ]);
  }

  private repeatRows(
    input: GPUBuffer,
    batch: number,
    dim: number,
    empty: (elements: number) => GPUBuffer,
    uniform: (
      data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    ) => GPUBuffer,
  ): GPUBuffer {
    const output = empty(batch * dim);
    this.submit(this.repeatRowsPipeline, Math.ceil((batch * dim) / 64), [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: output } },
      { binding: 2, resource: { buffer: uniform(new Uint32Array([dim, batch, 0, 0])) } },
    ]);
    return output;
  }

  private playerBoardHadamard(
    pairLow: GPUBuffer,
    boardLow: GPUBuffer,
    output: GPUBuffer,
    batch: number,
    numPlayers: number,
    interactionDim: number,
    uniform: (
      data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    ) => GPUBuffer,
  ): void {
    const elements = batch * numPlayers * interactionDim;
    this.submit(this.playerBoardHadamardPipeline, Math.ceil(elements / 64), [
      { binding: 0, resource: { buffer: pairLow } },
      { binding: 1, resource: { buffer: boardLow } },
      { binding: 2, resource: { buffer: output } },
      {
        binding: 3,
        resource: {
          buffer: uniform(
            new Uint32Array([elements, interactionDim, numPlayers, 0]),
          ),
        },
      },
    ]);
  }

  private stateFeatures(
    states: GPUBuffer,
    context: GPUBuffer,
    baseEmbedding: GPUBuffer,
    rankCounts: GPUBuffer,
    suitCounts: GPUBuffer,
    batch: number,
    hidden: number,
    uniform: (
      data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    ) => GPUBuffer,
  ): void {
    this.submit(this.stateFeaturePipeline, batch, [
      { binding: 0, resource: { buffer: states } },
      { binding: 1, resource: { buffer: this.tensor("street_embedding.weight").buffer } },
      { binding: 2, resource: { buffer: this.tensor("rank_embedding.weight").buffer } },
      { binding: 3, resource: { buffer: this.tensor("suit_embedding.weight").buffer } },
      { binding: 4, resource: { buffer: context } },
      { binding: 5, resource: { buffer: baseEmbedding } },
      { binding: 6, resource: { buffer: rankCounts } },
      { binding: 7, resource: { buffer: suitCounts } },
      { binding: 8, resource: { buffer: uniform(new Uint32Array([hidden, batch, 0, 0])) } },
    ]);
  }

  private scaledResidualAdd(
    residual: GPUBuffer,
    inner: GPUBuffer,
    output: GPUBuffer,
    dim: number,
    alpha: number,
    uniform: (
      data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    ) => GPUBuffer,
  ): void {
    const alphaBits = new Uint32Array(new Float32Array([alpha]).buffer)[0]!;
    this.submit(this.scaledResidualAddPipeline, Math.ceil(dim / 64), [
      { binding: 0, resource: { buffer: residual } },
      { binding: 1, resource: { buffer: inner } },
      { binding: 2, resource: { buffer: output } },
      {
        binding: 3,
        resource: { buffer: uniform(new Uint32Array([dim, 0, alphaBits, 0])) },
      },
    ]);
  }

  private zeroSumBatch(
    values: GPUBuffer,
    beliefs: GPUBuffer,
    batch: number,
    uniform: (
      data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    ) => GPUBuffer,
    beliefStride = 2 * NUM_HANDS,
  ): void {
    this.submit(this.zeroSumBatchPipeline, batch, [
      { binding: 0, resource: { buffer: values } },
      { binding: 1, resource: { buffer: beliefs } },
      {
        binding: 2,
        resource: { buffer: uniform(new Uint32Array([NUM_HANDS, batch, beliefStride, 0])) },
      },
    ]);
  }

  private acquireStorage(elements: number): TempBuffer {
    const key = Math.max(4, Math.ceil((elements * 4) / 4) * 4);
    const buffer = this.storagePool.get(key)?.pop();
    return {
      buffer:
        buffer ??
        this.device.createBuffer({
          size: key,
          usage:
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_SRC |
            GPUBufferUsage.COPY_DST,
        }),
      key,
      kind: "storage",
    };
  }

  private acquireUniform(byteLength: number): TempBuffer {
    const key = Math.max(16, Math.ceil(byteLength / 16) * 16);
    const buffer = this.uniformPool.get(key)?.pop();
    return {
      buffer:
        buffer ??
        this.device.createBuffer({
          size: key,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        }),
      key,
      kind: "uniform",
    };
  }

  private releaseTemp(temp: TempBuffer): void {
    const pool = temp.kind === "storage" ? this.storagePool : this.uniformPool;
    const buffers = pool.get(temp.key);
    if (buffers) {
      buffers.push(temp.buffer);
    } else {
      pool.set(temp.key, [temp.buffer]);
    }
  }

  private tensor(name: string): GpuTensor {
    const tensor = this.tensors.get(name);
    if (!tensor) {
      throw new Error(`model tensor ${name} is missing`);
    }
    return tensor;
  }

  private pipeline(source: string, label: string): GPUComputePipeline {
    return this.device.createComputePipeline({
      label,
      layout: "auto",
      compute: {
        module: this.device.createShaderModule({ label: `${label}.wgsl`, code: source }),
        entryPoint: "main",
      },
    });
  }

  private submit(
    pipeline: GPUComputePipeline,
    workgroups: number,
    entries: GPUBindGroupEntry[],
  ): void {
    const encoder = this.recordingEncoder ?? this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries,
      }),
    );
    pass.dispatchWorkgroups(workgroups);
    pass.end();
    if (!this.recordingEncoder) {
      this.device.queue.submit([encoder.finish()]);
    }
  }

  private submit2d(
    pipeline: GPUComputePipeline,
    workgroupsX: number,
    workgroupsY: number,
    entries: GPUBindGroupEntry[],
  ): void {
    const encoder = this.recordingEncoder ?? this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries,
      }),
    );
    pass.dispatchWorkgroups(workgroupsX, workgroupsY);
    pass.end();
    if (!this.recordingEncoder) {
      this.device.queue.submit([encoder.finish()]);
    }
  }
}
