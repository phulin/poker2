import {
  FILL_EXACT_PAIR_MASS_WGSL,
  MAT_VEC_BATCH_EXACT_BELIEF_LINEAR_IN_512_BATCH2_SUBGROUP_WGSL,
  PLAYER_BOARD_HADAMARD_WGSL,
} from "./modelKernels/beliefFeatures.js";
import {
  LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_WGSL,
  LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_BATCH2_SUBGROUP_WGSL,
  LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_SUBGROUP_WGSL,
  LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_COLS_512_WGSL,
  LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_COLS_512_SUBGROUP_WGSL,
  LEAKY_RELU_MAT_VEC_BATCH_EXACT_ROWS_WGSL,
  LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_WGSL,
  LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_BATCH2_SUBGROUP_WGSL,
  LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_SUBGROUP_WGSL,
  LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_EXACT_ROWS_COLS_512_WGSL,
  LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_EXACT_ROWS_WGSL,
  LEAKY_RELU_RESIDUAL_MAT_VEC_BATCH_WGSL,
  MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_WGSL,
  MAT_VEC_BATCH_EXACT_ROWS_COLS_1024_BATCH2_SUBGROUP_WGSL,
  MAT_VEC_BATCH_EXACT_ROWS_COLS_1326_BATCH2_SUBGROUP_WGSL,
  MAT_VEC_BATCH_EXACT_ROWS_COLS_1326_BATCH4_SUBGROUP_WGSL,
  MAT_VEC_BATCH_EXACT_ROWS_COLS_512_BATCH3_SUBGROUP_WGSL,
  MAT_VEC_BATCH_EXACT_ROWS_COLS_512_BATCH4_SUBGROUP_WGSL,
  MAT_VEC_BATCH_EXACT_ROWS_COLS_512_BATCH2_SUBGROUP_WGSL,
  MAT_VEC_BATCH_EXACT_ROWS_COLS_512_SUBGROUP_WGSL,
  MAT_VEC_BATCH_EXACT_ROWS_COLS_512_WGSL,
  MAT_VEC_BATCH_EXACT_ROWS_WGSL,
} from "./modelKernels/matVecGenerated.js";
import {
  LEAKY_RELU_MAT_VEC_BATCH_WGSL,
  MAT_VEC_BATCH_SMALL_COLS_WGSL,
  MAT_VEC_BATCH_WGSL,
  MAT_VEC_WGSL,
} from "./modelKernels/matVec.js";
import {
  RMS_NORM_BELIEF_EXACT_WGSL,
  RMS_NORM_BELIEF_EXACT_HALF_WGSL,
  RMS_NORM_BATCH_WGSL,
  RMS_NORM_BATCH_SMALL_WGSL,
  RMS_NORM_WGSL,
} from "./modelKernels/norm.js";
import {
  ADD3_WGSL,
  REPEAT_ROWS_WGSL,
  SCALED_RESIDUAL_ADD_WGSL,
  ZERO_SUM_BATCH_WGSL,
} from "./modelKernels/pointwise.js";
import { POKER_ENCODE_FEATURES_WGSL } from "./pokerStateKernels/modelFeatures.js";
import {
  makeStorageBuffer,
  readFloatBuffer,
} from "./gpuBuffers.js";
import { createComputePipeline, dispatchCompute } from "./gpuPipeline.js";
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
import type { AllInTableProvider } from "./allInTables.js";
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
  valuePreChance?: readonly boolean[];
}

const BATCH_ROW_BLOCK = 4;

export interface ExactBelief {
  player: number;
  hand: number;
}

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
  beliefPhaseShift?: GPUBuffer;
  beliefPhaseKeys?: Uint32Array<ArrayBuffer>;
  boardRankLow?: GPUBuffer;
  boardSuitLow?: GPUBuffer;
  dispose: () => void;
}

interface ExactBeliefPhaseShiftBuffers {
  embeddingRows: GPUBuffer;
  contributionRows: GPUBuffer;
}

export class BetterFfnWebGpuModel {
  readonly device: GPUDevice;
  readonly manifest: BetterFfnManifest;
  readonly actionLabels: string[];
  allInTableProvider: AllInTableProvider | undefined;
  private readonly tensors = new Map<string, GpuTensor>();
  private readonly dummyBias: GPUBuffer;
  private readonly handEmbeddingCpu: Float32Array<ArrayBuffer>;
  private readonly handEmbeddingT: GPUBuffer;
  private readonly beliefProjLinearInHalf0: GPUBuffer;
  private readonly beliefProjLinearInHalf1: GPUBuffer;
  private readonly beliefProjExactContribution0: GPUBuffer;
  private readonly beliefProjExactContribution1: GPUBuffer;
  private readonly rankPairOneHotT?: GPUBuffer;
  private readonly suitPairOneHotT?: GPUBuffer;
  private readonly rankPairLowT?: GPUBuffer;
  private readonly suitPairLowT?: GPUBuffer;
  private readonly rankPairLowByHandT?: GPUBuffer;
  private readonly suitPairLowByHandT?: GPUBuffer;
  private readonly matVecPipeline: GPUComputePipeline;
  private readonly matVecBatchPipeline: GPUComputePipeline;
  private readonly matVecBatchExactBeliefLinearInPipeline:
    | GPUComputePipeline
    | undefined;
  private readonly matVecBatchSmallColsPipeline: GPUComputePipeline;
  private readonly matVecBatchExactRowsPipeline: GPUComputePipeline;
  private readonly matVecBatchExactRowsCols512Pipeline: GPUComputePipeline;
  private readonly matVecBatchExactRowsCols512Batch2SubgroupPipeline:
    | GPUComputePipeline
    | undefined;
  private readonly matVecBatchExactRowsCols512Batch3SubgroupPipeline:
    | GPUComputePipeline
    | undefined;
  private readonly matVecBatchExactRowsCols512Batch4SubgroupPipeline:
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
  private readonly matVecBatchExactRowsCols1326Batch4SubgroupPipeline:
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
  private readonly fillExactPairMassPipeline: GPUComputePipeline;
  private readonly rmsNormPipeline: GPUComputePipeline;
  private readonly rmsNormBeliefExactPipeline: GPUComputePipeline;
  private readonly rmsNormBeliefExactHalfPipeline: GPUComputePipeline;
  private readonly rmsNormBatchPipeline: GPUComputePipeline;
  private readonly rmsNormBatchSmallPipeline: GPUComputePipeline;
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
    this.handEmbeddingCpu = this.buildHandEmbeddingT();
    this.handEmbeddingT = makeStorageBuffer(device, this.handEmbeddingCpu);
    this.beliefProjLinearInHalf0 = makeStorageBuffer(
      device,
      this.sliceColumns("belief_proj.linear_in.weight", 0, this.manifest.architecture.hiddenDim),
    );
    this.beliefProjLinearInHalf1 = makeStorageBuffer(
      device,
      this.sliceColumns(
        "belief_proj.linear_in.weight",
        this.manifest.architecture.hiddenDim,
        this.manifest.architecture.hiddenDim,
      ),
    );
    this.beliefProjExactContribution0 = makeStorageBuffer(
      device,
      this.buildBeliefExactContribution(0),
    );
    this.beliefProjExactContribution1 = makeStorageBuffer(
      device,
      this.buildBeliefExactContribution(1),
    );
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
      this.rankPairLowByHandT = makeStorageBuffer(
        device,
        this.buildPairLowByHandT("rank_pair_low_embedding.weight", 13),
      );
      this.suitPairLowByHandT = makeStorageBuffer(
        device,
        this.buildPairLowByHandT("suit_pair_low_embedding.weight", 4),
      );
    }
    this.matVecPipeline = this.pipeline(MAT_VEC_WGSL, "better-ffn-mat-vec");
    this.matVecBatchPipeline = this.pipeline(
      MAT_VEC_BATCH_WGSL,
      "better-ffn-mat-vec-batch",
    );
    this.matVecBatchExactBeliefLinearInPipeline = device.features.has(
      "subgroups" as GPUFeatureName,
    )
      ? this.pipeline(
        MAT_VEC_BATCH_EXACT_BELIEF_LINEAR_IN_512_BATCH2_SUBGROUP_WGSL,
        "better-ffn-mat-vec-exact-belief-linear-in-512-batch2-subgroup",
      )
      : undefined;
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
    this.matVecBatchExactRowsCols512Batch3SubgroupPipeline =
      device.features.has("subgroups" as GPUFeatureName)
        ? this.pipeline(
            MAT_VEC_BATCH_EXACT_ROWS_COLS_512_BATCH3_SUBGROUP_WGSL,
            "better-ffn-mat-vec-batch-exact-rows-cols-512-batch3-subgroup",
          )
        : undefined;
    this.matVecBatchExactRowsCols512Batch4SubgroupPipeline =
      device.features.has("subgroups" as GPUFeatureName)
        ? this.pipeline(
            MAT_VEC_BATCH_EXACT_ROWS_COLS_512_BATCH4_SUBGROUP_WGSL,
            "better-ffn-mat-vec-batch-exact-rows-cols-512-batch4-subgroup",
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
    this.matVecBatchExactRowsCols1326Batch4SubgroupPipeline =
      device.features.has("subgroups" as GPUFeatureName)
        ? this.pipeline(
            MAT_VEC_BATCH_EXACT_ROWS_COLS_1326_BATCH4_SUBGROUP_WGSL,
            "better-ffn-mat-vec-batch-exact-rows-cols-1326-batch4-subgroup",
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
    this.fillExactPairMassPipeline = this.pipeline(
      FILL_EXACT_PAIR_MASS_WGSL,
      "better-ffn-fill-exact-pair-mass",
    );
    this.rmsNormPipeline = this.pipeline(
      RMS_NORM_WGSL,
      "better-ffn-rms-norm",
    );
    this.rmsNormBeliefExactPipeline = this.pipeline(
      RMS_NORM_BELIEF_EXACT_WGSL,
      "better-ffn-rms-norm-belief-exact",
    );
    this.rmsNormBeliefExactHalfPipeline = this.pipeline(
      RMS_NORM_BELIEF_EXACT_HALF_WGSL,
      "better-ffn-rms-norm-belief-exact-half",
    );
    this.rmsNormBatchPipeline = this.pipeline(
      RMS_NORM_BATCH_WGSL,
      "better-ffn-rms-norm-batch",
    );
    this.rmsNormBatchSmallPipeline = this.pipeline(
      RMS_NORM_BATCH_SMALL_WGSL,
      "better-ffn-rms-norm-batch-small",
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
    valuePreChance?: readonly boolean[],
  ): Promise<Float32Array<ArrayBufferLike>> {
    return (
      await this.predictBatch(envs, beliefs, {
        includePolicy: false,
        ...(valuePreChance ? { valuePreChance } : {}),
      })
    ).handValues;
  }

  async predictBatchHandValuesGpu(
    envs: readonly PublicHunlEnv[],
    beliefs: Float32Array<ArrayBufferLike> | GPUBuffer,
    prepared?: PreparedBatchFeatures,
    beforeSubmit?: (encoder: GPUCommandEncoder, handValues: GPUBuffer) => void,
    exactBelief?: ExactBelief,
    beforeEncode?: (encoder: GPUCommandEncoder) => void,
  ): Promise<GpuHandValuePrediction> {
    const prediction = this.enqueuePredictBatch(envs, beliefs, {
      includePolicy: false,
    }, prepared, beforeSubmit, exactBelief, beforeEncode);
    return {
      buffer: prediction.handValuesBuffer,
      batch: prediction.batch,
      valuesPerSample: 2 * NUM_HANDS,
      dispose: prediction.dispose,
    };
  }

  prepareBatchFeatures(
    envs: readonly PublicHunlEnv[],
    valuePreChance?: readonly boolean[],
  ): PreparedBatchFeatures {
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
      const encodedFeatures = envs.map((env, i) =>
        encodeBetterFeatures(env, valuePreChance?.[i] ?? false),
      );
      const beliefPhaseKeys = this.beliefPhaseKeysForFeatures(encodedFeatures);
      const contextDim = this.manifest.architecture.contextDim ?? 11;
      const context = new Float32Array(batch * contextDim);
      const base = new Float32Array(batch * hiddenDim);
      const beliefPhaseShift = this.beliefPhaseShiftForFeatures(encodedFeatures);
      for (let i = 0; i < batch; i += 1) {
        const features = encodedFeatures[i]!;
        context.set(
          contextDim === 15 ? features.valueContext : features.context,
          i * contextDim,
        );
        base.set(this.baseEmbedding(features.street, features.board), i * hiddenDim);
      }
      const contextFeatures = this.leakyReluBlockBatch(
        "context_encoder",
        storage(context),
        batch,
        contextDim,
        hiddenDim,
        hiddenDim,
        empty,
        uniform,
      );
      const baseEmbedding = storage(base);
      const beliefPhaseShiftBuffer = beliefPhaseShift
        ? storage(beliefPhaseShift)
        : undefined;

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
      if (beliefPhaseShiftBuffer) detach(beliefPhaseShiftBuffer);
      if (boardRankLow) detach(boardRankLow);
      if (boardSuitLow) detach(boardSuitLow);
      let disposed = false;
      return {
        batch,
        baseEmbedding,
        contextFeatures,
        ...(beliefPhaseShiftBuffer ? { beliefPhaseShift: beliefPhaseShiftBuffer } : {}),
        beliefPhaseKeys,
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
      } else if (options.includePolicy && this.manifest.architecture.splitPolicyValue) {
        prediction.policyLogits = this.predictSplitPolicyCpu(envs, beliefs);
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
    exactBelief?: ExactBelief,
    beforeEncode?: (encoder: GPUCommandEncoder) => void,
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
      beforeEncode?.(encoder);
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
      const encodedFeatures = prepared
        ? undefined
        : envs.map((env, i) =>
            encodeBetterFeatures(env, options.valuePreChance?.[i] ?? false),
          );
      const computedBeliefPhaseKeys = encodedFeatures
        ? this.beliefPhaseKeysForFeatures(encodedFeatures)
        : undefined;
      const computedBeliefPhaseShift = encodedFeatures
        ? this.beliefPhaseShiftForFeatures(encodedFeatures)
        : undefined;
      const beliefPhaseShift =
        prepared?.beliefPhaseShift ??
        (computedBeliefPhaseShift ? storage(computedBeliefPhaseShift) : undefined);
      const exactPhaseShift = exactBelief && beliefPhaseShift
        ? this.exactBeliefPhaseShiftForKeys(
            prepared?.beliefPhaseKeys ?? computedBeliefPhaseKeys,
            exactBelief,
          )
        : undefined;
      const exactPhaseShiftBuffers = exactPhaseShift
        ? {
            embeddingRows: storage(exactPhaseShift.embeddingRows),
            contributionRows: storage(exactPhaseShift.contributionRows),
          }
        : undefined;

      const perPlayerBelief = empty(batch * numPlayers * hiddenDim);
      for (let player = 0; player < numPlayers; player += 1) {
        if (exactBelief && exactBelief.player === player) continue;
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
      let beliefInput = perPlayerBelief;
      if (beliefPhaseShift) {
        const shiftedBelief = empty(batch * numPlayers * hiddenDim);
        this.add3(
          perPlayerBelief,
          beliefPhaseShift,
          storage(new Float32Array(batch * numPlayers * hiddenDim)),
          shiftedBelief,
          batch * numPlayers * hiddenDim,
          uniform,
        );
        beliefInput = shiftedBelief;
      }

      const beliefFeatures = this.leakyReluBlockBatch(
        "belief_proj",
        beliefInput,
        batch,
        numPlayers * hiddenDim,
        rangeHiddenDim === 0 ? ffnDim : numPlayers * rangeHiddenDim,
        hiddenDim,
        empty,
        uniform,
        exactBelief,
        exactPhaseShiftBuffers,
      );

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
        exactBelief,
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
      if (options.includePolicy && !this.manifest.architecture.splitPolicyValue) {
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
      const contextDim = this.manifest.architecture.contextDim ?? 11;
      const context = empty(batch * contextDim);
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
        contextDim,
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
        contextDim,
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
      if (options.includePolicy && !this.manifest.architecture.splitPolicyValue) {
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
    this.beliefProjLinearInHalf0.destroy();
    this.beliefProjLinearInHalf1.destroy();
    this.beliefProjExactContribution0.destroy();
    this.beliefProjExactContribution1.destroy();
    this.rankPairOneHotT?.destroy();
    this.suitPairOneHotT?.destroy();
    this.rankPairLowT?.destroy();
    this.suitPairLowT?.destroy();
    this.rankPairLowByHandT?.destroy();
    this.suitPairLowByHandT?.destroy();
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
    const contextDim = this.manifest.architecture.contextDim ?? 11;
    this.requireLinearBlock(tensors, "context_encoder", contextDim, hidden, hidden);
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
    if (this.manifest.architecture.splitPolicyValue) {
      this.validateSplitPolicyTensors(tensors, hidden, ffn, actions);
    } else {
      this.requireHead(
        tensors,
        "policy_head",
        this.manifest.architecture.numPolicyLayers,
        hidden,
        ffn,
        actions * NUM_HANDS,
      );
    }
  }

  private predictSplitPolicyCpu(
    envs: readonly PublicHunlEnv[],
    beliefs: Float32Array<ArrayBufferLike>,
  ): Float32Array<ArrayBuffer> {
    const batch = envs.length;
    const hidden = this.manifest.architecture.hiddenDim;
    const ffn = this.manifest.architecture.ffnDim;
    const actions = this.manifest.architecture.numActions;
    const policyRank = this.manifest.architecture.policyRank ?? 64;
    const biasRank = this.manifest.architecture.policyHandBiasRank ?? 32;
    const singleBeliefSize = 2 * NUM_HANDS;
    const batchedBeliefs =
      beliefs.length === batch * singleBeliefSize
        ? beliefs
        : (() => {
            const out = new Float32Array(batch * singleBeliefSize);
            for (let i = 0; i < batch; i += 1) out.set(beliefs, i * singleBeliefSize);
            return out;
          })();
    const handEmb = this.buildExactCardHandEmbeddingRows("policy_model.");
    const handEmbNorm = this.rmsNormRowsCpu(
      handEmb,
      NUM_HANDS,
      hidden,
      "policy_model.policy_hand_norm",
    );
    const handVec = this.outputProjectionRowsCpu(
      handEmbNorm,
      NUM_HANDS,
      hidden,
      policyRank,
      "policy_model.policy_hand_proj",
    );
    const handBias = this.outputProjectionRowsCpu(
      handEmbNorm,
      NUM_HANDS,
      hidden,
      biasRank,
      "policy_model.policy_hand_bias",
    );
    const out = new Float32Array(batch * NUM_HANDS * actions);
    const combos = handCombos();
    const alpha = 1 / Math.sqrt(
      this.manifest.architecture.numHiddenLayers +
        this.manifest.architecture.numValueLayers,
    );
    for (let b = 0; b < batch; b += 1) {
      const env = envs[b]!;
      const features = encodeBetterFeatures(env);
      const beliefBase = b * singleBeliefSize;
      const perPlayer = new Float32Array(2 * hidden);
      for (let player = 0; player < 2; player += 1) {
        for (let hand = 0; hand < NUM_HANDS; hand += 1) {
          const belief = batchedBeliefs[beliefBase + player * NUM_HANDS + hand]!;
          if (belief === 0) continue;
          const embBase = hand * hidden;
          const outBase = player * hidden;
          for (let d = 0; d < hidden; d += 1) {
            const idx = outBase + d;
            perPlayer[idx] = perPlayer[idx]! + belief * handEmb[embBase + d]!;
          }
        }
      }
      const beliefFeatures = this.leakyBlockCpu(
        perPlayer,
        2 * hidden,
        this.manifest.architecture.rangeHiddenDim === 0
          ? ffn
          : 2 * this.manifest.architecture.rangeHiddenDim,
        hidden,
        "policy_model.belief_proj",
      );
      const contextFeatures = this.leakyBlockCpu(
        features.policyContext,
        this.manifest.architecture.contextDim ?? 15,
        hidden,
        hidden,
        "policy_model.context_encoder",
      );
      let x = this.baseEmbeddingCpu("policy_model.", features.street, features.board);
      const interaction = this.boardInteractionCpu(
        "policy_model.",
        batchedBeliefs,
        beliefBase,
        features.board,
      );
      for (let d = 0; d < hidden; d += 1) {
        x[d] = x[d]! + contextFeatures[d]! + beliefFeatures[d]! + interaction[d]!;
      }
      for (let i = 0; i < this.manifest.architecture.numHiddenLayers; i += 1) {
        x = this.residualBlockCpu(
          x,
          hidden,
          ffn,
          `policy_model.trunk.${i}.inner`,
          alpha,
        );
      }
      let policyState = x;
      for (let i = 0; i < this.manifest.architecture.numPolicyLayers; i += 1) {
        policyState = this.residualBlockCpu(
          policyState,
          hidden,
          ffn,
          `policy_model.policy_tower.${i}.inner`,
          alpha,
        );
      }
      const actionEmb = this.outputProjectionCpu(
        policyState,
        hidden,
        actions * policyRank,
        "policy_model.policy_action_head",
      );
      const handGate = this.outputProjectionCpu(
        policyState,
        hidden,
        policyRank,
        "policy_model.policy_hand_gate",
      );
      for (let r = 0; r < policyRank; r += 1) handGate[r] = 1 + Math.tanh(handGate[r]!);
      for (let a = 0; a < actions; a += 1) {
        const actionBase = a * policyRank;
        for (let r = 0; r < policyRank; r += 1) {
          const idx = actionBase + r;
          actionEmb[idx] = actionEmb[idx]! * handGate[r]!;
        }
      }
      const dynamicCoeff = this.outputProjectionCpu(
        policyState,
        hidden,
        actions * 15,
        "policy_model.policy_dynamic_coeff",
      );
      const actionBias = this.outputProjectionCpu(
        policyState,
        hidden,
        actions,
        "policy_model.policy_action_bias",
      );
      const handBiasAction = this.outputProjectionCpu(
        policyState,
        hidden,
        actions * biasRank,
        "policy_model.policy_hand_bias_action",
      );
      const policyScale = this.tensor("policy_model.policy_factor_scale").data[0]!;
      const cardMass = this.cardMasses(batchedBeliefs, beliefBase);
      const actor = env.toAct;
      const opp = 1 - actor;
      const boardStats = this.boardStats(features.board);
      for (let hand = 0; hand < NUM_HANDS; hand += 1) {
        const [c0, c1] = combos[hand]!;
        const ownBelief = batchedBeliefs[beliefBase + actor * NUM_HANDS + hand]!;
        const oppBelief = batchedBeliefs[beliefBase + opp * NUM_HANDS + hand]!;
        const ownCardMass = cardMass[actor]!;
        const oppCardMass = cardMass[opp]!;
        const oppUnblocked = Math.max(
          0,
          1 - oppCardMass[c0]! - oppCardMass[c1]! + oppBelief,
        );
        const dyn = this.dynamicHandFeatures(
          ownBelief,
          ownCardMass,
          oppCardMass,
          oppUnblocked,
          c0,
          c1,
          boardStats,
        );
        for (let a = 0; a < actions; a += 1) {
          let logit = actionBias[a]!;
          for (let r = 0; r < policyRank; r += 1) {
            logit +=
              (handVec[hand * policyRank + r]! * actionEmb[a * policyRank + r]!) /
              Math.sqrt(policyRank);
          }
          for (let r = 0; r < biasRank; r += 1) {
            logit += handBias[hand * biasRank + r]! * handBiasAction[a * biasRank + r]!;
          }
          for (let f = 0; f < 15; f += 1) {
            logit += dyn[f]! * dynamicCoeff[a * 15 + f]!;
          }
          out[(b * NUM_HANDS + hand) * actions + a] = logit * policyScale;
        }
      }
    }
    return out;
  }

  private validateSplitPolicyTensors(
    tensors: TensorMap,
    hidden: number,
    ffn: number,
    actions: number,
  ): void {
    const prefix = "policy_model.";
    requireTensor(tensors, `${prefix}policy_factor_scale`, []);
    for (let i = 0; i < this.manifest.architecture.numPolicyLayers; i += 1) {
      this.requireLinearBlock(tensors, `${prefix}policy_tower.${i}.inner`, hidden, ffn, hidden);
    }
    const policyRank = this.manifest.architecture.policyRank ?? 64;
    const biasRank = this.manifest.architecture.policyHandBiasRank ?? 32;
    this.requireOutputProjection(tensors, `${prefix}policy_hand_proj`, hidden, policyRank);
    this.requireOutputProjection(tensors, `${prefix}policy_action_head`, hidden, actions * policyRank);
    this.requireOutputProjection(tensors, `${prefix}policy_hand_gate`, hidden, policyRank);
    this.requireOutputProjection(tensors, `${prefix}policy_dynamic_coeff`, hidden, actions * 15);
    this.requireOutputProjection(tensors, `${prefix}policy_action_bias`, hidden, actions);
    this.requireOutputProjection(tensors, `${prefix}policy_hand_bias`, hidden, biasRank);
    this.requireOutputProjection(tensors, `${prefix}policy_hand_bias_action`, hidden, actions * biasRank);
    requireTensor(tensors, `${prefix}policy_hand_norm.weight`, [hidden]);
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

  private rmsNormVectorCpu(input: Float32Array<ArrayBufferLike>, prefix: string): Float32Array<ArrayBuffer> {
    const weight = this.tensors.has(`${prefix}.norm.weight`)
      ? this.tensor(`${prefix}.norm.weight`).data
      : this.tensor(`${prefix}.weight`).data;
    let sq = 0;
    for (let i = 0; i < input.length; i += 1) sq += input[i]! * input[i]!;
    const inv = 1 / Math.sqrt(sq / input.length + 1e-5);
    const out = new Float32Array(input.length);
    for (let i = 0; i < input.length; i += 1) out[i] = input[i]! * inv * weight[i]!;
    return out;
  }

  private rmsNormRowsCpu(
    input: Float32Array<ArrayBufferLike>,
    rows: number,
    cols: number,
    prefix: string,
  ): Float32Array<ArrayBuffer> {
    const out = new Float32Array(rows * cols);
    for (let row = 0; row < rows; row += 1) {
      out.set(
        this.rmsNormVectorCpu(input.subarray(row * cols, (row + 1) * cols), prefix),
        row * cols,
      );
    }
    return out;
  }

  private linearCpu(
    input: Float32Array<ArrayBufferLike>,
    inDim: number,
    outDim: number,
    prefix: string,
    bias = true,
  ): Float32Array<ArrayBuffer> {
    const weight = this.tensor(`${prefix}.weight`).data;
    const biasData = bias ? this.tensor(`${prefix}.bias`).data : undefined;
    const out = new Float32Array(outDim);
    for (let row = 0; row < outDim; row += 1) {
      let sum = biasData?.[row] ?? 0;
      const weightBase = row * inDim;
      for (let col = 0; col < inDim; col += 1) {
        sum += weight[weightBase + col]! * input[col]!;
      }
      out[row] = sum;
    }
    return out;
  }

  private leakyBlockCpu(
    input: Float32Array<ArrayBufferLike>,
    inDim: number,
    hiddenDim: number,
    outDim: number,
    prefix: string,
  ): Float32Array<ArrayBuffer> {
    const normed = this.rmsNormVectorCpu(input, prefix);
    const linear = this.linearCpu(normed, inDim, hiddenDim, `${prefix}.linear_in`, false);
    for (let i = 0; i < linear.length; i += 1) {
      if (linear[i]! < 0) linear[i] = linear[i]! * 0.01;
    }
    return this.linearCpu(linear, hiddenDim, outDim, `${prefix}.linear_out`, true);
  }

  private residualBlockCpu(
    input: Float32Array<ArrayBufferLike>,
    inDim: number,
    hiddenDim: number,
    prefix: string,
    alpha: number,
  ): Float32Array<ArrayBuffer> {
    const inner = this.leakyBlockCpu(input, inDim, hiddenDim, inDim, prefix);
    const out = new Float32Array(inDim);
    for (let i = 0; i < inDim; i += 1) out[i] = input[i]! + alpha * inner[i]!;
    return out;
  }

  private outputProjectionCpu(
    input: Float32Array<ArrayBufferLike>,
    inDim: number,
    outDim: number,
    prefix: string,
  ): Float32Array<ArrayBuffer> {
    const normed = this.rmsNormVectorCpu(input, prefix);
    return this.linearCpu(normed, inDim, outDim, `${prefix}.linear_out`, true);
  }

  private outputProjectionRowsCpu(
    input: Float32Array<ArrayBufferLike>,
    rows: number,
    inDim: number,
    outDim: number,
    prefix: string,
  ): Float32Array<ArrayBuffer> {
    const out = new Float32Array(rows * outDim);
    for (let row = 0; row < rows; row += 1) {
      out.set(
        this.outputProjectionCpu(
          input.subarray(row * inDim, (row + 1) * inDim),
          inDim,
          outDim,
          prefix,
        ),
        row * outDim,
      );
    }
    return out;
  }

  private buildExactCardHandEmbeddingRows(prefix: string): Float32Array<ArrayBuffer> {
    const hidden = this.manifest.architecture.hiddenDim;
    const cards = this.tensor(`${prefix}card_embedding.weight`);
    const featureProj = this.tensor(`${prefix}hand_feature_proj.weight`);
    const combos = handCombos();
    const out = new Float32Array(NUM_HANDS * hidden);
    for (let hand = 0; hand < combos.length; hand += 1) {
      const [c0, c1] = combos[hand]!;
      const staticFeatures = this.handStaticFeatures(c0, c1);
      for (let d = 0; d < hidden; d += 1) {
        let value = cards.data[c0 * hidden + d]! + cards.data[c1 * hidden + d]!;
        const projBase = d * 8;
        for (let f = 0; f < 8; f += 1) value += featureProj.data[projBase + f]! * staticFeatures[f]!;
        out[hand * hidden + d] = value;
      }
    }
    return out;
  }

  private baseEmbeddingCpu(prefix: string, street: number, board: readonly number[]): Float32Array<ArrayBuffer> {
    const hidden = this.manifest.architecture.hiddenDim;
    const streetEmbedding = this.tensor(`${prefix}street_embedding.weight`).data;
    const rank = this.tensor(`${prefix}rank_embedding.weight`).data;
    const suit = this.tensor(`${prefix}suit_embedding.weight`).data;
    const out = new Float32Array(hidden);
    const streetIndex = Math.max(0, Math.min(4, Math.trunc(street)));
    for (let d = 0; d < hidden; d += 1) out[d] = streetEmbedding[streetIndex * hidden + d]!;
    for (let i = 0; i < 5; i += 1) {
      const card = board[i] ?? -1;
      const rankIndex = card >= 0 ? card % 13 : 13;
      const suitIndex = card >= 0 ? Math.floor(card / 13) : 4;
      for (let d = 0; d < hidden; d += 1) {
        out[d] = out[d]! + rank[rankIndex * hidden + d]! + suit[suitIndex * hidden + d]!;
      }
    }
    return out;
  }

  private boardStats(board: readonly number[]): {
    ranks: Float32Array<ArrayBuffer>;
    suits: Float32Array<ArrayBuffer>;
    blocked: Uint8Array<ArrayBuffer>;
  } {
    const ranks = new Float32Array(13);
    const suits = new Float32Array(4);
    const blocked = new Uint8Array(52);
    for (let i = 0; i < 5; i += 1) {
      const card = board[i] ?? -1;
      if (card < 0) continue;
      const rank = card % 13;
      const suit = Math.floor(card / 13);
      ranks[rank] = ranks[rank]! + 1;
      suits[suit] = suits[suit]! + 1;
      blocked[card] = 1;
    }
    return { ranks, suits, blocked };
  }

  private boardInteractionCpu(
    prefix: string,
    beliefs: Float32Array<ArrayBufferLike>,
    beliefBase: number,
    board: readonly number[],
  ): Float32Array<ArrayBuffer> {
    const hidden = this.manifest.architecture.hiddenDim;
    const dim = this.manifest.architecture.boardInteractionDim;
    const out = new Float32Array(hidden);
    if (dim <= 0) return out;
    const stats = this.boardStats(board);
    const rankPair = this.tensor(`${prefix}rank_pair_low_embedding.weight`).data;
    const suitPair = this.tensor(`${prefix}suit_pair_low_embedding.weight`).data;
    const rankLow = this.linearCpu(stats.ranks, 13, dim, `${prefix}board_rank_low`, false);
    const suitLow = this.linearCpu(stats.suits, 4, dim, `${prefix}board_suit_low`, false);
    const rankGated = new Float32Array(2 * dim);
    const suitGated = new Float32Array(2 * dim);
    const combos = handCombos();
    for (let player = 0; player < 2; player += 1) {
      const rankMass = new Float32Array(91);
      const suitMass = new Float32Array(10);
      for (let hand = 0; hand < NUM_HANDS; hand += 1) {
        const belief = beliefs[beliefBase + player * NUM_HANDS + hand]!;
        if (belief === 0) continue;
        const [c0, c1] = combos[hand]!;
        const rankIdx = this.unorderedPairIndex(c0 % 13, c1 % 13, 13);
        const suitIdx = this.unorderedPairIndex(Math.floor(c0 / 13), Math.floor(c1 / 13), 4);
        rankMass[rankIdx] = rankMass[rankIdx]! + belief;
        suitMass[suitIdx] = suitMass[suitIdx]! + belief;
      }
      for (let d = 0; d < dim; d += 1) {
        let r = 0;
        for (let pair = 0; pair < 91; pair += 1) r += rankMass[pair]! * rankPair[pair * dim + d]!;
        rankGated[player * dim + d] = r * rankLow[d]!;
        let s = 0;
        for (let pair = 0; pair < 10; pair += 1) s += suitMass[pair]! * suitPair[pair * dim + d]!;
        suitGated[player * dim + d] = s * suitLow[d]!;
      }
    }
    const rankOut = this.linearCpu(rankGated, 2 * dim, hidden, `${prefix}rank_board_interaction_out`, false);
    const suitOut = this.linearCpu(suitGated, 2 * dim, hidden, `${prefix}suit_board_interaction_out`, false);
    for (let d = 0; d < hidden; d += 1) out[d] = rankOut[d]! + suitOut[d]!;
    return out;
  }

  private cardMasses(
    beliefs: Float32Array<ArrayBufferLike>,
    beliefBase: number,
  ): [Float32Array<ArrayBuffer>, Float32Array<ArrayBuffer>] {
    const combos = handCombos();
    const out: [Float32Array<ArrayBuffer>, Float32Array<ArrayBuffer>] = [
      new Float32Array(52),
      new Float32Array(52),
    ];
    for (let player = 0; player < 2; player += 1) {
      for (let hand = 0; hand < NUM_HANDS; hand += 1) {
        const belief = beliefs[beliefBase + player * NUM_HANDS + hand]!;
        const [c0, c1] = combos[hand]!;
        const playerOut = out[player]!;
        playerOut[c0] = playerOut[c0]! + belief;
        playerOut[c1] = playerOut[c1]! + belief;
      }
    }
    return out;
  }

  private dynamicHandFeatures(
    ownBelief: number,
    ownCardMass: Float32Array<ArrayBufferLike>,
    oppCardMass: Float32Array<ArrayBufferLike>,
    oppUnblocked: number,
    c0: number,
    c1: number,
    stats: ReturnType<BetterFfnWebGpuModel["boardStats"]>,
  ): Float32Array<ArrayBuffer> {
    const r0 = c0 % 13;
    const r1 = c1 % 13;
    const s0 = Math.floor(c0 / 13);
    const s1 = Math.floor(c1 / 13);
    const rankA = stats.ranks[r0]!;
    const rankB = stats.ranks[r1]!;
    const suitA = stats.suits[s0]!;
    const suitB = stats.suits[s1]!;
    return new Float32Array([
      ownBelief,
      Math.log(Math.max(ownBelief, 1e-8)),
      oppUnblocked,
      ownCardMass[c0]!,
      ownCardMass[c1]!,
      oppCardMass[c0]!,
      oppCardMass[c1]!,
      rankA / 4,
      rankB / 4,
      (rankA + rankB) / 4,
      (rankA * rankB) / 16,
      suitA / 5,
      suitB / 5,
      (suitA + suitB) / 5,
      stats.blocked[c0] || stats.blocked[c1] ? 1 : 0,
    ]);
  }

  private buildHandEmbeddingT(): Float32Array<ArrayBuffer> {
    const hidden = this.manifest.architecture.hiddenDim;
    if (this.tensors.has("card_embedding.weight") && this.tensors.has("hand_feature_proj.weight")) {
      return this.buildExactCardHandEmbeddingT("");
    }
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

  private buildExactCardHandEmbeddingT(prefix: string): Float32Array<ArrayBuffer> {
    const hidden = this.manifest.architecture.hiddenDim;
    const cards = this.tensor(`${prefix}card_embedding.weight`);
    const featureProj = this.tensor(`${prefix}hand_feature_proj.weight`);
    const combos = handCombos();
    const out = new Float32Array(hidden * NUM_HANDS);
    for (let hand = 0; hand < combos.length; hand += 1) {
      const [c0, c1] = combos[hand]!;
      const staticFeatures = this.handStaticFeatures(c0, c1);
      for (let d = 0; d < hidden; d += 1) {
        let value = cards.data[c0 * hidden + d]! + cards.data[c1 * hidden + d]!;
        const projBase = d * 8;
        for (let f = 0; f < 8; f += 1) {
          value += featureProj.data[projBase + f]! * staticFeatures[f]!;
        }
        out[d * NUM_HANDS + hand] = value;
      }
    }
    return out;
  }

  private handStaticFeatures(c0: number, c1: number): Float32Array<ArrayBuffer> {
    const r0 = c0 % 13;
    const r1 = c1 % 13;
    const s0 = Math.floor(c0 / 13);
    const s1 = Math.floor(c1 / 13);
    const hi = Math.max(r0, r1);
    const lo = Math.min(r0, r1);
    const gap = Math.max(hi - lo, 0);
    return new Float32Array([
      r0 === r1 ? 1 : 0,
      s0 === s1 ? 1 : 0,
      gap / 12,
      hi / 12,
      lo / 12,
      hi === 12 ? 1 : 0,
      lo >= 8 ? 1 : 0,
      gap <= 1 ? 1 : 0,
    ]);
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
    exactBelief?: ExactBelief,
  ): { rank: GPUBuffer; suit: GPUBuffer } | undefined {
    const interactionDim = this.manifest.architecture.boardInteractionDim;
    if (interactionDim <= 0) return undefined;
    if (
      !this.rankPairOneHotT ||
      !this.suitPairOneHotT ||
      !this.rankPairLowT ||
      !this.suitPairLowT ||
      !this.rankPairLowByHandT ||
      !this.suitPairLowByHandT
    ) {
      throw new Error("board interaction buffers are not initialized");
    }

    const rankMass = empty(batch * numPlayers * 91);
    const suitMass = empty(batch * numPlayers * 10);
    for (let player = 0; player < numPlayers; player += 1) {
      if (exactBelief?.player !== player) {
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
    }

    const rankPairLow = empty(batch * numPlayers * interactionDim);
    const suitPairLow = empty(batch * numPlayers * interactionDim);
    for (let player = 0; player < numPlayers; player += 1) {
      if (exactBelief?.player === player) {
        this.fillExactPairMass(
          this.rankPairLowByHandT,
          rankPairLow,
          interactionDim,
          batch,
          numPlayers,
          player,
          exactBelief.hand,
          uniform,
        );
        this.fillExactPairMass(
          this.suitPairLowByHandT,
          suitPairLow,
          interactionDim,
          batch,
          numPlayers,
          player,
          exactBelief.hand,
          uniform,
        );
      } else {
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
    const contextDim = this.manifest.architecture.contextDim ?? 11;
    const context = new Float32Array(batch * contextDim);
    for (let i = 0; i < batch; i += 1) {
      context.set(
        contextDim === 15 ? encodedFeatures[i]!.valueContext : encodedFeatures[i]!.context,
        i * contextDim,
      );
    }
    return this.leakyReluBlockBatch(
      "context_encoder",
      storage(context),
      batch,
      contextDim,
      hiddenDim,
      hiddenDim,
      empty,
      uniform,
    );
  }

  private beliefPhaseShiftForFeatures(
    encodedFeatures: readonly ReturnType<typeof encodeBetterFeatures>[],
  ): Float32Array<ArrayBuffer> | undefined {
    if (!this.tensors.has("belief_phase_shift.weight")) return undefined;
    const shift = this.tensor("belief_phase_shift.weight");
    const hiddenDim = this.manifest.architecture.hiddenDim;
    const numPlayers = this.manifest.architecture.numPlayers;
    const rowSize = numPlayers * hiddenDim;
    if (shift.shape[1] !== rowSize) {
      throw new Error(
        `belief_phase_shift width ${shift.shape[1]} does not match ${rowSize}`,
      );
    }
    const out = new Float32Array(encodedFeatures.length * rowSize);
    for (let row = 0; row < encodedFeatures.length; row += 1) {
      const features = encodedFeatures[row]!;
      const street = Math.max(0, Math.min(4, Math.trunc(features.street)));
      const phase = Math.max(0, Math.min(1, Math.round(features.valueContext[2]!)));
      const key = Math.max(0, Math.min(9, street * 2 + phase));
      out.set(shift.data.subarray(key * rowSize, (key + 1) * rowSize), row * rowSize);
    }
    return out;
  }

  private beliefPhaseKeysForFeatures(
    encodedFeatures: readonly ReturnType<typeof encodeBetterFeatures>[],
  ): Uint32Array<ArrayBuffer> {
    const out = new Uint32Array(encodedFeatures.length);
    for (let row = 0; row < encodedFeatures.length; row += 1) {
      const features = encodedFeatures[row]!;
      const street = Math.max(0, Math.min(4, Math.trunc(features.street)));
      const phase = Math.max(0, Math.min(1, Math.round(features.valueContext[2]!)));
      out[row] = Math.max(0, Math.min(9, street * 2 + phase));
    }
    return out;
  }

  private exactBeliefPhaseShiftForKeys(
    phaseKeys: Uint32Array<ArrayBufferLike> | undefined,
    exactBelief: ExactBelief,
  ): { embeddingRows: Float32Array<ArrayBuffer>; contributionRows: Float32Array<ArrayBuffer> } | undefined {
    if (!phaseKeys || !this.tensors.has("belief_phase_shift.weight")) return undefined;
    const batch = phaseKeys.length;
    const hidden = this.manifest.architecture.hiddenDim;
    const ffnDim = this.manifest.architecture.ffnDim;
    const playerOffset = exactBelief.player * hidden;
    const shift = this.tensor("belief_phase_shift.weight");
    const matrix = this.tensor("belief_proj.linear_in.weight");
    const norm = this.tensor("belief_proj.norm.weight");
    const embeddingRows = new Float32Array(batch * hidden);
    const contributionRows = new Float32Array(batch * ffnDim);
    const vectorsByKey = new Map<number, Float32Array<ArrayBuffer>>();
    const contributionsByKey = new Map<number, Float32Array<ArrayBuffer>>();

    const buildForKey = (phaseKey: number): {
      vector: Float32Array<ArrayBuffer>;
      contribution: Float32Array<ArrayBuffer>;
    } => {
      const key = Math.max(0, Math.min(9, phaseKey));
      let vector = vectorsByKey.get(key);
      let contribution = contributionsByKey.get(key);
      if (vector && contribution) return { vector, contribution };

      vector = new Float32Array(hidden);
      const shiftBase = key * 2 * hidden + playerOffset;
      for (let d = 0; d < hidden; d += 1) {
        vector[d] =
          this.handEmbeddingCpu[d * NUM_HANDS + exactBelief.hand]! +
          shift.data[shiftBase + d]!;
      }

      contribution = new Float32Array(ffnDim);
      for (let row = 0; row < ffnDim; row += 1) {
        const matrixBase = row * 2 * hidden + playerOffset;
        let sum = 0;
        for (let d = 0; d < hidden; d += 1) {
          sum +=
            matrix.data[matrixBase + d]! *
            vector[d]! *
            norm.data[playerOffset + d]!;
        }
        contribution[row] = sum;
      }
      vectorsByKey.set(key, vector);
      contributionsByKey.set(key, contribution);
      return { vector, contribution };
    };

    for (let row = 0; row < batch; row += 1) {
      const { vector, contribution } = buildForKey(phaseKeys[row]!);
      embeddingRows.set(vector, row * hidden);
      contributionRows.set(contribution, row * ffnDim);
    }
    return { embeddingRows, contributionRows };
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

  private buildPairLowByHandT(
    name: string,
    numItems: number,
  ): Float32Array<ArrayBuffer> {
    const tensor = this.tensor(name);
    if (tensor.shape.length !== 2) {
      throw new Error(`model tensor ${name} is not rank-2`);
    }
    const [, dim] = tensor.shape as [number, number];
    const combos = handCombos();
    const out = new Float32Array(dim * NUM_HANDS);
    for (let hand = 0; hand < combos.length; hand += 1) {
      const [c0, c1] = combos[hand]!;
      const first = numItems === 13 ? c0 % 13 : Math.floor(c0 / 13);
      const second = numItems === 13 ? c1 % 13 : Math.floor(c1 / 13);
      const pair = this.unorderedPairIndex(first, second, numItems);
      for (let d = 0; d < dim; d += 1) {
        out[d * NUM_HANDS + hand] = tensor.data[pair * dim + d]!;
      }
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

  private sliceColumns(
    name: string,
    startCol: number,
    colCount: number,
  ): Float32Array<ArrayBuffer> {
    const tensor = this.tensor(name);
    if (tensor.shape.length !== 2) {
      throw new Error(`model tensor ${name} is not rank-2`);
    }
    const [rows, cols] = tensor.shape as [number, number];
    const out = new Float32Array(rows * colCount);
    for (let row = 0; row < rows; row += 1) {
      const sourceBase = row * cols + startCol;
      const targetBase = row * colCount;
      for (let col = 0; col < colCount; col += 1) {
        out[targetBase + col] = tensor.data[sourceBase + col]!;
      }
    }
    return out;
  }

  private buildBeliefExactContribution(player: number): Float32Array<ArrayBuffer> {
    const hidden = this.manifest.architecture.hiddenDim;
    const ffnDim = this.manifest.architecture.ffnDim;
    const offset = player * hidden;
    const matrix = this.tensor("belief_proj.linear_in.weight");
    const norm = this.tensor("belief_proj.norm.weight");
    const out = new Float32Array(ffnDim * NUM_HANDS);
    for (let row = 0; row < ffnDim; row += 1) {
      const matrixBase = row * 2 * hidden + offset;
      const outBase = row * NUM_HANDS;
      for (let hand = 0; hand < NUM_HANDS; hand += 1) {
        let sum = 0.0;
        for (let dim = 0; dim < hidden; dim += 1) {
          sum +=
            matrix.data[matrixBase + dim]! *
            this.handEmbeddingCpu[dim * NUM_HANDS + hand]! *
            norm.data[offset + dim]!;
        }
        out[outBase + hand] = sum;
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
    exactBelief?: ExactBelief,
    exactPhaseShift?: ExactBeliefPhaseShiftBuffers,
  ): GPUBuffer {
    const modelHiddenDim = this.manifest.architecture.hiddenDim;
    const useExactBeliefLinearIn =
      exactBelief !== undefined &&
      this.matVecBatchExactBeliefLinearInPipeline !== undefined &&
      prefix === "belief_proj" &&
      inDim === 2 * modelHiddenDim &&
      hiddenDim === this.manifest.architecture.ffnDim &&
      outDim === modelHiddenDim;
    const linear = empty(batch * hiddenDim);
    if (useExactBeliefLinearIn) {
      const normedOpponent = empty(batch * modelHiddenDim);
      const invRms = empty(batch);
      this.rmsNormBeliefExactHalf(
        prefix,
        input,
        normedOpponent,
        invRms,
        batch,
        inDim,
        inDim,
        exactBelief,
        modelHiddenDim,
        exactPhaseShift?.embeddingRows,
        uniform,
      );
      this.matVecExactBeliefLinearIn(
        exactBelief.player === 0
          ? this.beliefProjLinearInHalf1
          : this.beliefProjLinearInHalf0,
        normedOpponent,
        invRms,
        exactBelief.player === 0
          ? this.beliefProjExactContribution0
          : this.beliefProjExactContribution1,
        linear,
        batch,
        hiddenDim,
        modelHiddenDim,
        exactBelief,
        exactPhaseShift?.contributionRows,
        uniform,
      );
    } else {
      const normed = empty(batch * inDim);
      if (exactBelief && prefix === "belief_proj" && inDim === 2 * modelHiddenDim) {
        this.rmsNormBeliefExact(
          prefix,
          input,
          normed,
          batch,
          inDim,
          inDim,
          inDim,
          exactBelief,
          modelHiddenDim,
          exactPhaseShift?.embeddingRows,
          uniform,
        );
      } else {
        this.rmsNormBatch(prefix, input, normed, batch, inDim, inDim, inDim, uniform);
      }
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
    }
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
    this.submit(dim <= 16 ? this.rmsNormBatchSmallPipeline : this.rmsNormBatchPipeline, batch, [
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

  private rmsNormBeliefExact(
    prefix: string,
    input: GPUBuffer,
    output: GPUBuffer,
    batch: number,
    dim: number,
    inputStride: number,
    outputStride: number,
    exactBelief: ExactBelief,
    hidden: number,
    exactEmbeddingRows: GPUBuffer | undefined,
    uniform: (
      data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    ) => GPUBuffer,
  ): void {
    this.submit(this.rmsNormBeliefExactPipeline, batch, [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: this.tensor(`${prefix}.norm.weight`).buffer } },
      { binding: 2, resource: { buffer: output } },
      {
        binding: 3,
        resource: {
          buffer: uniform(
            new Uint32Array([
              dim,
              batch,
              inputStride,
              outputStride,
              exactBelief.player * hidden,
              exactBelief.hand,
              NUM_HANDS,
              hidden,
              exactEmbeddingRows ? 1 : 0,
              0,
              0,
              0,
            ]),
          ),
        },
      },
      { binding: 4, resource: { buffer: exactEmbeddingRows ?? this.handEmbeddingT } },
    ]);
  }

  private rmsNormBeliefExactHalf(
    prefix: string,
    input: GPUBuffer,
    opponentOutput: GPUBuffer,
    invRmsOutput: GPUBuffer,
    batch: number,
    dim: number,
    inputStride: number,
    exactBelief: ExactBelief,
    hidden: number,
    exactEmbeddingRows: GPUBuffer | undefined,
    uniform: (
      data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    ) => GPUBuffer,
  ): void {
    this.submit(this.rmsNormBeliefExactHalfPipeline, batch, [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: this.tensor(`${prefix}.norm.weight`).buffer } },
      { binding: 2, resource: { buffer: opponentOutput } },
      { binding: 3, resource: { buffer: invRmsOutput } },
      { binding: 4, resource: { buffer: exactEmbeddingRows ?? this.handEmbeddingT } },
      {
        binding: 5,
        resource: {
          buffer: uniform(
            new Uint32Array([
              dim,
              batch,
              inputStride,
              exactBelief.player * hidden,
              exactBelief.hand,
              NUM_HANDS,
              hidden,
              exactEmbeddingRows ? 1 : 0,
              0,
              0,
              0,
              0,
            ]),
          ),
        },
      },
    ]);
  }

  private matVecExactBeliefLinearIn(
    matrix: GPUBuffer,
    input: GPUBuffer,
    invRms: GPUBuffer,
    contribution: GPUBuffer,
    output: GPUBuffer,
    batch: number,
    rows: number,
    cols: number,
    exactBelief: ExactBelief,
    exactContributionRows: GPUBuffer | undefined,
    uniform: (
      data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    ) => GPUBuffer,
  ): void {
    this.submit2d(
      this.matVecBatchExactBeliefLinearInPipeline!,
      this.batchRowGroups(rows),
      Math.ceil(batch / 2),
      [
        { binding: 0, resource: { buffer: matrix } },
        { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: invRms } },
        { binding: 3, resource: { buffer: exactContributionRows ?? contribution } },
        { binding: 4, resource: { buffer: output } },
        {
          binding: 5,
          resource: {
            buffer: uniform(
              new Uint32Array([
                rows,
                cols,
                batch,
                cols,
                rows,
                exactBelief.hand,
                NUM_HANDS,
                exactContributionRows ? 1 : 0,
              ]),
            ),
          },
        },
      ],
    );
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
    const useBatch3LinearIn =
      rows === 1024 &&
      cols === 512 &&
      inputStride === 512 &&
      outputStride === 1024 &&
      inputOffset === 0 &&
      outputOffset === 0 &&
      !biasPresent &&
      this.matVecBatchExactRowsCols512Batch3SubgroupPipeline;
    const useBatch4LinearIn =
      batch >= 4 &&
      rows === 1024 &&
      cols === 512 &&
      inputStride === 512 &&
      outputStride === 1024 &&
      inputOffset === 0 &&
      outputOffset === 0 &&
      !biasPresent &&
      this.matVecBatchExactRowsCols512Batch4SubgroupPipeline;
    const useHandEmbeddingBatch2 =
      rows === 512 &&
      cols === NUM_HANDS &&
      inputStride === 2 * NUM_HANDS &&
      outputStride === 1024 &&
      !biasPresent &&
      this.matVecBatchExactRowsCols1326Batch2SubgroupPipeline;
    const useHandEmbeddingBatch4 =
      batch >= 4 &&
      rows === 512 &&
      cols === NUM_HANDS &&
      inputStride === 2 * NUM_HANDS &&
      outputStride === 1024 &&
      !biasPresent &&
      this.matVecBatchExactRowsCols1326Batch4SubgroupPipeline;
    const useBatch2Cols1024 =
      rows === 1024 &&
      cols === 1024 &&
      inputStride === 1024 &&
      outputStride === 1024 &&
      inputOffset === 0 &&
      outputOffset === 0 &&
      !biasPresent &&
      this.matVecBatchExactRowsCols1024Batch2SubgroupPipeline;
    const useBatch2ValueHead =
      rows === 2 * NUM_HANDS &&
      cols === 512 &&
      inputStride === 512 &&
      outputStride === 2 * NUM_HANDS &&
      inputOffset === 0 &&
      outputOffset === 0 &&
      biasPresent &&
      this.matVecBatchExactRowsCols512Batch2SubgroupPipeline;
    const useSmallCols =
      cols <= 128 &&
      inputOffset === 0 &&
      outputOffset === 0 &&
      !biasPresent;
    const pipeline = useSmallCols
      ? this.matVecBatchSmallColsPipeline
      : useBatch4LinearIn
      ? this.matVecBatchExactRowsCols512Batch4SubgroupPipeline!
      : useBatch3LinearIn
      ? this.matVecBatchExactRowsCols512Batch3SubgroupPipeline!
      : useBatch2LinearIn || useBatch2ValueHead
      ? this.matVecBatchExactRowsCols512Batch2SubgroupPipeline!
      : useHandEmbeddingBatch4
        ? this.matVecBatchExactRowsCols1326Batch4SubgroupPipeline!
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
      useBatch4LinearIn || useHandEmbeddingBatch4
        ? Math.ceil(batch / 4)
        : useBatch3LinearIn
        ? Math.ceil(batch / 3)
        : useBatch2LinearIn ||
            useBatch2ValueHead ||
            useHandEmbeddingBatch2 ||
            useBatch2Cols1024
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

  private fillExactPairMass(
    pairOneHotT: GPUBuffer,
    output: GPUBuffer,
    rows: number,
    batch: number,
    numPlayers: number,
    player: number,
    hand: number,
    uniform: (
      data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    ) => GPUBuffer,
  ): void {
    this.submit(this.fillExactPairMassPipeline, Math.ceil((batch * rows) / 64), [
      { binding: 0, resource: { buffer: pairOneHotT } },
      { binding: 1, resource: { buffer: output } },
      {
        binding: 2,
        resource: {
          buffer: uniform(
            new Uint32Array([rows, batch, numPlayers, player, hand, 0, 0, 0]),
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
    contextDim: number,
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
      {
        binding: 8,
        resource: {
          buffer: uniform(
            new Float32Array([
              hidden,
              batch,
              contextDim,
              this.manifest.env.bb,
              this.manifest.env.maxStackBb ?? 400,
              0,
              0,
              0,
            ]),
          ),
        },
      },
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
    return createComputePipeline(this.device, source, label);
  }

  private submit(
    pipeline: GPUComputePipeline,
    workgroups: number,
    entries: GPUBindGroupEntry[],
  ): void {
    const encoder = this.recordingEncoder ?? this.device.createCommandEncoder();
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries,
    });
    dispatchCompute(encoder, pipeline, bindGroup, workgroups);
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
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries,
    });
    dispatchCompute(encoder, pipeline, bindGroup, workgroupsX, workgroupsY);
    if (!this.recordingEncoder) {
      this.device.queue.submit([encoder.finish()]);
    }
  }
}
