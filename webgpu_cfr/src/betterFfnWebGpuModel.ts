import {
  ADD3_WGSL,
  GELU_WGSL,
  LAYER_NORM_BATCH_WGSL,
  LAYER_NORM_WGSL,
  MAT_VEC_BATCH_WGSL,
  MAT_VEC_WGSL,
  SCALED_RESIDUAL_ADD_WGSL,
  SILU_MUL_WGSL,
} from "./modelKernels.js";
import {
  makeEmptyStorageBuffer,
  makeStorageBuffer,
  makeUniformBuffer,
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

interface PredictOptions {
  includePolicy?: boolean;
}

export interface BetterFfnPrediction {
  handValues: Float32Array<ArrayBufferLike>;
  policyLogits?: Float32Array<ArrayBufferLike>;
}

export class BetterFfnWebGpuModel {
  readonly device: GPUDevice;
  readonly manifest: BetterFfnManifest;
  readonly actionLabels: string[];
  private readonly tensors = new Map<string, GpuTensor>();
  private readonly dummyBias: GPUBuffer;
  private readonly handEmbeddingT: GPUBuffer;
  private readonly matVecPipeline: GPUComputePipeline;
  private readonly matVecBatchPipeline: GPUComputePipeline;
  private readonly layerNormPipeline: GPUComputePipeline;
  private readonly layerNormBatchPipeline: GPUComputePipeline;
  private readonly siluMulPipeline: GPUComputePipeline;
  private readonly geluPipeline: GPUComputePipeline;
  private readonly scaledResidualAddPipeline: GPUComputePipeline;
  private readonly add3Pipeline: GPUComputePipeline;

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
      this.tensors.set(name, {
        data: tensor.data,
        shape: tensor.manifest.shape,
        buffer: makeStorageBuffer(device, tensor.data),
      });
    }
    this.dummyBias = makeStorageBuffer(device, new Float32Array([0]));
    this.handEmbeddingT = makeStorageBuffer(device, this.buildHandEmbeddingT());
    this.matVecPipeline = this.pipeline(MAT_VEC_WGSL, "better-ffn-mat-vec");
    this.matVecBatchPipeline = this.pipeline(
      MAT_VEC_BATCH_WGSL,
      "better-ffn-mat-vec-batch",
    );
    this.layerNormPipeline = this.pipeline(
      LAYER_NORM_WGSL,
      "better-ffn-layer-norm",
    );
    this.layerNormBatchPipeline = this.pipeline(
      LAYER_NORM_BATCH_WGSL,
      "better-ffn-layer-norm-batch",
    );
    this.siluMulPipeline = this.pipeline(SILU_MUL_WGSL, "better-ffn-silu-mul");
    this.geluPipeline = this.pipeline(GELU_WGSL, "better-ffn-gelu");
    this.scaledResidualAddPipeline = this.pipeline(
      SCALED_RESIDUAL_ADD_WGSL,
      "better-ffn-scaled-residual-add",
    );
    this.add3Pipeline = this.pipeline(ADD3_WGSL, "better-ffn-add3");
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
    if (envs.length === 0) {
      return { handValues: new Float32Array(0) };
    }
    const hiddenDim = this.manifest.architecture.hiddenDim;
    const ffnDim = this.manifest.architecture.ffnDim;
    const rangeHiddenDim = this.manifest.architecture.rangeHiddenDim;
    const numPlayers = this.manifest.architecture.numPlayers;
    const numActions = this.manifest.architecture.numActions;
    const batch = envs.length;
    if (beliefs.length !== numPlayers * NUM_HANDS) {
      throw new Error(
        `belief vector has ${beliefs.length} entries, expected ${numPlayers * NUM_HANDS}`,
      );
    }

    const temps: GPUBuffer[] = [];
    const storage = (
      data: Float32Array<ArrayBufferLike> | Uint32Array<ArrayBufferLike>,
    ): GPUBuffer => {
      const buffer = makeStorageBuffer(this.device, data);
      temps.push(buffer);
      return buffer;
    };
    const empty = (elements: number): GPUBuffer => {
      const buffer = makeEmptyStorageBuffer(this.device, elements);
      temps.push(buffer);
      return buffer;
    };
    const uniform = (
      data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    ): GPUBuffer => {
      const buffer = makeUniformBuffer(this.device, data);
      temps.push(buffer);
      return buffer;
    };

    try {
      const batchedBeliefs = new Float32Array(batch * numPlayers * NUM_HANDS);
      for (let i = 0; i < batch; i += 1) {
        batchedBeliefs.set(beliefs, i * numPlayers * NUM_HANDS);
      }
      const beliefBuffer = storage(batchedBeliefs);
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

      const beliefFeatures = this.swigluBlockBatch(
        "belief_proj",
        perPlayerBelief,
        batch,
        numPlayers * hiddenDim,
        numPlayers * rangeHiddenDim,
        hiddenDim,
        empty,
        uniform,
      );

      const context = new Float32Array(batch * 11);
      const base = new Float32Array(batch * hiddenDim);
      for (let i = 0; i < batch; i += 1) {
        const features = encodeBetterFeatures(envs[i]!);
        context.set(features.context, i * 11);
        base.set(this.baseEmbedding(features.street, features.board), i * hiddenDim);
      }
      const contextFeatures = this.swigluBlockBatch(
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
      let x = empty(batch * hiddenDim);
      this.add3(
        baseEmbedding,
        contextFeatures,
        beliefFeatures,
        x,
        batch * hiddenDim,
        uniform,
      );

      const alpha = 1 / Math.sqrt(
        this.manifest.architecture.numHiddenLayers +
          this.manifest.architecture.numValueLayers,
      );
      for (let i = 0; i < this.manifest.architecture.numHiddenLayers; i += 1) {
        const inner = this.swigluBlockBatch(
          `trunk.${i}.inner`,
          x,
          batch,
          hiddenDim,
          ffnDim,
          hiddenDim,
          empty,
          uniform,
        );
        const out = empty(batch * hiddenDim);
        this.scaledResidualAdd(x, inner, out, batch * hiddenDim, alpha, uniform);
        x = out;
      }

      let valueInput = x;
      for (let i = 0; i < this.manifest.architecture.numValueLayers - 1; i += 1) {
        const inner = this.swigluBlockBatch(
          `hand_value_head.${i}.inner`,
          valueInput,
          batch,
          hiddenDim,
          ffnDim,
          hiddenDim,
          empty,
          uniform,
        );
        const out = empty(batch * hiddenDim);
        this.scaledResidualAdd(
          valueInput,
          inner,
          out,
          batch * hiddenDim,
          alpha,
          uniform,
        );
        valueInput = out;
      }
      const valueRawBuffer = this.geluBlockBatch(
        `hand_value_head.${this.manifest.architecture.numValueLayers - 1}`,
        valueInput,
        batch,
        hiddenDim,
        ffnDim,
        numPlayers * NUM_HANDS,
        empty,
        uniform,
      );

      let policyBuffer: GPUBuffer | undefined;
      if (options.includePolicy) {
        let policyInput = x;
        const policyAlpha = alpha;
        for (
          let i = 0;
          i < this.manifest.architecture.numPolicyLayers - 1;
          i += 1
        ) {
          const inner = this.swigluBlockBatch(
            `policy_head.${i}.inner`,
            policyInput,
            batch,
            hiddenDim,
            ffnDim,
            hiddenDim,
            empty,
            uniform,
          );
          const out = empty(batch * hiddenDim);
          this.scaledResidualAdd(
            policyInput,
            inner,
            out,
            batch * hiddenDim,
            policyAlpha,
            uniform,
          );
          policyInput = out;
        }
        policyBuffer = this.geluBlockBatch(
          `policy_head.${this.manifest.architecture.numPolicyLayers - 1}`,
          policyInput,
          batch,
          hiddenDim,
          ffnDim,
          numActions * NUM_HANDS,
          empty,
          uniform,
        );
      }

      const handValues = await readFloatBuffer(
        this.device,
        valueRawBuffer,
        batch * numPlayers * NUM_HANDS,
      );
      this.applyZeroSumBatchIfNeeded(handValues, beliefs, batch);

      const prediction: BetterFfnPrediction = { handValues };
      if (policyBuffer) {
        prediction.policyLogits = await readFloatBuffer(
          this.device,
          policyBuffer,
          batch * numActions * NUM_HANDS,
        );
      }
      return prediction;
    } finally {
      for (const buffer of temps) {
        buffer.destroy();
      }
    }
  }

  dispose(): void {
    for (const tensor of this.tensors.values()) {
      tensor.buffer.destroy();
    }
    this.dummyBias.destroy();
    this.handEmbeddingT.destroy();
  }

  private validateRequiredTensors(tensors: TensorMap): void {
    const hidden = this.manifest.architecture.hiddenDim;
    const ffn = this.manifest.architecture.ffnDim;
    const rangeHidden = this.manifest.architecture.rangeHiddenDim;
    const actions = this.manifest.architecture.numActions;
    requireTensor(tensors, "street_embedding.weight", [5, hidden]);
    requireTensor(tensors, "rank_embedding.weight", [14, hidden]);
    requireTensor(tensors, "suit_embedding.weight", [5, hidden]);
    this.requireSwiglu(tensors, "belief_proj", 2 * hidden, 2 * rangeHidden, hidden);
    this.requireSwiglu(tensors, "context_encoder", 11, hidden, hidden);
    for (let i = 0; i < this.manifest.architecture.numHiddenLayers; i += 1) {
      this.requireSwiglu(tensors, `trunk.${i}.inner`, hidden, ffn, hidden);
    }
    for (let i = 0; i < this.manifest.architecture.numValueLayers - 1; i += 1) {
      this.requireSwiglu(tensors, `hand_value_head.${i}.inner`, hidden, ffn, hidden);
    }
    this.requireGelu(
      tensors,
      `hand_value_head.${this.manifest.architecture.numValueLayers - 1}`,
      hidden,
      ffn,
      2 * NUM_HANDS,
    );
    for (let i = 0; i < this.manifest.architecture.numPolicyLayers - 1; i += 1) {
      this.requireSwiglu(tensors, `policy_head.${i}.inner`, hidden, ffn, hidden);
    }
    this.requireGelu(
      tensors,
      `policy_head.${this.manifest.architecture.numPolicyLayers - 1}`,
      hidden,
      ffn,
      actions * NUM_HANDS,
    );
  }

  private requireSwiglu(
    tensors: TensorMap,
    prefix: string,
    inDim: number,
    hiddenDim: number,
    outDim: number,
  ): void {
    requireTensor(tensors, `${prefix}.norm.weight`, [inDim]);
    requireTensor(tensors, `${prefix}.norm.bias`, [inDim]);
    requireTensor(tensors, `${prefix}.swiglu.gate.weight`, [hiddenDim, inDim]);
    requireTensor(tensors, `${prefix}.swiglu.up.weight`, [hiddenDim, inDim]);
    requireTensor(tensors, `${prefix}.swiglu.down.weight`, [outDim, hiddenDim]);
  }

  private requireGelu(
    tensors: TensorMap,
    prefix: string,
    inDim: number,
    hiddenDim: number,
    outDim: number,
  ): void {
    requireTensor(tensors, `${prefix}.norm.weight`, [inDim]);
    requireTensor(tensors, `${prefix}.norm.bias`, [inDim]);
    requireTensor(tensors, `${prefix}.linear_in.weight`, [hiddenDim, inDim]);
    requireTensor(tensors, `${prefix}.linear_out.weight`, [outDim, hiddenDim]);
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

  private swigluBlockBatch(
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
    this.layerNormBatch(prefix, input, normed, batch, inDim, inDim, inDim, uniform);
    const gate = empty(batch * hiddenDim);
    const up = empty(batch * hiddenDim);
    this.matVecBatch(
      this.tensor(`${prefix}.swiglu.gate.weight`).buffer,
      normed,
      this.dummyBias,
      gate,
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
    this.matVecBatch(
      this.tensor(`${prefix}.swiglu.up.weight`).buffer,
      normed,
      this.dummyBias,
      up,
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
    const gated = empty(batch * hiddenDim);
    this.siluMul(gate, up, gated, batch * hiddenDim, uniform);
    const out = empty(batch * outDim);
    this.matVecBatch(
      this.tensor(`${prefix}.swiglu.down.weight`).buffer,
      gated,
      this.dummyBias,
      out,
      outDim,
      hiddenDim,
      batch,
      hiddenDim,
      outDim,
      0,
      0,
      false,
      uniform,
    );
    return out;
  }

  private geluBlockBatch(
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
    this.layerNormBatch(prefix, input, normed, batch, inDim, inDim, inDim, uniform);
    const linear = empty(batch * hiddenDim);
    this.matVecBatch(
      this.tensor(`${prefix}.linear_in.weight`).buffer,
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
    const activated = empty(batch * hiddenDim);
    this.gelu(linear, activated, batch * hiddenDim, uniform);
    const out = empty(batch * outDim);
    this.matVecBatch(
      this.tensor(`${prefix}.linear_out.weight`).buffer,
      activated,
      this.tensor(`${prefix}.linear_out.bias`).buffer,
      out,
      outDim,
      hiddenDim,
      batch,
      hiddenDim,
      outDim,
      0,
      0,
      true,
      uniform,
    );
    return out;
  }

  private swigluBlock(
    prefix: string,
    input: GPUBuffer,
    inDim: number,
    hiddenDim: number,
    outDim: number,
    empty: (elements: number) => GPUBuffer,
    uniform: (
      data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    ) => GPUBuffer,
  ): GPUBuffer {
    const normed = empty(inDim);
    this.layerNorm(prefix, input, normed, inDim, uniform);
    const gate = empty(hiddenDim);
    const up = empty(hiddenDim);
    this.matVec(
      this.tensor(`${prefix}.swiglu.gate.weight`).buffer,
      normed,
      this.dummyBias,
      gate,
      hiddenDim,
      inDim,
      0,
      0,
      false,
      uniform,
    );
    this.matVec(
      this.tensor(`${prefix}.swiglu.up.weight`).buffer,
      normed,
      this.dummyBias,
      up,
      hiddenDim,
      inDim,
      0,
      0,
      false,
      uniform,
    );
    const gated = empty(hiddenDim);
    this.siluMul(gate, up, gated, hiddenDim, uniform);
    const out = empty(outDim);
    this.matVec(
      this.tensor(`${prefix}.swiglu.down.weight`).buffer,
      gated,
      this.dummyBias,
      out,
      outDim,
      hiddenDim,
      0,
      0,
      false,
      uniform,
    );
    return out;
  }

  private geluBlock(
    prefix: string,
    input: GPUBuffer,
    inDim: number,
    hiddenDim: number,
    outDim: number,
    empty: (elements: number) => GPUBuffer,
    uniform: (
      data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    ) => GPUBuffer,
  ): GPUBuffer {
    const normed = empty(inDim);
    this.layerNorm(prefix, input, normed, inDim, uniform);
    const linear = empty(hiddenDim);
    this.matVec(
      this.tensor(`${prefix}.linear_in.weight`).buffer,
      normed,
      this.dummyBias,
      linear,
      hiddenDim,
      inDim,
      0,
      0,
      false,
      uniform,
    );
    const activated = empty(hiddenDim);
    this.gelu(linear, activated, hiddenDim, uniform);
    const out = empty(outDim);
    this.matVec(
      this.tensor(`${prefix}.linear_out.weight`).buffer,
      activated,
      this.tensor(`${prefix}.linear_out.bias`).buffer,
      out,
      outDim,
      hiddenDim,
      0,
      0,
      true,
      uniform,
    );
    return out;
  }

  private layerNorm(
    prefix: string,
    input: GPUBuffer,
    output: GPUBuffer,
    dim: number,
    uniform: (
      data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    ) => GPUBuffer,
  ): void {
    this.submit(this.layerNormPipeline, 1, [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: this.tensor(`${prefix}.norm.weight`).buffer } },
      { binding: 2, resource: { buffer: this.tensor(`${prefix}.norm.bias`).buffer } },
      { binding: 3, resource: { buffer: output } },
      { binding: 4, resource: { buffer: uniform(new Uint32Array([dim, 0, 0, 0])) } },
    ]);
  }

  private layerNormBatch(
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
    this.submit(this.layerNormBatchPipeline, batch, [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: this.tensor(`${prefix}.norm.weight`).buffer } },
      { binding: 2, resource: { buffer: this.tensor(`${prefix}.norm.bias`).buffer } },
      { binding: 3, resource: { buffer: output } },
      {
        binding: 4,
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
    this.submit2d(this.matVecBatchPipeline, rows, batch, [
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
    ]);
  }

  private siluMul(
    gate: GPUBuffer,
    up: GPUBuffer,
    output: GPUBuffer,
    dim: number,
    uniform: (
      data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    ) => GPUBuffer,
  ): void {
    this.submit(this.siluMulPipeline, Math.ceil(dim / 64), [
      { binding: 0, resource: { buffer: gate } },
      { binding: 1, resource: { buffer: up } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: uniform(new Uint32Array([dim, 0, 0, 0])) } },
    ]);
  }

  private gelu(
    input: GPUBuffer,
    output: GPUBuffer,
    dim: number,
    uniform: (
      data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    ) => GPUBuffer,
  ): void {
    this.submit(this.geluPipeline, Math.ceil(dim / 64), [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: output } },
      { binding: 2, resource: { buffer: uniform(new Uint32Array([dim, 0, 0, 0])) } },
    ]);
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

  private applyZeroSumIfNeeded(
    handValues: Float32Array<ArrayBufferLike>,
    beliefs: Float32Array<ArrayBufferLike>,
  ): void {
    if (!this.manifest.architecture.enforceZeroSum) {
      return;
    }
    let p0 = 0;
    let p1 = 0;
    for (let h = 0; h < NUM_HANDS; h += 1) {
      p0 += handValues[h]! * beliefs[h]!;
      p1 += handValues[NUM_HANDS + h]! * beliefs[NUM_HANDS + h]!;
    }
    const offset = (p0 + p1) / 2;
    for (let i = 0; i < handValues.length; i += 1) {
      handValues[i] = handValues[i]! - offset;
    }
  }

  private applyZeroSumBatchIfNeeded(
    handValues: Float32Array<ArrayBufferLike>,
    beliefs: Float32Array<ArrayBufferLike>,
    batch: number,
  ): void {
    if (!this.manifest.architecture.enforceZeroSum) {
      return;
    }
    const stride = 2 * NUM_HANDS;
    for (let sample = 0; sample < batch; sample += 1) {
      const base = sample * stride;
      let p0 = 0;
      let p1 = 0;
      for (let h = 0; h < NUM_HANDS; h += 1) {
        p0 += handValues[base + h]! * beliefs[h]!;
        p1 += handValues[base + NUM_HANDS + h]! * beliefs[NUM_HANDS + h]!;
      }
      const offset = (p0 + p1) / 2;
      for (let i = 0; i < stride; i += 1) {
        handValues[base + i] = handValues[base + i]! - offset;
      }
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
    const encoder = this.device.createCommandEncoder();
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
    this.device.queue.submit([encoder.finish()]);
  }

  private submit2d(
    pipeline: GPUComputePipeline,
    workgroupsX: number,
    workgroupsY: number,
    entries: GPUBindGroupEntry[],
  ): void {
    const encoder = this.device.createCommandEncoder();
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
    this.device.queue.submit([encoder.finish()]);
  }
}
