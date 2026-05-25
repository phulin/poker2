import { makeStorageBuffer, makeUniformBuffer } from "../gpuBuffers.js";
import { createComputePipeline, dispatchCompute } from "../gpuPipeline.js";
import {
  LEAF_SAMPLE_CHUNK,
  MAX_DISPATCH_WORKGROUPS_PER_DIMENSION,
  SHOWDOWN_BOTH_PLAYER_SAMPLE_CHUNK,
  TERMINAL_SAMPLE_CHUNK,
  alignedSampleChunk,
  dispatchLimitedCount,
  rangeChunks,
} from "./dispatch.js";
import {
  type SparseGpuTreeBuffers,
  type SparseGpuTreeData,
} from "./treeBuffers.js";
import {
  SPARSE_AVERAGE_POLICY_WGSL,
  SPARSE_BACKUP_DEPTH_WGSL,
  SPARSE_BELIEF_PROPAGATE_FUSED_WGSL,
  SPARSE_REACH_APPLY_WGSL,
  SPARSE_REGRET_MATCH_WGSL,
  SPARSE_REGRET_TAIL_WGSL,
} from "./shaders/core.js";
import {
  SPARSE_OPPONENT_POLICY_AGGREGATE_PARALLEL_WGSL,
  SPARSE_OPPONENT_POLICY_AGGREGATE_WGSL,
  SPARSE_OPPONENT_POLICY_FROM_AGGREGATE_WGSL,
  SPARSE_OPPONENT_POLICY_WGSL,
  SPARSE_REGRET_WEIGHT_AGGREGATE_PARALLEL_WGSL,
  SPARSE_REGRET_WEIGHT_AGGREGATE_WGSL,
  SPARSE_REGRET_WEIGHT_FROM_AGGREGATE_WGSL,
  SPARSE_REGRET_WEIGHT_WGSL,
} from "./shaders/opponent.js";
import {
  SPARSE_SHOWDOWN_RANK_MASS_BY_HANDS_BOTH_PLAYERS_WGSL,
  SPARSE_SHOWDOWN_RANK_MASS_BY_HANDS_WGSL,
  SPARSE_SHOWDOWN_RANK_MASS_WGSL,
  SPARSE_SHOWDOWN_RANK_PREFIX_PACKED_WGSL,
  SPARSE_SHOWDOWN_RANK_PREFIX_WGSL,
  SPARSE_GATHER_NODE_BELIEFS_WGSL,
  SPARSE_SCATTER_NODE_VALUES_WGSL,
  SPARSE_SHOWDOWN_VALUES_FROM_RANKS_1326_BOTH_PLAYERS_PACKED_WGSL,
  SPARSE_SHOWDOWN_VALUES_FROM_RANKS_1326_BOTH_PLAYERS_WGSL,
  SPARSE_SHOWDOWN_VALUES_FROM_RANKS_1326_WGSL,
  SPARSE_SHOWDOWN_VALUES_FROM_RANKS_WGSL,
  SPARSE_SHOWDOWN_VALUES_WGSL,
} from "./shaders/terminal.js";
import {
  SPARSE_ALLIN_TABLE_VALUES_1326_NOPERM_BOTH_PLAYERS_WGSL,
  SPARSE_ALLIN_TABLE_VALUES_1326_NOPERM_WGSL,
  SPARSE_ALLIN_TABLE_VALUES_WGSL,
} from "./shaders/allIn.js";

type UniformWord = ["u32" | "f32", number];

function mixedUniform(words: UniformWord[]): Uint32Array<ArrayBuffer> {
  const buffer = new ArrayBuffer(words.length * Uint32Array.BYTES_PER_ELEMENT);
  const view = new DataView(buffer);
  for (let i = 0; i < words.length; i += 1) {
    const [kind, value] = words[i]!;
    const offset = i * Uint32Array.BYTES_PER_ELEMENT;
    if (kind === "f32") view.setFloat32(offset, value, true);
    else view.setUint32(offset, value, true);
  }
  return new Uint32Array(buffer);
}

export class SparseCfrGpuKernels {
  readonly device: GPUDevice;
  private readonly regretMatchPipeline: GPUComputePipeline;
  private readonly beliefPropagateFusedPipeline: GPUComputePipeline;
  private readonly reachApplyPipeline: GPUComputePipeline;
  private readonly averagePolicyPipeline: GPUComputePipeline;
  private readonly backupDepthPipeline: GPUComputePipeline;
  private readonly regretTailPipeline: GPUComputePipeline;
  private readonly opponentPolicyPipeline: GPUComputePipeline;
  private readonly opponentPolicyAggregatePipeline: GPUComputePipeline;
  private readonly opponentPolicyAggregateParallelPipeline: GPUComputePipeline;
  private readonly opponentPolicyFromAggregatePipeline: GPUComputePipeline;
  private readonly regretWeightPipeline: GPUComputePipeline;
  private readonly regretWeightAggregatePipeline: GPUComputePipeline;
  private readonly regretWeightAggregateParallelPipeline: GPUComputePipeline;
  private readonly regretWeightFromAggregatePipeline: GPUComputePipeline;
  private readonly gatherNodeBeliefsPipeline: GPUComputePipeline;
  private readonly scatterNodeValuesPipeline: GPUComputePipeline;
  private readonly showdownValuesPipeline: GPUComputePipeline;
  private readonly showdownRankMassPipeline: GPUComputePipeline;
  private readonly showdownRankMassByHandsPipeline: GPUComputePipeline;
  private readonly showdownRankMassByHandsBothPlayersPipeline: GPUComputePipeline;
  private readonly showdownRankPrefixPipeline: GPUComputePipeline;
  private readonly showdownRankPrefixPackedPipeline: GPUComputePipeline;
  private readonly showdownValuesFromRanksPipeline?: GPUComputePipeline;
  private readonly showdownValuesFromRanks1326Pipeline?: GPUComputePipeline;
  private readonly showdownValuesFromRanks1326BothPlayersPipeline?: GPUComputePipeline;
  private readonly showdownValuesFromRanks1326BothPlayersPackedPipeline: GPUComputePipeline;
  private readonly allInTableValuesPipeline?: GPUComputePipeline;
  private readonly allInTableValues1326NoPermPipeline: GPUComputePipeline;
  private readonly allInTableValues1326NoPermBothPlayersPipeline: GPUComputePipeline;

  constructor(device: GPUDevice) {
    this.device = device;
    const maxStorageBuffers = device.limits.maxStorageBuffersPerShaderStage;
    this.regretMatchPipeline = this.pipeline(
      SPARSE_REGRET_MATCH_WGSL,
      "sparse-cfr-regret-match",
    );
    this.beliefPropagateFusedPipeline = this.pipeline(
      SPARSE_BELIEF_PROPAGATE_FUSED_WGSL,
      "sparse-cfr-belief-propagate-fused",
    );
    this.reachApplyPipeline = this.pipeline(
      SPARSE_REACH_APPLY_WGSL,
      "sparse-cfr-reach-apply",
    );
    this.averagePolicyPipeline = this.pipeline(
      SPARSE_AVERAGE_POLICY_WGSL,
      "sparse-cfr-average-policy",
    );
    this.backupDepthPipeline = this.pipeline(
      SPARSE_BACKUP_DEPTH_WGSL,
      "sparse-cfr-backup-depth",
    );
    this.regretTailPipeline = this.pipeline(
      SPARSE_REGRET_TAIL_WGSL,
      "sparse-cfr-regret-tail",
    );
    this.opponentPolicyPipeline = this.pipeline(
      SPARSE_OPPONENT_POLICY_WGSL,
      "sparse-cfr-opponent-policy",
    );
    this.opponentPolicyAggregatePipeline = this.pipeline(
      SPARSE_OPPONENT_POLICY_AGGREGATE_WGSL,
      "sparse-cfr-opponent-policy-aggregate",
    );
    this.opponentPolicyAggregateParallelPipeline = this.pipeline(
      SPARSE_OPPONENT_POLICY_AGGREGATE_PARALLEL_WGSL,
      "sparse-cfr-opponent-policy-aggregate-parallel",
    );
    this.opponentPolicyFromAggregatePipeline = this.pipeline(
      SPARSE_OPPONENT_POLICY_FROM_AGGREGATE_WGSL,
      "sparse-cfr-opponent-policy-from-aggregate",
    );
    this.regretWeightPipeline = this.pipeline(
      SPARSE_REGRET_WEIGHT_WGSL,
      "sparse-cfr-regret-weight",
    );
    this.regretWeightAggregatePipeline = this.pipeline(
      SPARSE_REGRET_WEIGHT_AGGREGATE_WGSL,
      "sparse-cfr-regret-weight-aggregate",
    );
    this.regretWeightAggregateParallelPipeline = this.pipeline(
      SPARSE_REGRET_WEIGHT_AGGREGATE_PARALLEL_WGSL,
      "sparse-cfr-regret-weight-aggregate-parallel",
    );
    this.regretWeightFromAggregatePipeline = this.pipeline(
      SPARSE_REGRET_WEIGHT_FROM_AGGREGATE_WGSL,
      "sparse-cfr-regret-weight-from-aggregate",
    );
    this.gatherNodeBeliefsPipeline = this.pipeline(
      SPARSE_GATHER_NODE_BELIEFS_WGSL,
      "sparse-cfr-gather-node-beliefs",
    );
    this.scatterNodeValuesPipeline = this.pipeline(
      SPARSE_SCATTER_NODE_VALUES_WGSL,
      "sparse-cfr-scatter-node-values",
    );
    this.showdownValuesPipeline = this.pipeline(
      SPARSE_SHOWDOWN_VALUES_WGSL,
      "sparse-cfr-showdown-values",
    );
    this.showdownRankMassPipeline = this.pipeline(
      SPARSE_SHOWDOWN_RANK_MASS_WGSL,
      "sparse-cfr-showdown-rank-mass",
    );
    this.showdownRankMassByHandsPipeline = this.pipeline(
      SPARSE_SHOWDOWN_RANK_MASS_BY_HANDS_WGSL,
      "sparse-cfr-showdown-rank-mass-by-hands",
    );
    this.showdownRankMassByHandsBothPlayersPipeline = this.pipeline(
      SPARSE_SHOWDOWN_RANK_MASS_BY_HANDS_BOTH_PLAYERS_WGSL,
      "sparse-cfr-showdown-rank-mass-by-hands-both-players",
    );
    this.showdownRankPrefixPipeline = this.pipeline(
      SPARSE_SHOWDOWN_RANK_PREFIX_WGSL,
      "sparse-cfr-showdown-rank-prefix",
    );
    this.showdownRankPrefixPackedPipeline = this.pipeline(
      SPARSE_SHOWDOWN_RANK_PREFIX_PACKED_WGSL,
      "sparse-cfr-showdown-rank-prefix-packed",
    );
    this.showdownValuesFromRanks1326BothPlayersPackedPipeline = this.pipeline(
      SPARSE_SHOWDOWN_VALUES_FROM_RANKS_1326_BOTH_PLAYERS_PACKED_WGSL,
      "sparse-cfr-showdown-values-from-ranks-1326-both-players-packed",
    );
    if (maxStorageBuffers >= 10) {
      this.showdownValuesFromRanksPipeline = this.pipeline(
        SPARSE_SHOWDOWN_VALUES_FROM_RANKS_WGSL,
        "sparse-cfr-showdown-values-from-ranks",
      );
      this.showdownValuesFromRanks1326Pipeline = this.pipeline(
        SPARSE_SHOWDOWN_VALUES_FROM_RANKS_1326_WGSL,
        "sparse-cfr-showdown-values-from-ranks-1326",
      );
      this.showdownValuesFromRanks1326BothPlayersPipeline = this.pipeline(
        SPARSE_SHOWDOWN_VALUES_FROM_RANKS_1326_BOTH_PLAYERS_WGSL,
        "sparse-cfr-showdown-values-from-ranks-1326-both-players",
      );
    }
    if (maxStorageBuffers >= 9) {
      this.allInTableValuesPipeline = this.pipeline(
        SPARSE_ALLIN_TABLE_VALUES_WGSL,
        "sparse-cfr-allin-table-values",
      );
    }
    this.allInTableValues1326NoPermPipeline = this.pipeline(
      SPARSE_ALLIN_TABLE_VALUES_1326_NOPERM_WGSL,
      "sparse-cfr-allin-table-values-1326-noperm",
    );
    this.allInTableValues1326NoPermBothPlayersPipeline = this.pipeline(
      SPARSE_ALLIN_TABLE_VALUES_1326_NOPERM_BOTH_PLAYERS_WGSL,
      "sparse-cfr-allin-table-values-1326-noperm-both-players",
    );
  }

  createTreeBuffers(data: SparseGpuTreeData): SparseGpuTreeBuffers {
    const buffers = [
      makeStorageBuffer(this.device, data.childOffsets),
      makeStorageBuffer(this.device, data.childCount),
      makeStorageBuffer(this.device, data.childIndices),
      makeStorageBuffer(this.device, data.parentIndex),
      makeStorageBuffer(this.device, data.prevActor),
      makeStorageBuffer(this.device, data.toAct),
      makeStorageBuffer(this.device, data.allowedMask),
      makeStorageBuffer(this.device, data.allowedProb),
      makeStorageBuffer(this.device, data.handCard0),
      makeStorageBuffer(this.device, data.handCard1),
      makeStorageBuffer(this.device, data.overlapHands),
      makeStorageBuffer(this.device, data.overlapCounts),
    ] as const;
    return {
      nodeCount: data.nodeCount,
      numHands: data.numHands,
      childOffsets: buffers[0],
      childCount: buffers[1],
      childIndices: buffers[2],
      parentIndex: buffers[3],
      prevActor: buffers[4],
      toAct: buffers[5],
      allowedMask: buffers[6],
      allowedProb: buffers[7],
      handCard0: buffers[8],
      handCard1: buffers[9],
      overlapHands: buffers[10],
      overlapCounts: buffers[11],
      overlapSlots: data.overlapSlots,
      dispose: () => {
        for (const buffer of buffers) {
          buffer.destroy();
        }
      },
    };
  }

  encodeRegretMatch(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    regrets: GPUBuffer,
    policy: GPUBuffer,
  ): GPUBuffer[] {
    const paramsList: GPUBuffer[] = [];
    const nodeChunk = alignedSampleChunk(tree.numHands, 64);
    for (let start = 0; start < tree.nodeCount; start += nodeChunk) {
      const nodeCount = Math.min(nodeChunk, tree.nodeCount - start);
      const params = makeUniformBuffer(
        this.device,
        new Uint32Array([nodeCount, tree.numHands, 0, 0]),
      );
      const bindGroup = this.device.createBindGroup({
        layout: this.regretMatchPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: tree.childOffsets, offset: start * 4 } },
          { binding: 1, resource: { buffer: tree.childCount, offset: start * 4 } },
          { binding: 2, resource: { buffer: tree.childIndices } },
          { binding: 3, resource: { buffer: regrets } },
          { binding: 4, resource: { buffer: policy } },
          { binding: 5, resource: { buffer: params } },
        ],
      });
      this.encode(
        encoder,
        this.regretMatchPipeline,
        bindGroup,
        Math.ceil((nodeCount * tree.numHands) / 64),
      );
      paramsList.push(params);
    }
    return paramsList;
  }

  encodePropagateBeliefsDepth(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    policy: GPUBuffer,
    beliefs: GPUBuffer,
    denom: GPUBuffer,
    start: number,
    end: number,
  ): GPUBuffer[] {
    return this.encodePropagateBeliefsDepthFused(
      encoder,
      tree,
      policy,
      beliefs,
      denom,
      start,
      end,
    );
  }

  encodePropagateBeliefsDepthFused(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    policy: GPUBuffer,
    beliefs: GPUBuffer,
    denom: GPUBuffer,
    start: number,
    end: number,
  ): GPUBuffer[] {
    const paramsList: GPUBuffer[] = [];
    for (const [chunkStart, chunkEnd] of rangeChunks(
      start,
      end,
      MAX_DISPATCH_WORKGROUPS_PER_DIMENSION,
    )) {
      const params = makeUniformBuffer(
        this.device,
        new Uint32Array([tree.numHands, chunkStart, chunkEnd, 0]),
      );
      const bindGroup = this.device.createBindGroup({
        layout: this.beliefPropagateFusedPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: tree.parentIndex } },
          { binding: 1, resource: { buffer: tree.prevActor } },
          { binding: 2, resource: { buffer: tree.allowedMask } },
          { binding: 3, resource: { buffer: tree.allowedProb } },
          { binding: 4, resource: { buffer: policy } },
          { binding: 5, resource: { buffer: beliefs } },
          { binding: 6, resource: { buffer: denom } },
          { binding: 7, resource: { buffer: params } },
        ],
      });
      this.encode(
        encoder,
        this.beliefPropagateFusedPipeline,
        bindGroup,
        Math.max(0, chunkEnd - chunkStart),
        2,
      );
      paramsList.push(params);
    }
    return paramsList;
  }

  encodePropagateReachDepth(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    policy: GPUBuffer,
    reach: GPUBuffer,
    start: number,
    end: number,
  ): GPUBuffer[] {
    const paramsList: GPUBuffer[] = [];
    for (const [chunkStart, chunkEnd] of rangeChunks(
      start,
      end,
      dispatchLimitedCount(2 * tree.numHands, 64),
    )) {
      const params = makeUniformBuffer(
        this.device,
        new Uint32Array([tree.numHands, chunkStart, chunkEnd, 0]),
      );
      const bindGroup = this.device.createBindGroup({
        layout: this.reachApplyPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: tree.parentIndex } },
          { binding: 1, resource: { buffer: tree.prevActor } },
          { binding: 2, resource: { buffer: tree.allowedMask } },
          { binding: 3, resource: { buffer: policy } },
          { binding: 4, resource: { buffer: reach } },
          { binding: 5, resource: { buffer: params } },
        ],
      });
      this.encode(
        encoder,
        this.reachApplyPipeline,
        bindGroup,
        Math.ceil(((chunkEnd - chunkStart) * 2 * tree.numHands) / 64),
      );
      paramsList.push(params);
    }
    return paramsList;
  }

  encodeUpdateAveragePolicyRange(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    reach: GPUBuffer,
    policy: GPUBuffer,
    numerator: GPUBuffer,
    denominator: GPUBuffer,
    policyAvg: GPUBuffer,
    start: number,
    end: number,
    weight = 1,
  ): GPUBuffer[] {
    const paramsList: GPUBuffer[] = [];
    for (const [chunkStart, chunkEnd] of rangeChunks(
      start,
      end,
      dispatchLimitedCount(tree.numHands, 64),
    )) {
      const params = makeUniformBuffer(
        this.device,
        mixedUniform([
          ["u32", tree.numHands],
          ["u32", chunkStart],
          ["u32", chunkEnd],
          ["f32", weight],
        ]),
      );
      const bindGroup = this.device.createBindGroup({
        layout: this.averagePolicyPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: tree.parentIndex } },
          { binding: 1, resource: { buffer: tree.prevActor } },
          { binding: 2, resource: { buffer: reach } },
          { binding: 3, resource: { buffer: policy } },
          { binding: 4, resource: { buffer: numerator } },
          { binding: 5, resource: { buffer: denominator } },
          { binding: 6, resource: { buffer: policyAvg } },
          { binding: 7, resource: { buffer: params } },
        ],
      });
      this.encode(
        encoder,
        this.averagePolicyPipeline,
        bindGroup,
        Math.ceil(((chunkEnd - chunkStart) * tree.numHands) / 64),
      );
      paramsList.push(params);
    }
    return paramsList;
  }

  encodeBackupDepth(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    policy: GPUBuffer,
    opponentPolicy: GPUBuffer,
    values: GPUBuffer,
    start: number,
    end: number,
  ): GPUBuffer[] {
    const pipeline = this.backupDepthPipeline;
    const paramsList: GPUBuffer[] = [];
    for (const [chunkStart, chunkEnd] of rangeChunks(
      start,
      end,
      dispatchLimitedCount(2 * tree.numHands, 128),
    )) {
      const params = makeUniformBuffer(
        this.device,
        new Uint32Array([tree.numHands, chunkStart, chunkEnd, 0]),
      );
      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: tree.childOffsets } },
          { binding: 1, resource: { buffer: tree.childCount } },
          { binding: 2, resource: { buffer: tree.childIndices } },
          { binding: 3, resource: { buffer: tree.toAct } },
          { binding: 4, resource: { buffer: policy } },
          { binding: 5, resource: { buffer: opponentPolicy } },
          { binding: 6, resource: { buffer: values } },
          { binding: 7, resource: { buffer: params } },
        ],
      });
      this.encode(
        encoder,
        pipeline,
        bindGroup,
        Math.ceil(((chunkEnd - chunkStart) * 2 * tree.numHands) / 128),
      );
      paramsList.push(params);
    }
    return paramsList;
  }

  encodeAccumulateRegretsRange(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    regretWeights: GPUBuffer,
    values: GPUBuffer,
    regrets: GPUBuffer,
    start: number,
    end: number,
    options: {
      discountPositive?: number;
      discountNegative?: number;
      linearSkipActor?: number;
      cfrPlus?: boolean;
    } = {},
  ): GPUBuffer[] {
    const paramsList: GPUBuffer[] = [];
    for (const [chunkStart, chunkEnd] of rangeChunks(
      start,
      end,
      dispatchLimitedCount(tree.numHands, 64),
    )) {
      const params = makeUniformBuffer(
        this.device,
        mixedUniform([
          ["u32", tree.numHands],
          ["u32", chunkStart],
          ["u32", chunkEnd],
          ["f32", options.discountPositive ?? 1],
          ["f32", options.discountNegative ?? 1],
          ["u32", options.linearSkipActor ?? 2],
          ["u32", options.cfrPlus ? 1 : 0],
          ["u32", 0],
        ]),
      );
      const bindGroup = this.device.createBindGroup({
        layout: this.regretTailPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: tree.parentIndex } },
          { binding: 1, resource: { buffer: tree.prevActor } },
          { binding: 2, resource: { buffer: regretWeights } },
          { binding: 3, resource: { buffer: values } },
          { binding: 4, resource: { buffer: regrets } },
          { binding: 5, resource: { buffer: params } },
        ],
      });
      this.encode(
        encoder,
        this.regretTailPipeline,
        bindGroup,
        Math.ceil(((chunkEnd - chunkStart) * tree.numHands) / 64),
      );
      paramsList.push(params);
    }
    return paramsList;
  }

  encodeComputeOpponentPolicyReferenceRange(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    beliefs: GPUBuffer,
    policy: GPUBuffer,
    opponentPolicy: GPUBuffer,
    start: number,
    end: number,
  ): GPUBuffer {
    const params = makeUniformBuffer(
      this.device,
      new Uint32Array([tree.numHands, start, end, 0]),
    );
    const bindGroup = this.device.createBindGroup({
      layout: this.opponentPolicyPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tree.parentIndex } },
        { binding: 1, resource: { buffer: tree.prevActor } },
        { binding: 2, resource: { buffer: tree.handCard0 } },
        { binding: 3, resource: { buffer: tree.handCard1 } },
        { binding: 4, resource: { buffer: beliefs } },
        { binding: 5, resource: { buffer: policy } },
        { binding: 6, resource: { buffer: opponentPolicy } },
        { binding: 7, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.opponentPolicyPipeline,
      bindGroup,
      Math.ceil(((end - start) * tree.numHands) / 128),
    );
    return params;
  }

  encodeComputeOpponentPolicyAggregateRange(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    beliefs: GPUBuffer,
    policy: GPUBuffer,
    aggregates: GPUBuffer,
    opponentPolicy: GPUBuffer,
    start: number,
    end: number,
  ): GPUBuffer[] {
    return this.encodeComputeOpponentPolicyAggregateParallelRange(
      encoder,
      tree,
      beliefs,
      policy,
      aggregates,
      opponentPolicy,
      start,
      end,
    );
  }

  encodeComputeOpponentPolicyAggregateReferenceRange(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    beliefs: GPUBuffer,
    policy: GPUBuffer,
    aggregates: GPUBuffer,
    opponentPolicy: GPUBuffer,
    start: number,
    end: number,
  ): GPUBuffer[] {
    const params = makeUniformBuffer(
      this.device,
      new Uint32Array([tree.numHands, start, end, 0]),
    );
    const aggregateBindGroup = this.device.createBindGroup({
      layout: this.opponentPolicyAggregatePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tree.parentIndex } },
        { binding: 1, resource: { buffer: tree.prevActor } },
        { binding: 2, resource: { buffer: tree.handCard0 } },
        { binding: 3, resource: { buffer: tree.handCard1 } },
        { binding: 4, resource: { buffer: beliefs } },
        { binding: 5, resource: { buffer: policy } },
        { binding: 6, resource: { buffer: aggregates } },
        { binding: 7, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.opponentPolicyAggregatePipeline,
      aggregateBindGroup,
      Math.max(0, end - start),
    );

    const applyBindGroup = this.device.createBindGroup({
      layout: this.opponentPolicyFromAggregatePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tree.parentIndex } },
        { binding: 1, resource: { buffer: tree.prevActor } },
        { binding: 2, resource: { buffer: tree.handCard0 } },
        { binding: 3, resource: { buffer: tree.handCard1 } },
        { binding: 4, resource: { buffer: beliefs } },
        { binding: 5, resource: { buffer: policy } },
        { binding: 6, resource: { buffer: aggregates } },
        { binding: 7, resource: { buffer: opponentPolicy } },
        { binding: 8, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.opponentPolicyFromAggregatePipeline,
      applyBindGroup,
      Math.ceil(((end - start) * tree.numHands) / 64),
    );
    return [params];
  }

  encodeComputeOpponentPolicyAggregateParallelRange(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    beliefs: GPUBuffer,
    policy: GPUBuffer,
    aggregates: GPUBuffer,
    opponentPolicy: GPUBuffer,
    start: number,
    end: number,
  ): GPUBuffer[] {
    const paramsList: GPUBuffer[] = [];
    for (const [chunkStart, chunkEnd] of rangeChunks(
      start,
      end,
      dispatchLimitedCount(tree.numHands, 64),
    )) {
      const params = makeUniformBuffer(
        this.device,
        new Uint32Array([tree.numHands, chunkStart, chunkEnd, 0]),
      );
      const aggregateBindGroup = this.device.createBindGroup({
        layout: this.opponentPolicyAggregateParallelPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: tree.parentIndex } },
          { binding: 1, resource: { buffer: tree.prevActor } },
          { binding: 2, resource: { buffer: tree.handCard0 } },
          { binding: 3, resource: { buffer: tree.handCard1 } },
          { binding: 4, resource: { buffer: beliefs } },
          { binding: 5, resource: { buffer: policy } },
          { binding: 6, resource: { buffer: aggregates } },
          { binding: 7, resource: { buffer: params } },
        ],
      });
      this.encode(
        encoder,
        this.opponentPolicyAggregateParallelPipeline,
        aggregateBindGroup,
        Math.max(0, chunkEnd - chunkStart),
      );

      const applyBindGroup = this.device.createBindGroup({
        layout: this.opponentPolicyFromAggregatePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: tree.parentIndex } },
          { binding: 1, resource: { buffer: tree.prevActor } },
          { binding: 2, resource: { buffer: tree.handCard0 } },
          { binding: 3, resource: { buffer: tree.handCard1 } },
          { binding: 4, resource: { buffer: beliefs } },
          { binding: 5, resource: { buffer: policy } },
          { binding: 6, resource: { buffer: aggregates } },
          { binding: 7, resource: { buffer: opponentPolicy } },
          { binding: 8, resource: { buffer: params } },
        ],
      });
      this.encode(
        encoder,
        this.opponentPolicyFromAggregatePipeline,
        applyBindGroup,
        Math.ceil(((chunkEnd - chunkStart) * tree.numHands) / 64),
      );
      paramsList.push(params);
    }
    return paramsList;
  }

  encodeComputeRegretWeightsReferenceRange(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    beliefs: GPUBuffer,
    regretWeights: GPUBuffer,
    start: number,
    end: number,
  ): GPUBuffer {
    const params = makeUniformBuffer(
      this.device,
      new Uint32Array([tree.numHands, start, end, 0]),
    );
    const bindGroup = this.device.createBindGroup({
      layout: this.regretWeightPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tree.toAct } },
        { binding: 1, resource: { buffer: tree.allowedMask } },
        { binding: 2, resource: { buffer: tree.handCard0 } },
        { binding: 3, resource: { buffer: tree.handCard1 } },
        { binding: 4, resource: { buffer: beliefs } },
        { binding: 5, resource: { buffer: regretWeights } },
        { binding: 6, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.regretWeightPipeline,
      bindGroup,
      Math.ceil(((end - start) * tree.numHands) / 128),
    );
    return params;
  }

  encodeComputeRegretWeightsAggregateRange(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    beliefs: GPUBuffer,
    aggregates: GPUBuffer,
    regretWeights: GPUBuffer,
    start: number,
    end: number,
  ): GPUBuffer[] {
    return this.encodeComputeRegretWeightsAggregateParallelRange(
      encoder,
      tree,
      beliefs,
      aggregates,
      regretWeights,
      start,
      end,
    );
  }

  encodeComputeRegretWeightsAggregateReferenceRange(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    beliefs: GPUBuffer,
    aggregates: GPUBuffer,
    regretWeights: GPUBuffer,
    start: number,
    end: number,
  ): GPUBuffer[] {
    const params = makeUniformBuffer(
      this.device,
      new Uint32Array([tree.numHands, start, end, 0]),
    );
    const aggregateBindGroup = this.device.createBindGroup({
      layout: this.regretWeightAggregatePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tree.toAct } },
        { binding: 1, resource: { buffer: tree.handCard0 } },
        { binding: 2, resource: { buffer: tree.handCard1 } },
        { binding: 3, resource: { buffer: beliefs } },
        { binding: 4, resource: { buffer: aggregates } },
        { binding: 5, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.regretWeightAggregatePipeline,
      aggregateBindGroup,
      Math.max(0, end - start),
    );

    const applyBindGroup = this.device.createBindGroup({
      layout: this.regretWeightFromAggregatePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tree.toAct } },
        { binding: 1, resource: { buffer: tree.allowedMask } },
        { binding: 2, resource: { buffer: tree.handCard0 } },
        { binding: 3, resource: { buffer: tree.handCard1 } },
        { binding: 4, resource: { buffer: beliefs } },
        { binding: 5, resource: { buffer: aggregates } },
        { binding: 6, resource: { buffer: regretWeights } },
        { binding: 7, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.regretWeightFromAggregatePipeline,
      applyBindGroup,
      Math.ceil(((end - start) * tree.numHands) / 64),
    );
    return [params];
  }

  encodeComputeRegretWeightsAggregateParallelRange(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    beliefs: GPUBuffer,
    aggregates: GPUBuffer,
    regretWeights: GPUBuffer,
    start: number,
    end: number,
  ): GPUBuffer[] {
    const paramsList: GPUBuffer[] = [];
    for (const [chunkStart, chunkEnd] of rangeChunks(
      start,
      end,
      dispatchLimitedCount(tree.numHands, 64),
    )) {
      const params = makeUniformBuffer(
        this.device,
        new Uint32Array([tree.numHands, chunkStart, chunkEnd, 0]),
      );
      const aggregateBindGroup = this.device.createBindGroup({
        layout: this.regretWeightAggregateParallelPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: tree.toAct } },
          { binding: 1, resource: { buffer: tree.handCard0 } },
          { binding: 2, resource: { buffer: tree.handCard1 } },
          { binding: 3, resource: { buffer: beliefs } },
          { binding: 4, resource: { buffer: aggregates } },
          { binding: 5, resource: { buffer: params } },
        ],
      });
      this.encode(
        encoder,
        this.regretWeightAggregateParallelPipeline,
        aggregateBindGroup,
        Math.max(0, chunkEnd - chunkStart),
      );

      const applyBindGroup = this.device.createBindGroup({
        layout: this.regretWeightFromAggregatePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: tree.toAct } },
          { binding: 1, resource: { buffer: tree.allowedMask } },
          { binding: 2, resource: { buffer: tree.handCard0 } },
          { binding: 3, resource: { buffer: tree.handCard1 } },
          { binding: 4, resource: { buffer: beliefs } },
          { binding: 5, resource: { buffer: aggregates } },
          { binding: 6, resource: { buffer: regretWeights } },
          { binding: 7, resource: { buffer: params } },
        ],
      });
      this.encode(
        encoder,
        this.regretWeightFromAggregatePipeline,
        applyBindGroup,
        Math.ceil(((chunkEnd - chunkStart) * tree.numHands) / 64),
      );
      paramsList.push(params);
    }
    return paramsList;
  }

  encodeGatherNodeBeliefs(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    nodeIndices: GPUBuffer,
    beliefs: GPUBuffer,
    out: GPUBuffer,
    batch: number,
  ): GPUBuffer[] {
    const paramsList: GPUBuffer[] = [];
    for (let start = 0; start < batch; start += LEAF_SAMPLE_CHUNK) {
      const chunkBatch = Math.min(LEAF_SAMPLE_CHUNK, batch - start);
      const params = makeUniformBuffer(
        this.device,
        new Uint32Array([tree.numHands, chunkBatch, 0, 0]),
      );
      const bindGroup = this.device.createBindGroup({
        layout: this.gatherNodeBeliefsPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: nodeIndices, offset: start * 4 } },
          { binding: 1, resource: { buffer: beliefs } },
          { binding: 2, resource: { buffer: out, offset: start * 2 * tree.numHands * 4 } },
          { binding: 3, resource: { buffer: params } },
        ],
      });
      this.encode(
        encoder,
        this.gatherNodeBeliefsPipeline,
        bindGroup,
        Math.ceil((chunkBatch * 2 * tree.numHands) / 64),
      );
      paramsList.push(params);
    }
    return paramsList;
  }

  encodeScatterNodeValues(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    nodeIndices: GPUBuffer,
    sourceValues: GPUBuffer,
    values: GPUBuffer,
    batch: number,
  ): GPUBuffer[] {
    const paramsList: GPUBuffer[] = [];
    for (let start = 0; start < batch; start += LEAF_SAMPLE_CHUNK) {
      const chunkBatch = Math.min(LEAF_SAMPLE_CHUNK, batch - start);
      const params = makeUniformBuffer(
        this.device,
        new Uint32Array([tree.numHands, chunkBatch, 0, 0]),
      );
      const bindGroup = this.device.createBindGroup({
        layout: this.scatterNodeValuesPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: nodeIndices, offset: start * 4 } },
          { binding: 1, resource: { buffer: sourceValues, offset: start * 2 * tree.numHands * 4 } },
          { binding: 2, resource: { buffer: values } },
          { binding: 3, resource: { buffer: params } },
        ],
      });
      this.encode(
        encoder,
        this.scatterNodeValuesPipeline,
        bindGroup,
        Math.ceil((chunkBatch * 2 * tree.numHands) / 64),
      );
      paramsList.push(params);
    }
    return paramsList;
  }

  encodeShowdownValues(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    nodeIndices: GPUBuffer,
    rankCodes: GPUBuffer,
    payoffs: GPUBuffer,
    beliefs: GPUBuffer,
    values: GPUBuffer,
    batch: number,
  ): GPUBuffer[] {
    const paramsList: GPUBuffer[] = [];
    for (let start = 0; start < batch; start += TERMINAL_SAMPLE_CHUNK) {
      const chunkBatch = Math.min(TERMINAL_SAMPLE_CHUNK, batch - start);
      const params = makeUniformBuffer(
        this.device,
        new Uint32Array([tree.numHands, chunkBatch, 0, 0]),
      );
      const bindGroup = this.device.createBindGroup({
        layout: this.showdownValuesPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: nodeIndices, offset: start * 4 } },
          { binding: 1, resource: { buffer: tree.allowedMask } },
          { binding: 2, resource: { buffer: tree.handCard0 } },
          { binding: 3, resource: { buffer: tree.handCard1 } },
          { binding: 4, resource: { buffer: rankCodes, offset: start * tree.numHands * 4 } },
          { binding: 5, resource: { buffer: payoffs, offset: start * 3 * 4 } },
          { binding: 6, resource: { buffer: beliefs } },
          { binding: 7, resource: { buffer: values } },
          { binding: 8, resource: { buffer: params } },
        ],
      });
      this.encode(
        encoder,
        this.showdownValuesPipeline,
        bindGroup,
        Math.ceil((chunkBatch * 2 * tree.numHands) / 128),
      );
      paramsList.push(params);
    }
    return paramsList;
  }

  encodeShowdownValuesFromRankAggregates(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    nodeIndices: GPUBuffer,
    rankOrdinals: GPUBuffer,
    rankCounts: GPUBuffer,
    payoffs: GPUBuffer,
    beliefs: GPUBuffer,
    values: GPUBuffer,
    rankMass: GPUBuffer,
    rankPrefixLess: GPUBuffer,
    rankTotal: GPUBuffer,
    batch: number,
    maxRanks: number,
    rankHandOffsets?: GPUBuffer,
    rankHandCounts?: GPUBuffer,
    rankHands?: GPUBuffer,
  ): GPUBuffer[] {
    if (!rankHandOffsets || !rankHandCounts || !rankHands) {
      throw new Error("production showdown rank aggregates require rank-hand buffers");
    }
    if (tree.numHands !== 1326 || tree.overlapSlots !== 101) {
      throw new Error("production showdown rank aggregates require 1326 HUNL hands");
    }
    const params = this.encodeShowdownRankMassByHandsBothPlayers(
      encoder,
      tree,
      nodeIndices,
      rankHandOffsets,
      rankHandCounts,
      rankHands,
      rankCounts,
      beliefs,
      rankMass,
      batch,
      maxRanks,
    );
    const paramsList = [params];
    this.encodeShowdownRankPrefix(
      encoder,
      rankCounts,
      rankMass,
      rankPrefixLess,
      rankTotal,
      batch,
      maxRanks,
      params,
    );
    for (let start = 0; start < batch; start += SHOWDOWN_BOTH_PLAYER_SAMPLE_CHUNK) {
      paramsList.push(this.encodeShowdownValuesFromRanks1326BothPlayers(
        encoder,
        tree,
        nodeIndices,
        rankOrdinals,
        payoffs,
        beliefs,
        values,
        rankMass,
        rankPrefixLess,
        rankTotal,
        Math.min(SHOWDOWN_BOTH_PLAYER_SAMPLE_CHUNK, batch - start),
        maxRanks,
        undefined,
        start,
      ));
    }
    return paramsList;
  }

  encodeShowdownValuesFromRankAggregatesPacked(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    nodeIndices: GPUBuffer,
    rankOrdinals: GPUBuffer,
    rankCounts: GPUBuffer,
    payoffs: GPUBuffer,
    beliefs: GPUBuffer,
    values: GPUBuffer,
    rankScratch: GPUBuffer,
    batch: number,
    maxRanks: number,
    rankHandOffsets: GPUBuffer,
    rankHandCounts: GPUBuffer,
    rankHands: GPUBuffer,
  ): GPUBuffer[] {
    if (tree.numHands !== 1326 || tree.overlapSlots !== 101) {
      throw new Error("packed showdown rank aggregates require 1326 HUNL hands");
    }
    const aggregateCount = batch * 2 * maxRanks;
    const rankMassOffset = 0;
    const rankPrefixOffset = aggregateCount;
    const rankTotalOffset = aggregateCount * 2;
    const params = makeUniformBuffer(
      this.device,
      new Uint32Array([
        tree.numHands,
        batch,
        maxRanks,
        tree.overlapSlots,
        rankMassOffset,
        rankPrefixOffset,
        rankTotalOffset,
        0,
      ]),
    );
    const paramsList = [params];
    this.encodeShowdownRankMassByHandsBothPlayers(
      encoder,
      tree,
      nodeIndices,
      rankHandOffsets,
      rankHandCounts,
      rankHands,
      rankCounts,
      beliefs,
      rankScratch,
      batch,
      maxRanks,
      params,
    );
    this.encodeShowdownRankPrefixPacked(
      encoder,
      rankCounts,
      rankScratch,
      batch,
      maxRanks,
      rankMassOffset,
      rankPrefixOffset,
      rankTotalOffset,
      params,
    );
    for (let start = 0; start < batch; start += SHOWDOWN_BOTH_PLAYER_SAMPLE_CHUNK) {
      paramsList.push(this.encodeShowdownValuesFromRanks1326BothPlayersPacked(
        encoder,
        tree,
        nodeIndices,
        rankOrdinals,
        payoffs,
        beliefs,
        values,
        rankScratch,
        Math.min(SHOWDOWN_BOTH_PLAYER_SAMPLE_CHUNK, batch - start),
        maxRanks,
        rankMassOffset + start * 2 * maxRanks,
        rankPrefixOffset + start * 2 * maxRanks,
        rankTotalOffset + start * 2,
        start,
      ));
    }
    return paramsList;
  }

  encodeShowdownValuesFromRankAggregatesReference(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    nodeIndices: GPUBuffer,
    rankOrdinals: GPUBuffer,
    rankCounts: GPUBuffer,
    payoffs: GPUBuffer,
    beliefs: GPUBuffer,
    values: GPUBuffer,
    rankMass: GPUBuffer,
    rankPrefixLess: GPUBuffer,
    rankTotal: GPUBuffer,
    batch: number,
    maxRanks: number,
  ): GPUBuffer {
    const params = this.encodeShowdownRankMass(
      encoder,
      tree,
      nodeIndices,
      rankOrdinals,
      rankCounts,
      beliefs,
      rankMass,
      batch,
      maxRanks,
    );
    this.encodeShowdownRankPrefix(
      encoder,
      rankCounts,
      rankMass,
      rankPrefixLess,
      rankTotal,
      batch,
      maxRanks,
      params,
    );
    this.encodeShowdownValuesFromRanksReference(
      encoder,
      tree,
      nodeIndices,
      rankOrdinals,
      payoffs,
      beliefs,
      values,
      rankMass,
      rankPrefixLess,
      rankTotal,
      batch,
      maxRanks,
      params,
    );
    return params;
  }

  encodeShowdownRankMassByHandsBothPlayers(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    nodeIndices: GPUBuffer,
    rankHandOffsets: GPUBuffer,
    rankHandCounts: GPUBuffer,
    rankHands: GPUBuffer,
    rankCounts: GPUBuffer,
    beliefs: GPUBuffer,
    rankMass: GPUBuffer,
    batch: number,
    maxRanks: number,
    existingParams?: GPUBuffer,
  ): GPUBuffer {
    const params =
      existingParams ??
      makeUniformBuffer(
        this.device,
        new Uint32Array([tree.numHands, batch, maxRanks, tree.overlapSlots]),
      );
    const bindGroup = this.device.createBindGroup({
      layout: this.showdownRankMassByHandsBothPlayersPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: nodeIndices } },
        { binding: 1, resource: { buffer: tree.allowedMask } },
        { binding: 2, resource: { buffer: rankHandOffsets } },
        { binding: 3, resource: { buffer: rankHandCounts } },
        { binding: 4, resource: { buffer: rankHands } },
        { binding: 5, resource: { buffer: rankCounts } },
        { binding: 6, resource: { buffer: beliefs } },
        { binding: 7, resource: { buffer: rankMass } },
        { binding: 8, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.showdownRankMassByHandsBothPlayersPipeline,
      bindGroup,
      maxRanks,
      batch,
    );
    return params;
  }

  encodeShowdownRankMassByHands(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    nodeIndices: GPUBuffer,
    rankHandOffsets: GPUBuffer,
    rankHandCounts: GPUBuffer,
    rankHands: GPUBuffer,
    rankCounts: GPUBuffer,
    beliefs: GPUBuffer,
    rankMass: GPUBuffer,
    batch: number,
    maxRanks: number,
    existingParams?: GPUBuffer,
  ): GPUBuffer {
    const params =
      existingParams ??
      makeUniformBuffer(
        this.device,
        new Uint32Array([tree.numHands, batch, maxRanks, tree.overlapSlots]),
      );
    const bindGroup = this.device.createBindGroup({
      layout: this.showdownRankMassByHandsPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: nodeIndices } },
        { binding: 1, resource: { buffer: tree.allowedMask } },
        { binding: 2, resource: { buffer: rankHandOffsets } },
        { binding: 3, resource: { buffer: rankHandCounts } },
        { binding: 4, resource: { buffer: rankHands } },
        { binding: 5, resource: { buffer: rankCounts } },
        { binding: 6, resource: { buffer: beliefs } },
        { binding: 7, resource: { buffer: rankMass } },
        { binding: 8, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.showdownRankMassByHandsPipeline,
      bindGroup,
      maxRanks,
      batch * 2,
    );
    return params;
  }

  encodeShowdownRankMass(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    nodeIndices: GPUBuffer,
    rankOrdinals: GPUBuffer,
    rankCounts: GPUBuffer,
    beliefs: GPUBuffer,
    rankMass: GPUBuffer,
    batch: number,
    maxRanks: number,
    existingParams?: GPUBuffer,
  ): GPUBuffer {
    const params =
      existingParams ??
      makeUniformBuffer(
        this.device,
        new Uint32Array([tree.numHands, batch, maxRanks, tree.overlapSlots]),
      );
    const massBindGroup = this.device.createBindGroup({
      layout: this.showdownRankMassPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: nodeIndices } },
        { binding: 1, resource: { buffer: tree.allowedMask } },
        { binding: 2, resource: { buffer: rankOrdinals } },
        { binding: 3, resource: { buffer: rankCounts } },
        { binding: 4, resource: { buffer: beliefs } },
        { binding: 5, resource: { buffer: rankMass } },
        { binding: 6, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.showdownRankMassPipeline,
      massBindGroup,
      maxRanks,
      batch * 2,
    );
    return params;
  }

  encodeShowdownRankPrefix(
    encoder: GPUCommandEncoder,
    rankCounts: GPUBuffer,
    rankMass: GPUBuffer,
    rankPrefixLess: GPUBuffer,
    rankTotal: GPUBuffer,
    batch: number,
    maxRanks: number,
    existingParams?: GPUBuffer,
  ): GPUBuffer {
    const params =
      existingParams ??
      makeUniformBuffer(
        this.device,
        new Uint32Array([0, batch, maxRanks, 0]),
      );
    const prefixBindGroup = this.device.createBindGroup({
      layout: this.showdownRankPrefixPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: rankCounts } },
        { binding: 1, resource: { buffer: rankMass } },
        { binding: 2, resource: { buffer: rankPrefixLess } },
        { binding: 3, resource: { buffer: rankTotal } },
        { binding: 4, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.showdownRankPrefixPipeline,
      prefixBindGroup,
      batch * 2,
    );
    return params;
  }

  encodeShowdownRankPrefixPacked(
    encoder: GPUCommandEncoder,
    rankCounts: GPUBuffer,
    rankScratch: GPUBuffer,
    batch: number,
    maxRanks: number,
    rankMassOffset: number,
    rankPrefixOffset: number,
    rankTotalOffset: number,
    existingParams?: GPUBuffer,
  ): GPUBuffer {
    const params =
      existingParams ??
      makeUniformBuffer(
        this.device,
        new Uint32Array([
          0,
          batch,
          maxRanks,
          0,
          rankMassOffset,
          rankPrefixOffset,
          rankTotalOffset,
          0,
        ]),
      );
    const prefixBindGroup = this.device.createBindGroup({
      layout: this.showdownRankPrefixPackedPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: rankCounts } },
        { binding: 1, resource: { buffer: rankScratch } },
        { binding: 2, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.showdownRankPrefixPackedPipeline,
      prefixBindGroup,
      batch * 2,
    );
    return params;
  }

  encodeShowdownValuesFromRanksReference(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    nodeIndices: GPUBuffer,
    rankOrdinals: GPUBuffer,
    payoffs: GPUBuffer,
    beliefs: GPUBuffer,
    values: GPUBuffer,
    rankMass: GPUBuffer,
    rankPrefixLess: GPUBuffer,
    rankTotal: GPUBuffer,
    batch: number,
    maxRanks: number,
    existingParams?: GPUBuffer,
  ): GPUBuffer {
    const pipeline = this.showdownValuesFromRanksPipeline;
    if (!pipeline) {
      throw new Error(
        "reference showdown rank values require maxStorageBuffersPerShaderStage >= 10",
      );
    }
    const params =
      existingParams ??
      makeUniformBuffer(
        this.device,
        new Uint32Array([tree.numHands, batch, maxRanks, tree.overlapSlots]),
      );
    const valuesBindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: nodeIndices } },
        { binding: 1, resource: { buffer: rankOrdinals } },
        { binding: 2, resource: { buffer: tree.overlapHands } },
        { binding: 3, resource: { buffer: tree.overlapCounts } },
        { binding: 4, resource: { buffer: payoffs } },
        { binding: 5, resource: { buffer: beliefs } },
        { binding: 6, resource: { buffer: rankMass } },
        { binding: 7, resource: { buffer: rankPrefixLess } },
        { binding: 8, resource: { buffer: rankTotal } },
        { binding: 9, resource: { buffer: values } },
        { binding: 10, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      pipeline,
      valuesBindGroup,
      Math.ceil((batch * 2 * tree.numHands) / 128),
    );
    return params;
  }

  encodeShowdownValuesFromRanks1326(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    nodeIndices: GPUBuffer,
    rankOrdinals: GPUBuffer,
    payoffs: GPUBuffer,
    beliefs: GPUBuffer,
    values: GPUBuffer,
    rankMass: GPUBuffer,
    rankPrefixLess: GPUBuffer,
    rankTotal: GPUBuffer,
    batch: number,
    maxRanks: number,
    existingParams?: GPUBuffer,
  ): GPUBuffer {
    if (tree.numHands !== 1326 || tree.overlapSlots !== 101) {
      throw new Error("1326 showdown values require 1326 HUNL hands");
    }
    const pipeline = this.showdownValuesFromRanks1326Pipeline;
    if (!pipeline) {
      throw new Error(
        "1326 showdown rank values require maxStorageBuffersPerShaderStage >= 10",
      );
    }
    const params =
      existingParams ??
      makeUniformBuffer(
        this.device,
        new Uint32Array([tree.numHands, batch, maxRanks, tree.overlapSlots]),
      );
    const valuesBindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: nodeIndices } },
        { binding: 1, resource: { buffer: rankOrdinals } },
        { binding: 2, resource: { buffer: tree.overlapHands } },
        { binding: 3, resource: { buffer: tree.overlapCounts } },
        { binding: 4, resource: { buffer: payoffs } },
        { binding: 5, resource: { buffer: beliefs } },
        { binding: 6, resource: { buffer: rankMass } },
        { binding: 7, resource: { buffer: rankPrefixLess } },
        { binding: 8, resource: { buffer: rankTotal } },
        { binding: 9, resource: { buffer: values } },
        { binding: 10, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      pipeline,
      valuesBindGroup,
      Math.ceil((batch * 2652) / 128),
    );
    return params;
  }

  encodeShowdownValuesFromRanks1326BothPlayers(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    nodeIndices: GPUBuffer,
    rankOrdinals: GPUBuffer,
    payoffs: GPUBuffer,
    beliefs: GPUBuffer,
    values: GPUBuffer,
    rankMass: GPUBuffer,
    rankPrefixLess: GPUBuffer,
    rankTotal: GPUBuffer,
    batch: number,
    maxRanks: number,
    existingParams?: GPUBuffer,
    sampleStart = 0,
  ): GPUBuffer {
    if (tree.numHands !== 1326 || tree.overlapSlots !== 101) {
      throw new Error("both-player showdown values require 1326 HUNL hands");
    }
    const pipeline = this.showdownValuesFromRanks1326BothPlayersPipeline;
    if (!pipeline) {
      throw new Error(
        "both-player showdown rank values require maxStorageBuffersPerShaderStage >= 10",
      );
    }
    const params =
      existingParams ??
      makeUniformBuffer(
        this.device,
        new Uint32Array([tree.numHands, batch, maxRanks, tree.overlapSlots]),
      );
    const valuesBindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: nodeIndices, offset: sampleStart * 4 } },
        { binding: 1, resource: { buffer: rankOrdinals, offset: sampleStart * 1326 * 4 } },
        { binding: 2, resource: { buffer: tree.overlapHands } },
        { binding: 3, resource: { buffer: tree.overlapCounts } },
        { binding: 4, resource: { buffer: payoffs, offset: sampleStart * 3 * 4 } },
        { binding: 5, resource: { buffer: beliefs } },
        { binding: 6, resource: { buffer: rankMass, offset: sampleStart * 2 * maxRanks * 4 } },
        { binding: 7, resource: { buffer: rankPrefixLess, offset: sampleStart * 2 * maxRanks * 4 } },
        { binding: 8, resource: { buffer: rankTotal, offset: sampleStart * 2 * 4 } },
        { binding: 9, resource: { buffer: values } },
        { binding: 10, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      pipeline,
      valuesBindGroup,
      Math.ceil((batch * 1326) / 128),
    );
    return params;
  }

  encodeShowdownValuesFromRanks1326BothPlayersPacked(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    nodeIndices: GPUBuffer,
    rankOrdinals: GPUBuffer,
    payoffs: GPUBuffer,
    beliefs: GPUBuffer,
    values: GPUBuffer,
    rankScratch: GPUBuffer,
    batch: number,
    maxRanks: number,
    rankMassOffset: number,
    rankPrefixOffset: number,
    rankTotalOffset: number,
    sampleStart = 0,
  ): GPUBuffer {
    if (tree.numHands !== 1326 || tree.overlapSlots !== 101) {
      throw new Error("packed both-player showdown values require 1326 HUNL hands");
    }
    const params = makeUniformBuffer(
      this.device,
      new Uint32Array([
        tree.numHands,
        batch,
        maxRanks,
        tree.overlapSlots,
        rankMassOffset,
        rankPrefixOffset,
        rankTotalOffset,
        0,
      ]),
    );
    const valuesBindGroup = this.device.createBindGroup({
      layout: this.showdownValuesFromRanks1326BothPlayersPackedPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: nodeIndices, offset: sampleStart * 4 } },
        { binding: 1, resource: { buffer: rankOrdinals, offset: sampleStart * 1326 * 4 } },
        { binding: 2, resource: { buffer: tree.overlapHands } },
        { binding: 3, resource: { buffer: tree.overlapCounts } },
        { binding: 4, resource: { buffer: payoffs, offset: sampleStart * 3 * 4 } },
        { binding: 5, resource: { buffer: beliefs } },
        { binding: 6, resource: { buffer: rankScratch } },
        { binding: 7, resource: { buffer: values } },
        { binding: 8, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.showdownValuesFromRanks1326BothPlayersPackedPipeline,
      valuesBindGroup,
      Math.ceil((batch * 1326) / 128),
    );
    return params;
  }

  encodeAllInTableValues(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    nodeIndices: GPUBuffer,
    tablePacked: GPUBuffer,
    comboPerms: GPUBuffer,
    scaleFactors: GPUBuffer,
    beliefs: GPUBuffer,
    values: GPUBuffer,
    batch: number,
    tableScale: number,
    permId: number,
    hasPerm: boolean,
  ): GPUBuffer[] {
    if (tree.numHands !== 1326 || tree.overlapSlots !== 101 || hasPerm) {
      throw new Error("production all-in table values require 1326 HUNL hands without permutations");
    }
    void comboPerms;
    void permId;
    const paramsList: GPUBuffer[] = [];
    for (let start = 0; start < batch; start += TERMINAL_SAMPLE_CHUNK) {
      paramsList.push(this.encodeAllInTableValues1326NoPermBothPlayersChunk(
        encoder,
        this.allInTableValues1326NoPermBothPlayersPipeline,
        tree,
        nodeIndices,
        tablePacked,
        scaleFactors,
        beliefs,
        values,
        tree.handCard0,
        tree.handCard1,
        Math.min(TERMINAL_SAMPLE_CHUNK, batch - start),
        tableScale,
        start,
      ));
    }
    return paramsList;
  }

  private encodeAllInTableValues1326NoPermBothPlayersChunk(
    encoder: GPUCommandEncoder,
    pipeline: GPUComputePipeline,
    tree: SparseGpuTreeBuffers,
    nodeIndices: GPUBuffer,
    tablePacked: GPUBuffer,
    scaleFactors: GPUBuffer,
    beliefs: GPUBuffer,
    values: GPUBuffer,
    aux0: GPUBuffer,
    aux1: GPUBuffer,
    batch: number,
    tableScale: number,
    sampleStart: number,
  ): GPUBuffer {
    const words = new ArrayBuffer(32);
    const u32 = new Uint32Array(words);
    const f32 = new Float32Array(words);
    u32[0] = tree.numHands;
    u32[1] = batch;
    f32[4] = tableScale;
    const params = makeUniformBuffer(this.device, u32);
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: nodeIndices, offset: sampleStart * 4 } },
        { binding: 1, resource: { buffer: tree.allowedMask } },
        { binding: 2, resource: { buffer: aux0 } },
        { binding: 3, resource: { buffer: aux1 } },
        { binding: 4, resource: { buffer: tablePacked } },
        { binding: 6, resource: { buffer: scaleFactors, offset: sampleStart * 2 * 4 } },
        { binding: 7, resource: { buffer: beliefs } },
        { binding: 8, resource: { buffer: values } },
        { binding: 9, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      pipeline,
      bindGroup,
      Math.ceil((batch * 1326) / 64),
    );
    return params;
  }

  encodeAllInTableValuesReference(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    nodeIndices: GPUBuffer,
    tablePacked: GPUBuffer,
    comboPerms: GPUBuffer,
    scaleFactors: GPUBuffer,
    beliefs: GPUBuffer,
    values: GPUBuffer,
    batch: number,
    tableScale: number,
    permId: number,
    hasPerm: boolean,
  ): GPUBuffer {
    const pipeline = this.allInTableValuesPipeline;
    if (!pipeline) {
      throw new Error(
        "reference all-in table values require maxStorageBuffersPerShaderStage >= 9",
      );
    }
    const words = new ArrayBuffer(32);
    const u32 = new Uint32Array(words);
    const f32 = new Float32Array(words);
    u32[0] = tree.numHands;
    u32[1] = batch;
    u32[2] = permId;
    u32[3] = hasPerm ? 1 : 0;
    f32[4] = tableScale;
    const params = makeUniformBuffer(this.device, u32);
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: nodeIndices } },
        { binding: 1, resource: { buffer: tree.allowedMask } },
        { binding: 2, resource: { buffer: tree.handCard0 } },
        { binding: 3, resource: { buffer: tree.handCard1 } },
        { binding: 4, resource: { buffer: tablePacked } },
        { binding: 5, resource: { buffer: comboPerms } },
        { binding: 6, resource: { buffer: scaleFactors } },
        { binding: 7, resource: { buffer: beliefs } },
        { binding: 8, resource: { buffer: values } },
        { binding: 9, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      pipeline,
      bindGroup,
      Math.ceil((batch * 2 * tree.numHands) / 64),
    );
    return params;
  }

  encodeAllInTableValues1326NoPerm(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    nodeIndices: GPUBuffer,
    tablePacked: GPUBuffer,
    comboPerms: GPUBuffer,
    scaleFactors: GPUBuffer,
    beliefs: GPUBuffer,
    values: GPUBuffer,
    batch: number,
    tableScale: number,
    permId: number,
    hasPerm: boolean,
  ): GPUBuffer {
    if (tree.numHands !== 1326 || hasPerm) {
      throw new Error("1326 no-permutation all-in values require 1326 hands without permutations");
    }
    const words = new ArrayBuffer(32);
    const u32 = new Uint32Array(words);
    const f32 = new Float32Array(words);
    u32[0] = tree.numHands;
    u32[1] = batch;
    u32[2] = permId;
    u32[3] = hasPerm ? 1 : 0;
    f32[4] = tableScale;
    const params = makeUniformBuffer(this.device, u32);
    const bindGroup = this.device.createBindGroup({
      layout: this.allInTableValues1326NoPermPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: nodeIndices } },
        { binding: 1, resource: { buffer: tree.allowedMask } },
        { binding: 2, resource: { buffer: tree.handCard0 } },
        { binding: 3, resource: { buffer: tree.handCard1 } },
        { binding: 4, resource: { buffer: tablePacked } },
        { binding: 6, resource: { buffer: scaleFactors } },
        { binding: 7, resource: { buffer: beliefs } },
        { binding: 8, resource: { buffer: values } },
        { binding: 9, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.allInTableValues1326NoPermPipeline,
      bindGroup,
      Math.ceil((batch * 2652) / 64),
    );
    return params;
  }

  encodeAllInTableValues1326NoPermBothPlayers(
    encoder: GPUCommandEncoder,
    tree: SparseGpuTreeBuffers,
    nodeIndices: GPUBuffer,
    tablePacked: GPUBuffer,
    comboPerms: GPUBuffer,
    scaleFactors: GPUBuffer,
    beliefs: GPUBuffer,
    values: GPUBuffer,
    batch: number,
    tableScale: number,
    permId: number,
    hasPerm: boolean,
  ): GPUBuffer {
    if (tree.numHands !== 1326 || hasPerm) {
      throw new Error("both-player all-in values require 1326 hands without permutations");
    }
    const words = new ArrayBuffer(32);
    const u32 = new Uint32Array(words);
    const f32 = new Float32Array(words);
    u32[0] = tree.numHands;
    u32[1] = batch;
    u32[2] = permId;
    u32[3] = hasPerm ? 1 : 0;
    f32[4] = tableScale;
    const params = makeUniformBuffer(this.device, u32);
    const bindGroup = this.device.createBindGroup({
      layout: this.allInTableValues1326NoPermBothPlayersPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: nodeIndices } },
        { binding: 1, resource: { buffer: tree.allowedMask } },
        { binding: 2, resource: { buffer: tree.handCard0 } },
        { binding: 3, resource: { buffer: tree.handCard1 } },
        { binding: 4, resource: { buffer: tablePacked } },
        { binding: 6, resource: { buffer: scaleFactors } },
        { binding: 7, resource: { buffer: beliefs } },
        { binding: 8, resource: { buffer: values } },
        { binding: 9, resource: { buffer: params } },
      ],
    });
    this.encode(
      encoder,
      this.allInTableValues1326NoPermBothPlayersPipeline,
      bindGroup,
      Math.ceil((batch * 1326) / 64),
    );
    return params;
  }

  private encode(
    encoder: GPUCommandEncoder,
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    x: number,
    y = 1,
  ): void {
    dispatchCompute(encoder, pipeline, bindGroup, x, y);
  }

  private pipeline(source: string, label: string): GPUComputePipeline {
    return createComputePipeline(this.device, source, label);
  }
}
