import type {
  BetterFfnWebGpuModel,
  ExactBelief,
  PreparedBatchFeatures,
} from "./betterFfnWebGpuModel.js";
import {
  computeAllInTableValues,
  packInt16Table,
  type AllInTableContext,
} from "./allInTables.js";
import { normalizeBeliefVector } from "./beliefs.js";
import { HAND_COMBOS, handOverlapsCards } from "./cards.js";
import {
  makeEmptyStorageBuffer,
  makeStorageBuffer,
  readFloatBuffer,
} from "./gpuBuffers.js";
import {
  NUM_HANDS,
  PublicHunlEnv,
  showdownTerminalRankCodes,
  showdownTerminalValues,
} from "./hunlEnv.js";
import {
  SparseCfrGpuKernels,
  type SparseGpuTreeData,
  type SparseGpuTreeBuffers,
} from "./sparseCfrKernels.js";
import type { LocalSolveResult, PlayerIndex } from "./types.js";

const EPS = 1.0e-8;

const HAND_CARD0 = new Uint8Array(NUM_HANDS);
const HAND_CARD1 = new Uint8Array(NUM_HANDS);
for (let hand = 0; hand < NUM_HANDS; hand += 1) {
  const combo = HAND_COMBOS[hand]!;
  HAND_CARD0[hand] = combo[0];
  HAND_CARD1[hand] = combo[1];
}

interface SparseNode {
  env: PublicHunlEnv;
  parent: number;
  actionFromParent: number;
  reward: number;
  depth: number;
  legalMask: number[];
  allowedMask: Uint8Array;
  children: number[];
  actionToChild: Int32Array;
  leaf: boolean;
  terminal: boolean;
  newStreet: boolean;
}

interface SparseTree {
  nodes: SparseNode[];
  depthOffsets: number[];
  treeDepth: number;
}

interface SparseLeafGpuBatch {
  modelEnvs: PublicHunlEnv[];
  modelNodeIndices: Uint32Array<ArrayBuffer>;
  showdownNodeIndices: Uint32Array<ArrayBuffer>;
  showdownRankCodes: Uint32Array<ArrayBuffer>;
  showdownPayoffs: Float32Array<ArrayBuffer>;
  allInNodeIndices: Uint32Array<ArrayBuffer>;
  allInScaleFactors: Float32Array<ArrayBuffer>;
}

interface SparseStaticGpuData {
  treeBuffers: SparseGpuTreeBuffers;
  leafBatch: SparseLeafGpuBatch;
  foldTerminalValues: Float32Array;
  modelLeafNodeBuffer: GPUBuffer;
  showdownNodeBuffer: GPUBuffer;
  showdownRankBuffer: GPUBuffer;
  showdownPayoffBuffer: GPUBuffer;
  allInNodeBuffer: GPUBuffer;
  allInScaleBuffer: GPUBuffer;
  allInTableBuffer?: GPUBuffer;
  allInComboPermBuffer?: GPUBuffer;
  allInContext?: AllInTableContext;
  preparedLeafFeatures?: PreparedBatchFeatures;
  dispose: () => void;
}

interface SparseTreeCacheEntry {
  tree: SparseTree;
  initialStates: Map<string, SparseInitialState>;
  gpu?: SparseStaticGpuData;
}

interface SparseInitialState {
  policy: Float32Array;
  beliefs: Float32Array;
}

export interface SparseResolveOptions {
  depth: number;
  iterations: number;
  cfrAvg?: boolean;
  selectedAction?: number;
  readPolicy?: boolean;
  readActionProbs?: boolean;
  readBeliefs?: boolean;
}

export class SparseCfrResolver {
  readonly model: BetterFfnWebGpuModel;
  readonly numActions: number;
  private readonly gpuKernels?: SparseCfrGpuKernels;
  private readonly treeCache = new Map<string, SparseTreeCacheEntry>();

  constructor(model: BetterFfnWebGpuModel) {
    this.model = model;
    this.numActions = model.manifest.architecture.numActions;
    if (model.device) {
      this.gpuKernels = new SparseCfrGpuKernels(model.device);
    }
  }

  async solve(
    env: PublicHunlEnv,
    inputBeliefs: Float32Array<ArrayBufferLike>,
    options: SparseResolveOptions,
  ): Promise<LocalSolveResult> {
    if (!Number.isInteger(options.depth) || options.depth <= 0) {
      throw new Error("depth must be a positive integer");
    }
    if (!Number.isInteger(options.iterations) || options.iterations <= 0) {
      throw new Error("iterations must be a positive integer");
    }
    const cfrAvg = options.cfrAvg ?? true;

    const cachedTree = this.cachedTree(env, options.depth);
    const tree = cachedTree.tree;
    const allInContext = await this.resolveAllInContext(tree);
    if (this.gpuKernels && !cachedTree.gpu) {
      cachedTree.gpu = this.createStaticGpuData(tree, allInContext);
    }
    const totalNodes = tree.nodes.length;
    const policy = new Float32Array(totalNodes * NUM_HANDS);
    const policyAvg = new Float32Array(totalNodes * NUM_HANDS);
    const cumulativeRegrets = new Float32Array(totalNodes * NUM_HANDS);
    const avgNumerator = new Float32Array(totalNodes * NUM_HANDS);
    const avgDenominator = new Float32Array(totalNodes * NUM_HANDS);
    const beliefs = new Float32Array(totalNodes * 2 * NUM_HANDS);
    const beliefsAvg = new Float32Array(totalNodes * 2 * NUM_HANDS);

    const rootBeliefs = this.rootBeliefsForEnv(tree.nodes[0]!.env, inputBeliefs);
    const exactBelief = this.exactBelief(rootBeliefs);
    this.copyBeliefsToNode(rootBeliefs, beliefs, 0);
    this.copyBeliefsToNode(rootBeliefs, beliefsAvg, 0);

    const initialStateKey = this.beliefsCacheKey(rootBeliefs);
    const cachedInitialState = cachedTree.initialStates.get(initialStateKey);
    if (cachedInitialState) {
      policy.set(cachedInitialState.policy);
      beliefs.set(cachedInitialState.beliefs);
    } else {
      await this.initializePolicyAndBeliefs(tree, rootBeliefs, policy, beliefs);
      cachedTree.initialStates.set(initialStateKey, {
        policy: new Float32Array(policy),
        beliefs: new Float32Array(beliefs),
      });
    }
    policyAvg.set(policy);
    beliefsAvg.set(beliefs);

    if (!(this.gpuKernels && cachedTree.gpu)) {
      const latestValues = new Float32Array(totalNodes * 2 * NUM_HANDS);
      await this.setLeafValues(
        tree,
        cfrAvg ? beliefsAvg : beliefs,
        latestValues,
        allInContext,
      );
      let values = this.computeExpectedValues(tree, policy, beliefs, latestValues);

      for (let t = 0; t < options.iterations; t += 1) {
        this.accumulateRegrets(
          tree,
          cfrAvg ? beliefsAvg : beliefs,
          values,
          cumulativeRegrets,
        );
        this.updatePolicy(tree, cumulativeRegrets, policy);

        const current = this.propagateReachAndBeliefs(tree, rootBeliefs, policy);
        beliefs.set(current.beliefs);
        this.updateAveragePolicy(
          tree,
          policy,
          current.reach,
          avgNumerator,
          avgDenominator,
          policyAvg,
        );
        if (cfrAvg) {
          const avg = this.propagateReachAndBeliefs(tree, rootBeliefs, policyAvg);
          beliefsAvg.set(avg.beliefs);
        }

        await this.setLeafValues(
          tree,
          cfrAvg ? beliefsAvg : beliefs,
          latestValues,
          allInContext,
        );
        values = this.computeExpectedValues(tree, policy, beliefs, latestValues);
      }
    } else {
      await this.solveIterationsGpuResident(
        tree,
        rootBeliefs,
        policy,
        policyAvg,
        beliefs,
        beliefsAvg,
        cumulativeRegrets,
        avgNumerator,
        avgDenominator,
        cachedTree.gpu,
        options.iterations,
        cfrAvg,
        (options.readPolicy ?? true) ||
          (options.readActionProbs ?? true) ||
          (options.selectedAction !== undefined && (options.readBeliefs ?? true)),
        exactBelief,
      );
    }

    const readPolicy = options.readPolicy ?? true;
    const readActionProbs = options.readActionProbs ?? true;
    const readBeliefs = options.readBeliefs ?? true;
    const rootPolicy = readPolicy
      ? this.rootPolicy(tree, policyAvg)
      : new Float32Array(0);
    const actionProbs = readActionProbs
      ? this.rootActionProbs(tree, rootBeliefs, policyAvg)
      : new Float32Array(0);
    const result: LocalSolveResult = {
      policy: rootPolicy,
      actionProbs,
    };

    if (options.selectedAction !== undefined && readBeliefs) {
      result.beliefsAfter = this.nextBeliefs(
        tree,
        rootBeliefs,
        policyAvg,
        options.selectedAction,
      );
    }
    return result;
  }

  dispose(): void {
    for (const entry of this.treeCache.values()) {
      entry.gpu?.dispose();
    }
    this.treeCache.clear();
  }

  private cachedTree(rootEnv: PublicHunlEnv, maxDepth: number): SparseTreeCacheEntry {
    const key = this.treeCacheKey(rootEnv, maxDepth);
    const cached = this.treeCache.get(key);
    if (cached) return cached;
    const tree = this.buildTree(rootEnv, maxDepth);
    const entry: SparseTreeCacheEntry = { tree, initialStates: new Map() };
    this.treeCache.set(key, entry);
    return entry;
  }

  private createStaticGpuData(
    tree: SparseTree,
    allInContext?: AllInTableContext,
  ): SparseStaticGpuData {
    if (!this.gpuKernels) {
      throw new Error("GPU sparse kernels are not initialized");
    }
    const device = this.model.device;
    const treeBuffers = this.gpuKernels.createTreeBuffers(this.gpuTreeData(tree));
    const leafBatch = this.leafGpuBatch(tree, allInContext);
    const foldTerminalValues = this.foldTerminalLeafValues(tree);
    const modelLeafNodeBuffer = makeStorageBuffer(device, leafBatch.modelNodeIndices);
    const showdownNodeBuffer = makeStorageBuffer(device, leafBatch.showdownNodeIndices);
    const showdownRankBuffer = makeStorageBuffer(device, leafBatch.showdownRankCodes);
    const showdownPayoffBuffer = makeStorageBuffer(device, leafBatch.showdownPayoffs);
    const allInNodeBuffer = makeStorageBuffer(device, leafBatch.allInNodeIndices);
    const allInScaleBuffer = makeStorageBuffer(device, leafBatch.allInScaleFactors);
    const allInTableBuffer = allInContext
      ? makeStorageBuffer(device, packInt16Table(allInContext.table))
      : undefined;
    const identityPerms = new Uint32Array(NUM_HANDS);
    for (let hand = 0; hand < NUM_HANDS; hand += 1) identityPerms[hand] = hand;
    const allInComboPermBuffer = allInContext
      ? makeStorageBuffer(device, allInContext.comboPerms ?? identityPerms)
      : undefined;
    const preparedLeafFeatures =
      leafBatch.modelEnvs.length > 0 && this.model.prepareBatchFeatures
        ? this.model.prepareBatchFeatures(leafBatch.modelEnvs)
        : undefined;
    return {
      treeBuffers,
      leafBatch,
      foldTerminalValues,
      modelLeafNodeBuffer,
      showdownNodeBuffer,
      showdownRankBuffer,
      showdownPayoffBuffer,
      allInNodeBuffer,
      allInScaleBuffer,
      ...(allInTableBuffer ? { allInTableBuffer } : {}),
      ...(allInComboPermBuffer ? { allInComboPermBuffer } : {}),
      ...(allInContext ? { allInContext } : {}),
      ...(preparedLeafFeatures ? { preparedLeafFeatures } : {}),
      dispose: () => {
        treeBuffers.dispose();
        modelLeafNodeBuffer.destroy();
        showdownNodeBuffer.destroy();
        showdownRankBuffer.destroy();
        showdownPayoffBuffer.destroy();
        allInNodeBuffer.destroy();
        allInScaleBuffer.destroy();
        allInTableBuffer?.destroy();
        allInComboPermBuffer?.destroy();
        preparedLeafFeatures?.dispose();
      },
    };
  }

  private treeCacheKey(rootEnv: PublicHunlEnv, maxDepth: number): string {
    return JSON.stringify([
      maxDepth,
      rootEnv.betBins,
      rootEnv.sb,
      rootEnv.bb,
      rootEnv.flopShowdown,
      rootEnv.button,
      rootEnv.street,
      rootEnv.toAct,
      rootEnv.lastToAct,
      rootEnv.pot,
      rootEnv.minRaise,
      rootEnv.actionsThisRound,
      rootEnv.actionsLastRound,
      rootEnv.actedSinceReset,
      rootEnv.stacks,
      rootEnv.committed,
      rootEnv.startingStacks,
      rootEnv.scale,
      rootEnv.isAllin,
      rootEnv.hasFolded,
      rootEnv.done,
      rootEnv.winner,
      rootEnv.deck,
      rootEnv.deckPos,
      rootEnv.boardIndices,
      rootEnv.holeIndices,
    ]);
  }

  private async resolveAllInContext(
    tree: SparseTree,
  ): Promise<AllInTableContext | undefined> {
    const provider = this.model.allInTableProvider;
    if (!provider) return undefined;
    const root = tree.nodes[0]!.env;
    if (root.street > 2) return undefined;
    return await provider.tableForRoot(root);
  }

  private isAllInCallLeaf(tree: SparseTree, nodeIndex: number): boolean {
    const node = tree.nodes[nodeIndex]!;
    if (!node.leaf || node.env.done || !node.newStreet || node.parent < 0) return false;
    if (node.actionFromParent !== 1) return false;
    const parent = tree.nodes[node.parent]!;
    const actor = parent.env.toAct;
    const opp = (1 - actor) as PlayerIndex;
    return parent.env.isAllin[opp] && parent.env.committed[opp] > parent.env.committed[actor];
  }

  private beliefsCacheKey(beliefs: Float32Array<ArrayBuffer>): string {
    return Array.from(
      new Uint32Array(beliefs.buffer, beliefs.byteOffset, beliefs.length),
    ).join(",");
  }

  private exactBelief(beliefs: Float32Array<ArrayBuffer>): ExactBelief | undefined {
    for (let player = 0; player < 2; player += 1) {
      const offset = player * NUM_HANDS;
      let hand = -1;
      let mass = 0;
      for (let i = 0; i < NUM_HANDS; i += 1) {
        const value = beliefs[offset + i]!;
        if (value <= EPS) continue;
        if (hand >= 0) {
          hand = -1;
          break;
        }
        hand = i;
        mass = value;
      }
      if (hand >= 0 && Math.abs(mass - 1) <= 1.0e-6) {
        return { player, hand };
      }
    }
    return undefined;
  }

  private buildTree(rootEnv: PublicHunlEnv, maxDepth: number): SparseTree {
    const nodes: SparseNode[] = [
      this.makeNode(rootEnv.clone(), -1, -1, 0, 0),
    ];
    const depthOffsets = [0, 1];

    for (let depth = 0; depth < maxDepth; depth += 1) {
      const start = depthOffsets[depth]!;
      const end = depthOffsets[depth + 1]!;
      for (let nodeIndex = start; nodeIndex < end; nodeIndex += 1) {
        const node = nodes[nodeIndex]!;
        if (this.shouldStop(node, maxDepth)) {
          continue;
        }
        const callFoldOnly = depth >= maxDepth - 1;
        for (let action = 0; action < this.numActions; action += 1) {
          if (callFoldOnly && action > 1) continue;
          if (!node.legalMask[action]) continue;
          const childEnv = node.env.clone();
          const step = childEnv.stepBin(action, {
            amounts: node.env.legalBinsAmountAndMask().amounts,
            mask: node.legalMask,
          });
          const childIndex = nodes.length;
          node.children.push(childIndex);
          node.actionToChild[action] = childIndex;
          nodes.push(
            this.makeNode(childEnv, nodeIndex, action, step.reward, depth + 1),
          );
        }
      }
      if (nodes.length === end) {
        break;
      }
      depthOffsets.push(nodes.length);
    }

    const treeDepth = Math.max(0, depthOffsets.length - 2);
    for (const node of nodes) {
      node.leaf = node.children.length === 0 || this.shouldStop(node, maxDepth);
    }
    return { nodes, depthOffsets, treeDepth };
  }

  private makeNode(
    env: PublicHunlEnv,
    parent: number,
    actionFromParent: number,
    reward: number,
    depth: number,
  ): SparseNode {
    const legal = env.legalBinsAmountAndMask();
    const actionToChild = new Int32Array(this.numActions);
    actionToChild.fill(-1);
    const newStreet = depth > 0 && env.actionsThisRound === 0 && !env.done;
    return {
      env,
      parent,
      actionFromParent,
      reward,
      depth,
      legalMask: legal.mask,
      allowedMask: this.allowedMask(env),
      children: [],
      actionToChild,
      leaf: false,
      terminal: env.done,
      newStreet,
    };
  }

  private shouldStop(node: SparseNode, maxDepth: number): boolean {
    return node.env.done || node.newStreet || node.depth >= maxDepth;
  }

  private async initializePolicyAndBeliefs(
    tree: SparseTree,
    rootBeliefs: Float32Array<ArrayBuffer>,
    policy: Float32Array,
    beliefs: Float32Array,
  ): Promise<void> {
    const beliefSize = 2 * NUM_HANDS;
    for (let depth = 0; depth < tree.treeDepth; depth += 1) {
      const start = tree.depthOffsets[depth]!;
      const end = tree.depthOffsets[depth + 1]!;
      const batchEnvs: PublicHunlEnv[] = [];
      const batchNodeIndices: number[] = [];
      for (let nodeIndex = start; nodeIndex < end; nodeIndex += 1) {
        const node = tree.nodes[nodeIndex]!;
        if (node.leaf || node.children.length === 0) continue;
        batchEnvs.push(node.env);
        batchNodeIndices.push(nodeIndex);
      }
      if (batchEnvs.length === 0) continue;

      const batchedBeliefs = new Float32Array(batchEnvs.length * beliefSize);
      for (let i = 0; i < batchNodeIndices.length; i += 1) {
        batchedBeliefs.set(
          beliefs.subarray(
            batchNodeIndices[i]! * beliefSize,
            (batchNodeIndices[i]! + 1) * beliefSize,
          ),
          i * beliefSize,
        );
      }
      const prediction = await this.model.predictBatch(batchEnvs, batchedBeliefs, {
        includePolicy: true,
      });
      if (!prediction.policyLogits) {
        throw new Error("model did not return policy logits");
      }
      const stride = NUM_HANDS * this.numActions;
      for (let i = 0; i < batchNodeIndices.length; i += 1) {
        const nodeIndex = batchNodeIndices[i]!;
        const node = tree.nodes[nodeIndex]!;
        const slice = prediction.policyLogits.subarray(i * stride, (i + 1) * stride);
        const modelPolicy = this.softmaxPolicy(slice, node.legalMask);
        this.writeChildPolicy(node, modelPolicy, policy);
        for (const childIndex of node.children) {
          this.propagateChildBelief(
            tree.nodes[childIndex]!,
            node.env.toAct,
            nodeIndex,
            childIndex,
            policy,
            beliefs,
          );
        }
      }
    }

    this.copyBeliefsToNode(rootBeliefs, beliefs, 0);
  }

  private softmaxPolicy(
    logits: Float32Array<ArrayBufferLike>,
    legalMask: readonly number[],
  ): Float32Array<ArrayBuffer> {
    const out = new Float32Array(NUM_HANDS * this.numActions);
    let legalCount = 0;
    for (let action = 0; action < this.numActions; action += 1) {
      legalCount += legalMask[action] ? 1 : 0;
    }
    if (legalCount === 0) return out;

    for (let hand = 0; hand < NUM_HANDS; hand += 1) {
      const base = hand * this.numActions;
      let maxLogit = -Infinity;
      for (let action = 0; action < this.numActions; action += 1) {
        if (!legalMask[action]) continue;
        const value = logits[base + action]!;
        if (value > maxLogit) maxLogit = value;
      }
      let denom = 0;
      for (let action = 0; action < this.numActions; action += 1) {
        if (!legalMask[action]) continue;
        const expValue = Math.exp(logits[base + action]! - maxLogit);
        out[base + action] = expValue;
        denom += expValue;
      }
      if (denom <= EPS) {
        const uniform = 1 / legalCount;
        for (let action = 0; action < this.numActions; action += 1) {
          out[base + action] = legalMask[action] ? uniform : 0;
        }
      } else {
        for (let action = 0; action < this.numActions; action += 1) {
          out[base + action] = legalMask[action] ? out[base + action]! / denom : 0;
        }
      }
    }
    return out;
  }

  private writeChildPolicy(
    node: SparseNode,
    modelPolicy: Float32Array<ArrayBuffer>,
    policy: Float32Array,
  ): void {
    for (let action = 0; action < this.numActions; action += 1) {
      const childIndex = node.actionToChild[action]!;
      if (childIndex < 0) continue;
      const childBase = childIndex * NUM_HANDS;
      for (let hand = 0; hand < NUM_HANDS; hand += 1) {
        policy[childBase + hand] = modelPolicy[hand * this.numActions + action]!;
      }
    }
  }

  private async solveIterationsGpuResident(
    tree: SparseTree,
    rootBeliefs: Float32Array<ArrayBuffer>,
    policy: Float32Array,
    policyAvg: Float32Array,
    beliefs: Float32Array,
    beliefsAvg: Float32Array,
    cumulativeRegrets: Float32Array,
    avgNumerator: Float32Array,
    avgDenominator: Float32Array,
    staticGpu: SparseStaticGpuData,
    iterations: number,
    cfrAvg: boolean,
    readPolicyAvg: boolean,
    exactBelief?: ExactBelief,
  ): Promise<void> {
    if (!this.gpuKernels) {
      throw new Error("GPU sparse kernels are not initialized");
    }

    const device = this.model.device;
    const totalNodes = tree.nodes.length;
    const valueCount = totalNodes * 2 * NUM_HANDS;
    const policyCount = totalNodes * NUM_HANDS;
    const { treeBuffers, leafBatch } = staticGpu;
    const rootReach = this.rootReach();
    const policyBuffer = makeStorageBuffer(device, policy);
    const policyAvgBuffer = makeStorageBuffer(device, policyAvg);
    const regretsBuffer = makeEmptyStorageBuffer(device, cumulativeRegrets.length);
    const avgNumeratorBuffer = makeEmptyStorageBuffer(device, avgNumerator.length);
    const avgDenominatorBuffer = makeEmptyStorageBuffer(device, avgDenominator.length);
    const beliefsBuffer = makeStorageBuffer(device, beliefs);
    const beliefsAvgBuffer = cfrAvg
      ? makeStorageBuffer(device, beliefsAvg)
      : undefined;
    const valuesBuffer = makeStorageBuffer(device, staticGpu.foldTerminalValues);
    const reachBuffer = makeEmptyStorageBuffer(device, valueCount);
    const denomBuffer = makeEmptyStorageBuffer(device, totalNodes * 2);
    const opponentPolicyBuffer = makeEmptyStorageBuffer(device, policyCount);
    const opponentPolicyAggregatesBuffer = makeEmptyStorageBuffer(
      device,
      totalNodes * 106,
    );
    const regretWeightsBuffer = makeEmptyStorageBuffer(device, policyCount);
    const regretWeightAggregatesBuffer = makeEmptyStorageBuffer(device, totalNodes * 53);
    const modelLeafBeliefsBuffer = makeEmptyStorageBuffer(
      device,
      leafBatch.modelNodeIndices.length * 2 * NUM_HANDS,
    );
    const pendingLeafDisposals: Array<() => void> = [];
    device.queue.writeBuffer(reachBuffer, 0, rootReach);
    device.queue.writeBuffer(beliefsBuffer, 0, rootBeliefs);
    if (beliefsAvgBuffer) {
      device.queue.writeBuffer(beliefsAvgBuffer, 0, rootBeliefs);
    }

    const profile = Boolean(globalThis.process?.env?.P2_PROFILE);
    const phaseTotals = new Map<string, number>();
    const phaseCounts = new Map<string, number>();
    const time = async <T>(label: string, fn: () => T | Promise<T>): Promise<T> => {
      if (!profile) return await fn();
      const start = performance.now();
      const out = await fn();
      await device.queue.onSubmittedWorkDone();
      const elapsed = performance.now() - start;
      phaseTotals.set(label, (phaseTotals.get(label) ?? 0) + elapsed);
      phaseCounts.set(label, (phaseCounts.get(label) ?? 0) + 1);
      return out;
    };

    try {
      const initialLeafDispose = await time("leafValues0", () =>
        this.updateLeafValuesGpuBridge(
          treeBuffers,
          leafBatch,
          cfrAvg ? beliefsAvgBuffer! : beliefsBuffer,
          valuesBuffer,
          staticGpu.modelLeafNodeBuffer,
          modelLeafBeliefsBuffer,
          staticGpu.showdownNodeBuffer,
          staticGpu.showdownRankBuffer,
          staticGpu.showdownPayoffBuffer,
          staticGpu.allInNodeBuffer,
          staticGpu.allInScaleBuffer,
          staticGpu.allInTableBuffer,
          staticGpu.allInComboPermBuffer,
          staticGpu.allInContext,
          staticGpu.preparedLeafFeatures,
          exactBelief,
        ),
      );
      pendingLeafDisposals.push(initialLeafDispose);
      await time("backup0", () => {
        this.encodeExpectedValuesGpuResident(
          tree,
          treeBuffers,
          policyBuffer,
          beliefsBuffer,
          opponentPolicyBuffer,
          opponentPolicyAggregatesBuffer,
          valuesBuffer,
        );
      });

      for (let t = 0; t < iterations; t += 1) {
        const regretBeliefsBuffer = cfrAvg ? beliefsAvgBuffer! : beliefsBuffer;
        await time("iterationPrefix", () => {
          this.encodeIterationPrefixGpuResident(
            tree,
            treeBuffers,
            regretBeliefsBuffer,
            valuesBuffer,
            regretWeightsBuffer,
            regretWeightAggregatesBuffer,
            regretsBuffer,
            policyBuffer,
            reachBuffer,
            beliefsBuffer,
            denomBuffer,
            avgNumeratorBuffer,
            avgDenominatorBuffer,
            policyAvgBuffer,
            cfrAvg ? beliefsAvgBuffer! : undefined,
          );
        });

        if (t + 1 < iterations) {
          const disposeLeafPrediction = await time("leafValues", () =>
            this.updateLeafValuesGpuBridge(
              treeBuffers,
              leafBatch,
              cfrAvg ? beliefsAvgBuffer! : beliefsBuffer,
              valuesBuffer,
              staticGpu.modelLeafNodeBuffer,
              modelLeafBeliefsBuffer,
              staticGpu.showdownNodeBuffer,
              staticGpu.showdownRankBuffer,
              staticGpu.showdownPayoffBuffer,
              staticGpu.allInNodeBuffer,
              staticGpu.allInScaleBuffer,
              staticGpu.allInTableBuffer,
              staticGpu.allInComboPermBuffer,
              staticGpu.allInContext,
              staticGpu.preparedLeafFeatures,
              exactBelief,
            ),
          );
          pendingLeafDisposals.push(disposeLeafPrediction);
          await time("backup", () => {
            this.encodeExpectedValuesGpuResident(
              tree,
              treeBuffers,
              policyBuffer,
              beliefsBuffer,
              opponentPolicyBuffer,
              opponentPolicyAggregatesBuffer,
              valuesBuffer,
            );
          });
        }
      }
      if (profile) {
        const stats = Array.from(phaseTotals.entries()).map(([label, total]) => ({
          phase: label,
          totalMs: total,
          count: phaseCounts.get(label) ?? 0,
          avgMs: total / (phaseCounts.get(label) ?? 1),
        }));
        const treeStats = {
          totalNodes: tree.nodes.length,
          treeDepth: tree.treeDepth,
          showdownLeaves: leafBatch.showdownNodeIndices.length,
          allInLeaves: leafBatch.allInNodeIndices.length,
          modelLeaves: leafBatch.modelNodeIndices.length,
          street: tree.nodes[0]!.env.street,
        };
        console.error("PROFILE", JSON.stringify({ tree: treeStats, phases: stats }));
      }

      if (readPolicyAvg) {
        policyAvg.set(await readFloatBuffer(device, policyAvgBuffer, policyAvg.length));
      }
      for (const dispose of pendingLeafDisposals.splice(0)) dispose();
    } finally {
      for (const dispose of pendingLeafDisposals.splice(0)) dispose();
      policyBuffer.destroy();
      policyAvgBuffer.destroy();
      regretsBuffer.destroy();
      avgNumeratorBuffer.destroy();
      avgDenominatorBuffer.destroy();
      beliefsBuffer.destroy();
      beliefsAvgBuffer?.destroy();
      valuesBuffer.destroy();
      reachBuffer.destroy();
      denomBuffer.destroy();
      opponentPolicyBuffer.destroy();
      opponentPolicyAggregatesBuffer.destroy();
      regretWeightsBuffer.destroy();
      regretWeightAggregatesBuffer.destroy();
      modelLeafBeliefsBuffer.destroy();
    }
  }

  private async updateLeafValuesGpuBridge(
    treeBuffers: SparseGpuTreeBuffers,
    leafBatch: SparseLeafGpuBatch,
    leafBeliefsBuffer: GPUBuffer,
    valuesBuffer: GPUBuffer,
    modelLeafNodeBuffer: GPUBuffer,
    modelLeafBeliefsBuffer: GPUBuffer,
    showdownNodeBuffer: GPUBuffer,
    showdownRankBuffer: GPUBuffer,
    showdownPayoffBuffer: GPUBuffer,
    allInNodeBuffer: GPUBuffer,
    allInScaleBuffer: GPUBuffer,
    allInTableBuffer?: GPUBuffer,
    allInComboPermBuffer?: GPUBuffer,
    allInContext?: AllInTableContext,
    preparedLeafFeatures?: PreparedBatchFeatures,
    exactBelief?: ExactBelief,
  ): Promise<() => void> {
    if (!this.gpuKernels) return () => undefined;
    const device = this.model.device;

    if (leafBatch.showdownNodeIndices.length > 0) {
      const showdownEncoder = device.createCommandEncoder();
      const showdownParams = this.gpuKernels.encodeShowdownValues(
        showdownEncoder,
        treeBuffers,
        showdownNodeBuffer,
        showdownRankBuffer,
        showdownPayoffBuffer,
        leafBeliefsBuffer,
        valuesBuffer,
        leafBatch.showdownNodeIndices.length,
      );
      device.queue.submit([showdownEncoder.finish()]);
      showdownParams.destroy();
    }

    if (
      leafBatch.allInNodeIndices.length > 0 &&
      allInContext &&
      allInTableBuffer &&
      allInComboPermBuffer
    ) {
      const allInEncoder = device.createCommandEncoder();
      const allInParams = this.gpuKernels.encodeAllInTableValues(
        allInEncoder,
        treeBuffers,
        allInNodeBuffer,
        allInTableBuffer,
        allInComboPermBuffer,
        allInScaleBuffer,
        leafBeliefsBuffer,
        valuesBuffer,
        leafBatch.allInNodeIndices.length,
        allInContext.scale,
        allInContext.permId ?? 0,
        allInContext.permId !== undefined && allInContext.comboPerms !== undefined,
      );
      device.queue.submit([allInEncoder.finish()]);
      allInParams.destroy();
    }

    if (leafBatch.modelNodeIndices.length === 0) return () => undefined;

    const gatherEncoder = device.createCommandEncoder();
    const gatherParams = this.gpuKernels.encodeGatherNodeBeliefs(
      gatherEncoder,
      treeBuffers,
      modelLeafNodeBuffer,
      leafBeliefsBuffer,
      modelLeafBeliefsBuffer,
      leafBatch.modelNodeIndices.length,
    );
    device.queue.submit([gatherEncoder.finish()]);
    gatherParams.destroy();

    let scatterParams: GPUBuffer | undefined;
    const prediction = await this.model.predictBatchHandValuesGpu(
      leafBatch.modelEnvs,
      modelLeafBeliefsBuffer,
      preparedLeafFeatures,
      (encoder, handValues) => {
        scatterParams = this.gpuKernels!.encodeScatterNodeValues(
          encoder,
          treeBuffers,
          modelLeafNodeBuffer,
          handValues,
          valuesBuffer,
          leafBatch.modelNodeIndices.length,
        );
      },
      exactBelief,
    );
    scatterParams?.destroy();
    return () => prediction.dispose();
  }

  private leafGpuBatch(
    tree: SparseTree,
    allInContext?: AllInTableContext,
  ): SparseLeafGpuBatch {
    const modelEnvs: PublicHunlEnv[] = [];
    const modelNodeIndices: number[] = [];
    const showdownNodeIndices: number[] = [];
    const showdownRankCodes: number[] = [];
    const showdownPayoffs: number[] = [];
    const allInNodeIndices: number[] = [];
    const allInScaleFactors: number[] = [];
    for (let nodeIndex = 0; nodeIndex < tree.nodes.length; nodeIndex += 1) {
      const node = tree.nodes[nodeIndex]!;
      if (!node.leaf) continue;
      if (allInContext && this.isAllInCallLeaf(tree, nodeIndex)) {
        allInNodeIndices.push(nodeIndex);
        allInScaleFactors.push(
          (node.env.stacks[0] + node.env.pot - node.env.startingStacks[0]) /
            Math.max(node.env.scale, 1.0e-8),
          (node.env.stacks[1] + node.env.pot - node.env.startingStacks[1]) /
            Math.max(node.env.scale, 1.0e-8),
        );
      } else if (!node.env.done) {
        modelEnvs.push(node.env);
        modelNodeIndices.push(nodeIndex);
      } else if (!node.env.hasFolded[0] && !node.env.hasFolded[1]) {
        showdownNodeIndices.push(nodeIndex);
        showdownRankCodes.push(...showdownTerminalRankCodes(node.env));
        showdownPayoffs.push(...this.showdownPayoffs(node.env));
      }
    }
    return {
      modelEnvs,
      modelNodeIndices: new Uint32Array(modelNodeIndices),
      showdownNodeIndices: new Uint32Array(showdownNodeIndices),
      showdownRankCodes: new Uint32Array(showdownRankCodes),
      showdownPayoffs: new Float32Array(showdownPayoffs),
      allInNodeIndices: new Uint32Array(allInNodeIndices),
      allInScaleFactors: new Float32Array(allInScaleFactors),
    };
  }

  private showdownPayoffs(env: PublicHunlEnv): [number, number, number] {
    return [
      (env.stacks[0] + env.pot - env.startingStacks[0]) / env.scale,
      (env.stacks[0] - env.startingStacks[0]) / env.scale,
      (env.stacks[0] + env.pot / 2 - env.startingStacks[0]) / env.scale,
    ];
  }

  private foldTerminalLeafValues(tree: SparseTree): Float32Array {
    const values = new Float32Array(tree.nodes.length * 2 * NUM_HANDS);
    for (let nodeIndex = 0; nodeIndex < tree.nodes.length; nodeIndex += 1) {
      const node = tree.nodes[nodeIndex]!;
      if (!node.leaf || !node.env.done) continue;
      const valueBase = nodeIndex * 2 * NUM_HANDS;
      if (node.env.hasFolded[0] || node.env.hasFolded[1]) {
        values.fill(node.reward, valueBase, valueBase + NUM_HANDS);
        values.fill(-node.reward, valueBase + NUM_HANDS, valueBase + 2 * NUM_HANDS);
      }
    }
    return values;
  }

  private encodeExpectedValuesGpuResident(
    tree: SparseTree,
    treeBuffers: SparseGpuTreeBuffers,
    policyBuffer: GPUBuffer,
    beliefsBuffer: GPUBuffer,
    opponentPolicyBuffer: GPUBuffer,
    opponentPolicyAggregatesBuffer: GPUBuffer,
    valuesBuffer: GPUBuffer,
  ): void {
    if (!this.gpuKernels) return;
    const encoder = this.model.device.createCommandEncoder();
    const params: GPUBuffer[] = [
      ...this.gpuKernels.encodeComputeOpponentPolicyAggregateRange(
        encoder,
        treeBuffers,
        beliefsBuffer,
        policyBuffer,
        opponentPolicyAggregatesBuffer,
        opponentPolicyBuffer,
        1,
        tree.nodes.length,
      ),
    ];
    for (let depth = tree.treeDepth - 1; depth >= 0; depth -= 1) {
      params.push(
        this.gpuKernels.encodeBackupDepth(
          encoder,
          treeBuffers,
          policyBuffer,
          opponentPolicyBuffer,
          valuesBuffer,
          tree.depthOffsets[depth]!,
          tree.depthOffsets[depth + 1]!,
        ),
      );
    }
    this.model.device.queue.submit([encoder.finish()]);
    for (const param of params) param.destroy();
  }

  private encodeAccumulateRegretsGpuResident(
    tree: SparseTree,
    treeBuffers: SparseGpuTreeBuffers,
    beliefsAvgBuffer: GPUBuffer,
    valuesBuffer: GPUBuffer,
    regretWeightsBuffer: GPUBuffer,
    regretsBuffer: GPUBuffer,
  ): void {
    if (!this.gpuKernels) return;
    const encoder = this.model.device.createCommandEncoder();
    const regretWeightEnd = tree.depthOffsets[tree.treeDepth] ?? tree.nodes.length;
    const params = [
      this.gpuKernels.encodeComputeRegretWeightsRange(
        encoder,
        treeBuffers,
        beliefsAvgBuffer,
        regretWeightsBuffer,
        0,
        regretWeightEnd,
      ),
      this.gpuKernels.encodeAccumulateRegretsRange(
        encoder,
        treeBuffers,
        regretWeightsBuffer,
        valuesBuffer,
        regretsBuffer,
        1,
        tree.nodes.length,
      ),
    ];
    this.model.device.queue.submit([encoder.finish()]);
    for (const param of params) param.destroy();
  }

  private encodeIterationPrefixGpuResident(
    tree: SparseTree,
    treeBuffers: SparseGpuTreeBuffers,
    regretBeliefsBuffer: GPUBuffer,
    valuesBuffer: GPUBuffer,
    regretWeightsBuffer: GPUBuffer,
    regretWeightAggregatesBuffer: GPUBuffer,
    regretsBuffer: GPUBuffer,
    policyBuffer: GPUBuffer,
    reachBuffer: GPUBuffer,
    beliefsBuffer: GPUBuffer,
    denomBuffer: GPUBuffer,
    numeratorBuffer: GPUBuffer,
    denominatorBuffer: GPUBuffer,
    policyAvgBuffer: GPUBuffer,
    beliefsAvgBuffer: GPUBuffer | undefined,
  ): void {
    if (!this.gpuKernels) return;
    const encoder = this.model.device.createCommandEncoder();
    const regretWeightEnd = tree.depthOffsets[tree.treeDepth] ?? tree.nodes.length;
    const params: GPUBuffer[] = [
      ...this.gpuKernels.encodeComputeRegretWeightsAggregateRange(
        encoder,
        treeBuffers,
        regretBeliefsBuffer,
        regretWeightAggregatesBuffer,
        regretWeightsBuffer,
        0,
        regretWeightEnd,
      ),
      this.gpuKernels.encodeAccumulateRegretsRange(
        encoder,
        treeBuffers,
        regretWeightsBuffer,
        valuesBuffer,
        regretsBuffer,
        1,
        tree.nodes.length,
      ),
      this.gpuKernels.encodeRegretMatch(
        encoder,
        treeBuffers,
        regretsBuffer,
        policyBuffer,
      ),
    ];
    this.encodePropagateInto(
      encoder,
      params,
      tree,
      treeBuffers,
      policyBuffer,
      reachBuffer,
      beliefsBuffer,
      denomBuffer,
    );
    params.push(
      this.gpuKernels.encodeUpdateAveragePolicyRange(
        encoder,
        treeBuffers,
        reachBuffer,
        policyBuffer,
        numeratorBuffer,
        denominatorBuffer,
        policyAvgBuffer,
        1,
        tree.nodes.length,
      ),
    );
    if (beliefsAvgBuffer) {
      this.encodePropagateInto(
        encoder,
        params,
        tree,
        treeBuffers,
        policyAvgBuffer,
        reachBuffer,
        beliefsAvgBuffer,
        denomBuffer,
      );
    }
    this.model.device.queue.submit([encoder.finish()]);
    for (const param of params) param.destroy();
  }

  private encodeRegretMatchGpuResident(
    tree: SparseTree,
    treeBuffers: SparseGpuTreeBuffers,
    regretsBuffer: GPUBuffer,
    policyBuffer: GPUBuffer,
  ): void {
    if (!this.gpuKernels) return;
    const encoder = this.model.device.createCommandEncoder();
    const params = this.gpuKernels.encodeRegretMatch(
      encoder,
      treeBuffers,
      regretsBuffer,
      policyBuffer,
    );
    this.model.device.queue.submit([encoder.finish()]);
    params.destroy();
  }

  private encodePropagateGpuResident(
    tree: SparseTree,
    treeBuffers: SparseGpuTreeBuffers,
    rootBeliefs: Float32Array<ArrayBuffer>,
    rootReach: Float32Array<ArrayBuffer>,
    policyBuffer: GPUBuffer,
    reachBuffer: GPUBuffer,
    beliefsBuffer: GPUBuffer,
    denomBuffer: GPUBuffer,
  ): void {
    if (!this.gpuKernels) return;
    const device = this.model.device;
    device.queue.writeBuffer(reachBuffer, 0, rootReach);
    device.queue.writeBuffer(beliefsBuffer, 0, rootBeliefs);
    const encoder = device.createCommandEncoder();
    const params: GPUBuffer[] = [];
    this.encodePropagateInto(
      encoder,
      params,
      tree,
      treeBuffers,
      policyBuffer,
      reachBuffer,
      beliefsBuffer,
      denomBuffer,
    );
    device.queue.submit([encoder.finish()]);
    for (const param of params) param.destroy();
  }

  private encodePropagateInto(
    encoder: GPUCommandEncoder,
    params: GPUBuffer[],
    tree: SparseTree,
    treeBuffers: SparseGpuTreeBuffers,
    policyBuffer: GPUBuffer,
    reachBuffer: GPUBuffer,
    beliefsBuffer: GPUBuffer,
    denomBuffer: GPUBuffer,
  ): void {
    if (!this.gpuKernels) return;
    for (let depth = 0; depth < tree.treeDepth; depth += 1) {
      const start = tree.depthOffsets[depth + 1]!;
      const end = tree.depthOffsets[depth + 2]!;
      params.push(
        this.gpuKernels.encodePropagateReachDepth(
          encoder,
          treeBuffers,
          policyBuffer,
          reachBuffer,
          start,
          end,
        ),
      );
      params.push(
        this.gpuKernels.encodePropagateBeliefsDepth(
          encoder,
          treeBuffers,
          policyBuffer,
          beliefsBuffer,
          denomBuffer,
          start,
          end,
        ),
      );
    }
  }

  private encodeAveragePolicyGpuResident(
    tree: SparseTree,
    treeBuffers: SparseGpuTreeBuffers,
    reachBuffer: GPUBuffer,
    policyBuffer: GPUBuffer,
    numeratorBuffer: GPUBuffer,
    denominatorBuffer: GPUBuffer,
    policyAvgBuffer: GPUBuffer,
  ): void {
    if (!this.gpuKernels) return;
    const encoder = this.model.device.createCommandEncoder();
    const params = this.gpuKernels.encodeUpdateAveragePolicyRange(
      encoder,
      treeBuffers,
      reachBuffer,
      policyBuffer,
      numeratorBuffer,
      denominatorBuffer,
      policyAvgBuffer,
      1,
      tree.nodes.length,
    );
    this.model.device.queue.submit([encoder.finish()]);
    params.destroy();
  }

  private updatePolicy(
    tree: SparseTree,
    cumulativeRegrets: Float32Array,
    policy: Float32Array,
  ): void {
    policy.fill(0);
    for (const node of tree.nodes) {
      if (node.leaf || node.children.length === 0) continue;
      const legalCount = node.children.length;
      for (let hand = 0; hand < NUM_HANDS; hand += 1) {
        let positiveSum = 0;
        for (const childIndex of node.children) {
          positiveSum += Math.max(cumulativeRegrets[childIndex * NUM_HANDS + hand]!, 0);
        }
        const uniform = 1 / legalCount;
        for (const childIndex of node.children) {
          const idx = childIndex * NUM_HANDS + hand;
          policy[idx] =
            positiveSum > EPS
              ? Math.max(cumulativeRegrets[idx]!, 0) / positiveSum
              : uniform;
        }
      }
    }
  }

  private updateAveragePolicy(
    tree: SparseTree,
    policy: Float32Array,
    reach: Float32Array,
    numerator: Float32Array,
    denominator: Float32Array,
    policyAvg: Float32Array,
  ): void {
    for (let nodeIndex = 0; nodeIndex < tree.nodes.length; nodeIndex += 1) {
      const node = tree.nodes[nodeIndex]!;
      if (node.leaf || node.children.length === 0) continue;
      const actor = node.env.toAct;
      const reachBase = (nodeIndex * 2 + actor) * NUM_HANDS;
      for (const childIndex of node.children) {
        const policyBase = childIndex * NUM_HANDS;
        for (let hand = 0; hand < NUM_HANDS; hand += 1) {
          const weight = reach[reachBase + hand]!;
          const idx = policyBase + hand;
          numerator[idx] = numerator[idx]! + weight * policy[idx]!;
          denominator[idx] = denominator[idx]! + weight;
          policyAvg[idx] =
            denominator[idx]! > EPS ? numerator[idx]! / denominator[idx]! : policy[idx]!;
        }
      }
    }
  }

  private propagateReachAndBeliefs(
    tree: SparseTree,
    rootBeliefs: Float32Array<ArrayBuffer>,
    policy: Float32Array,
  ): { reach: Float32Array; beliefs: Float32Array } {
    const reach = new Float32Array(tree.nodes.length * 2 * NUM_HANDS);
    const beliefs = new Float32Array(tree.nodes.length * 2 * NUM_HANDS);
    for (let player = 0; player < 2; player += 1) {
      reach.fill(1, (player * NUM_HANDS), (player + 1) * NUM_HANDS);
    }
    this.copyBeliefsToNode(rootBeliefs, beliefs, 0);

    for (let depth = 0; depth < tree.treeDepth; depth += 1) {
      const start = tree.depthOffsets[depth]!;
      const end = tree.depthOffsets[depth + 1]!;
      for (let parentIndex = start; parentIndex < end; parentIndex += 1) {
        const parent = tree.nodes[parentIndex]!;
        if (parent.leaf || parent.children.length === 0) continue;
        const actor = parent.env.toAct;
        for (const childIndex of parent.children) {
          const child = tree.nodes[childIndex]!;
          const childAllowed = child.allowedMask;
          for (let player = 0; player < 2; player += 1) {
            const parentReachBase = (parentIndex * 2 + player) * NUM_HANDS;
            const childReachBase = (childIndex * 2 + player) * NUM_HANDS;
            for (let hand = 0; hand < NUM_HANDS; hand += 1) {
              let value = reach[parentReachBase + hand]!;
              if (player === actor) {
                value *= policy[childIndex * NUM_HANDS + hand]!;
              }
              reach[childReachBase + hand] = childAllowed[hand] ? value : 0;
              beliefs[childReachBase + hand] =
                childAllowed[hand] ? rootBeliefs[player * NUM_HANDS + hand]! * value : 0;
            }
          }
          this.normalizeNodeBeliefsWithAllowed(beliefs, childIndex, childAllowed);
        }
      }
    }
    return { reach, beliefs };
  }


  private async setLeafValues(
    tree: SparseTree,
    beliefs: Float32Array,
    latestValues: Float32Array,
    allInContext?: AllInTableContext,
  ): Promise<void> {
    latestValues.fill(0);
    const modelEnvs: PublicHunlEnv[] = [];
    const modelBeliefs: Float32Array<ArrayBuffer>[] = [];
    const modelValueBases: number[] = [];
    for (let nodeIndex = 0; nodeIndex < tree.nodes.length; nodeIndex += 1) {
      const node = tree.nodes[nodeIndex]!;
      if (!node.leaf) continue;
      const valueBase = nodeIndex * 2 * NUM_HANDS;
      const nodeBeliefs = this.nodeBeliefs(beliefs, nodeIndex);
      if (allInContext && this.isAllInCallLeaf(tree, nodeIndex)) {
        latestValues.set(
          computeAllInTableValues(allInContext, nodeBeliefs, 0, node.env),
          valueBase,
        );
      } else if (node.env.done) {
        if (node.env.hasFolded[0] || node.env.hasFolded[1]) {
          latestValues.fill(node.reward, valueBase, valueBase + NUM_HANDS);
          latestValues.fill(-node.reward, valueBase + NUM_HANDS, valueBase + 2 * NUM_HANDS);
        } else {
          latestValues.set(showdownTerminalValues(node.env, nodeBeliefs), valueBase);
        }
      } else {
        modelEnvs.push(node.env);
        modelBeliefs.push(nodeBeliefs);
        modelValueBases.push(valueBase);
      }
    }

    if (modelEnvs.length > 0) {
      const beliefSize = 2 * NUM_HANDS;
      const batchedBeliefs = new Float32Array(modelEnvs.length * beliefSize);
      for (let i = 0; i < modelBeliefs.length; i += 1) {
        batchedBeliefs.set(modelBeliefs[i]!, i * beliefSize);
      }
      const handValues = await this.model.predictBatchHandValues(
        modelEnvs,
        batchedBeliefs,
      );
      for (let i = 0; i < modelValueBases.length; i += 1) {
        latestValues.set(
          handValues.subarray(i * beliefSize, (i + 1) * beliefSize),
          modelValueBases[i]!,
        );
      }
    }
  }

  private computeExpectedValues(
    tree: SparseTree,
    policy: Float32Array,
    beliefs: Float32Array,
    leafValues: Float32Array,
  ): Float32Array {
    const values = new Float32Array(leafValues.length);
    for (let nodeIndex = 0; nodeIndex < tree.nodes.length; nodeIndex += 1) {
      if (tree.nodes[nodeIndex]!.leaf) {
        values.set(
          leafValues.subarray(nodeIndex * 2 * NUM_HANDS, (nodeIndex + 1) * 2 * NUM_HANDS),
          nodeIndex * 2 * NUM_HANDS,
        );
      }
    }

    const actorBelief = new Float32Array(NUM_HANDS);
    const marginal = new Float32Array(NUM_HANDS);
    const denom = new Float32Array(NUM_HANDS);
    const numer = new Float32Array(NUM_HANDS);
    for (let depth = tree.treeDepth - 1; depth >= 0; depth -= 1) {
      const start = tree.depthOffsets[depth]!;
      const end = tree.depthOffsets[depth + 1]!;
      for (let parentIndex = start; parentIndex < end; parentIndex += 1) {
        const parent = tree.nodes[parentIndex]!;
        if (parent.leaf || parent.children.length === 0) continue;
        const actor = parent.env.toAct;
        const other = (1 - actor) as PlayerIndex;
        const parentActorBase = (parentIndex * 2 + actor) * NUM_HANDS;
        actorBelief.set(beliefs.subarray(parentActorBase, parentActorBase + NUM_HANDS));
        this.unblockedMass(actorBelief, denom);

        for (const childIndex of parent.children) {
          const policyBase = childIndex * NUM_HANDS;
          for (let hand = 0; hand < NUM_HANDS; hand += 1) {
            marginal[hand] = actorBelief[hand]! * policy[policyBase + hand]!;
          }
          this.unblockedMass(marginal, numer);

          const parentValueActorBase = (parentIndex * 2 + actor) * NUM_HANDS;
          const parentValueOtherBase = (parentIndex * 2 + other) * NUM_HANDS;
          const childValueActorBase = (childIndex * 2 + actor) * NUM_HANDS;
          const childValueOtherBase = (childIndex * 2 + other) * NUM_HANDS;
          for (let hand = 0; hand < NUM_HANDS; hand += 1) {
            values[parentValueActorBase + hand] =
              values[parentValueActorBase + hand]! +
              policy[policyBase + hand]! * values[childValueActorBase + hand]!;
            const oppWeight = denom[hand]! > EPS ? numer[hand]! / denom[hand]! : 0;
            values[parentValueOtherBase + hand] =
              values[parentValueOtherBase + hand]! +
              oppWeight * values[childValueOtherBase + hand]!;
          }
        }
      }
    }
    return values;
  }

  private accumulateRegrets(
    tree: SparseTree,
    beliefs: Float32Array,
    values: Float32Array,
    cumulativeRegrets: Float32Array,
  ): void {
    const oppBelief = new Float32Array(NUM_HANDS);
    const weights = new Float32Array(NUM_HANDS);
    for (let parentIndex = 0; parentIndex < tree.nodes.length; parentIndex += 1) {
      const parent = tree.nodes[parentIndex]!;
      if (parent.leaf || parent.children.length === 0) continue;
      const actor = parent.env.toAct;
      const opp = (1 - actor) as PlayerIndex;
      const oppBase = (parentIndex * 2 + opp) * NUM_HANDS;
      oppBelief.set(beliefs.subarray(oppBase, oppBase + NUM_HANDS));
      this.unblockedMass(oppBelief, weights);
      const allowed = parent.allowedMask;
      const parentValueBase = (parentIndex * 2 + actor) * NUM_HANDS;
      for (const childIndex of parent.children) {
        const childValueBase = (childIndex * 2 + actor) * NUM_HANDS;
        const regretBase = childIndex * NUM_HANDS;
        for (let hand = 0; hand < NUM_HANDS; hand += 1) {
          if (!allowed[hand]) continue;
          cumulativeRegrets[regretBase + hand] =
            cumulativeRegrets[regretBase + hand]! +
            weights[hand]! *
              (values[childValueBase + hand]! - values[parentValueBase + hand]!);
        }
      }
    }
  }

  private rootPolicy(tree: SparseTree, policy: Float32Array): Float32Array<ArrayBuffer> {
    const out = new Float32Array(NUM_HANDS * this.numActions);
    const root = tree.nodes[0]!;
    for (let action = 0; action < this.numActions; action += 1) {
      const childIndex = root.actionToChild[action]!;
      if (childIndex < 0) continue;
      for (let hand = 0; hand < NUM_HANDS; hand += 1) {
        out[hand * this.numActions + action] = policy[childIndex * NUM_HANDS + hand]!;
      }
    }
    return out;
  }

  private rootActionProbs(
    tree: SparseTree,
    rootBeliefs: Float32Array<ArrayBuffer>,
    policy: Float32Array,
  ): Float32Array<ArrayBuffer> {
    const out = new Float32Array(this.numActions);
    const root = tree.nodes[0]!;
    const actor = root.env.toAct;
    const beliefBase = actor * NUM_HANDS;
    for (let action = 0; action < this.numActions; action += 1) {
      const childIndex = root.actionToChild[action]!;
      if (childIndex < 0) continue;
      let sum = 0;
      for (let hand = 0; hand < NUM_HANDS; hand += 1) {
        sum += rootBeliefs[beliefBase + hand]! * policy[childIndex * NUM_HANDS + hand]!;
      }
      out[action] = sum;
    }
    return out;
  }

  private nextBeliefs(
    tree: SparseTree,
    rootBeliefs: Float32Array<ArrayBuffer>,
    policy: Float32Array,
    selectedAction: number,
  ): Float32Array<ArrayBuffer> {
    const root = tree.nodes[0]!;
    if (!Number.isInteger(selectedAction) || selectedAction < 0 || selectedAction >= this.numActions) {
      throw new Error(`action ${selectedAction} is outside [0, ${this.numActions})`);
    }
    const childIndex = root.actionToChild[selectedAction]!;
    if (childIndex < 0) {
      throw new Error(`action ${selectedAction} is not legal for the current public state`);
    }
    const actor = root.env.toAct;
    const out = new Float32Array(rootBeliefs);
    const actorBase = actor * NUM_HANDS;
    let denom = 0;
    for (let hand = 0; hand < NUM_HANDS; hand += 1) {
      const value = rootBeliefs[actorBase + hand]! * policy[childIndex * NUM_HANDS + hand]!;
      out[actorBase + hand] = value;
      denom += value;
    }
    if (denom > EPS) {
      for (let hand = 0; hand < NUM_HANDS; hand += 1) {
        out[actorBase + hand] = out[actorBase + hand]! / denom;
      }
    } else {
      out.set(rootBeliefs.subarray(actorBase, actorBase + NUM_HANDS), actorBase);
    }
    const childAllowed = tree.nodes[childIndex]!.allowedMask;
    for (let player = 0; player < 2; player += 1) {
      const offset = player * NUM_HANDS;
      for (let hand = 0; hand < NUM_HANDS; hand += 1) {
        if (!childAllowed[hand]) out[offset + hand] = 0;
      }
      normalizeBeliefVector(out, player as PlayerIndex);
    }
    return out;
  }

  private propagateChildBelief(
    child: SparseNode,
    actor: PlayerIndex,
    parentIndex: number,
    childIndex: number,
    policy: Float32Array,
    beliefs: Float32Array,
  ): void {
    const parent = parentIndex;
    const parentBase = parent * 2 * NUM_HANDS;
    const childBase = childIndex * 2 * NUM_HANDS;
    beliefs.set(
      beliefs.subarray(parentBase, parentBase + 2 * NUM_HANDS),
      childBase,
    );
    const actorBase = childBase + actor * NUM_HANDS;
    const policyBase = childIndex * NUM_HANDS;
    for (let hand = 0; hand < NUM_HANDS; hand += 1) {
      beliefs[actorBase + hand] =
        beliefs[actorBase + hand]! * policy[policyBase + hand]!;
    }
    this.normalizeNodeBeliefsWithAllowed(beliefs, childIndex, child.allowedMask);
  }

  private nodeBeliefs(
    beliefs: Float32Array,
    nodeIndex: number,
  ): Float32Array<ArrayBuffer> {
    return new Float32Array(
      beliefs.slice(nodeIndex * 2 * NUM_HANDS, (nodeIndex + 1) * 2 * NUM_HANDS),
    );
  }

  private copyBeliefsToNode(
    source: Float32Array<ArrayBufferLike>,
    target: Float32Array,
    nodeIndex: number,
  ): void {
    target.set(source, nodeIndex * 2 * NUM_HANDS);
  }

  private rootBeliefsForEnv(
    env: PublicHunlEnv,
    inputBeliefs: Float32Array<ArrayBufferLike>,
  ): Float32Array<ArrayBuffer> {
    const out = new Float32Array(inputBeliefs);
    const allowed = this.allowedMask(env);
    for (let player = 0; player < 2; player += 1) {
      const offset = player * NUM_HANDS;
      for (let hand = 0; hand < NUM_HANDS; hand += 1) {
        if (!allowed[hand]) out[offset + hand] = 0;
      }
      normalizeBeliefVector(out, player as PlayerIndex);
    }
    return out;
  }

  private rootReach(): Float32Array<ArrayBuffer> {
    const reach = new Float32Array(2 * NUM_HANDS);
    reach.fill(1, 0, NUM_HANDS);
    reach.fill(1, NUM_HANDS, 2 * NUM_HANDS);
    return reach;
  }

  private gpuTreeData(tree: SparseTree): SparseGpuTreeData {
    const nodeCount = tree.nodes.length;
    const childOffsets = new Uint32Array(nodeCount);
    const childCount = new Uint32Array(nodeCount);
    const childIndices = new Uint32Array(
      tree.nodes.reduce((sum, node) => sum + node.children.length, 0),
    );
    const parentIndex = new Uint32Array(nodeCount);
    const prevActor = new Uint32Array(nodeCount);
    const toAct = new Uint32Array(nodeCount);
    const allowedMask = new Uint32Array(nodeCount * NUM_HANDS);
    const allowedProb = new Float32Array(nodeCount * NUM_HANDS);
    const handCard0 = new Uint32Array(NUM_HANDS);
    const handCard1 = new Uint32Array(NUM_HANDS);
    for (let hand = 0; hand < NUM_HANDS; hand += 1) {
      handCard0[hand] = HAND_CARD0[hand]!;
      handCard1[hand] = HAND_CARD1[hand]!;
    }

    let cursor = 0;
    for (let nodeIndex = 0; nodeIndex < nodeCount; nodeIndex += 1) {
      const node = tree.nodes[nodeIndex]!;
      childOffsets[nodeIndex] = cursor;
      childCount[nodeIndex] = node.children.length;
      for (const childIndex of node.children) {
        childIndices[cursor] = childIndex;
        cursor += 1;
      }
      parentIndex[nodeIndex] = Math.max(0, node.parent);
      prevActor[nodeIndex] =
        node.parent >= 0 ? tree.nodes[node.parent]!.env.toAct : node.env.toAct;
      toAct[nodeIndex] = node.env.toAct;
      const maskBase = nodeIndex * NUM_HANDS;
      let allowedCount = 0;
      for (let hand = 0; hand < NUM_HANDS; hand += 1) {
        const allowed = node.allowedMask[hand] ? 1 : 0;
        allowedMask[maskBase + hand] = allowed;
        allowedCount += allowed;
      }
      const fallback = allowedCount > 0 ? 1 / allowedCount : 0;
      for (let hand = 0; hand < NUM_HANDS; hand += 1) {
        allowedProb[maskBase + hand] = node.allowedMask[hand] ? fallback : 0;
      }
    }

    return {
      nodeCount,
      numHands: NUM_HANDS,
      childOffsets,
      childCount,
      childIndices,
      parentIndex,
      prevActor,
      toAct,
      allowedMask,
      allowedProb,
      handCard0,
      handCard1,
    };
  }

  private normalizeNodeBeliefsWithAllowed(
    beliefs: Float32Array,
    nodeIndex: number,
    allowed: Uint8Array,
  ): void {
    const base = nodeIndex * 2 * NUM_HANDS;
    let allowedCount = 0;
    for (let hand = 0; hand < NUM_HANDS; hand += 1) {
      allowedCount += allowed[hand] ? 1 : 0;
    }
    const fallback = allowedCount > 0 ? 1 / allowedCount : 0;
    for (let player = 0; player < 2; player += 1) {
      const offset = base + player * NUM_HANDS;
      let sum = 0;
      for (let hand = 0; hand < NUM_HANDS; hand += 1) {
        if (!allowed[hand]) {
          beliefs[offset + hand] = 0;
        }
        sum += beliefs[offset + hand]!;
      }
      if (sum > EPS) {
        const inv = 1 / sum;
        for (let hand = 0; hand < NUM_HANDS; hand += 1) {
          beliefs[offset + hand] = beliefs[offset + hand]! * inv;
        }
      } else {
        for (let hand = 0; hand < NUM_HANDS; hand += 1) {
          beliefs[offset + hand] = allowed[hand] ? fallback : 0;
        }
      }
    }
  }

  private allowedMask(env: PublicHunlEnv): Uint8Array {
    const board = env.boardIndices.filter((card) => card >= 0);
    const out = new Uint8Array(NUM_HANDS);
    for (let hand = 0; hand < NUM_HANDS; hand += 1) {
      out[hand] = handOverlapsCards(hand, board) ? 0 : 1;
    }
    return out;
  }

  private unblockedMass(values: Float32Array, out: Float32Array): void {
    const cardSums = new Float32Array(52);
    let total = 0;
    for (let hand = 0; hand < NUM_HANDS; hand += 1) {
      const value = values[hand]!;
      total += value;
      cardSums[HAND_CARD0[hand]!] = cardSums[HAND_CARD0[hand]!]! + value;
      cardSums[HAND_CARD1[hand]!] = cardSums[HAND_CARD1[hand]!]! + value;
    }
    for (let hand = 0; hand < NUM_HANDS; hand += 1) {
      out[hand] =
        total -
        cardSums[HAND_CARD0[hand]!]! -
        cardSums[HAND_CARD1[hand]!]! +
        values[hand]!;
    }
  }
}
