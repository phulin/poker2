import type { BetterFfnWebGpuModel } from "./betterFfnWebGpuModel.js";
import { normalizeBeliefVector } from "./beliefs.js";
import { HAND_COMBOS, handOverlapsCards } from "./cards.js";
import {
  NUM_HANDS,
  PublicHunlEnv,
  showdownTerminalValues,
} from "./hunlEnv.js";
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

export interface SparseResolveOptions {
  depth: number;
  iterations: number;
  selectedAction?: number;
  readPolicy?: boolean;
  readActionProbs?: boolean;
  readBeliefs?: boolean;
}

export class SparseCfrResolver {
  readonly model: BetterFfnWebGpuModel;
  readonly numActions: number;

  constructor(model: BetterFfnWebGpuModel) {
    this.model = model;
    this.numActions = model.manifest.architecture.numActions;
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

    const tree = this.buildTree(env, options.depth);
    const totalNodes = tree.nodes.length;
    const policy = new Float32Array(totalNodes * NUM_HANDS);
    const policyAvg = new Float32Array(totalNodes * NUM_HANDS);
    const cumulativeRegrets = new Float32Array(totalNodes * NUM_HANDS);
    const avgNumerator = new Float32Array(totalNodes * NUM_HANDS);
    const avgDenominator = new Float32Array(totalNodes * NUM_HANDS);
    const beliefs = new Float32Array(totalNodes * 2 * NUM_HANDS);
    const beliefsAvg = new Float32Array(totalNodes * 2 * NUM_HANDS);
    const latestValues = new Float32Array(totalNodes * 2 * NUM_HANDS);

    const rootBeliefs = this.rootBeliefsForEnv(tree.nodes[0]!.env, inputBeliefs);
    this.copyBeliefsToNode(rootBeliefs, beliefs, 0);
    this.copyBeliefsToNode(rootBeliefs, beliefsAvg, 0);

    await this.initializePolicyAndBeliefs(tree, rootBeliefs, policy, beliefs);
    policyAvg.set(policy);
    beliefsAvg.set(beliefs);

    await this.setLeafValues(tree, beliefsAvg, latestValues);
    let values = this.computeExpectedValues(tree, policy, beliefs, latestValues);

    for (let t = 0; t < options.iterations; t += 1) {
      this.accumulateRegrets(tree, beliefsAvg, values, cumulativeRegrets);
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
      const avg = this.propagateReachAndBeliefs(tree, rootBeliefs, policyAvg);
      beliefsAvg.set(avg.beliefs);

      await this.setLeafValues(tree, beliefsAvg, latestValues);
      values = this.computeExpectedValues(tree, policy, beliefs, latestValues);
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
        for (let action = 0; action < this.numActions; action += 1) {
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
    for (let depth = 0; depth < tree.treeDepth; depth += 1) {
      const start = tree.depthOffsets[depth]!;
      const end = tree.depthOffsets[depth + 1]!;
      for (let nodeIndex = start; nodeIndex < end; nodeIndex += 1) {
        const node = tree.nodes[nodeIndex]!;
        if (node.leaf || node.children.length === 0) continue;
        const nodeBeliefs = this.nodeBeliefs(beliefs, nodeIndex);
        const modelPolicy = await this.modelPolicy(node.env, nodeBeliefs, node.legalMask);
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

  private async modelPolicy(
    env: PublicHunlEnv,
    beliefs: Float32Array<ArrayBuffer>,
    legalMask: readonly number[],
  ): Promise<Float32Array<ArrayBuffer>> {
    const prediction = await this.model.predict(env, beliefs, {
      includePolicy: true,
    });
    if (!prediction.policyLogits) {
      throw new Error("model did not return policy logits");
    }
    const logits = prediction.policyLogits;
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
  ): Promise<void> {
    latestValues.fill(0);
    for (let nodeIndex = 0; nodeIndex < tree.nodes.length; nodeIndex += 1) {
      const node = tree.nodes[nodeIndex]!;
      if (!node.leaf) continue;
      const valueBase = nodeIndex * 2 * NUM_HANDS;
      const nodeBeliefs = this.nodeBeliefs(beliefs, nodeIndex);
      if (node.env.done) {
        if (node.env.hasFolded[0] || node.env.hasFolded[1]) {
          latestValues.fill(node.reward, valueBase, valueBase + NUM_HANDS);
          latestValues.fill(-node.reward, valueBase + NUM_HANDS, valueBase + 2 * NUM_HANDS);
        } else {
          latestValues.set(showdownTerminalValues(node.env, nodeBeliefs), valueBase);
        }
      } else {
        const handValues = await this.model.predictHandValues(node.env, nodeBeliefs);
        latestValues.set(handValues, valueBase);
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
