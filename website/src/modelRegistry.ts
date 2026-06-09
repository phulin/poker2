import type { BetterFfnWebGpuModel } from "./betterFfnWebGpuModel.js";
import type { BetterFfnCurriculumStreet, BetterFfnManifest } from "./types.js";

export type BetterFfnRuntime = BetterFfnWebGpuModel | BetterFfnModelRegistry;

const STREET_BY_INDEX = new Map<number, BetterFfnCurriculumStreet>([
  [1, "flop"],
  [2, "turn"],
  [3, "river"],
]);

function arraysEqual<T>(a: readonly T[], b: readonly T[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

function closeArrays(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    if (Math.abs(a[i]! - b[i]!) > 1.0e-9) return false;
  }
  return true;
}

function assertCompatibleManifest(
  street: BetterFfnCurriculumStreet,
  manifest: BetterFfnManifest,
  root: BetterFfnManifest,
): void {
  const arch = manifest.architecture;
  const rootArch = root.architecture;
  if (arch.numHands !== rootArch.numHands) {
    throw new Error(`curriculum ${street} hand count does not match ${rootArch.numHands}`);
  }
  if (arch.numPlayers !== rootArch.numPlayers) {
    throw new Error(`curriculum ${street} player count does not match ${rootArch.numPlayers}`);
  }
  if (arch.numActions !== rootArch.numActions) {
    throw new Error(`curriculum ${street} action count does not match ${rootArch.numActions}`);
  }
  if (!arraysEqual(manifest.actionLabels, root.actionLabels)) {
    throw new Error(`curriculum ${street} action labels do not match default stage`);
  }
  if (
    manifest.env.stack !== root.env.stack ||
    manifest.env.sb !== root.env.sb ||
    manifest.env.bb !== root.env.bb ||
    manifest.env.flopShowdown !== root.env.flopShowdown ||
    manifest.env.maxStackBb !== root.env.maxStackBb ||
    !closeArrays(manifest.env.betBins, root.env.betBins)
  ) {
    throw new Error(`curriculum ${street} env settings do not match default stage`);
  }
}

export class BetterFfnModelRegistry {
  readonly root: BetterFfnWebGpuModel;
  readonly defaultStage: BetterFfnCurriculumStreet;
  readonly stages: Readonly<Record<BetterFfnCurriculumStreet, BetterFfnWebGpuModel>>;

  constructor(
    stages: Record<BetterFfnCurriculumStreet, BetterFfnWebGpuModel>,
    defaultStage: BetterFfnCurriculumStreet,
  ) {
    this.defaultStage = defaultStage;
    this.stages = stages;
    this.root = stages[defaultStage];
    if (!this.root) {
      throw new Error(`curriculum default stage ${defaultStage} is missing`);
    }
    const device = this.root.device;
    for (const street of ["flop", "turn", "river"] as const) {
      const model = stages[street];
      if (!model) throw new Error(`curriculum ${street} model is missing`);
      if (model.device !== device) {
        throw new Error("all curriculum models must use the same GPU device");
      }
      assertCompatibleManifest(street, model.manifest, this.root.manifest);
    }
  }

  modelForStreet(street: number): BetterFfnWebGpuModel {
    const streetName = STREET_BY_INDEX.get(street);
    if (!streetName) {
      throw new Error("curriculum model sets support only flop, turn, and river public states");
    }
    return this.stages[streetName];
  }

  dispose(): void {
    const disposed = new Set<BetterFfnWebGpuModel>();
    for (const model of Object.values(this.stages)) {
      if (disposed.has(model)) continue;
      disposed.add(model);
      model.dispose();
    }
  }
}

export function isBetterFfnModelRegistry(value: BetterFfnRuntime): value is BetterFfnModelRegistry {
  return value instanceof BetterFfnModelRegistry;
}

export function modelForRuntimeStreet(
  runtime: BetterFfnRuntime,
  street: number,
): BetterFfnWebGpuModel {
  return isBetterFfnModelRegistry(runtime) ? runtime.modelForStreet(street) : runtime;
}

export function rootModelForRuntime(runtime: BetterFfnRuntime): BetterFfnWebGpuModel {
  return isBetterFfnModelRegistry(runtime) ? runtime.root : runtime;
}
