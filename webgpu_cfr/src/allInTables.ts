import { HAND_COMBOS } from "./cards.js";
import { WebGpuAllInTableGenerator } from "./allInTableGenerator.js";
import {
  readCachedAllInTable,
  writeCachedAllInTable,
  type CachedAllInTableKind,
} from "./allInTableCache.js";
import { NUM_HANDS, type PublicHunlEnv } from "./hunlEnv.js";
import type { BetterFfnAllInManifest } from "./types.js";

export const ALL_IN_I16_SCALE = 32768;

export type AllInStreet = 0 | 1 | 2;

export interface AllInTableContext {
  street: AllInStreet;
  table: Int16Array<ArrayBufferLike>;
  scale: number;
  permId?: number;
  comboPerms?: Uint32Array<ArrayBufferLike>;
}

export interface AllInTableProvider {
  tableForRoot(env: PublicHunlEnv): Promise<AllInTableContext | undefined>;
  prefetchForBoard?(board: readonly number[]): Promise<void>;
}

type BinaryLoader = (url: string) => Promise<ArrayBuffer>;

export interface AllInTableProviderOptions {
  isOffline?: () => boolean;
  persistentCache?: boolean;
}

const HAND_CARD0 = new Uint8Array(NUM_HANDS);
const HAND_CARD1 = new Uint8Array(NUM_HANDS);
for (let hand = 0; hand < NUM_HANDS; hand += 1) {
  const combo = HAND_COMBOS[hand]!;
  HAND_CARD0[hand] = combo[0];
  HAND_CARD1[hand] = combo[1];
}

export function packInt16Table(table: Int16Array<ArrayBufferLike>): Uint32Array<ArrayBuffer> {
  const packed = new Uint32Array(Math.ceil(table.length / 2));
  for (let i = 0; i < table.length; i += 2) {
    const lo = table[i]! & 0xffff;
    const hi = (table[i + 1] ?? 0) & 0xffff;
    packed[i >> 1] = lo | (hi << 16);
  }
  return packed;
}

export function computeAllInTableValues(
  context: AllInTableContext,
  beliefs: Float32Array<ArrayBufferLike>,
  nodeBase: number,
  env: PublicHunlEnv,
): Float32Array<ArrayBuffer> {
  const out = new Float32Array(2 * NUM_HANDS);
  const p0Base = nodeBase;
  const p1Base = nodeBase + NUM_HANDS;
  const permBase =
    context.permId !== undefined && context.comboPerms
      ? context.permId * NUM_HANDS
      : -1;
  const tableHand = (hand: number): number =>
    permBase >= 0 ? context.comboPerms![permBase + hand]! : hand;
  const tableAt = (hero: number, opp: number): number =>
    context.table[tableHand(hero) * NUM_HANDS + tableHand(opp)]! / context.scale;

  const total0 = beliefCardStats(beliefs, p0Base);
  const total1 = beliefCardStats(beliefs, p1Base);
  const scale0 =
    (env.stacks[0] + env.pot - env.startingStacks[0]) / Math.max(env.scale, 1.0e-8);
  const scale1 =
    (env.stacks[1] + env.pot - env.startingStacks[1]) / Math.max(env.scale, 1.0e-8);

  for (let hand = 0; hand < NUM_HANDS; hand += 1) {
    const c0 = HAND_CARD0[hand]!;
    const c1 = HAND_CARD1[hand]!;
    const denom0 = Math.max(
      0,
      total1.total - total1.cards[c0]! - total1.cards[c1]! + beliefs[p1Base + hand]!,
    );
    const denom1 = Math.max(
      0,
      total0.total - total0.cards[c0]! - total0.cards[c1]! + beliefs[p0Base + hand]!,
    );
    let numer0 = 0;
    let numer1 = 0;
    for (let opp = 0; opp < NUM_HANDS; opp += 1) {
      numer0 += tableAt(hand, opp) * beliefs[p1Base + opp]!;
      numer1 += tableAt(hand, opp) * beliefs[p0Base + opp]!;
    }
    out[hand] = denom0 > 1.0e-8 ? (numer0 / denom0) * scale0 : 0;
    out[NUM_HANDS + hand] = denom1 > 1.0e-8 ? (numer1 / denom1) * scale1 : 0;
  }
  return out;
}

function beliefCardStats(
  beliefs: Float32Array<ArrayBufferLike>,
  base: number,
): { total: number; cards: Float32Array<ArrayBuffer> } {
  const cards = new Float32Array(52);
  let total = 0;
  for (let hand = 0; hand < NUM_HANDS; hand += 1) {
    const value = beliefs[base + hand]!;
    total += value;
    const c0 = HAND_CARD0[hand]!;
    const c1 = HAND_CARD1[hand]!;
    cards[c0] = cards[c0]! + value;
    cards[c1] = cards[c1]! + value;
  }
  return { total, cards };
}

export function createManifestAllInTableProvider(
  manifest: { allIn?: BetterFfnAllInManifest },
  manifestUrl: string,
  loadBinary: BinaryLoader = fetchBinary,
  device?: GPUDevice,
  options: AllInTableProviderOptions = {},
): AllInTableProvider | undefined {
  const config = manifest.allIn;
  if (!config || config.enabled === false) return undefined;
  return new ManifestAllInTableProvider(config, manifestUrl, loadBinary, device, options);
}

class ManifestAllInTableProvider implements AllInTableProvider {
  private readonly cache = new Map<string, ArrayBuffer>();

  constructor(
    private readonly config: BetterFfnAllInManifest,
    private readonly manifestUrl: string,
    private readonly loadBinary: BinaryLoader,
    device?: GPUDevice,
    options: AllInTableProviderOptions = {},
  ) {
    this.generator = device ? new WebGpuAllInTableGenerator(device) : undefined;
    this.isOffline = options.isOffline ?? defaultIsOffline;
    this.persistentCache = options.persistentCache ?? true;
  }

  private readonly generator: WebGpuAllInTableGenerator | undefined;
  private readonly isOffline: () => boolean;
  private readonly persistentCache: boolean;

  async tableForRoot(env: PublicHunlEnv): Promise<AllInTableContext | undefined> {
    const scale = this.config.scale ?? ALL_IN_I16_SCALE;
    if (env.street === 0) {
      if (!this.config.preflop) return undefined;
      return {
        street: 0,
        table: await this.loadInt16(this.config.preflop.file, "preflop"),
        scale: this.config.preflop.scale ?? scale,
      };
    }
    if (env.street === 1) {
      const flop = env.boardIndices.slice(0, 3).filter((card) => card >= 0).sort((a, b) => a - b);
      if (flop.length !== 3) throw new Error("flop all-in table requires three public cards");
      if (!this.config.flop) return this.generatedFlopIfOffline(flop);
      try {
        const [actualToCanon, actualPerm, comboPerms] = await Promise.all([
          this.loadUint32(this.config.flop.actualToCanonFile),
          this.loadUint32(this.config.flop.actualPermFile),
          this.loadUint32(this.config.flop.comboPermsFile),
        ]);
        const actual = flopCombinationIndex(flop[0]!, flop[1]!, flop[2]!);
        const canonical = actualToCanon[actual]!;
        const permId = actualPerm[actual]!;
        return {
          street: 1,
          table: await this.loadInt16(
            templatePath(this.config.flop.tablePathTemplate, canonical, flop),
            "flop",
          ),
          scale: this.config.flop.scale ?? scale,
          permId,
          comboPerms,
        };
      } catch (error) {
        const generated = await this.generatedFlopIfOffline(flop);
        if (generated) return generated;
        throw error;
      }
    }
    if (env.street === 2) {
      const turn = env.boardIndices.slice(0, 4).filter((card) => card >= 0);
      if (turn.length !== 4) throw new Error("turn all-in table requires four public cards");
      if (!this.config.turn) return this.generated(turn);
      return {
        street: 2,
        table: await this.loadInt16OrGenerate(
          templatePath(this.config.turn.tablePathTemplate, 0, turn),
          turn,
        ),
        scale: this.config.turn.scale ?? scale,
      };
    }
    return undefined;
  }

  async prefetchForBoard(board: readonly number[]): Promise<void> {
    if (this.isOffline() || board.length !== 3 || !this.config.flop) return;
    const flop = [...board].sort((a, b) => a - b);
    const [actualToCanon] = await Promise.all([
      this.loadUint32(this.config.flop.actualToCanonFile),
      this.loadUint32(this.config.flop.actualPermFile),
      this.loadUint32(this.config.flop.comboPermsFile),
    ]);
    const actual = flopCombinationIndex(flop[0]!, flop[1]!, flop[2]!);
    const canonical = actualToCanon[actual]!;
    await this.loadInt16(
      templatePath(this.config.flop.tablePathTemplate, canonical, flop),
      "flop",
    );
  }

  private async generatedFlopIfOffline(
    board: readonly number[],
  ): Promise<AllInTableContext | undefined> {
    if (!this.isOffline()) return undefined;
    return await this.generated(board);
  }

  private async generated(board: readonly number[]): Promise<AllInTableContext | undefined> {
    const table = await this.generator?.tableForBoard(board);
    if (!table) return undefined;
    return {
      street: board.length === 3 ? 1 : 2,
      table,
      scale: ALL_IN_I16_SCALE,
    };
  }

  private async loadInt16OrGenerate(
    path: string,
    board: readonly number[],
  ): Promise<Int16Array<ArrayBuffer>> {
    try {
      return await this.loadInt16(path);
    } catch (error) {
      if (!this.generator) throw error;
      return await this.generator.tableForBoard(board);
    }
  }

  private async loadInt16(
    path: string,
    cacheKind?: CachedAllInTableKind,
  ): Promise<Int16Array<ArrayBuffer>> {
    const buffer = await this.load(path, cacheKind);
    if (buffer.byteLength % Int16Array.BYTES_PER_ELEMENT !== 0) {
      throw new Error(`${path} byte length is not aligned for int16`);
    }
    return new Int16Array(buffer.slice(0));
  }

  private async loadUint32(path: string): Promise<Uint32Array<ArrayBuffer>> {
    const buffer = await this.load(path);
    if (buffer.byteLength % Uint32Array.BYTES_PER_ELEMENT !== 0) {
      throw new Error(`${path} byte length is not aligned for uint32`);
    }
    return new Uint32Array(buffer.slice(0));
  }

  private async load(
    path: string,
    cacheKind?: CachedAllInTableKind,
  ): Promise<ArrayBuffer> {
    const url = new URL(path, this.manifestUrl).toString();
    const cached = this.cache.get(url);
    if (cached) return cached;
    if (cacheKind && this.persistentCache) {
      const persistent = await readCachedAllInTable(url);
      if (persistent) {
        this.cache.set(url, persistent);
        return persistent;
      }
    }
    const buffer = await this.loadBinary(url);
    this.cache.set(url, buffer);
    if (cacheKind && this.persistentCache) {
      await writeCachedAllInTable(url, cacheKind, buffer).catch(() => undefined);
    }
    return buffer;
  }
}

function defaultIsOffline(): boolean {
  return typeof navigator !== "undefined" && navigator.onLine === false;
}

function templatePath(template: string, canonical: number, cards: readonly number[]): string {
  return template
    .replaceAll("{id}", String(canonical))
    .replaceAll("{id4}", canonical.toString().padStart(4, "0"))
    .replaceAll("{canonical}", String(canonical))
    .replaceAll("{canonical4}", canonical.toString().padStart(4, "0"))
    .replaceAll("{cards}", cards.join("-"));
}

async function fetchBinary(url: string): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`failed to fetch ${url}: ${response.status}`);
  }
  return await response.arrayBuffer();
}

function flopCombinationIndex(a: number, b: number, c: number): number {
  const total = (52 * 51 * 50) / 6;
  const prefixA = total - ((52 - a) * (51 - a) * (50 - a)) / 6;
  const prefixB = ((51 - a) * (50 - a)) / 2 - ((52 - b) * (51 - b)) / 2;
  return prefixA + prefixB + c - b - 1;
}
