import { HAND_COMBOS } from "./cards.js";
import {
  makeEmptyStorageBuffer,
  makeStorageBuffer,
  makeUniformBuffer,
  readUintBuffer,
} from "./gpuBuffers.js";
import { createComputePipeline, dispatchCompute } from "./gpuPipeline.js";
import { NUM_HANDS } from "./hunlEnv.js";

const ALL_IN_I16_SCALE = 32768;
const TABLE_CELLS = NUM_HANDS * NUM_HANDS;
const PACKED_TABLE_WORDS = Math.ceil(TABLE_CELLS / 2);

const HAND_CARD0_U32 = new Uint32Array(NUM_HANDS);
const HAND_CARD1_U32 = new Uint32Array(NUM_HANDS);
for (let hand = 0; hand < NUM_HANDS; hand += 1) {
  const [c0, c1] = HAND_COMBOS[hand]!;
  HAND_CARD0_U32[hand] = c0;
  HAND_CARD1_U32[hand] = c1;
}

const RANK_CODES_WGSL = /* wgsl */ `
struct Params {
  numHands: u32,
  runoutCount: u32,
  boardCount: u32,
  _pad0: u32,
  board0: u32,
  board1: u32,
  board2: u32,
  board3: u32,
};

@group(0) @binding(0) var<storage, read> runouts: array<u32>;
@group(0) @binding(1) var<storage, read> handCard0: array<u32>;
@group(0) @binding(2) var<storage, read> handCard1: array<u32>;
@group(0) @binding(3) var<storage, read_write> ranks: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

fn card_rank(card: u32) -> u32 {
  return card % 13u;
}

fn card_suit(card: u32) -> u32 {
  return card / 13u;
}

fn encode_rank(category: u32, k0: u32, k1: u32, k2: u32, k3: u32, k4: u32) -> u32 {
  return (category << 20u) | (k0 << 16u) | (k1 << 12u) | (k2 << 8u) | (k3 << 4u) | k4;
}

fn evaluate_standard_five(c0: u32, c1: u32, c2: u32, c3: u32, c4: u32) -> u32 {
  var counts: array<u32, 13>;
  for (var i = 0u; i < 13u; i = i + 1u) {
    counts[i] = 0u;
  }
  counts[card_rank(c0)] = counts[card_rank(c0)] + 1u;
  counts[card_rank(c1)] = counts[card_rank(c1)] + 1u;
  counts[card_rank(c2)] = counts[card_rank(c2)] + 1u;
  counts[card_rank(c3)] = counts[card_rank(c3)] + 1u;
  counts[card_rank(c4)] = counts[card_rank(c4)] + 1u;

  let flush = card_suit(c0) == card_suit(c1) &&
    card_suit(c0) == card_suit(c2) &&
    card_suit(c0) == card_suit(c3) &&
    card_suit(c0) == card_suit(c4);

  var straightHigh = 4294967295u;
  if (counts[12] != 0u && counts[3] != 0u && counts[2] != 0u && counts[1] != 0u && counts[0] != 0u) {
    straightHigh = 3u;
  }
  for (var h = 12u; h >= 4u; h = h - 1u) {
    if (
      counts[h] != 0u &&
      counts[h - 1u] != 0u &&
      counts[h - 2u] != 0u &&
      counts[h - 3u] != 0u &&
      counts[h - 4u] != 0u
    ) {
      straightHigh = h;
      break;
    }
    if (h == 4u) {
      break;
    }
  }

  var quad = 4294967295u;
  var trip = 4294967295u;
  var pair0 = 4294967295u;
  var pair1 = 4294967295u;
  var kickers: array<u32, 5>;
  var kickerCount = 0u;
  for (var offset = 0u; offset < 13u; offset = offset + 1u) {
    let r = 12u - offset;
    let count = counts[r];
    if (count == 4u) {
      quad = r;
    } else if (count == 3u) {
      trip = r;
    } else if (count == 2u) {
      if (pair0 == 4294967295u) {
        pair0 = r;
      } else {
        pair1 = r;
      }
    } else if (count == 1u) {
      kickers[kickerCount] = r;
      kickerCount = kickerCount + 1u;
    }
  }

  if (flush && straightHigh != 4294967295u) {
    return encode_rank(9u, straightHigh, 0u, 0u, 0u, 0u);
  }
  if (quad != 4294967295u) {
    return encode_rank(8u, quad, kickers[0], 0u, 0u, 0u);
  }
  if (trip != 4294967295u && pair0 != 4294967295u) {
    return encode_rank(7u, trip, pair0, 0u, 0u, 0u);
  }
  if (flush) {
    return encode_rank(6u, kickers[0], kickers[1], kickers[2], kickers[3], kickers[4]);
  }
  if (straightHigh != 4294967295u) {
    return encode_rank(5u, straightHigh, 0u, 0u, 0u, 0u);
  }
  if (trip != 4294967295u) {
    return encode_rank(4u, trip, kickers[0], kickers[1], 0u, 0u);
  }
  if (pair0 != 4294967295u && pair1 != 4294967295u) {
    return encode_rank(3u, pair0, pair1, kickers[0], 0u, 0u);
  }
  if (pair0 != 4294967295u) {
    return encode_rank(2u, pair0, kickers[0], kickers[1], kickers[2], 0u);
  }
  return encode_rank(1u, kickers[0], kickers[1], kickers[2], kickers[3], kickers[4]);
}

fn evaluate_standard_seven(c0: u32, c1: u32, c2: u32, c3: u32, c4: u32, c5: u32, c6: u32) -> u32 {
  var cards = array<u32, 7>(c0, c1, c2, c3, c4, c5, c6);
  var best = 0u;
  for (var omit0 = 0u; omit0 < 6u; omit0 = omit0 + 1u) {
    for (var omit1 = omit0 + 1u; omit1 < 7u; omit1 = omit1 + 1u) {
      var chosen: array<u32, 5>;
      var out = 0u;
      for (var i = 0u; i < 7u; i = i + 1u) {
        if (i != omit0 && i != omit1) {
          chosen[out] = cards[i];
          out = out + 1u;
        }
      }
      let rank = evaluate_standard_five(chosen[0], chosen[1], chosen[2], chosen[3], chosen[4]);
      if (rank > best) {
        best = rank;
      }
    }
  }
  return best;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let linear = gid.x;
  let total = params.runoutCount * params.numHands;
  if (linear >= total) {
    return;
  }
  let hand = linear % params.numHands;
  let runout = linear / params.numHands;
  let r0 = runouts[runout * 2u];
  let r1 = runouts[runout * 2u + 1u];
  let public3 = select(r0, params.board3, params.boardCount == 4u);
  let public4 = select(r1, r0, params.boardCount == 4u);
  ranks[linear] = evaluate_standard_seven(
    handCard0[hand],
    handCard1[hand],
    params.board0,
    params.board1,
    params.board2,
    public3,
    public4,
  );
}
`;

const PAYOFF_TABLE_WGSL = /* wgsl */ `
struct Params {
  numHands: u32,
  runoutCount: u32,
  boardCount: u32,
  tableScale: u32,
  board0: u32,
  board1: u32,
  board2: u32,
  board3: u32,
};

@group(0) @binding(0) var<storage, read> runouts: array<u32>;
@group(0) @binding(1) var<storage, read> handCard0: array<u32>;
@group(0) @binding(2) var<storage, read> handCard1: array<u32>;
@group(0) @binding(3) var<storage, read> ranks: array<u32>;
@group(0) @binding(4) var<storage, read_write> outPacked: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;

fn hand_hits_board(hand: u32) -> bool {
  let c0 = handCard0[hand];
  let c1 = handCard1[hand];
  return c0 == params.board0 || c1 == params.board0 ||
    c0 == params.board1 || c1 == params.board1 ||
    c0 == params.board2 || c1 == params.board2 ||
    (params.boardCount == 4u && (c0 == params.board3 || c1 == params.board3));
}

fn hands_overlap(a: u32, b: u32) -> bool {
  return handCard0[a] == handCard0[b] ||
    handCard0[a] == handCard1[b] ||
    handCard1[a] == handCard0[b] ||
    handCard1[a] == handCard1[b];
}

fn hand_hits_runout(hand: u32, runout: u32) -> bool {
  let c0 = handCard0[hand];
  let c1 = handCard1[hand];
  let r0 = runouts[runout * 2u];
  let r1 = runouts[runout * 2u + 1u];
  return c0 == r0 || c1 == r0 ||
    (params.boardCount == 3u && (c0 == r1 || c1 == r1));
}

fn quantize_score(score: i32, count: i32) -> u32 {
  if (count <= 0) {
    return 0u;
  }
  let positive = score >= 0;
  let absScore = select(-score, score, positive);
  let numerator = absScore * i32(params.tableScale);
  var q = numerator / count;
  let rem = numerator - q * count;
  let twiceRem = rem * 2;
  if (twiceRem > count || (twiceRem == count && (q % 2) != 0)) {
    q = q + 1;
  }
  var signed = select(-q, q, positive);
  signed = clamp(signed, -32768, 32767);
  return u32(signed) & 0xffffu;
}

fn table_entry(pair: u32) -> u32 {
  let hero = pair / params.numHands;
  let opp = pair - hero * params.numHands;
  if (
    hero >= params.numHands ||
    hand_hits_board(hero) ||
    hand_hits_board(opp) ||
    hands_overlap(hero, opp)
  ) {
    return 0u;
  }
  var score = 0;
  var count = 0;
  for (var runout = 0u; runout < params.runoutCount; runout = runout + 1u) {
    if (hand_hits_runout(hero, runout) || hand_hits_runout(opp, runout)) {
      continue;
    }
    let heroRank = ranks[runout * params.numHands + hero];
    let oppRank = ranks[runout * params.numHands + opp];
    if (heroRank > oppRank) {
      score = score + 1;
    } else if (heroRank < oppRank) {
      score = score - 1;
    }
    count = count + 1;
  }
  return quantize_score(score, count);
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let word = gid.x;
  let pair0 = word * 2u;
  if (pair0 >= params.numHands * params.numHands) {
    return;
  }
  let lo = table_entry(pair0);
  let hi = table_entry(pair0 + 1u);
  outPacked[word] = lo | (hi << 16u);
}
`;

function boardKey(board: readonly number[]): string {
  return [...board].sort((a, b) => a - b).join("-");
}

function runoutsForBoard(board: readonly number[]): Uint32Array<ArrayBuffer> {
  const blocked = new Set(board);
  const cards: number[] = [];
  for (let card = 0; card < 52; card += 1) {
    if (!blocked.has(card)) cards.push(card);
  }
  const values: number[] = [];
  if (board.length === 3) {
    for (let i = 0; i < cards.length; i += 1) {
      for (let j = i + 1; j < cards.length; j += 1) {
        values.push(cards[i]!, cards[j]!);
      }
    }
  } else if (board.length === 4) {
    for (const card of cards) values.push(card, 0);
  } else {
    throw new Error(`all-in table generation requires flop or turn board, got ${board.length} cards`);
  }
  return new Uint32Array(values);
}

function paramsForBoard(board: readonly number[], runoutCount: number): Uint32Array<ArrayBuffer> {
  if (board.length !== 3 && board.length !== 4) {
    throw new Error(`all-in table generation requires flop or turn board, got ${board.length} cards`);
  }
  return new Uint32Array([
    NUM_HANDS,
    runoutCount,
    board.length,
    ALL_IN_I16_SCALE,
    board[0]!,
    board[1]!,
    board[2]!,
    board[3] ?? 0,
  ]);
}

function unpackInt16(words: Uint32Array<ArrayBufferLike>): Int16Array<ArrayBuffer> {
  const out = new Int16Array(TABLE_CELLS);
  for (let i = 0; i < words.length; i += 1) {
    const word = words[i]!;
    out[i * 2] = word & 0xffff;
    out[i * 2 + 1] = (word >>> 16) & 0xffff;
  }
  return out;
}

export class WebGpuAllInTableGenerator {
  private readonly rankPipeline: GPUComputePipeline;
  private readonly payoffPipeline: GPUComputePipeline;
  private readonly handCard0: GPUBuffer;
  private readonly handCard1: GPUBuffer;
  private readonly cache = new Map<string, Promise<Int16Array<ArrayBuffer>>>();

  constructor(private readonly device: GPUDevice) {
    this.rankPipeline = createComputePipeline(
      device,
      RANK_CODES_WGSL,
      "allin-table-rank-codes",
    );
    this.payoffPipeline = createComputePipeline(
      device,
      PAYOFF_TABLE_WGSL,
      "allin-table-payoff",
    );
    this.handCard0 = makeStorageBuffer(device, HAND_CARD0_U32);
    this.handCard1 = makeStorageBuffer(device, HAND_CARD1_U32);
  }

  tableForBoard(board: readonly number[]): Promise<Int16Array<ArrayBuffer>> {
    const sorted = [...board].sort((a, b) => a - b);
    const key = boardKey(sorted);
    let cached = this.cache.get(key);
    if (!cached) {
      cached = this.generate(sorted);
      this.cache.set(key, cached);
    }
    return cached;
  }

  private async generate(board: readonly number[]): Promise<Int16Array<ArrayBuffer>> {
    const runouts = runoutsForBoard(board);
    const runoutCount = runouts.length / 2;
    const paramsWords = paramsForBoard(board, runoutCount);
    const runoutBuffer = makeStorageBuffer(this.device, runouts);
    const rankBuffer = makeEmptyStorageBuffer(this.device, runoutCount * NUM_HANDS);
    const tableBuffer = makeEmptyStorageBuffer(this.device, PACKED_TABLE_WORDS);
    const params = makeUniformBuffer(this.device, paramsWords);

    try {
      const encoder = this.device.createCommandEncoder();
      dispatchCompute(
        encoder,
        this.rankPipeline,
        this.device.createBindGroup({
          layout: this.rankPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: runoutBuffer } },
            { binding: 1, resource: { buffer: this.handCard0 } },
            { binding: 2, resource: { buffer: this.handCard1 } },
            { binding: 3, resource: { buffer: rankBuffer } },
            { binding: 4, resource: { buffer: params } },
          ],
        }),
        Math.ceil((runoutCount * NUM_HANDS) / 128),
      );
      dispatchCompute(
        encoder,
        this.payoffPipeline,
        this.device.createBindGroup({
          layout: this.payoffPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: runoutBuffer } },
            { binding: 1, resource: { buffer: this.handCard0 } },
            { binding: 2, resource: { buffer: this.handCard1 } },
            { binding: 3, resource: { buffer: rankBuffer } },
            { binding: 4, resource: { buffer: tableBuffer } },
            { binding: 5, resource: { buffer: params } },
          ],
        }),
        Math.ceil(PACKED_TABLE_WORDS / 128),
      );
      this.device.queue.submit([encoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      return unpackInt16(await readUintBuffer(this.device, tableBuffer, PACKED_TABLE_WORDS));
    } finally {
      runoutBuffer.destroy();
      rankBuffer.destroy();
      tableBuffer.destroy();
      params.destroy();
    }
  }
}
