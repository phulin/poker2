import type {
  BetterFfnManifest,
  BrowserCfrInitialState,
  PlayerIndex,
} from "./types.js";
import {
  assertUniqueCards,
  HAND_COMBOS,
  handComboCards,
  handsOverlap,
  NUM_CARDS,
} from "./cards.js";

export const NUM_HANDS = 1326;
export const DEFAULT_FORCE_DECK = [12, 25, 38, 51, 0, 13, 26, 1, 14];

export interface LegalBins {
  amounts: number[];
  mask: number[];
}

export interface StepResult {
  reward: number;
  newStreet: number;
  dealtCards: number[];
}

function completeDeck(seed: readonly number[], blocked: readonly number[] = []): number[] {
  const out: number[] = [];
  const used = new Set<number>();
  for (const card of [...blocked, ...seed]) {
    if (!Number.isInteger(card) || card < 0 || card >= NUM_CARDS) continue;
    if (used.has(card)) continue;
    used.add(card);
    if (!blocked.includes(card)) out.push(card);
  }
  for (let card = 0; card < NUM_CARDS; card += 1) {
    if (!used.has(card)) {
      used.add(card);
      out.push(card);
    }
  }
  return out;
}

function streetForPublicCardCount(count: number): number {
  if (count === 0) return 0;
  if (count === 3) return 1;
  if (count === 4) return 2;
  if (count === 5) return 3;
  throw new Error("publicCards must contain 0, 3, 4, or 5 cards");
}

export class PublicHunlEnv {
  readonly betBins: number[];
  readonly sb: number;
  readonly bb: number;
  readonly flopShowdown: boolean;

  button: PlayerIndex;
  street = 0;
  toAct: PlayerIndex;
  lastToAct: PlayerIndex;
  pot: number;
  minRaise: number;
  actionsThisRound = 0;
  actionsLastRound = 0;
  actedSinceReset = false;
  stacks: [number, number];
  committed: [number, number];
  startingStacks: [number, number];
  scale: number;
  isAllin: [boolean, boolean] = [false, false];
  hasFolded: [boolean, boolean] = [false, false];
  done = false;
  winner = -1;
  deck: number[];
  deckPos = 4;
  boardIndices: number[] = [-1, -1, -1, -1, -1];
  lastBoardIndices: number[] = [-1, -1, -1, -1, -1];
  holeIndices: [[number, number], [number, number]];

  constructor(options: {
    stack: number;
    sb: number;
    bb: number;
    betBins: number[];
    button: PlayerIndex;
    forceDeck: number[];
    flopShowdown?: boolean;
  }) {
    this.betBins = [...options.betBins];
    this.sb = options.sb;
    this.bb = options.bb;
    this.flopShowdown = options.flopShowdown ?? false;
    this.button = options.button;
    this.toAct = options.button;
    this.lastToAct = options.button;
    this.pot = options.sb + options.bb;
    this.minRaise = options.bb;
    this.startingStacks = [options.stack, options.stack];
    this.scale = options.stack;
    this.deck = completeDeck(options.forceDeck).slice(0, 9);
    this.holeIndices = [
      [this.deck[0]!, this.deck[1]!],
      [this.deck[2]!, this.deck[3]!],
    ];

    const pSb = options.button;
    const pBb = (1 - pSb) as PlayerIndex;
    this.stacks = [options.stack, options.stack];
    this.committed = [0, 0];
    this.stacks[pSb] -= options.sb;
    this.committed[pSb] = options.sb;
    this.stacks[pBb] -= options.bb;
    this.committed[pBb] = options.bb;
  }

  static fromManifest(
    manifest: BetterFfnManifest,
    initialState?: BrowserCfrInitialState,
  ): PublicHunlEnv {
    return new PublicHunlEnv({
      stack: initialState?.stack ?? manifest.env.stack,
      sb: initialState?.sb ?? manifest.env.sb,
      bb: initialState?.bb ?? manifest.env.bb,
      betBins: initialState?.betBins ?? manifest.env.betBins,
      button: initialState?.button ?? manifest.env.defaultButton,
      forceDeck:
        initialState?.forceDeck ??
        manifest.env.defaultForceDeck ??
        DEFAULT_FORCE_DECK,
      flopShowdown: initialState?.flopShowdown ?? manifest.env.flopShowdown,
    });
  }

  clone(): PublicHunlEnv {
    const env = Object.create(PublicHunlEnv.prototype) as PublicHunlEnv;
    Object.assign(env, {
      betBins: [...this.betBins],
      sb: this.sb,
      bb: this.bb,
      flopShowdown: this.flopShowdown,
      button: this.button,
      street: this.street,
      toAct: this.toAct,
      lastToAct: this.lastToAct,
      pot: this.pot,
      minRaise: this.minRaise,
      actionsThisRound: this.actionsThisRound,
      actionsLastRound: this.actionsLastRound,
      actedSinceReset: this.actedSinceReset,
      stacks: [...this.stacks] as [number, number],
      committed: [...this.committed] as [number, number],
      startingStacks: [...this.startingStacks] as [number, number],
      scale: this.scale,
      isAllin: [...this.isAllin] as [boolean, boolean],
      hasFolded: [...this.hasFolded] as [boolean, boolean],
      done: this.done,
      winner: this.winner,
      deck: [...this.deck],
      deckPos: this.deckPos,
      boardIndices: [...this.boardIndices],
      lastBoardIndices: [...this.lastBoardIndices],
      holeIndices: [
        [...this.holeIndices[0]] as [number, number],
        [...this.holeIndices[1]] as [number, number],
      ],
    });
    return env;
  }

  configureKnownCards(options: {
    publicCards?: readonly number[];
    heroPlayer?: PlayerIndex;
    heroHand?: readonly [number, number];
  }): void {
    const publicCards = options.publicCards ?? [];
    assertUniqueCards([
      ...publicCards,
      ...(options.heroHand ? [...options.heroHand] : []),
    ]);
    const street = streetForPublicCardCount(publicCards.length);

    const knownPrivate: Array<number | undefined> = [
      this.holeIndices[0][0],
      this.holeIndices[0][1],
      this.holeIndices[1][0],
      this.holeIndices[1][1],
    ];
    if (options.heroHand) {
      const offset = (options.heroPlayer ?? 0) * 2;
      knownPrivate[offset] = options.heroHand[0];
      knownPrivate[offset + 1] = options.heroHand[1];
    }

    const blocked = new Set<number>([...publicCards]);
    const privateCards: number[] = [];
    const candidates = completeDeck(
      [...knownPrivate.filter((card): card is number => card !== undefined), ...this.deck],
      publicCards,
    );
    for (let slot = 0; slot < 4; slot += 1) {
      const known = knownPrivate[slot];
      if (known !== undefined && !blocked.has(known) && !privateCards.includes(known)) {
        privateCards.push(known);
        blocked.add(known);
        continue;
      }
      const next = candidates.find(
        (card) => !blocked.has(card) && !privateCards.includes(card),
      );
      if (next === undefined) {
        throw new Error("could not construct deterministic deck placeholders");
      }
      privateCards.push(next);
      blocked.add(next);
    }

    this.holeIndices = [
      [privateCards[0]!, privateCards[1]!],
      [privateCards[2]!, privateCards[3]!],
    ];
    this.boardIndices = [-1, -1, -1, -1, -1];
    for (let i = 0; i < publicCards.length; i += 1) {
      this.boardIndices[i] = publicCards[i]!;
    }
    this.lastBoardIndices = [...this.boardIndices];
    this.street = street;

    if (street > 0 && this.actionsThisRound === 0) {
      this.toAct = (1 - this.button) as PlayerIndex;
      this.lastToAct = this.button;
      this.committed = [0, 0];
      this.minRaise = this.bb;
      this.actedSinceReset = false;
    }

    const future = completeDeck(this.deck, [...blocked]);
    const deck = [...privateCards, ...publicCards];
    for (const card of future) {
      if (deck.length >= 9) break;
      deck.push(card);
    }
    this.deck = deck;
    this.deckPos = 4 + publicCards.length;
  }

  legalBinsAmountAndMask(): LegalBins {
    const numBins = this.betBins.length + 3;
    const amounts = new Array<number>(numBins).fill(-1);
    const mask = new Array<number>(numBins).fill(0);
    const me = this.toAct;
    const opp = (1 - me) as PlayerIndex;
    const meStack = this.stacks[me];
    const oppStack = this.stacks[opp];
    const meCommitted = this.committed[me];
    const oppCommitted = this.committed[opp];
    const toCall = oppCommitted - meCommitted;

    mask[0] = toCall > 0 ? 1 : 0;
    mask[1] = 1;

    const canBetRaise = meStack > 0 && oppStack > 0;
    for (let i = 0; i < this.betBins.length; i += 1) {
      const bin = i + 2;
      const additional = Math.trunc(this.betBins[i]! * this.pot);
      const amount = toCall + additional;
      amounts[bin] = amount;
      mask[bin] =
        canBetRaise && amount <= meStack && additional >= this.minRaise ? 1 : 0;
    }

    const allInIndex = numBins - 1;
    amounts[allInIndex] = meStack;
    mask[allInIndex] = meStack > 0 ? 1 : 0;

    if (this.isAllin[opp]) {
      for (let i = 0; i < numBins; i += 1) mask[i] = 0;
      mask[0] = 1;
      mask[1] = 1;
    }
    if (this.isAllin[me]) {
      for (let i = 0; i < numBins; i += 1) mask[i] = 0;
      mask[1] = 1;
    }
    if (this.done) {
      for (let i = 0; i < numBins; i += 1) mask[i] = 0;
    }
    return { amounts, mask };
  }

  stepBin(bin: number, legal?: LegalBins): StepResult {
    const bins = legal ?? this.legalBinsAmountAndMask();
    const numBins = this.betBins.length + 3;
    const allInIndex = numBins - 1;

    let action = -1;
    let amount = 0;
    if (bin === 0) action = 0;
    else if (bin === 1) action = 1;
    else if (bin >= 2 && bin < allInIndex) {
      action = 2;
      amount = bins.amounts[bin] ?? -1;
    } else if (bin === allInIndex) {
      action = 3;
    }

    if (action < 0) {
      throw new Error(`unsupported action bin ${bin}`);
    }
    return this.step(action, amount);
  }

  private step(action: number, betAmount: number): StepResult {
    const actor = this.toAct;
    const other = (1 - actor) as PlayerIndex;
    const toCall = this.committed[other] - this.committed[actor];
    let reward = 0;
    let newStreet = -1;
    const dealtCards = [-1, -1, -1];

    this.actedSinceReset = true;

    if (action === 0) {
      reward = this.finishAndAssignRewards(other);
      this.hasFolded[actor] = true;
    } else if (action === 1) {
      this.bet(Math.min(toCall, this.stacks[actor]));
    } else if (action === 2) {
      this.bet(betAmount);
      this.minRaise = Math.max(this.minRaise, betAmount - toCall);
    } else if (action === 3) {
      const allInAmount = this.stacks[actor];
      this.bet(allInAmount);
      this.isAllin[actor] = true;
    } else {
      throw new Error(`unsupported concrete action ${action}`);
    }

    this.lastToAct = this.toAct;
    this.toAct = other;
    this.actionsThisRound += 1;
    this.lastBoardIndices = [...this.boardIndices];

    const equalCommitted = this.committed[0] === this.committed[1];
    const allInCommitted =
      (this.isAllin[0] && this.isAllin[1]) ||
      (this.isAllin[0] && this.committed[0] <= this.committed[1]) ||
      (this.isAllin[1] && this.committed[1] <= this.committed[0]);
    const roundClosed =
      !this.done &&
      (equalCommitted || allInCommitted) &&
      this.actionsThisRound >= 2;

    if (roundClosed) {
      this.committed = [0, 0];
      this.actionsLastRound = this.actionsThisRound;
      this.actionsThisRound = 0;
      this.toAct = (1 - this.button) as PlayerIndex;
      this.minRaise = this.bb;

      const startingStreet = this.street;
      const shouldShowdown = startingStreet === (this.flopShowdown ? 0 : 3);
      if (shouldShowdown) {
        reward = this.finishShowdown();
      }

      if (startingStreet === 0) {
        this.dealBoard(0, 3, dealtCards);
        newStreet = 1;
      } else if (startingStreet === 1) {
        this.dealBoard(3, 1, dealtCards);
        newStreet = 2;
      } else if (startingStreet === 2) {
        this.dealBoard(4, 1, dealtCards);
        newStreet = 3;
      }
      this.street += 1;
    }

    return { reward, newStreet, dealtCards };
  }

  private bet(chips: number): void {
    const amount = Math.max(0, Math.trunc(chips));
    const player = this.toAct;
    this.stacks[player] -= amount;
    this.committed[player] += amount;
    this.pot += amount;
  }

  private dealBoard(start: number, count: number, dealtCards: number[]): void {
    for (let i = 0; i < count; i += 1) {
      const card = this.deck[this.deckPos + i];
      if (card === undefined) {
        throw new Error("deterministic deck ran out of board cards");
      }
      this.boardIndices[start + i] = card;
      dealtCards[i] = card;
    }
    this.deckPos += count;
  }

  private finishAndAssignRewards(winner: number): number {
    this.winner = winner;
    this.done = true;
    const potShare =
      winner === 0 ? this.pot : winner === 2 ? this.pot / 2.0 : 0.0;
    return (this.stacks[0] + potShare - this.startingStacks[0]) / this.scale;
  }

  private finishShowdown(): number {
    const board = this.boardIndices.filter((card) => card >= 0);
    if (board.length < 5) {
      throw new Error("showdown requires a complete five-card board");
    }
    const p0 = evaluateSeven([...this.holeIndices[0], ...board]);
    const p1 = evaluateSeven([...this.holeIndices[1], ...board]);
    const cmp = compareRanks(p0, p1);
    return this.finishAndAssignRewards(cmp > 0 ? 0 : cmp < 0 ? 1 : 2);
  }
}

export function initialUniformBeliefs(): Float32Array<ArrayBuffer> {
  const beliefs = new Float32Array(2 * NUM_HANDS);
  beliefs.fill(1 / NUM_HANDS);
  return beliefs;
}

export function handCombos(): Array<[number, number]> {
  return HAND_COMBOS.map(([c0, c1]) => [c0, c1]);
}

export function showdownTerminalValues(
  env: PublicHunlEnv,
  beliefs: Float32Array<ArrayBufferLike>,
): Float32Array<ArrayBuffer> {
  const board = env.boardIndices.filter((card) => card >= 0);
  if (board.length !== 5) {
    throw new Error("showdown terminal values require a complete five-card board");
  }

  const boardSet = new Set(board);
  const handRanks: Array<RankVector | undefined> = new Array(NUM_HANDS);
  for (let hand = 0; hand < NUM_HANDS; hand += 1) {
    const [c0, c1] = handComboCards(hand);
    if (boardSet.has(c0) || boardSet.has(c1)) continue;
    handRanks[hand] = evaluateSeven([c0, c1, ...board]);
  }

  const winValue = (env.stacks[0] + env.pot - env.startingStacks[0]) / env.scale;
  const loseValue = (env.stacks[0] - env.startingStacks[0]) / env.scale;
  const tieValue =
    (env.stacks[0] + env.pot / 2 - env.startingStacks[0]) / env.scale;
  const payoffForPlayer0 = (cmp: number): number =>
    cmp > 0 ? winValue : cmp < 0 ? loseValue : tieValue;

  const out = new Float32Array(2 * NUM_HANDS);
  for (let hand = 0; hand < NUM_HANDS; hand += 1) {
    const rank = handRanks[hand];
    if (!rank) continue;

    let p0Value = 0;
    let p0Mass = 0;
    let p1Value = 0;
    let p1Mass = 0;
    for (let oppHand = 0; oppHand < NUM_HANDS; oppHand += 1) {
      const oppRank = handRanks[oppHand];
      if (!oppRank || handsOverlap(hand, oppHand)) continue;

      const cmpP0Hand = compareRanks(rank, oppRank);
      const p0Payoff = payoffForPlayer0(cmpP0Hand);
      const p1Belief = beliefs[NUM_HANDS + oppHand]!;
      p0Value += p1Belief * p0Payoff;
      p0Mass += p1Belief;

      const cmpOppP0Hand = compareRanks(oppRank, rank);
      const p0PayoffAgainstP1 = payoffForPlayer0(cmpOppP0Hand);
      const p0Belief = beliefs[oppHand]!;
      p1Value += p0Belief * -p0PayoffAgainstP1;
      p1Mass += p0Belief;
    }
    out[hand] = p0Mass > 1.0e-12 ? p0Value / p0Mass : 0;
    out[NUM_HANDS + hand] = p1Mass > 1.0e-12 ? p1Value / p1Mass : 0;
  }
  return out;
}

export function showdownTerminalRankCodes(env: PublicHunlEnv): Uint32Array<ArrayBuffer> {
  const board = env.boardIndices.filter((card) => card >= 0);
  if (board.length !== 5) {
    throw new Error("showdown terminal ranks require a complete five-card board");
  }

  const boardSet = new Set(board);
  const out = new Uint32Array(NUM_HANDS);
  for (let hand = 0; hand < NUM_HANDS; hand += 1) {
    const [c0, c1] = handComboCards(hand);
    if (boardSet.has(c0) || boardSet.has(c1)) continue;
    out[hand] = encodeRankCode(evaluateSeven([c0, c1, ...board]));
  }
  return out;
}

function encodeRankCode(rank: RankVector): number {
  let out = rank[0] ?? 0;
  for (let i = 1; i <= 5; i += 1) {
    out = out * 13 + (rank[i] ?? 0);
  }
  return out;
}

export function encodeBetterFeatures(env: PublicHunlEnv): {
  context: Float32Array<ArrayBuffer>;
  street: number;
  board: number[];
} {
  const context = new Float32Array(11);
  context[0] = env.toAct;
  context[1] = (env.toAct - env.button + 2) % 2;
  context[2] = env.actionsThisRound;
  context[3] = env.pot;
  context[4] = env.minRaise;
  context[5] = env.stacks[0];
  context[6] = env.stacks[1];
  context[7] = env.committed[0];
  context[8] = env.committed[1];
  context[9] = env.stacks[0] / env.pot;
  context[10] = env.stacks[1] / env.pot;
  return {
    context,
    street: env.street,
    board: [...env.boardIndices],
  };
}

type RankVector = number[];

function evaluateSeven(cards: number[]): RankVector {
  if (cards.length !== 7) {
    throw new Error(`evaluateSeven expected 7 cards, got ${cards.length}`);
  }
  let best: RankVector | undefined;
  for (let a = 0; a < 3; a += 1) {
    for (let b = a + 1; b < 4; b += 1) {
      for (let c = b + 1; c < 5; c += 1) {
        for (let d = c + 1; d < 6; d += 1) {
          for (let e = d + 1; e < 7; e += 1) {
            const rank = evaluateFive([
              cards[a]!,
              cards[b]!,
              cards[c]!,
              cards[d]!,
              cards[e]!,
            ]);
            if (!best || compareRanks(rank, best) > 0) best = rank;
          }
        }
      }
    }
  }
  if (!best) throw new Error("internal showdown evaluation error");
  return best;
}

function evaluateFive(cards: number[]): RankVector {
  const ranks = cards.map((card) => card % 13).sort((a, b) => b - a);
  const suits = cards.map((card) => Math.floor(card / 13));
  const flush = suits.every((suit) => suit === suits[0]);
  const straightHigh = getStraightHigh(ranks);
  const counts = new Map<number, number>();
  for (const rank of ranks) counts.set(rank, (counts.get(rank) ?? 0) + 1);
  const groups = [...counts.entries()].sort(
    (a, b) => b[1] - a[1] || b[0] - a[0],
  );

  if (flush && straightHigh >= 0) return [8, straightHigh];
  if (groups[0]?.[1] === 4) {
    const quad = groups[0][0];
    const kicker = groups.find(([rank]) => rank !== quad)?.[0] ?? 0;
    return [7, quad, kicker];
  }
  if (groups[0]?.[1] === 3 && groups[1]?.[1] === 2) {
    return [6, groups[0][0], groups[1][0]];
  }
  if (flush) return [5, ...ranks];
  if (straightHigh >= 0) return [4, straightHigh];
  if (groups[0]?.[1] === 3) {
    return [
      3,
      groups[0][0],
      ...groups.filter(([, count]) => count === 1).map(([rank]) => rank),
    ];
  }
  if (groups[0]?.[1] === 2 && groups[1]?.[1] === 2) {
    const pairRanks = groups
      .filter(([, count]) => count === 2)
      .map(([rank]) => rank)
      .sort((a, b) => b - a);
    const kicker = groups.find(([, count]) => count === 1)?.[0] ?? 0;
    return [2, ...pairRanks, kicker];
  }
  if (groups[0]?.[1] === 2) {
    return [
      1,
      groups[0][0],
      ...groups.filter(([, count]) => count === 1).map(([rank]) => rank),
    ];
  }
  return [0, ...ranks];
}

function getStraightHigh(sortedRanksDesc: number[]): number {
  const unique = [...new Set(sortedRanksDesc)].sort((a, b) => b - a);
  if ([12, 3, 2, 1, 0].every((rank) => unique.includes(rank))) {
    return 3;
  }
  for (let high = 12; high >= 4; high -= 1) {
    if ([0, 1, 2, 3, 4].every((delta) => unique.includes(high - delta))) {
      return high;
    }
  }
  return -1;
}

function compareRanks(a: RankVector, b: RankVector): number {
  const n = Math.max(a.length, b.length);
  for (let i = 0; i < n; i += 1) {
    const av = a[i] ?? 0;
    const bv = b[i] ?? 0;
    if (av !== bv) return av - bv;
  }
  return 0;
}
