export const NUM_CARDS = 52;
export const NUM_HAND_COMBOS = 1326;

const RANK_TO_INDEX = new Map<string, number>([
  ["2", 0],
  ["3", 1],
  ["4", 2],
  ["5", 3],
  ["6", 4],
  ["7", 5],
  ["8", 6],
  ["9", 7],
  ["T", 8],
  ["10", 8],
  ["J", 9],
  ["Q", 10],
  ["K", 11],
  ["A", 12],
]);
const INDEX_TO_RANK = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"];
const SUIT_TO_INDEX = new Map<string, number>([
  ["c", 0],
  ["d", 1],
  ["h", 2],
  ["s", 3],
]);
const INDEX_TO_SUIT = ["c", "d", "h", "s"];

export const HAND_COMBOS: Array<[number, number]> = (() => {
  const combos: Array<[number, number]> = [];
  for (let c0 = 0; c0 < NUM_CARDS; c0 += 1) {
    for (let c1 = c0 + 1; c1 < NUM_CARDS; c1 += 1) {
      combos.push([c0, c1]);
    }
  }
  return combos;
})();

export function assertCardIndex(card: number, label = "card"): void {
  if (!Number.isInteger(card) || card < 0 || card >= NUM_CARDS) {
    throw new Error(`${label} must be an integer in [0, ${NUM_CARDS})`);
  }
}

export function cardIndex(rank: string, suit: string): number {
  const rankIndex = RANK_TO_INDEX.get(rank.toUpperCase());
  const suitIndex = SUIT_TO_INDEX.get(suit.toLowerCase());
  if (rankIndex === undefined || suitIndex === undefined) {
    throw new Error(`invalid card "${rank}${suit}"`);
  }
  return suitIndex * 13 + rankIndex;
}

export function parseCard(input: string): number {
  const value = input.trim();
  const match = /^([2-9TJQKA]|10)([cdhs])$/i.exec(value);
  if (!match) {
    throw new Error(`invalid card "${input}"`);
  }
  return cardIndex(match[1]!, match[2]!);
}

export function parseCards(input: string | readonly string[]): number[] {
  const parts =
    typeof input === "string"
      ? input
          .split(/[\s,]+/)
          .map((part) => part.trim())
          .filter(Boolean)
      : [...input];
  const cards = parts.map((part) => parseCard(part));
  assertUniqueCards(cards);
  return cards;
}

export function formatCard(card: number): string {
  assertCardIndex(card);
  const rank = card % 13;
  const suit = Math.floor(card / 13);
  return `${INDEX_TO_RANK[rank]}${INDEX_TO_SUIT[suit]}`;
}

export function assertUniqueCards(cards: readonly number[], label = "cards"): void {
  const seen = new Set<number>();
  for (const card of cards) {
    assertCardIndex(card, label);
    if (seen.has(card)) {
      throw new Error(`${label} contain duplicate card ${formatCard(card)}`);
    }
    seen.add(card);
  }
}

export function handComboIndex(cardA: number, cardB: number): number {
  assertCardIndex(cardA, "cardA");
  assertCardIndex(cardB, "cardB");
  if (cardA === cardB) {
    throw new Error(`hand combo cannot contain duplicate card ${formatCard(cardA)}`);
  }
  const first = Math.min(cardA, cardB);
  const second = Math.max(cardA, cardB);
  const preceding = first * 51 - (first * (first - 1)) / 2;
  return preceding + (second - first - 1);
}

export function handComboCards(index: number): [number, number] {
  if (!Number.isInteger(index) || index < 0 || index >= NUM_HAND_COMBOS) {
    throw new Error(`hand combo index must be in [0, ${NUM_HAND_COMBOS})`);
  }
  return HAND_COMBOS[index]!;
}

export function cardsOverlap(a: readonly number[], b: readonly number[]): boolean {
  const seen = new Set(a);
  return b.some((card) => seen.has(card));
}

export function handOverlapsCards(
  handIndex: number,
  blockedCards: ReadonlySet<number> | readonly number[],
): boolean {
  const [c0, c1] = handComboCards(handIndex);
  const blocked = blockedCards instanceof Set ? blockedCards : new Set<number>(blockedCards);
  return blocked.has(c0) || blocked.has(c1);
}

export function handsOverlap(handA: number, handB: number): boolean {
  const [a0, a1] = handComboCards(handA);
  const [b0, b1] = handComboCards(handB);
  return a0 === b0 || a0 === b1 || a1 === b0 || a1 === b1;
}
