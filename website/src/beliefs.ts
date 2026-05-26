import { assertUniqueCards, handOverlapsCards, NUM_HAND_COMBOS } from "./cards.js";
import type { PlayerIndex } from "./types.js";

export function normalizeBeliefVector(
  values: Float32Array<ArrayBufferLike>,
  player: PlayerIndex,
): number {
  const offset = player * NUM_HAND_COMBOS;
  let sum = 0;
  for (let i = 0; i < NUM_HAND_COMBOS; i += 1) {
    const value = values[offset + i]!;
    if (!Number.isFinite(value) || value < 0) {
      throw new Error(`belief for player ${player} hand ${i} must be finite and non-negative`);
    }
    sum += value;
  }
  if (sum <= 0) {
    throw new Error(`belief vector for player ${player} has zero mass`);
  }
  const scale = 1 / sum;
  for (let i = 0; i < NUM_HAND_COMBOS; i += 1) {
    values[offset + i] = values[offset + i]! * scale;
  }
  return sum;
}

export function normalizeBeliefs(
  input: Float32Array<ArrayBufferLike> | ArrayLike<number>,
): Float32Array<ArrayBuffer> {
  if (input.length !== 2 * NUM_HAND_COMBOS) {
    throw new Error(`initialBeliefs has ${input.length} entries, expected ${2 * NUM_HAND_COMBOS}`);
  }
  const beliefs = new Float32Array(2 * NUM_HAND_COMBOS);
  for (let i = 0; i < input.length; i += 1) {
    beliefs[i] = Number(input[i]);
  }
  normalizeBeliefVector(beliefs, 0);
  normalizeBeliefVector(beliefs, 1);
  return beliefs;
}

export function buildPublicBeliefs(
  options: { publicCards?: readonly number[] } = {},
): Float32Array<ArrayBuffer> {
  const publicCards = options.publicCards ?? [];
  assertUniqueCards(publicCards);

  const beliefs = new Float32Array(2 * NUM_HAND_COMBOS);
  const blocked = new Set<number>(publicCards);
  for (let player = 0; player < 2; player += 1) {
    const offset = player * NUM_HAND_COMBOS;
    for (let hand = 0; hand < NUM_HAND_COMBOS; hand += 1) {
      beliefs[offset + hand] = handOverlapsCards(hand, blocked) ? 0 : 1;
    }
    normalizeBeliefVector(beliefs, player as PlayerIndex);
  }
  return beliefs;
}

export function compatibleHandMask(blockedCards: readonly number[]): Uint8Array {
  assertUniqueCards(blockedCards);
  const blocked = new Set(blockedCards);
  const mask = new Uint8Array(NUM_HAND_COMBOS);
  for (let hand = 0; hand < NUM_HAND_COMBOS; hand += 1) {
    mask[hand] = handOverlapsCards(hand, blocked) ? 0 : 1;
  }
  return mask;
}
