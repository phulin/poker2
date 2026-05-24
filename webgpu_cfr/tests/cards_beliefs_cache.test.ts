import assert from "node:assert/strict";
import { test } from "node:test";
import {
  formatCard,
  handComboCards,
  handComboIndex,
  NUM_HAND_COMBOS,
  parseCard,
  parseCards,
} from "../src/cards.js";
import {
  buildHeroOnlyBeliefs,
  compatibleHandMask,
} from "../src/beliefs.js";
import { isCachedModelFresh } from "../src/modelCache.js";
import { resolveCfrDefaults } from "../src/modelFormat.js";
import type { BetterFfnManifest } from "../src/types.js";

function sum(values: Float32Array, start: number, count: number): number {
  let total = 0;
  for (let i = 0; i < count; i += 1) total += values[start + i]!;
  return total;
}

test("card parser accepts standard notation and rejects duplicates", () => {
  assert.equal(parseCard("As"), 51);
  assert.equal(parseCard("kd"), 24);
  assert.equal(parseCard("10h"), parseCard("Th"));
  assert.deepEqual(parseCards("As, Kd 2c"), [51, 24, 0]);
  assert.throws(() => parseCards(["As", "as"]), /duplicate card As/i);
});

test("hand combo index round-trips sorted and unsorted cards", () => {
  const first = parseCard("2c");
  const second = parseCard("As");
  const index = handComboIndex(second, first);
  assert.equal(index, 50);
  assert.deepEqual(handComboCards(index), [first, second]);
  assert.equal(formatCard(handComboCards(NUM_HAND_COMBOS - 1)[1]), "As");
});

test("hero-only beliefs set exact hero hand and mask villain blockers", () => {
  const heroHand = [parseCard("As"), parseCard("Kd")] as [number, number];
  const publicCards = [parseCard("2c"), parseCard("7d"), parseCard("Th")];
  const beliefs = buildHeroOnlyBeliefs({
    heroPlayer: 1,
    heroHand,
    publicCards,
  });
  const heroIndex = handComboIndex(heroHand[0], heroHand[1]);
  assert.equal(beliefs[NUM_HAND_COMBOS + heroIndex], 1);
  assert.ok(Math.abs(sum(beliefs, NUM_HAND_COMBOS, NUM_HAND_COMBOS) - 1) < 1e-6);
  assert.ok(Math.abs(sum(beliefs, 0, NUM_HAND_COMBOS) - 1) < 1e-6);

  const blockedByHero = handComboIndex(parseCard("As"), parseCard("Ac"));
  const blockedByBoard = handComboIndex(parseCard("2c"), parseCard("3c"));
  assert.equal(beliefs[blockedByHero], 0);
  assert.equal(beliefs[blockedByBoard], 0);
});

test("compatible hand mask blocks public cards", () => {
  const mask = compatibleHandMask([parseCard("As"), parseCard("Kd")]);
  assert.equal(mask[handComboIndex(parseCard("As"), parseCard("2c"))], 0);
  assert.equal(mask[handComboIndex(parseCard("Qh"), parseCard("Jd"))], 1);
});

test("cached model metadata invalidates on size or sha mismatch", () => {
  const manifest = {
    weights: { byteLength: 1234, sha256: "abc" },
  } as BetterFfnManifest;
  assert.equal(
    isCachedModelFresh({ byteLength: 1234, sha256: "abc" }, manifest),
    true,
  );
  assert.equal(
    isCachedModelFresh({ byteLength: 1235, sha256: "abc" }, manifest),
    false,
  );
  assert.equal(
    isCachedModelFresh({ byteLength: 1234, sha256: "def" }, manifest),
    false,
  );
});

test("CFR defaults resolve from manifest configuration", () => {
  const manifest = {
    cfr: { iterations: 1000, depth: 4, cfrAvg: false },
  } as BetterFfnManifest;
  assert.deepEqual(resolveCfrDefaults(manifest), {
    iterations: 1000,
    depth: 4,
    cfrAvg: false,
  });
  assert.deepEqual(resolveCfrDefaults({} as BetterFfnManifest), {
    iterations: 16,
    depth: 6,
    cfrAvg: true,
  });
});
