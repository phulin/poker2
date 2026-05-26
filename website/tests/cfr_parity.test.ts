import assert from "node:assert/strict";
import { test } from "node:test";
import { evaluateFixture } from "../src/evaluator.js";
import { createDawnDevice } from "../src/gpu.js";
import { loadPythonReference } from "../src/pythonBridge.js";

function assertCloseArray(
  actual: Float32Array,
  expected: number[],
  atol: number,
  label: string,
): void {
  assert.equal(actual.length, expected.length, `${label} length`);
  let maxDiff = 0;
  for (let i = 0; i < actual.length; i += 1) {
    const diff = Math.abs(actual[i]! - expected[i]!);
    if (diff > maxDiff) maxDiff = diff;
  }
  assert.ok(maxDiff <= atol, `${label} max diff ${maxDiff} > ${atol}`);
}

test("Dawn WebGPU CFR kernel matches BetterFFN Python reference", async () => {
  const fixture = loadPythonReference({
    snapshot: "checkpoints-rebel/rebel_latest.pt",
    spot: [1],
    iterations: 8,
  });
  assert.ok(fixture.expected);
  const device = await createDawnDevice();
  try {
    const result = await evaluateFixture(device, fixture);

    assertCloseArray(result.beliefsAtSpot, fixture.expected.beliefsAtSpot, 5e-6, "beliefsAtSpot");
    assertCloseArray(result.actionProbs, fixture.expected.actionProbs, 5e-6, "actionProbs");
    assertCloseArray(result.policy, fixture.expected.policy, 5e-6, "policy");
  } finally {
    device.destroy();
  }
});
