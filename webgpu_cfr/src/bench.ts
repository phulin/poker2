import { dirname, resolve } from "node:path";
import { performance } from "node:perf_hooks";
import { BrowserCfrEvaluator } from "./browserEvaluator.js";
import { createDawnDevice } from "./gpu.js";
import { loadNodeModel } from "./nodeModel.js";

interface BenchOptions {
  manifest: string;
  weights?: string;
  spot: number[];
  iterations: number;
  warmups: number;
  runs: number;
}

function getArg(args: string[], name: string, fallback?: string): string {
  const index = args.indexOf(name);
  const value = index >= 0 ? args[index + 1] : undefined;
  if (value) return value;
  if (fallback !== undefined) return fallback;
  throw new Error(`Missing ${name}`);
}

function parsePositiveInt(value: string, name: string): number {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${name} must be a positive integer`);
  }
  return parsed;
}

function parseNonNegativeInt(value: string, name: string): number {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`${name} must be a non-negative integer`);
  }
  return parsed;
}

function parseSpot(value: string): number[] {
  return value
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => {
      const action = Number.parseInt(part, 10);
      if (!Number.isInteger(action)) {
        throw new Error(`invalid spot action "${part}"`);
      }
      return action;
    });
}

function readArgs(): BenchOptions {
  const args = process.argv.slice(2);
  const weights = getArg(args, "--weights", "");
  const options: BenchOptions = {
    manifest: getArg(args, "--manifest"),
    spot: parseSpot(getArg(args, "--spot", "1")),
    iterations: parsePositiveInt(getArg(args, "--iterations", "8"), "iterations"),
    warmups: parseNonNegativeInt(getArg(args, "--warmups", "1"), "warmups"),
    runs: parsePositiveInt(getArg(args, "--runs", "5"), "runs"),
  };
  if (weights) options.weights = weights;
  return options;
}

function stats(samples: number[]): {
  meanMs: number;
  medianMs: number;
  minMs: number;
  maxMs: number;
} {
  const sorted = [...samples].sort((a, b) => a - b);
  const mean = samples.reduce((sum, sample) => sum + sample, 0) / samples.length;
  return {
    meanMs: mean,
    medianMs: sorted[Math.floor(sorted.length / 2)]!,
    minMs: sorted[0]!,
    maxMs: sorted[sorted.length - 1]!,
  };
}

const options = readArgs();
const weightsPath =
  options.weights ?? resolve(dirname(options.manifest), "weights.bin");
const device = await createDawnDevice();
const model = await loadNodeModel(device, options.manifest, weightsPath);
const evaluator = new BrowserCfrEvaluator(device, model);

try {
  for (let i = 0; i < options.warmups; i += 1) {
    await evaluator.evaluateSpot({
      spot: options.spot,
      iterations: options.iterations,
    });
  }

  const samples: number[] = [];
  let actionLabels: string[] = [];
  let actionProbs: number[] = [];
  for (let i = 0; i < options.runs; i += 1) {
    const start = performance.now();
    const result = await evaluator.evaluateSpot({
      spot: options.spot,
      iterations: options.iterations,
    });
    samples.push(performance.now() - start);
    actionLabels = result.actionLabels;
    actionProbs = Array.from(result.actionProbs);
  }

  console.log(
    JSON.stringify(
      {
        spot: options.spot,
        iterations: options.iterations,
        warmups: options.warmups,
        runs: options.runs,
        ...stats(samples),
        samplesMs: samples,
        actionLabels,
        actionProbs,
      },
      null,
      2,
    ),
  );
} finally {
  evaluator.dispose();
  model.dispose();
  device.destroy();
}
