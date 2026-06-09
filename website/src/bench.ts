import { performance } from "node:perf_hooks";
import { BrowserCfrEvaluator } from "./browserEvaluator.js";
import { createDawnDevice } from "./gpu.js";
import { rootModelForRuntime } from "./modelRegistry.js";
import { resolveCfrDefaults } from "./modelFormat.js";
import { loadNodeRuntime } from "./nodeModel.js";

interface BenchOptions {
  manifest: string;
  weights?: string;
  spot: number[];
  iterations?: number;
  depth?: number;
  cfrAvg?: boolean;
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
  const iterations = getArg(args, "--iterations", "");
  const depth = getArg(args, "--depth", "");
  const options: BenchOptions = {
    manifest: getArg(args, "--manifest"),
    spot: parseSpot(getArg(args, "--spot", "1")),
    warmups: parseNonNegativeInt(getArg(args, "--warmups", "1"), "warmups"),
    runs: parsePositiveInt(getArg(args, "--runs", "5"), "runs"),
  };
  if (iterations) options.iterations = parsePositiveInt(iterations, "iterations");
  if (depth) options.depth = parsePositiveInt(depth, "depth");
  if (args.includes("--cfr-avg")) options.cfrAvg = true;
  if (args.includes("--no-cfr-avg")) options.cfrAvg = false;
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
const device = await createDawnDevice();
const runtime = await loadNodeRuntime(device, options.manifest, options.weights);
const model = rootModelForRuntime(runtime);
const evaluator = new BrowserCfrEvaluator(device, runtime);
const cfrDefaults = resolveCfrDefaults(model.manifest);
const iterations = options.iterations ?? cfrDefaults.iterations;
const depth = options.depth ?? cfrDefaults.depth;
const cfrAvg = options.cfrAvg ?? cfrDefaults.cfrAvg;

try {
  for (let i = 0; i < options.warmups; i += 1) {
    await evaluator.evaluateSpot({
      spot: options.spot,
      iterations,
      depth,
      cfrAvg,
    });
  }

  const samples: number[] = [];
  let actionLabels: string[] = [];
  let actionProbs: number[] = [];
  for (let i = 0; i < options.runs; i += 1) {
    const start = performance.now();
    const result = await evaluator.evaluateSpot({
      spot: options.spot,
      iterations,
      depth,
      cfrAvg,
    });
    samples.push(performance.now() - start);
    actionLabels = result.actionLabels;
    actionProbs = Array.from(result.actionProbs);
  }

  console.log(
    JSON.stringify(
      {
        spot: options.spot,
        iterations,
        depth,
        cfrAvg,
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
  runtime.dispose();
  device.destroy();
}
