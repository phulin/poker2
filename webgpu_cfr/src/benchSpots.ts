import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { performance } from "node:perf_hooks";
import { BrowserCfrEvaluator } from "./browserEvaluator.js";
import { createDawnDevice } from "./gpu.js";
import { resolveCfrDefaults } from "./modelFormat.js";
import { loadNodeModel } from "./nodeModel.js";
import type { EvaluateSpotRequest, PlayerIndex } from "./types.js";

interface BenchSpot {
  spot_index: number;
  street: number;
  street_name: string;
  button: 0 | 1;
  hero_player: 0 | 1;
  hero_hand: [number, number];
  public_cards: number[];
  pot: number;
}

interface CliOptions {
  manifest: string;
  weights?: string;
  spotsFile: string;
  iterations?: number;
  depth: number;
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

function readOptions(): CliOptions {
  const args = process.argv.slice(2);
  const iterations = getArg(args, "--iterations", "");
  const opts: CliOptions = {
    manifest: getArg(
      args,
      "--manifest",
      "public/models/rebel_latest/model.json",
    ),
    spotsFile: getArg(args, "--spots", "bench_spots.json"),
    depth: parsePositiveInt(getArg(args, "--depth", "4"), "depth"),
    warmups: parseNonNegativeInt(getArg(args, "--warmups", "1"), "warmups"),
    runs: parsePositiveInt(getArg(args, "--runs", "3"), "runs"),
  };
  if (iterations) opts.iterations = parsePositiveInt(iterations, "iterations");
  if (args.includes("--cfr-avg")) opts.cfrAvg = true;
  if (args.includes("--no-cfr-avg")) opts.cfrAvg = false;
  const weights = getArg(args, "--weights", "");
  if (weights) opts.weights = weights;
  return opts;
}

function stats(samples: number[]): {
  meanMs: number;
  medianMs: number;
  minMs: number;
  maxMs: number;
} {
  const sorted = [...samples].sort((a, b) => a - b);
  const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
  return {
    meanMs: mean,
    medianMs: sorted[Math.floor(sorted.length / 2)]!,
    minMs: sorted[0]!,
    maxMs: sorted[sorted.length - 1]!,
  };
}

const options = readOptions();
const weightsPath =
  options.weights ?? resolve(dirname(options.manifest), "weights.bin");

const spots: BenchSpot[] = JSON.parse(
  await readFile(options.spotsFile, "utf8"),
);

const device = await createDawnDevice();
const model = await loadNodeModel(device, options.manifest, weightsPath);
const evaluator = new BrowserCfrEvaluator(device, model);
const cfrDefaults = resolveCfrDefaults(model.manifest);
const iterations = options.iterations ?? cfrDefaults.iterations;
const cfrAvg = options.cfrAvg ?? cfrDefaults.cfrAvg;

function makeRequest(spot: BenchSpot): EvaluateSpotRequest {
  const req: EvaluateSpotRequest = {
    spot: [],
    iterations,
    depth: options.depth,
    cfrAvg,
    initialState: { button: spot.button as PlayerIndex },
    heroPlayer: spot.hero_player as PlayerIndex,
    heroHand: spot.hero_hand,
  };
  if (spot.public_cards.length > 0) {
    req.publicCards = spot.public_cards;
  }
  return req;
}

interface SpotResult {
  spot_index: number;
  street: string;
  meanMs: number;
  medianMs: number;
  minMs: number;
  maxMs: number;
  samplesMs: number[];
}

const results: SpotResult[] = [];
try {
  console.error(
    `running ${spots.length} spots at depth=${options.depth}, ` +
      `iterations=${iterations}, cfrAvg=${cfrAvg}, ` +
      `warmups=${options.warmups}, runs=${options.runs}`,
  );
  for (const spot of spots) {
    const request = makeRequest(spot);
    for (let i = 0; i < options.warmups; i += 1) {
      await evaluator.evaluateSpot(request);
    }
    const samples: number[] = [];
    for (let i = 0; i < options.runs; i += 1) {
      const start = performance.now();
      await evaluator.evaluateSpot(request);
      samples.push(performance.now() - start);
    }
    const summary = stats(samples);
    results.push({
      spot_index: spot.spot_index,
      street: spot.street_name,
      ...summary,
      samplesMs: samples,
    });
    console.error(
      `  spot ${spot.spot_index} (${spot.street_name}): ` +
        `mean=${summary.meanMs.toFixed(1)}ms ` +
        `median=${summary.medianMs.toFixed(1)}ms ` +
        `min=${summary.minMs.toFixed(1)}ms ` +
        `max=${summary.maxMs.toFixed(1)}ms`,
    );
  }

  const perStreet: Record<string, number[]> = {};
  for (const r of results) {
    (perStreet[r.street] ??= []).push(r.meanMs);
  }
  const byStreet = Object.fromEntries(
    Object.entries(perStreet).map(([street, means]) => [
      street,
      { ...stats(means), spots: means.length },
    ]),
  );

  console.log(
    JSON.stringify(
      {
        depth: options.depth,
        iterations,
        cfrAvg,
        warmups: options.warmups,
        runs: options.runs,
        spotsFile: options.spotsFile,
        byStreet,
        results,
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
