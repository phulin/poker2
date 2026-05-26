import { readFile } from "node:fs/promises";
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
  variantEnv: string;
  baselineValue?: string;
  candidateValue: string;
}

interface TimingSet {
  meanMs: number;
  medianMs: number;
  minMs: number;
  maxMs: number;
}

interface VariantResult extends TimingSet {
  samplesMs: number[];
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
  const baselineValue = getArg(args, "--baseline-value", "");
  const weights = getArg(args, "--weights", "");
  const opts: CliOptions = {
    manifest: getArg(args, "--manifest", "public/models/rebel_latest/model.json"),
    spotsFile: getArg(args, "--spots", "bench_spots.json"),
    depth: parsePositiveInt(getArg(args, "--depth", "4"), "depth"),
    warmups: parseNonNegativeInt(getArg(args, "--warmups", "1"), "warmups"),
    runs: parsePositiveInt(getArg(args, "--runs", "3"), "runs"),
    variantEnv: getArg(args, "--variant-env"),
    candidateValue: getArg(args, "--candidate-value"),
  };
  if (iterations) opts.iterations = parsePositiveInt(iterations, "iterations");
  if (args.includes("--cfr-avg")) opts.cfrAvg = true;
  if (args.includes("--no-cfr-avg")) opts.cfrAvg = false;
  if (baselineValue) opts.baselineValue = baselineValue;
  if (weights) opts.weights = weights;
  return opts;
}

function stats(samples: number[]): TimingSet {
  const sorted = [...samples].sort((a, b) => a - b);
  const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
  return {
    meanMs: mean,
    medianMs: sorted[Math.floor(sorted.length / 2)]!,
    minMs: sorted[0]!,
    maxMs: sorted[sorted.length - 1]!,
  };
}

function summarize(samples: number[]): VariantResult {
  return { ...stats(samples), samplesMs: samples };
}

function setVariant(envName: string, value: string | undefined): void {
  if (value === undefined || value.length === 0) {
    delete process.env[envName];
  } else {
    process.env[envName] = value;
  }
}

function variantLabel(value: string | undefined): string {
  return value && value.length > 0 ? value : "baseline";
}

const options = readOptions();
const spots: BenchSpot[] = JSON.parse(await readFile(options.spotsFile, "utf8"));

const originalVariant = process.env[options.variantEnv];
const device = await createDawnDevice();
const model = await loadNodeModel(device, options.manifest, options.weights);
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
    readPolicy: false,
    readActionProbs: false,
    readBeliefs: false,
  };
  if (spot.public_cards.length > 0) {
    req.publicCards = spot.public_cards;
  }
  return req;
}

async function timeVariant(
  request: EvaluateSpotRequest,
  envValue: string | undefined,
): Promise<number> {
  setVariant(options.variantEnv, envValue);
  const start = performance.now();
  await evaluator.evaluateSpot(request);
  return performance.now() - start;
}

interface SpotResult {
  spot_index: number;
  street: string;
  baseline: VariantResult;
  candidate: VariantResult;
  deltaMs: number;
  speedup: number;
}

const results: SpotResult[] = [];
try {
  console.error(
    `running ${spots.length} spots interleaved at depth=${options.depth}, ` +
      `iterations=${iterations}, cfrAvg=${cfrAvg}, ` +
      `warmups=${options.warmups}, runs=${options.runs}, ` +
      `${options.variantEnv}: ${variantLabel(options.baselineValue)} vs ` +
      `${variantLabel(options.candidateValue)}`,
  );
  for (const spot of spots) {
    const request = makeRequest(spot);
    for (let i = 0; i < options.warmups; i += 1) {
      const baselineFirst = i % 2 === 0;
      if (baselineFirst) {
        await timeVariant(request, options.baselineValue);
        await timeVariant(request, options.candidateValue);
      } else {
        await timeVariant(request, options.candidateValue);
        await timeVariant(request, options.baselineValue);
      }
    }

    const baselineSamples: number[] = [];
    const candidateSamples: number[] = [];
    for (let i = 0; i < options.runs; i += 1) {
      const baselineFirst = i % 2 === 0;
      if (baselineFirst) {
        baselineSamples.push(await timeVariant(request, options.baselineValue));
        candidateSamples.push(await timeVariant(request, options.candidateValue));
      } else {
        candidateSamples.push(await timeVariant(request, options.candidateValue));
        baselineSamples.push(await timeVariant(request, options.baselineValue));
      }
    }

    const baseline = summarize(baselineSamples);
    const candidate = summarize(candidateSamples);
    const deltaMs = candidate.meanMs - baseline.meanMs;
    const speedup = baseline.meanMs / candidate.meanMs;
    results.push({
      spot_index: spot.spot_index,
      street: spot.street_name,
      baseline,
      candidate,
      deltaMs,
      speedup,
    });
    console.error(
      `  spot ${spot.spot_index} (${spot.street_name}): ` +
        `baseline=${baseline.meanMs.toFixed(1)}ms ` +
        `candidate=${candidate.meanMs.toFixed(1)}ms ` +
        `speedup=${speedup.toFixed(3)}x`,
    );
  }

  const perStreet: Record<string, { baseline: number[]; candidate: number[]; speedup: number[] }> =
    {};
  for (const r of results) {
    const bucket = (perStreet[r.street] ??= { baseline: [], candidate: [], speedup: [] });
    bucket.baseline.push(r.baseline.meanMs);
    bucket.candidate.push(r.candidate.meanMs);
    bucket.speedup.push(r.speedup);
  }
  const byStreet = Object.fromEntries(
    Object.entries(perStreet).map(([street, bucket]) => [
      street,
      {
        baseline: { ...stats(bucket.baseline), spots: bucket.baseline.length },
        candidate: { ...stats(bucket.candidate), spots: bucket.candidate.length },
        speedup: stats(bucket.speedup).meanMs,
        deltaMs: stats(bucket.candidate).meanMs - stats(bucket.baseline).meanMs,
      },
    ]),
  );
  const baselineMeans = results.map((r) => r.baseline.meanMs);
  const candidateMeans = results.map((r) => r.candidate.meanMs);
  const overallBaseline = stats(baselineMeans);
  const overallCandidate = stats(candidateMeans);

  console.log(
    JSON.stringify(
      {
        depth: options.depth,
        iterations,
        cfrAvg,
        warmups: options.warmups,
        runs: options.runs,
        spotsFile: options.spotsFile,
        variantEnv: options.variantEnv,
        baselineValue: options.baselineValue ?? "",
        candidateValue: options.candidateValue,
        overall: {
          baseline: overallBaseline,
          candidate: overallCandidate,
          deltaMs: overallCandidate.meanMs - overallBaseline.meanMs,
          speedup: overallBaseline.meanMs / overallCandidate.meanMs,
        },
        byStreet,
        results,
      },
      null,
      2,
    ),
  );
} finally {
  setVariant(options.variantEnv, originalVariant);
  evaluator.dispose();
  model.dispose();
  device.destroy();
}
