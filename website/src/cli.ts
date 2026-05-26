import { BrowserCfrEvaluator } from "./browserEvaluator.js";
import { evaluateFixture } from "./evaluator.js";
import { createDawnDevice } from "./gpu.js";
import { resolveCfrDefaults } from "./modelFormat.js";
import { loadNodeModel } from "./nodeModel.js";
import { loadPythonReference } from "./pythonBridge.js";

interface CliOptions {
  snapshot: string;
  spot: number[];
  iterations?: number;
  depth?: number;
  cfrAvg?: boolean;
  manifest?: string;
  weights?: string;
}

function readArgs(): CliOptions {
  const args = process.argv.slice(2);
  const get = (name: string, fallback?: string): string => {
    const idx = args.indexOf(name);
    const value = idx >= 0 ? args[idx + 1] : undefined;
    if (value) return value;
    if (fallback !== undefined) return fallback;
    throw new Error(`Missing ${name}`);
  };
  const manifest = get("--manifest", "");
  const weights = get("--weights", "");
  const iterations = get("--iterations", "");
  const depth = get("--depth", "");
  const options: CliOptions = {
    snapshot: get("--snapshot", "checkpoints-rebel/rebel_latest.pt"),
    spot: get("--spot", "1")
      .split(",")
      .filter(Boolean)
      .map((v) => Number.parseInt(v, 10)),
  };
  if (iterations) options.iterations = Number.parseInt(iterations, 10);
  if (depth) options.depth = Number.parseInt(depth, 10);
  if (args.includes("--cfr-avg")) options.cfrAvg = true;
  if (args.includes("--no-cfr-avg")) options.cfrAvg = false;
  if (manifest) options.manifest = manifest;
  if (weights) options.weights = weights;
  return options;
}

const options = readArgs();
const device = await createDawnDevice();
let output: unknown;
try {
  if (options.manifest) {
    const manifestPath = options.manifest;
    const model = await loadNodeModel(device, manifestPath, options.weights);
    const evaluator = new BrowserCfrEvaluator(device, model);
    try {
      const cfrDefaults = resolveCfrDefaults(model.manifest);
      const iterations = options.iterations ?? cfrDefaults.iterations;
      const depth = options.depth ?? cfrDefaults.depth;
      const cfrAvg = options.cfrAvg ?? cfrDefaults.cfrAvg;
      const result = await evaluator.evaluateSpot({
        spot: options.spot,
        iterations,
        depth,
        cfrAvg,
      });
      output = {
        spot: options.spot,
        iterations,
        depth,
        cfrAvg,
        actionLabels: result.actionLabels,
        actionProbs: Array.from(result.actionProbs),
      };
    } finally {
      evaluator.dispose();
      model.dispose();
    }
  } else {
    const iterations = options.iterations ?? 8;
    const cfrAvg = options.cfrAvg ?? true;
    const fixture = loadPythonReference({ ...options, iterations });
    const result = await evaluateFixture(device, fixture);
    output = {
      spot: fixture.spot,
      depth: 1,
      iterations,
      cfrAvg,
      actionLabels: fixture.actionLabels,
      actionProbs: Array.from(result.actionProbs),
    };
  }
} finally {
  device.destroy();
}

console.log(JSON.stringify(output, null, 2));
