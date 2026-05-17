import { dirname, resolve } from "node:path";
import { createDawnDevice } from "./gpu.js";
import { BrowserCfrEvaluator } from "./browserEvaluator.js";
import { evaluateFixture } from "./evaluator.js";
import { loadNodeModel } from "./nodeModel.js";
import { loadPythonReference } from "./pythonBridge.js";

interface CliOptions {
  snapshot: string;
  spot: number[];
  iterations: number;
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
  const options: CliOptions = {
    snapshot: get("--snapshot", "checkpoints-rebel/rebel_latest.pt"),
    spot: get("--spot", "1")
      .split(",")
      .filter(Boolean)
      .map((v) => Number.parseInt(v, 10)),
    iterations: Number.parseInt(get("--iterations", "8"), 10),
  };
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
    const weightsPath =
      options.weights ?? resolve(dirname(manifestPath), "weights.bin");
    const model = await loadNodeModel(device, manifestPath, weightsPath);
    try {
      const result = await new BrowserCfrEvaluator(device, model).evaluateSpot({
        spot: options.spot,
        iterations: options.iterations,
      });
      output = {
        spot: options.spot,
        actionLabels: result.actionLabels,
        actionProbs: Array.from(result.actionProbs),
      };
    } finally {
      model.dispose();
    }
  } else {
    const fixture = loadPythonReference(options);
    const result = await evaluateFixture(device, fixture);
    output = {
      spot: fixture.spot,
      actionLabels: fixture.actionLabels,
      actionProbs: Array.from(result.actionProbs),
    };
  }
} finally {
  device.destroy();
}

console.log(JSON.stringify(output, null, 2));
