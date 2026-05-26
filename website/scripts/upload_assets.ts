#!/usr/bin/env tsx
import { spawnSync } from "node:child_process";
import { cp, mkdir, readdir, readFile, stat, writeFile } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";

const DEFAULT_BUCKET = "p2-webgpu-cfr-assets";
const DEFAULT_MODEL_VERSION = "rebel_296_4000";
const DEFAULT_ALLIN_VERSION = "holdem_v1";
const DEFAULT_ASSET_ORIGIN = "https://assets.holdem.computer";

interface CliArgs {
  bucket?: string;
  "model-version"?: string;
  "allin-version"?: string;
  "asset-origin"?: string;
  "source-model"?: string;
  "source-allin"?: string;
  staging?: string;
  "dry-run"?: string;
  latest?: string;
  wrangler?: string;
}

interface UploadSpec {
  key: string;
  file: string;
  contentType: string;
  cacheControl: string;
}

interface WebGpuModelManifest {
  weights: {
    file: string;
  };
  allIn?: AllInManifest;
  [key: string]: unknown;
}

interface AllInManifest {
  enabled?: boolean;
  scale?: number;
  preflop?: {
    file?: string;
    dtype?: "int16";
    scale?: number;
    [key: string]: unknown;
  };
  flop?: {
    actualToCanonFile: string;
    actualPermFile: string;
    comboPermsFile: string;
    tablePathTemplate: string;
    [key: string]: unknown;
  };
  metadata?: Record<string, unknown>;
  [key: string]: unknown;
}

const args = parseArgs(process.argv.slice(2));
const bucket = args.bucket ?? DEFAULT_BUCKET;
const modelVersion = args["model-version"] ?? DEFAULT_MODEL_VERSION;
const allinVersion = args["allin-version"] ?? DEFAULT_ALLIN_VERSION;
const assetOrigin = stripTrailingSlash(args["asset-origin"] ?? DEFAULT_ASSET_ORIGIN);
const sourceModel = resolve(args["source-model"] ?? "public/models/rebel_latest");
const sourceAllIn = resolve(args["source-allin"] ?? join(sourceModel, "allin"));
const stagingRoot = resolve(args.staging ?? "dist/r2-assets");
const dryRun = isTruthy(args["dry-run"]);
const latestAlias = isTruthy(args.latest);
const wrangler = args.wrangler ?? process.env.WRANGLER_BIN ?? "wrangler";

const modelPrefix = `models/${modelVersion}`;
const allinPrefix = `allin/${allinVersion}`;
const immutableCache = "public, max-age=31536000, immutable";
const latestCache = "public, max-age=300";

await mkdir(stagingRoot, { recursive: true });

const modelJsonPath = join(sourceModel, "model.json");
const model = JSON.parse(await readFile(modelJsonPath, "utf8")) as WebGpuModelManifest;
const weightsName = basename(model.weights.file);
const sourceWeights = join(sourceModel, weightsName);

const stagedModelDir = join(stagingRoot, modelPrefix);
await mkdir(stagedModelDir, { recursive: true });
await cp(sourceWeights, join(stagedModelDir, weightsName));

const allIn = await buildAllInManifest(sourceAllIn, allinPrefix);
if (allIn) model.allIn = allIn;
model.weights.file = weightsName;
await writeJson(join(stagedModelDir, "model.json"), model);

const uploads: UploadSpec[] = [
  uploadSpec(
    `${modelPrefix}/model.json`,
    join(stagedModelDir, "model.json"),
    "application/json",
    immutableCache,
  ),
  uploadSpec(
    `${modelPrefix}/${weightsName}`,
    join(stagedModelDir, weightsName),
    "application/gzip",
    immutableCache,
  ),
];

if (latestAlias) {
  const latestDir = join(stagingRoot, "models/rebel_latest");
  await mkdir(latestDir, { recursive: true });
  await cp(join(stagedModelDir, "model.json"), join(latestDir, "model.json"));
  await cp(join(stagedModelDir, weightsName), join(latestDir, weightsName));
  uploads.push(
    uploadSpec(
      "models/rebel_latest/model.json",
      join(latestDir, "model.json"),
      "application/json",
      latestCache,
    ),
    uploadSpec(
      `models/rebel_latest/${weightsName}`,
      join(latestDir, weightsName),
      "application/gzip",
      immutableCache,
    ),
  );
}

const stagedAllInDir = join(stagingRoot, allinPrefix);
if (await exists(stagedAllInDir)) {
  for (const file of await listFiles(stagedAllInDir)) {
    const key = toObjectKey(stagingRoot, file);
    uploads.push(uploadSpec(key, file, contentTypeFor(file), immutableCache));
  }
}

console.log(
  JSON.stringify(
    {
      bucket,
      modelManifestUrl: `${assetOrigin}/${modelPrefix}/model.json`,
      stagedFiles: uploads.length,
      stagingRoot,
    },
    null,
    2,
  ),
);

if (dryRun) process.exit(0);

for (const upload of uploads) {
  runWrangler([
    "r2",
    "object",
    "put",
    `${bucket}/${upload.key}`,
    "--remote",
    "--file",
    upload.file,
    "--content-type",
    upload.contentType,
    "--cache-control",
    upload.cacheControl,
  ]);
}

async function buildAllInManifest(
  sourceDir: string,
  prefix: string,
): Promise<AllInManifest | undefined> {
  if (!(await exists(sourceDir))) return undefined;
  const sourceManifestPath = join(sourceDir, "allin_manifest.json");
  const sourceManifest = (await exists(sourceManifestPath))
    ? (JSON.parse(await readFile(sourceManifestPath, "utf8")) as AllInManifest)
    : { enabled: true, scale: 32768 };
  const allIn = structuredClone(sourceManifest);
  const stagedDir = join(stagingRoot, prefix);
  await mkdir(stagedDir, { recursive: true });

  const sourcePreflop = join(sourceDir, "preflop.i16");
  if (await exists(sourcePreflop)) {
    await cp(sourcePreflop, join(stagedDir, "preflop.i16"));
    allIn.preflop = {
      ...(allIn.preflop ?? {}),
      file: `${assetOrigin}/${prefix}/preflop.i16`,
      dtype: "int16",
      scale: allIn.preflop?.scale ?? allIn.scale ?? 32768,
    };
  }

  const sourceFlop = join(sourceDir, "flop");
  if (await exists(sourceFlop)) {
    await cp(sourceFlop, join(stagedDir, "flop"), { recursive: true });
    if (allIn.flop) {
      allIn.flop.actualToCanonFile = publicAllInPath(prefix, allIn.flop.actualToCanonFile);
      allIn.flop.actualPermFile = publicAllInPath(prefix, allIn.flop.actualPermFile);
      allIn.flop.comboPermsFile = publicAllInPath(prefix, allIn.flop.comboPermsFile);
      allIn.flop.tablePathTemplate = publicAllInPath(prefix, allIn.flop.tablePathTemplate);
    }
  }

  allIn.metadata = {
    ...(allIn.metadata ?? {}),
    assetPrefix: `${assetOrigin}/${prefix}/`,
  };
  await writeJson(join(stagedDir, "allin_manifest.json"), allIn);
  return allIn;
}

function publicAllInPath(prefix: string, value: string): string {
  if (value.startsWith("http://") || value.startsWith("https://")) return value;
  const stripped = value.replace(/^\/+/, "").replace(/^allin\//, "");
  return `${assetOrigin}/${prefix}/${stripped}`;
}

function uploadSpec(
  key: string,
  file: string,
  contentType: string,
  cacheControl: string,
): UploadSpec {
  return { key, file, contentType, cacheControl };
}

function contentTypeFor(file: string): string {
  if (file.endsWith(".json")) return "application/json";
  if (file.endsWith(".gz")) return "application/gzip";
  return "application/octet-stream";
}

function runWrangler(commandArgs: string[]): void {
  const result = spawnSync(wrangler, commandArgs, { stdio: "inherit" });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`${wrangler} ${commandArgs.join(" ")} exited with ${result.status}`);
  }
}

async function listFiles(root: string): Promise<string[]> {
  const out: string[] = [];
  const entries = await readdir(root);
  for (const entry of entries) {
    const path = join(root, entry);
    const entryStat = await stat(path);
    if (entryStat.isDirectory()) {
      out.push(...(await listFiles(path)));
    } else {
      out.push(path);
    }
  }
  return out;
}

function toObjectKey(root: string, file: string): string {
  return file
    .slice(resolve(root).length + 1)
    .split("/")
    .join("/");
}

async function writeJson(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, `${JSON.stringify(value, null, 2)}\n`);
}

async function exists(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

function parseArgs(argv: string[]): CliArgs {
  const parsed: Record<string, string> = {};
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (!arg?.startsWith("--")) continue;
    const key = arg.slice(2);
    const next = argv[i + 1];
    if (next && !next.startsWith("--")) {
      parsed[key] = next;
      i += 1;
    } else {
      parsed[key] = "true";
    }
  }
  return parsed;
}

function isTruthy(value: string | undefined): boolean {
  return value === "1" || value === "true";
}

function stripTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}
