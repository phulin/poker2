import { parseBetterFfnManifest } from "./modelFormat.js";
import type { BetterFfnManifest } from "./types.js";

const DB_NAME = "p2-webgpu-cfr-model-cache";
const STORE_NAME = "models";
const DB_VERSION = 1;

export interface ModelCacheProgress {
  phase: "manifest" | "cache-hit" | "weights" | "stored" | "uncached";
  message: string;
  loadedBytes?: number;
  totalBytes?: number;
}

export interface CachedModelMetadata {
  byteLength: number;
  sha256: string;
}

interface CachedModelRecord extends CachedModelMetadata {
  key: string;
  manifest: BetterFfnManifest;
  weights: ArrayBuffer;
  updatedAt: number;
}

export interface LoadedModelBytes {
  manifest: BetterFfnManifest;
  weights: ArrayBuffer;
  cached: boolean;
  cacheKey: string;
}

export function isCachedModelFresh(
  cached: CachedModelMetadata | undefined,
  manifest: BetterFfnManifest,
): boolean {
  return (
    cached !== undefined &&
    cached.byteLength === manifest.weights.byteLength &&
    cached.sha256 === manifest.weights.sha256
  );
}

export async function loadModelBytesWithCache(
  manifestUrl: string,
  options: {
    weightsUrl?: string;
    cacheKey?: string;
    onProgress?: (progress: ModelCacheProgress) => void;
  } = {},
): Promise<LoadedModelBytes> {
  options.onProgress?.({ phase: "manifest", message: "Fetching model manifest" });
  const manifestResponse = await fetch(manifestUrl);
  if (!manifestResponse.ok) {
    throw new Error(`failed to fetch ${manifestUrl}: ${manifestResponse.status}`);
  }
  const manifest = parseBetterFfnManifest(await manifestResponse.json());
  const baseUrl =
    typeof globalThis.location === "undefined"
      ? "http://localhost/"
      : globalThis.location.href;
  const manifestAbsoluteUrl = new URL(manifestUrl, baseUrl);
  const weightsUrl =
    options.weightsUrl ?? new URL(manifest.weights.file, manifestAbsoluteUrl).toString();
  const cacheKey = options.cacheKey ?? manifestAbsoluteUrl.toString();

  const db = await openModelCacheDb();
  if (!db) {
    options.onProgress?.({
      phase: "uncached",
      message: "IndexedDB unavailable; downloading model without cache",
    });
    const weights = await fetchWeights(weightsUrl, manifest, options.onProgress);
    return { manifest, weights, cached: false, cacheKey };
  }

  try {
    const cached = await getCachedRecord(db, cacheKey);
    if (cached && isCachedModelFresh(cached, manifest)) {
      options.onProgress?.({ phase: "cache-hit", message: "Using cached model weights" });
      return { manifest, weights: cached.weights, cached: true, cacheKey };
    }

    const weights = await fetchWeights(weightsUrl, manifest, options.onProgress);
    await putCachedRecord(db, {
      key: cacheKey,
      manifest,
      weights,
      byteLength: manifest.weights.byteLength,
      sha256: manifest.weights.sha256,
      updatedAt: Date.now(),
    });
    options.onProgress?.({ phase: "stored", message: "Model cached locally" });
    return { manifest, weights, cached: false, cacheKey };
  } finally {
    db.close();
  }
}

async function fetchWeights(
  weightsUrl: string,
  manifest: BetterFfnManifest,
  onProgress?: (progress: ModelCacheProgress) => void,
): Promise<ArrayBuffer> {
  const response = await fetch(weightsUrl);
  if (!response.ok) {
    throw new Error(`failed to fetch ${weightsUrl}: ${response.status}`);
  }
  const contentLength = Number.parseInt(response.headers.get("Content-Length") ?? "", 10);
  const expectedDownloadBytes =
    manifest.weights.compression?.byteLength ?? manifest.weights.byteLength;
  const totalBytes = Number.isFinite(contentLength)
    ? contentLength
    : expectedDownloadBytes;

  if (!response.body) {
    onProgress?.({
      phase: "weights",
      message: "Downloading model weights",
      totalBytes,
    });
    const buffer = await response.arrayBuffer();
    return await decodeModelWeights(buffer, manifest);
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loadedBytes = 0;
  onProgress?.({
    phase: "weights",
    message: "Downloading model weights",
    loadedBytes,
    totalBytes,
  });

  for (;;) {
    const read = await reader.read();
    if (read.done) break;
    chunks.push(read.value);
    loadedBytes += read.value.byteLength;
    onProgress?.({
      phase: "weights",
      message: "Downloading model weights",
      loadedBytes,
      totalBytes,
    });
  }

  const bytes = new Uint8Array(loadedBytes);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return await decodeModelWeights(bytes.buffer, manifest);
}

export async function decodeModelWeights(
  payload: ArrayBuffer,
  manifest: BetterFfnManifest,
): Promise<ArrayBuffer> {
  const compression = manifest.weights.compression;
  if (!compression) {
    assertWeightsLength(payload, manifest.weights.byteLength, "weights.bin");
    return payload;
  }
  if (payload.byteLength === manifest.weights.byteLength) {
    return payload;
  }
  assertWeightsLength(payload, compression.byteLength, manifest.weights.file);
  if (compression.format !== "gzip") {
    throw new Error(`unsupported weights compression ${compression.format}`);
  }
  if (typeof DecompressionStream === "undefined") {
    throw new Error("this browser does not support DecompressionStream for gzip weights");
  }
  const stream = new Blob([payload]).stream().pipeThrough(
    new DecompressionStream("gzip"),
  );
  const decoded = await new Response(stream).arrayBuffer();
  assertWeightsLength(decoded, manifest.weights.byteLength, "decoded weights");
  return decoded;
}

function assertWeightsLength(
  weights: ArrayBuffer,
  expectedBytes: number,
  label: string,
): void {
  if (weights.byteLength !== expectedBytes) {
    throw new Error(
      `${label} has ${weights.byteLength} bytes, expected ${expectedBytes}`,
    );
  }
}

function openModelCacheDb(): Promise<IDBDatabase | undefined> {
  if (!("indexedDB" in globalThis)) return Promise.resolve(undefined);
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: "key" });
      }
    };
    request.onerror = () => reject(request.error ?? new Error("failed to open IndexedDB"));
    request.onsuccess = () => resolve(request.result);
  });
}

function getCachedRecord(
  db: IDBDatabase,
  key: string,
): Promise<CachedModelRecord | undefined> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const request = tx.objectStore(STORE_NAME).get(key);
    request.onerror = () => reject(request.error ?? new Error("failed to read model cache"));
    request.onsuccess = () => resolve(request.result as CachedModelRecord | undefined);
  });
}

function putCachedRecord(
  db: IDBDatabase,
  record: CachedModelRecord,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.onerror = () => reject(tx.error ?? new Error("failed to write model cache"));
    tx.oncomplete = () => resolve();
    tx.objectStore(STORE_NAME).put(record);
  });
}
