export const MAX_DISPATCH_WORKGROUPS_PER_DIMENSION = 65535;

export function alignedSampleChunk(lanesPerSample: number, workgroupSize: number): number {
  const maxSamples = Math.floor(
    (MAX_DISPATCH_WORKGROUPS_PER_DIMENSION * workgroupSize) / lanesPerSample,
  );
  return Math.floor(maxSamples / 64) * 64;
}

export function dispatchLimitedCount(lanesPerItem: number, workgroupSize: number): number {
  return Math.max(
    1,
    Math.floor((MAX_DISPATCH_WORKGROUPS_PER_DIMENSION * workgroupSize) / lanesPerItem),
  );
}

export function rangeChunks(start: number, end: number, chunkSize: number): Array<readonly [number, number]> {
  const chunks: Array<readonly [number, number]> = [];
  for (let chunkStart = start; chunkStart < end; chunkStart += chunkSize) {
    chunks.push([chunkStart, Math.min(end, chunkStart + chunkSize)]);
  }
  return chunks;
}

export const LEAF_SAMPLE_CHUNK = alignedSampleChunk(2 * 1326, 64);
export const TERMINAL_SAMPLE_CHUNK = alignedSampleChunk(1326, 64);
export const SHOWDOWN_BOTH_PLAYER_SAMPLE_CHUNK = alignedSampleChunk(1326, 128);
