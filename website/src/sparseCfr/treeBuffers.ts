export interface SparseGpuTreeData {
  nodeCount: number;
  numHands: number;
  childOffsets: Uint32Array<ArrayBufferLike>;
  childCount: Uint32Array<ArrayBufferLike>;
  childIndices: Uint32Array<ArrayBufferLike>;
  parentIndex: Uint32Array<ArrayBufferLike>;
  prevActor: Uint32Array<ArrayBufferLike>;
  toAct: Uint32Array<ArrayBufferLike>;
  allowedMask: Uint32Array<ArrayBufferLike>;
  allowedProb: Float32Array<ArrayBufferLike>;
  handCard0: Uint32Array<ArrayBufferLike>;
  handCard1: Uint32Array<ArrayBufferLike>;
  overlapHands: Uint32Array<ArrayBufferLike>;
  overlapCounts: Uint32Array<ArrayBufferLike>;
  overlapSlots: number;
}

export interface SparseGpuTreeBuffers {
  nodeCount: number;
  numHands: number;
  childOffsets: GPUBuffer;
  childCount: GPUBuffer;
  childIndices: GPUBuffer;
  parentIndex: GPUBuffer;
  prevActor: GPUBuffer;
  toAct: GPUBuffer;
  allowedMask: GPUBuffer;
  allowedProb: GPUBuffer;
  handCard0: GPUBuffer;
  handCard1: GPUBuffer;
  overlapHands: GPUBuffer;
  overlapCounts: GPUBuffer;
  overlapSlots: number;
  dispose: () => void;
}
