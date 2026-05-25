export const REDUCE_4X_256_WGSL = /* wgsl */ `
  if (lane < 128u) {
    partial0[lane] = partial0[lane] + partial0[lane + 128u];
    partial1[lane] = partial1[lane] + partial1[lane + 128u];
    partial2[lane] = partial2[lane] + partial2[lane + 128u];
    partial3[lane] = partial3[lane] + partial3[lane + 128u];
  }
  workgroupBarrier();
  if (lane < 64u) {
    partial0[lane] = partial0[lane] + partial0[lane + 64u];
    partial1[lane] = partial1[lane] + partial1[lane + 64u];
    partial2[lane] = partial2[lane] + partial2[lane + 64u];
    partial3[lane] = partial3[lane] + partial3[lane + 64u];
  }
  workgroupBarrier();
  if (lane < 32u) {
    partial0[lane] = partial0[lane] + partial0[lane + 32u];
    partial1[lane] = partial1[lane] + partial1[lane + 32u];
    partial2[lane] = partial2[lane] + partial2[lane + 32u];
    partial3[lane] = partial3[lane] + partial3[lane + 32u];
  }
  workgroupBarrier();
  if (lane < 16u) {
    partial0[lane] = partial0[lane] + partial0[lane + 16u];
    partial1[lane] = partial1[lane] + partial1[lane + 16u];
    partial2[lane] = partial2[lane] + partial2[lane + 16u];
    partial3[lane] = partial3[lane] + partial3[lane + 16u];
  }
  workgroupBarrier();
  if (lane < 8u) {
    partial0[lane] = partial0[lane] + partial0[lane + 8u];
    partial1[lane] = partial1[lane] + partial1[lane + 8u];
    partial2[lane] = partial2[lane] + partial2[lane + 8u];
    partial3[lane] = partial3[lane] + partial3[lane + 8u];
  }
  workgroupBarrier();
  if (lane < 4u) {
    partial0[lane] = partial0[lane] + partial0[lane + 4u];
    partial1[lane] = partial1[lane] + partial1[lane + 4u];
    partial2[lane] = partial2[lane] + partial2[lane + 4u];
    partial3[lane] = partial3[lane] + partial3[lane + 4u];
  }
  workgroupBarrier();
  if (lane < 2u) {
    partial0[lane] = partial0[lane] + partial0[lane + 2u];
    partial1[lane] = partial1[lane] + partial1[lane + 2u];
    partial2[lane] = partial2[lane] + partial2[lane + 2u];
    partial3[lane] = partial3[lane] + partial3[lane + 2u];
  }
  workgroupBarrier();
  if (lane < 1u) {
    partial0[lane] = partial0[lane] + partial0[lane + 1u];
    partial1[lane] = partial1[lane] + partial1[lane + 1u];
    partial2[lane] = partial2[lane] + partial2[lane + 1u];
    partial3[lane] = partial3[lane] + partial3[lane + 1u];
  }
`;

export const REDUCE_PARTIAL_SUM_256_WGSL = /* wgsl */ `
  if (lane < 128u) {
    partialSum[lane] = partialSum[lane] + partialSum[lane + 128u];
  }
  workgroupBarrier();
  if (lane < 64u) {
    partialSum[lane] = partialSum[lane] + partialSum[lane + 64u];
  }
  workgroupBarrier();
  if (lane < 32u) {
    partialSum[lane] = partialSum[lane] + partialSum[lane + 32u];
  }
  workgroupBarrier();
  if (lane < 16u) {
    partialSum[lane] = partialSum[lane] + partialSum[lane + 16u];
  }
  workgroupBarrier();
  if (lane < 8u) {
    partialSum[lane] = partialSum[lane] + partialSum[lane + 8u];
  }
  workgroupBarrier();
  if (lane < 4u) {
    partialSum[lane] = partialSum[lane] + partialSum[lane + 4u];
  }
  workgroupBarrier();
  if (lane < 2u) {
    partialSum[lane] = partialSum[lane] + partialSum[lane + 2u];
  }
  workgroupBarrier();
  if (lane < 1u) {
    partialSum[lane] = partialSum[lane] + partialSum[lane + 1u];
  }
  workgroupBarrier();
`;

export const REDUCE_PARTIAL_SQ_256_WGSL = /* wgsl */ `
  if (lane < 128u) {
    partialSq[lane] = partialSq[lane] + partialSq[lane + 128u];
  }
  workgroupBarrier();
  if (lane < 64u) {
    partialSq[lane] = partialSq[lane] + partialSq[lane + 64u];
  }
  workgroupBarrier();
  if (lane < 32u) {
    partialSq[lane] = partialSq[lane] + partialSq[lane + 32u];
  }
  workgroupBarrier();
  if (lane < 16u) {
    partialSq[lane] = partialSq[lane] + partialSq[lane + 16u];
  }
  workgroupBarrier();
  if (lane < 8u) {
    partialSq[lane] = partialSq[lane] + partialSq[lane + 8u];
  }
  workgroupBarrier();
  if (lane < 4u) {
    partialSq[lane] = partialSq[lane] + partialSq[lane + 4u];
  }
  workgroupBarrier();
  if (lane < 2u) {
    partialSq[lane] = partialSq[lane] + partialSq[lane + 2u];
  }
  workgroupBarrier();
  if (lane < 1u) {
    partialSq[lane] = partialSq[lane] + partialSq[lane + 1u];
  }
  workgroupBarrier();
`;
