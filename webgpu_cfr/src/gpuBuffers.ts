function paddedSize(byteLength: number): number {
  return Math.max(4, Math.ceil(byteLength / 4) * 4);
}

export function makeStorageBuffer(
  device: GPUDevice,
  data: Float32Array<ArrayBufferLike> | Uint32Array<ArrayBufferLike>,
  extraUsage: GPUBufferUsageFlags = 0,
): GPUBuffer {
  const size = paddedSize(data.byteLength);
  const buffer = device.createBuffer({
    size,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST |
      extraUsage,
    mappedAtCreation: true,
  });
  const range = buffer.getMappedRange();
  if (data instanceof Float32Array) {
    new Float32Array(range, 0, data.length).set(data);
  } else {
    new Uint32Array(range, 0, data.length).set(data);
  }
  buffer.unmap();
  return buffer;
}

export function makeEmptyStorageBuffer(
  device: GPUDevice,
  elementCount: number,
  extraUsage: GPUBufferUsageFlags = 0,
): GPUBuffer {
  return device.createBuffer({
    size: paddedSize(elementCount * Float32Array.BYTES_PER_ELEMENT),
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST |
      extraUsage,
  });
}

export function makeUniformBuffer(
  device: GPUDevice,
  words: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
): GPUBuffer {
  const size = Math.ceil(words.byteLength / 16) * 16;
  const buffer = device.createBuffer({
    size,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, words);
  return buffer;
}

export async function readFloatBuffer(
  device: GPUDevice,
  source: GPUBuffer,
  elementCount: number,
): Promise<Float32Array<ArrayBufferLike>> {
  const size = paddedSize(elementCount * Float32Array.BYTES_PER_ELEMENT);
  const readback = device.createBuffer({
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, readback, 0, size);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await readback.mapAsync(GPUMapMode.READ);
  const copy = new Float32Array(readback.getMappedRange()).slice(0, elementCount);
  readback.unmap();
  readback.destroy();
  return copy;
}

export async function readUintBuffer(
  device: GPUDevice,
  source: GPUBuffer,
  elementCount: number,
): Promise<Uint32Array<ArrayBufferLike>> {
  const size = paddedSize(elementCount * Uint32Array.BYTES_PER_ELEMENT);
  const readback = device.createBuffer({
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, readback, 0, size);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await readback.mapAsync(GPUMapMode.READ);
  const copy = new Uint32Array(readback.getMappedRange()).slice(0, elementCount);
  readback.unmap();
  readback.destroy();
  return copy;
}
