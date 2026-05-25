export function createComputePipeline(
  device: GPUDevice,
  source: string,
  label: string,
): GPUComputePipeline {
  return device.createComputePipeline({
    label,
    layout: "auto",
    compute: {
      module: device.createShaderModule({ label: `${label}.wgsl`, code: source }),
      entryPoint: "main",
    },
  });
}

export function dispatchCompute(
  encoder: GPUCommandEncoder,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  x: number,
  y = 1,
  z = 1,
): void {
  if (x <= 0 || y <= 0 || z <= 0) return;
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(x, y, z);
  pass.end();
}
