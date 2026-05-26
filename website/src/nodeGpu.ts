export async function createDawnDevice(): Promise<GPUDevice> {
  const defaultBackend = process.platform === "darwin" ? "metal" : "vulkan";
  const backend = process.env.WEBGPU_BACKEND ?? defaultBackend;
  process.env.WEBGPU_BACKEND ??= backend;

  let mod: typeof import("webgpu");
  try {
    mod = await import("webgpu");
  } catch (error) {
    throw new Error("Dawn WebGPU bindings are unavailable. Run `npm install` in website first.", {
      cause: error,
    });
  }

  Object.assign(globalThis, mod.globals);
  const gpu = mod.create(backend ? [`backend=${backend}`] : []);
  const adapter = await gpu.requestAdapter();
  if (!adapter) {
    throw new Error(`Dawn WebGPU did not return an adapter for backend ${backend}.`);
  }
  const requiredFeatures: GPUFeatureName[] = [];
  if (adapter.features.has("subgroups" as GPUFeatureName)) {
    requiredFeatures.push("subgroups" as GPUFeatureName);
  }
  const requiredLimits: Record<string, number> = {};
  if (adapter.limits.maxStorageBuffersPerShaderStage >= 10) {
    requiredLimits.maxStorageBuffersPerShaderStage = 10;
  }
  const device = await adapter.requestDevice({
    requiredFeatures,
    ...(Object.keys(requiredLimits).length > 0 ? { requiredLimits } : {}),
  });
  const root = globalThis as typeof globalThis & {
    __p2DawnKeepAlive?: unknown[];
  };
  root.__p2DawnKeepAlive ??= [];
  root.__p2DawnKeepAlive.push(mod, gpu, adapter);
  return device;
}
