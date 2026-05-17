export async function createDawnDevice(): Promise<GPUDevice> {
  let mod: typeof import("webgpu");
  try {
    mod = await import("webgpu");
  } catch (error) {
    throw new Error(
      "Dawn WebGPU bindings are unavailable. Run `npm install` in webgpu_cfr first.",
      { cause: error },
    );
  }

  Object.assign(globalThis, mod.globals);
  const backend = process.env.WEBGPU_BACKEND ?? "vulkan";
  const gpu = mod.create(backend ? [`backend=${backend}`] : []);
  const adapter = await gpu.requestAdapter();
  if (!adapter) {
    throw new Error("Dawn WebGPU did not return an adapter.");
  }
  const device = await adapter.requestDevice();
  const root = globalThis as typeof globalThis & {
    __p2DawnKeepAlive?: unknown[];
  };
  root.__p2DawnKeepAlive ??= [];
  root.__p2DawnKeepAlive.push(mod, gpu, adapter);
  return device;
}
