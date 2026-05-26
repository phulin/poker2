declare module "webgpu" {
  export const globals: Record<string, unknown>;
  export function create(options: string[]): GPU;
}
