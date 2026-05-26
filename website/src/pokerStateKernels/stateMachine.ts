import { STATE_WGSL } from "./layout.js";

export const POKER_LEGAL_WGSL = /* wgsl */ `
${STATE_WGSL}
@group(0) @binding(0) var<storage, read> state: array<f32>;
@group(0) @binding(1) var<storage, read> betBins: array<f32>;
@group(0) @binding(2) var<storage, read_write> legalMask: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let action = gid.x;
  if (action >= params.numActions) { return; }
  legalMask[action] = legal_for_bin(&state, &betBins, 0u, action, params.numBetBins);
}
`;

export const POKER_STEP_WGSL = /* wgsl */ `
${STATE_WGSL}
@group(0) @binding(0) var<storage, read_write> state: array<f32>;
@group(0) @binding(1) var<storage, read> betBins: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main() {
  _ = step_state(&state, &betBins, 0u, params.action, params);
}
`;

export const POKER_BUILD_CHILD_STATES_WGSL = /* wgsl */ `
${STATE_WGSL}
@group(0) @binding(0) var<storage, read> stateIn: array<f32>;
@group(0) @binding(1) var<storage, read_write> childStates: array<f32>;
@group(0) @binding(2) var<storage, read> betBins: array<f32>;
@group(0) @binding(3) var<storage, read_write> legalMask: array<u32>;
@group(0) @binding(4) var<storage, read_write> terminalMask: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let action = gid.x;
  if (action >= params.numActions) { return; }
  let base = action * STATE_STRIDE;
  for (var i = 0u; i < STATE_STRIDE; i = i + 1u) {
    childStates[base + i] = stateIn[i];
  }
  let legal = legal_for_bin(&stateIn, &betBins, 0u, action, params.numBetBins);
  legalMask[action] = legal;
  if (legal != 0u) {
    _ = step_state(&childStates, &betBins, base, action, params);
  }
  terminalMask[action] = select(0u, 1u, legal != 0u && childStates[base + DONE] != 0.0);
}
`;

export const POKER_COMPACT_MODEL_STATES_WGSL = /* wgsl */ `
${STATE_WGSL}
@group(0) @binding(0) var<storage, read> childStates: array<f32>;
@group(0) @binding(1) var<storage, read> legalMask: array<u32>;
@group(0) @binding(2) var<storage, read> terminalMask: array<u32>;
@group(0) @binding(3) var<storage, read_write> modelStates: array<f32>;
@group(0) @binding(4) var<storage, read_write> modelActionMap: array<u32>;
@group(0) @binding(5) var<storage, read_write> modelCount: atomic<u32>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let action = gid.x;
  if (action >= params.numActions) { return; }
  if (legalMask[action] == 0u || terminalMask[action] != 0u) { return; }
  let dst = atomicAdd(&modelCount, 1u);
  let srcBase = action * STATE_STRIDE;
  let dstBase = dst * STATE_STRIDE;
  for (var i = 0u; i < STATE_STRIDE; i = i + 1u) {
    modelStates[dstBase + i] = childStates[srcBase + i];
  }
  modelActionMap[dst] = action;
}
`;

export const POKER_SCATTER_MODEL_VALUES_WGSL = /* wgsl */ `
${STATE_WGSL}
struct ScatterParams {
  numModelActions: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> modelValues: array<f32>;
@group(0) @binding(1) var<storage, read> modelActionMap: array<u32>;
@group(0) @binding(2) var<storage, read_write> childValues: array<f32>;
@group(0) @binding(3) var<uniform> params: ScatterParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let hand = gid.x;
  let player = gid.y;
  let modelAction = gid.z;
  if (hand >= NUM_HANDS || player >= 2u || modelAction >= params.numModelActions) {
    return;
  }
  let action = modelActionMap[modelAction];
  let src = (modelAction * 2u + player) * NUM_HANDS + hand;
  let dst = (action * 2u + player) * NUM_HANDS + hand;
  childValues[dst] = modelValues[src];
}
`;
