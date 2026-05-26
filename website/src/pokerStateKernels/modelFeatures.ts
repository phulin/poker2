import { STATE_LAYOUT_WGSL } from "./layout.js";

export const POKER_ENCODE_FEATURES_WGSL = /* wgsl */ `
${STATE_LAYOUT_WGSL}
struct EncodeParams {
  hidden: f32,
  batch: f32,
  contextDim: f32,
  bb: f32,
  maxStackBb: f32,
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
};

@group(0) @binding(0) var<storage, read> states: array<f32>;
@group(0) @binding(1) var<storage, read> streetEmbedding: array<f32>;
@group(0) @binding(2) var<storage, read> rankEmbedding: array<f32>;
@group(0) @binding(3) var<storage, read> suitEmbedding: array<f32>;
@group(0) @binding(4) var<storage, read_write> context: array<f32>;
@group(0) @binding(5) var<storage, read_write> baseEmbedding: array<f32>;
@group(0) @binding(6) var<storage, read_write> rankCounts: array<f32>;
@group(0) @binding(7) var<storage, read_write> suitCounts: array<f32>;
@group(0) @binding(8) var<uniform> params: EncodeParams;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let sample = wid.x;
  let lane = lid.x;
  let batch = u32(params.batch);
  let hidden = u32(params.hidden);
  let contextDim = u32(params.contextDim);
  if (sample >= batch) { return; }
  let base = sample * STATE_STRIDE;
  if (lane == 0u) {
    let contextBase = sample * contextDim;
    if (contextDim == 15u) {
      let scale = max(states[base + SCALE], 1.0);
      let bb = max(params.bb, 1.0);
      let pot = states[base + POT];
      context[contextBase + 0u] = states[base + TO_ACT];
      context[contextBase + 1u] = f32((as_u(states[base + TO_ACT]) + 2u - as_u(states[base + BUTTON])) % 2u);
      context[contextBase + 2u] = 0.0;
      context[contextBase + 3u] = pot / scale;
      context[contextBase + 4u] = states[base + MIN_RAISE] / scale;
      context[contextBase + 5u] = log(max(scale / bb, 1.0)) / log(max(params.maxStackBb, 2.0));
      context[contextBase + 6u] = log(1.0 + pot / bb);
      context[contextBase + 7u] = states[base + STACK0] / scale;
      context[contextBase + 8u] = states[base + STACK1] / scale;
      context[contextBase + 9u] = states[base + COMMITTED0] / scale;
      context[contextBase + 10u] = states[base + COMMITTED1] / scale;
      context[contextBase + 11u] = states[base + STACK0] / max(pot, 1.0);
      context[contextBase + 12u] = states[base + STACK1] / max(pot, 1.0);
      context[contextBase + 13u] = log(1.0 + states[base + COMMITTED0] / bb);
      context[contextBase + 14u] = log(1.0 + states[base + COMMITTED1] / bb);
    } else {
      context[contextBase + 0u] = states[base + TO_ACT];
      context[contextBase + 1u] = f32((as_u(states[base + TO_ACT]) + 2u - as_u(states[base + BUTTON])) % 2u);
      context[contextBase + 2u] = states[base + ACTIONS_THIS_ROUND];
      context[contextBase + 3u] = states[base + POT];
      context[contextBase + 4u] = states[base + MIN_RAISE];
      context[contextBase + 5u] = states[base + STACK0];
      context[contextBase + 6u] = states[base + STACK1];
      context[contextBase + 7u] = states[base + COMMITTED0];
      context[contextBase + 8u] = states[base + COMMITTED1];
      context[contextBase + 9u] = states[base + STACK0] / states[base + POT];
      context[contextBase + 10u] = states[base + STACK1] / states[base + POT];
    }
  }
  if (lane < 13u) {
    var count = 0.0;
    for (var i = 0u; i < 5u; i = i + 1u) {
      let card = states[base + BOARD0 + i];
      if (card >= 0.0 && as_u(card) % 13u == lane) { count = count + 1.0; }
    }
    rankCounts[sample * 13u + lane] = count;
  }
  if (lane < 4u) {
    var count = 0.0;
    for (var i = 0u; i < 5u; i = i + 1u) {
      let card = states[base + BOARD0 + i];
      if (card >= 0.0 && as_u(card) / 13u == lane) { count = count + 1.0; }
    }
    suitCounts[sample * 4u + lane] = count;
  }
  let street = min(as_u(states[base + STREET]), 4u);
  for (var d = lane; d < hidden; d = d + 256u) {
    var out = streetEmbedding[street * hidden + d];
    for (var i = 0u; i < 5u; i = i + 1u) {
      let card = states[base + BOARD0 + i];
      let rank = select(13u, as_u(card) % 13u, card >= 0.0);
      let suit = select(4u, as_u(card) / 13u, card >= 0.0);
      out = out + rankEmbedding[rank * hidden + d] + suitEmbedding[suit * hidden + d];
    }
    baseEmbedding[sample * hidden + d] = out;
  }
}
`;
