import { STATE_WGSL } from "./layout.js";

export const POKER_TERMINAL_RANKS_WGSL = /* wgsl */ `
${STATE_WGSL}
@group(0) @binding(0) var<storage, read_write> childStates: array<f32>;
@group(0) @binding(1) var<storage, read> legalMask: array<u32>;
@group(0) @binding(2) var<storage, read> terminalMask: array<u32>;
@group(0) @binding(3) var<storage, read> handCards: array<u32>;
@group(0) @binding(4) var<storage, read_write> terminalRanks: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let hand = gid.x;
  let action = gid.z;
  if (hand >= NUM_HANDS || action >= params.numActions) { return; }
  let out = action * NUM_HANDS + hand;
  terminalRanks[out] = 0u;
  if (legalMask[action] == 0u || terminalMask[action] == 0u) {
    return;
  }
  let base = action * STATE_STRIDE;
  if (childStates[base + HAS_FOLDED0] != 0.0 || childStates[base + HAS_FOLDED1] != 0.0) {
    return;
  }
  let c0 = handCards[2u * hand];
  let c1 = handCards[2u * hand + 1u];
  if (overlaps_board(base, &childStates, c0, c1)) {
    return;
  }
  terminalRanks[out] = evaluate7(
    c0,
    c1,
    as_u(childStates[base + BOARD0]),
    as_u(childStates[base + BOARD0 + 1u]),
    as_u(childStates[base + BOARD0 + 2u]),
    as_u(childStates[base + BOARD0 + 3u]),
    as_u(childStates[base + BOARD0 + 4u]),
  );
}
`;

export const POKER_TERMINAL_VALUES_WGSL = /* wgsl */ `
${STATE_WGSL}
@group(0) @binding(0) var<storage, read_write> childStates: array<f32>;
@group(0) @binding(1) var<storage, read> legalMask: array<u32>;
@group(0) @binding(2) var<storage, read> terminalMask: array<u32>;
@group(0) @binding(3) var<storage, read> beliefs: array<f32>;
@group(0) @binding(4) var<storage, read> handCards: array<u32>;
@group(0) @binding(5) var<storage, read> terminalRanks: array<u32>;
@group(0) @binding(6) var<storage, read_write> childValues: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let hand = gid.x;
  let player = gid.y;
  let action = gid.z;
  if (hand >= NUM_HANDS || player >= 2u || action >= params.numActions) { return; }
  let out = (action * 2u + player) * NUM_HANDS + hand;
  if (legalMask[action] == 0u) {
    childValues[out] = 0.0;
    return;
  }
  if (terminalMask[action] == 0u) {
    return;
  }
  let base = action * STATE_STRIDE;
  if (childStates[base + HAS_FOLDED0] != 0.0 || childStates[base + HAS_FOLDED1] != 0.0) {
    let reward = select(
      (childStates[base + STACK0] - childStates[base + STARTING0]) / childStates[base + SCALE],
      (childStates[base + STACK0] + childStates[base + POT] - childStates[base + STARTING0]) / childStates[base + SCALE],
      childStates[base + WINNER] == 0.0,
    );
    childValues[out] = select(reward, -reward, player == 1u);
    return;
  }
  let c0 = handCards[2u * hand];
  let c1 = handCards[2u * hand + 1u];
  if (overlaps_board(base, &childStates, c0, c1)) {
    childValues[out] = 0.0;
    return;
  }
  let rank = terminalRanks[action * NUM_HANDS + hand];
  let winValue = (childStates[base + STACK0] + childStates[base + POT] - childStates[base + STARTING0]) / childStates[base + SCALE];
  let loseValue = (childStates[base + STACK0] - childStates[base + STARTING0]) / childStates[base + SCALE];
  let tieValue = (childStates[base + STACK0] + 0.5 * childStates[base + POT] - childStates[base + STARTING0]) / childStates[base + SCALE];
  var value = 0.0;
  var mass = 0.0;
  for (var opp = 0u; opp < NUM_HANDS; opp = opp + 1u) {
    let o0 = handCards[2u * opp];
    let o1 = handCards[2u * opp + 1u];
    if (overlaps(c0, c1, o0, o1) || overlaps_board(base, &childStates, o0, o1)) {
      continue;
    }
    let oppRank = terminalRanks[action * NUM_HANDS + opp];
    if (player == 0u) {
      let belief = beliefs[NUM_HANDS + opp];
      let payoff = select(select(tieValue, loseValue, rank < oppRank), winValue, rank > oppRank);
      value = value + belief * payoff;
      mass = mass + belief;
    } else {
      let belief = beliefs[opp];
      let p0Payoff = select(select(tieValue, loseValue, oppRank < rank), winValue, oppRank > rank);
      value = value + belief * -p0Payoff;
      mass = mass + belief;
    }
  }
  childValues[out] = select(0.0, value / mass, mass > 1.0e-12);
}
`;
