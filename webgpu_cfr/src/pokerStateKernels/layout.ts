export const STATE_STRIDE = 40;
export const STATE_BUTTON = 0;
export const STATE_STREET = 1;
export const STATE_TO_ACT = 2;
export const STATE_LAST_TO_ACT = 3;
export const STATE_POT = 4;
export const STATE_MIN_RAISE = 5;
export const STATE_ACTIONS_THIS_ROUND = 6;
export const STATE_ACTIONS_LAST_ROUND = 7;
export const STATE_ACTED_SINCE_RESET = 8;
export const STATE_STACK0 = 9;
export const STATE_STACK1 = 10;
export const STATE_COMMITTED0 = 11;
export const STATE_COMMITTED1 = 12;
export const STATE_STARTING0 = 13;
export const STATE_STARTING1 = 14;
export const STATE_SCALE = 15;
export const STATE_IS_ALLIN0 = 16;
export const STATE_IS_ALLIN1 = 17;
export const STATE_HAS_FOLDED0 = 18;
export const STATE_HAS_FOLDED1 = 19;
export const STATE_DONE = 20;
export const STATE_WINNER = 21;
export const STATE_DECK_POS = 22;
export const STATE_BOARD0 = 23;
export const STATE_DECK0 = 28;

export const STATE_WGSL = /* wgsl */ `
const STATE_STRIDE = ${STATE_STRIDE}u;
const BUTTON = ${STATE_BUTTON}u;
const STREET = ${STATE_STREET}u;
const TO_ACT = ${STATE_TO_ACT}u;
const LAST_TO_ACT = ${STATE_LAST_TO_ACT}u;
const POT = ${STATE_POT}u;
const MIN_RAISE = ${STATE_MIN_RAISE}u;
const ACTIONS_THIS_ROUND = ${STATE_ACTIONS_THIS_ROUND}u;
const ACTIONS_LAST_ROUND = ${STATE_ACTIONS_LAST_ROUND}u;
const ACTED_SINCE_RESET = ${STATE_ACTED_SINCE_RESET}u;
const STACK0 = ${STATE_STACK0}u;
const STACK1 = ${STATE_STACK1}u;
const COMMITTED0 = ${STATE_COMMITTED0}u;
const COMMITTED1 = ${STATE_COMMITTED1}u;
const STARTING0 = ${STATE_STARTING0}u;
const STARTING1 = ${STATE_STARTING1}u;
const SCALE = ${STATE_SCALE}u;
const IS_ALLIN0 = ${STATE_IS_ALLIN0}u;
const IS_ALLIN1 = ${STATE_IS_ALLIN1}u;
const HAS_FOLDED0 = ${STATE_HAS_FOLDED0}u;
const HAS_FOLDED1 = ${STATE_HAS_FOLDED1}u;
const DONE = ${STATE_DONE}u;
const WINNER = ${STATE_WINNER}u;
const DECK_POS = ${STATE_DECK_POS}u;
const BOARD0 = ${STATE_BOARD0}u;
const DECK0 = ${STATE_DECK0}u;
const NUM_HANDS = 1326u;

struct Params {
  numActions: u32,
  action: u32,
  numBetBins: u32,
  flopShowdown: u32,
};

fn idx(base: u32, slot: u32) -> u32 {
  return base + slot;
}

fn as_u(v: f32) -> u32 {
  return u32(max(v, 0.0));
}

fn stack_slot(player: u32) -> u32 {
  return select(STACK0, STACK1, player == 1u);
}

fn committed_slot(player: u32) -> u32 {
  return select(COMMITTED0, COMMITTED1, player == 1u);
}

fn allin_slot(player: u32) -> u32 {
  return select(IS_ALLIN0, IS_ALLIN1, player == 1u);
}

fn folded_slot(player: u32) -> u32 {
  return select(HAS_FOLDED0, HAS_FOLDED1, player == 1u);
}

fn finish_reward(state: ptr<storage, array<f32>, read_write>, base: u32, winner: u32) -> f32 {
  (*state)[idx(base, WINNER)] = f32(winner);
  (*state)[idx(base, DONE)] = 1.0;
  let pot = (*state)[idx(base, POT)];
  var potShare = 0.0;
  if (winner == 0u) {
    potShare = pot;
  } else if (winner == 2u) {
    potShare = 0.5 * pot;
  }
  return ((*state)[idx(base, STACK0)] + potShare - (*state)[idx(base, STARTING0)]) / (*state)[idx(base, SCALE)];
}

fn bet(state: ptr<storage, array<f32>, read_write>, base: u32, player: u32, chips: f32) {
  let amount = max(0.0, floor(chips));
  (*state)[idx(base, stack_slot(player))] = (*state)[idx(base, stack_slot(player))] - amount;
  (*state)[idx(base, committed_slot(player))] = (*state)[idx(base, committed_slot(player))] + amount;
  (*state)[idx(base, POT)] = (*state)[idx(base, POT)] + amount;
}

fn legal_amount(state: ptr<storage, array<f32>, read_write>, betBins: ptr<storage, array<f32>, read>, base: u32, bin: u32, numBetBins: u32) -> f32 {
  let actor = as_u((*state)[idx(base, TO_ACT)]);
  let other = 1u - actor;
  let toCall = (*state)[idx(base, committed_slot(other))] - (*state)[idx(base, committed_slot(actor))];
  if (bin >= 2u && bin < 2u + numBetBins) {
    return toCall + floor((*betBins)[bin - 2u] * (*state)[idx(base, POT)]);
  }
  if (bin == 2u + numBetBins) {
    return (*state)[idx(base, stack_slot(actor))];
  }
  return -1.0;
}

fn legal_for_bin(state: ptr<storage, array<f32>, read>, betBins: ptr<storage, array<f32>, read>, base: u32, bin: u32, numBetBins: u32) -> u32 {
  if ((*state)[idx(base, DONE)] != 0.0) {
    return 0u;
  }
  let actor = as_u((*state)[idx(base, TO_ACT)]);
  let other = 1u - actor;
  let meStack = (*state)[idx(base, stack_slot(actor))];
  let oppStack = (*state)[idx(base, stack_slot(other))];
  let meCommitted = (*state)[idx(base, committed_slot(actor))];
  let oppCommitted = (*state)[idx(base, committed_slot(other))];
  let toCall = oppCommitted - meCommitted;
  let allIn = 2u + numBetBins;
  if ((*state)[idx(base, allin_slot(actor))] != 0.0) {
    return select(0u, 1u, bin == 1u);
  }
  if ((*state)[idx(base, allin_slot(other))] != 0.0) {
    return select(0u, 1u, bin <= 1u);
  }
  if (bin == 0u) {
    return select(0u, 1u, toCall > 0.0);
  }
  if (bin == 1u) {
    return 1u;
  }
  if (bin == allIn) {
    return select(0u, 1u, meStack > 0.0);
  }
  if (bin > 1u && bin < allIn) {
    let additional = floor((*betBins)[bin - 2u] * (*state)[idx(base, POT)]);
    let amount = toCall + additional;
    let ok = meStack > 0.0 && oppStack > 0.0 && amount <= meStack && additional >= (*state)[idx(base, MIN_RAISE)];
    return select(0u, 1u, ok);
  }
  return 0u;
}

fn deal_board(state: ptr<storage, array<f32>, read_write>, base: u32, start: u32, count: u32) {
  let pos = as_u((*state)[idx(base, DECK_POS)]);
  for (var i = 0u; i < count; i = i + 1u) {
    (*state)[idx(base, BOARD0 + start + i)] = (*state)[idx(base, DECK0 + pos + i)];
  }
  (*state)[idx(base, DECK_POS)] = f32(pos + count);
}

fn step_state(state: ptr<storage, array<f32>, read_write>, betBins: ptr<storage, array<f32>, read>, base: u32, bin: u32, params: Params) -> f32 {
  let actor = as_u((*state)[idx(base, TO_ACT)]);
  let other = 1u - actor;
  let toCall = (*state)[idx(base, committed_slot(other))] - (*state)[idx(base, committed_slot(actor))];
  let allIn = 2u + params.numBetBins;
  var reward = 0.0;

  (*state)[idx(base, ACTED_SINCE_RESET)] = 1.0;
  if (bin == 0u) {
    reward = finish_reward(state, base, other);
    (*state)[idx(base, folded_slot(actor))] = 1.0;
  } else if (bin == 1u) {
    bet(state, base, actor, min(toCall, (*state)[idx(base, stack_slot(actor))]));
  } else if (bin >= 2u && bin < allIn) {
    let amount = legal_amount(state, betBins, base, bin, params.numBetBins);
    bet(state, base, actor, amount);
    (*state)[idx(base, MIN_RAISE)] = max((*state)[idx(base, MIN_RAISE)], amount - toCall);
  } else if (bin == allIn) {
    let amount = (*state)[idx(base, stack_slot(actor))];
    bet(state, base, actor, amount);
    (*state)[idx(base, allin_slot(actor))] = 1.0;
  }

  (*state)[idx(base, LAST_TO_ACT)] = f32(actor);
  (*state)[idx(base, TO_ACT)] = f32(other);
  (*state)[idx(base, ACTIONS_THIS_ROUND)] = (*state)[idx(base, ACTIONS_THIS_ROUND)] + 1.0;

  let equalCommitted = (*state)[idx(base, COMMITTED0)] == (*state)[idx(base, COMMITTED1)];
  let allInCommitted =
    ((*state)[idx(base, IS_ALLIN0)] != 0.0 && (*state)[idx(base, IS_ALLIN1)] != 0.0) ||
    ((*state)[idx(base, IS_ALLIN0)] != 0.0 && (*state)[idx(base, COMMITTED0)] <= (*state)[idx(base, COMMITTED1)]) ||
    ((*state)[idx(base, IS_ALLIN1)] != 0.0 && (*state)[idx(base, COMMITTED1)] <= (*state)[idx(base, COMMITTED0)]);
  let roundClosed = (*state)[idx(base, DONE)] == 0.0 &&
    (equalCommitted || allInCommitted) &&
    (*state)[idx(base, ACTIONS_THIS_ROUND)] >= 2.0;
  if (roundClosed) {
    (*state)[idx(base, COMMITTED0)] = 0.0;
    (*state)[idx(base, COMMITTED1)] = 0.0;
    (*state)[idx(base, ACTIONS_LAST_ROUND)] = (*state)[idx(base, ACTIONS_THIS_ROUND)];
    (*state)[idx(base, ACTIONS_THIS_ROUND)] = 0.0;
    (*state)[idx(base, TO_ACT)] = 1.0 - (*state)[idx(base, BUTTON)];
    (*state)[idx(base, MIN_RAISE)] = (*state)[idx(base, DECK0 + 9u)];
    let street = as_u((*state)[idx(base, STREET)]);
    let showdownStreet = select(3u, 0u, params.flopShowdown != 0u);
    if (street == showdownStreet) {
      (*state)[idx(base, DONE)] = 1.0;
    }
    if (street == 0u) {
      deal_board(state, base, 0u, 3u);
    } else if (street == 1u) {
      deal_board(state, base, 3u, 1u);
    } else if (street == 2u) {
      deal_board(state, base, 4u, 1u);
    }
    (*state)[idx(base, STREET)] = (*state)[idx(base, STREET)] + 1.0;
  }
  return reward;
}

fn overlaps(c0: u32, c1: u32, d0: u32, d1: u32) -> bool {
  return c0 == d0 || c0 == d1 || c1 == d0 || c1 == d1;
}

fn overlaps_board(base: u32, state: ptr<storage, array<f32>, read_write>, c0: u32, c1: u32) -> bool {
  for (var i = 0u; i < 5u; i = i + 1u) {
    let card = (*state)[idx(base, BOARD0 + i)];
    if (card >= 0.0 && (u32(card) == c0 || u32(card) == c1)) {
      return true;
    }
  }
  return false;
}

fn encode_rank(category: u32, k0: u32, k1: u32, k2: u32, k3: u32, k4: u32) -> u32 {
  return (((((category * 13u + k0) * 13u + k1) * 13u + k2) * 13u + k3) * 13u + k4);
}

fn evaluate5(a: u32, b: u32, c: u32, d: u32, e: u32) -> u32 {
  var counts: array<u32, 13>;
  for (var i = 0u; i < 13u; i = i + 1u) { counts[i] = 0u; }
  let r0 = a % 13u; let r1 = b % 13u; let r2 = c % 13u; let r3 = d % 13u; let r4 = e % 13u;
  counts[r0] = counts[r0] + 1u; counts[r1] = counts[r1] + 1u; counts[r2] = counts[r2] + 1u; counts[r3] = counts[r3] + 1u; counts[r4] = counts[r4] + 1u;
  let flush = a / 13u == b / 13u && a / 13u == c / 13u && a / 13u == d / 13u && a / 13u == e / 13u;
  var straight = 99u;
  if (counts[12] > 0u && counts[3] > 0u && counts[2] > 0u && counts[1] > 0u && counts[0] > 0u) { straight = 3u; }
  for (var high = 12u; high >= 4u; high = high - 1u) {
    if (counts[high] > 0u && counts[high - 1u] > 0u && counts[high - 2u] > 0u && counts[high - 3u] > 0u && counts[high - 4u] > 0u) {
      straight = high;
      break;
    }
    if (high == 4u) { break; }
  }
  if (flush && straight != 99u) { return encode_rank(8u, straight, 0u, 0u, 0u, 0u); }
  var four = 99u; var three = 99u; var pair0 = 99u; var pair1 = 99u;
  for (var rr = 0u; rr < 13u; rr = rr + 1u) {
    let r = 12u - rr;
    if (counts[r] == 4u) { four = r; }
    if (counts[r] == 3u) { three = r; }
    if (counts[r] == 2u) {
      if (pair0 == 99u) { pair0 = r; } else if (pair1 == 99u) { pair1 = r; }
    }
  }
  if (four != 99u) {
    var kicker = 0u;
    for (var rr = 0u; rr < 13u; rr = rr + 1u) { let r = 12u - rr; if (counts[r] == 1u) { kicker = r; break; } }
    return encode_rank(7u, four, kicker, 0u, 0u, 0u);
  }
  if (three != 99u && pair0 != 99u) { return encode_rank(6u, three, pair0, 0u, 0u, 0u); }
  var hs: array<u32, 5>; var n = 0u;
  for (var rr = 0u; rr < 13u; rr = rr + 1u) {
    let r = 12u - rr;
    for (var j = 0u; j < counts[r]; j = j + 1u) { hs[n] = r; n = n + 1u; }
  }
  if (flush) { return encode_rank(5u, hs[0], hs[1], hs[2], hs[3], hs[4]); }
  if (straight != 99u) { return encode_rank(4u, straight, 0u, 0u, 0u, 0u); }
  if (three != 99u) {
    var k0 = 0u; var k1 = 0u; var ki = 0u;
    for (var rr = 0u; rr < 13u; rr = rr + 1u) { let r = 12u - rr; if (counts[r] == 1u) { if (ki == 0u) { k0 = r; } else { k1 = r; } ki = ki + 1u; } }
    return encode_rank(3u, three, k0, k1, 0u, 0u);
  }
  if (pair0 != 99u && pair1 != 99u) {
    var kicker = 0u;
    for (var rr = 0u; rr < 13u; rr = rr + 1u) { let r = 12u - rr; if (counts[r] == 1u) { kicker = r; break; } }
    return encode_rank(2u, pair0, pair1, kicker, 0u, 0u);
  }
  if (pair0 != 99u) {
    var k0 = 0u; var k1 = 0u; var k2 = 0u; var ki = 0u;
    for (var rr = 0u; rr < 13u; rr = rr + 1u) { let r = 12u - rr; if (counts[r] == 1u) { if (ki == 0u) { k0 = r; } else if (ki == 1u) { k1 = r; } else { k2 = r; } ki = ki + 1u; } }
    return encode_rank(1u, pair0, k0, k1, k2, 0u);
  }
  return encode_rank(0u, hs[0], hs[1], hs[2], hs[3], hs[4]);
}

fn evaluate7(c0: u32, c1: u32, b0: u32, b1: u32, b2: u32, b3: u32, b4: u32) -> u32 {
  var cards: array<u32, 7>;
  cards[0] = c0; cards[1] = c1; cards[2] = b0; cards[3] = b1; cards[4] = b2; cards[5] = b3; cards[6] = b4;
  var best = 0u;
  for (var a = 0u; a < 3u; a = a + 1u) {
    for (var b = a + 1u; b < 4u; b = b + 1u) {
      for (var c = b + 1u; c < 5u; c = c + 1u) {
        for (var d = c + 1u; d < 6u; d = d + 1u) {
          for (var e = d + 1u; e < 7u; e = e + 1u) {
            best = max(best, evaluate5(cards[a], cards[b], cards[c], cards[d], cards[e]));
          }
        }
      }
    }
  }
  return best;
}
`;
