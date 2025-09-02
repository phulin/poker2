from __future__ import annotations

# Conceptual bins (indices): 0=fold,1=check/call,2=1/2,3=3/4,4=1.0,5=1.5,6=2.0,7=allin
DEFAULT_BINS = [
    ("fold", 0.0),
    ("check_call", 0.0),
    ("bet_0.5", 0.5),
    ("bet_0.75", 0.75),
    ("bet_1.0", 1.0),
    ("bet_1.5", 1.5),
    ("bet_2.0", 2.0),
    ("allin", -1.0),
]


def choose_bin(pot_size: int, stack: int, nb: int, desired: float) -> int:
    bins = DEFAULT_BINS[:nb]
    best_idx = 1
    best_dist = 1e18
    for i, (_, mult) in enumerate(bins):
        dist = abs(desired - mult) if mult >= 0 else abs(stack)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def bin_to_amount(pot_size: int, to_call: int, stack: int, idx: int) -> int:
    label, mult = DEFAULT_BINS[idx]
    if label == "fold":
        return 0
    if label == "check_call":
        return to_call
    if label == "allin":
        return stack + to_call
    target = int(pot_size * mult)
    return max(to_call, min(stack + to_call, target))
