"""Profile BetterStreetValueFFN forward pass step-by-step with CUDA events."""
import sys, types
sys.path.insert(0, "src")

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

from p2.models.mlp.better_ffn import BetterStreetValueFFN, NUM_HANDS
from p2.models.mlp.mlp_features import MLPFeatures
from p2.core.structured_config import NonlinearityType, StreetValueHeads

DEVICE = "cuda"
B = 4096

# ── production config ───────────────────────────────────────────────────────
H, F = 384, 768
range_hd = 192
board_int = 64
P = 2
C = 41

torch.manual_seed(0)

model = BetterStreetValueFFN(
    num_actions=8,
    hidden_dim=H,
    range_hidden_dim=range_hd,
    ffn_dim=F,
    num_hidden_layers=0,
    num_value_layers=7,
    num_policy_layers=6,
    board_interaction_dim=board_int,
    policy_rank=96,
    policy_hand_bias_rank=24,
    nonlinearity=NonlinearityType.leaky_relu,
    value_heads=StreetValueHeads.both,
).to(DEVICE).eval()

print(f"BetterStreetValueFFN params: {sum(p.numel() for p in model.parameters()):,}")

# ── synthetic inputs ────────────────────────────────────────────────────────
def make_features(b: int) -> MLPFeatures:
    raw = torch.rand(b, P * NUM_HANDS, device=DEVICE)
    raw = raw / raw.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return MLPFeatures(
        context=torch.randn(b, C, device=DEVICE),
        street=torch.zeros(b, dtype=torch.long, device=DEVICE),
        to_act=torch.zeros(b, dtype=torch.long, device=DEVICE),
        board=torch.randint(0, 52, (b, 5), device=DEVICE),
        beliefs=raw,
    )

features = make_features(B)

# ── event-based step timer ──────────────────────────────────────────────────
step_events: dict[str, list[tuple]] = {}

class Step:
    def __init__(self, name: str):
        self.name = name
    def __enter__(self):
        self._s = torch.cuda.Event(enable_timing=True)
        self._s.record()
        return self
    def __exit__(self, *_):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        step_events.setdefault(self.name, []).append((self._s, e))

# ── instrumented _forward_base_from_static ──────────────────────────────────
def _instrumented_forward_base(self, features, static_base_features):
    player_beliefs = features.beliefs.view(-1, self.num_players, NUM_HANDS)

    with Step("board_ctx"):
        board_context = (
            self._board_context(features.board)
            if (self.board_conditioned_hand_embedding_dim > 0
                or self.belief_low_rank_board_conditioned)
            else None
        )

    with Step("hand_emb"):
        hand_emb = self._hand_embedding(board_context)

    with Step("belief_moments"):
        per_player_belief, per_player_variance = self._belief_moments(
            player_beliefs, hand_emb, board_context
        )

    with Step("belief_proj"):
        belief_features = self.belief_proj(
            self._belief_projection_input(per_player_belief, per_player_variance)
        )

    flat_features = static_base_features + belief_features

    with Step("range_ctx_delta"):
        delta = self._range_context_delta(features.context, player_beliefs)
        if delta is not None:
            flat_features = flat_features + delta

    with Step("cross_range"):
        cross = self._cross_range_interaction(per_player_belief)
        if cross is not None:
            flat_features = flat_features + cross

    with Step("board_interaction"):
        board_stats = self._board_stats(features.board, player_beliefs.dtype)
        inter = self._belief_board_interaction(player_beliefs, board_stats)
        if inter is not None:
            flat_features = flat_features + inter

    with Step("trunk"):
        x = (
            self._postflop_trunk_output(static_base_features, per_player_belief)
            if self.postflop_multi_token_trunk
            else self._postflop_trunk_output(flat_features, per_player_belief)
        )

    return player_beliefs, flat_features, x, hand_emb, board_stats

# Intercept at _forward_value_head level to also time value head
_orig_forward_value_head = BetterStreetValueFFN._forward_value_head

def _instrumented_forward_value_head(self, features, head,
                                     static_base_features=None,
                                     apply_zero_sum=True):
    with Step("static_prefix"):
        if static_base_features is None:
            static_base_features = self.static_feature_base(features)

    player_beliefs, flat, x, hand_emb, _ = _instrumented_forward_base(
        self, features, static_base_features
    )

    with Step("value_head"):
        result = self._value_tensor_from_base(
            player_beliefs, x, hand_emb, head, features,
            apply_zero_sum=apply_zero_sum
        )
    return result

# Bind patched methods to model instance
model._forward_value_head = types.MethodType(_instrumented_forward_value_head, model)

# ── warmup ──────────────────────────────────────────────────────────────────
print("Warming up (5 passes)...")
with torch.no_grad():
    for _ in range(5):
        model.forward_pre(features)
torch.cuda.synchronize()
step_events.clear()

# ── timed passes ────────────────────────────────────────────────────────────
RUNS = 20
print(f"Running {RUNS} timed passes (B={B})...")
with torch.no_grad():
    for _ in range(RUNS):
        model.forward_pre(features)
torch.cuda.synchronize()

# ── report ──────────────────────────────────────────────────────────────────
print(f"\n{'Step':<22} {'Mean μs':>10}  {'% total':>8}")
print("─" * 46)

step_us = {}
for name, pairs in step_events.items():
    ms = sum(s.elapsed_time(e) for s, e in pairs) / len(pairs)
    step_us[name] = ms * 1000

total_us = sum(step_us.values())
for name, us in sorted(step_us.items(), key=lambda x: -x[1]):
    print(f"  {name:<20} {us:>9.1f}μs  {us/total_us*100:>7.1f}%")
print(f"  {'─'*42}")
print(f"  {'TOTAL':<20} {total_us:>9.1f}μs")

# ── torch profiler ──────────────────────────────────────────────────────────
print("\n\nTop CUDA kernels (torch profiler, 3 passes):")
# Warmup outside profiler
with torch.no_grad():
    for _ in range(3):
        model.forward_pre(features)
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
    with torch.no_grad():
        for _ in range(3):
            model.forward_pre(features)
    torch.cuda.synchronize()

print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=25,
    max_name_column_width=55,
))
