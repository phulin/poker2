"""
Profile BetterStreetValueFFN with CUDA graphs.

Events inside a captured graph can't be elapsed_time()-queried.
Instead we time each step as an isolated mini-graph (replay-loop).
We also compare overall wall-clock: eager vs full-graph vs torch.compile.
"""
import sys, types
sys.path.insert(0, "src")

import torch
from p2.models.mlp.better_ffn import BetterStreetValueFFN, NUM_HANDS
from p2.models.mlp.mlp_features import MLPFeatures
from p2.core.structured_config import NonlinearityType, StreetValueHeads

DEVICE = "cuda"
B = 4096
H, F = 384, 768
range_hd, board_int = 192, 64
P, C = 2, 41
RUNS = 50

torch.manual_seed(0)

# ── model ───────────────────────────────────────────────────────────────────
def make_model():
    return BetterStreetValueFFN(
        num_actions=8, hidden_dim=H, range_hidden_dim=range_hd, ffn_dim=F,
        num_hidden_layers=0, num_value_layers=7, num_policy_layers=6,
        board_interaction_dim=board_int, policy_rank=96, policy_hand_bias_rank=24,
        nonlinearity=NonlinearityType.leaky_relu, value_heads=StreetValueHeads.both,
    ).to(DEVICE).eval()

model = make_model()

# ── static tensors (fixed addresses required for graph capture) ──────────────
raw = torch.rand(B, P * NUM_HANDS, device=DEVICE)
raw /= raw.sum(-1, keepdim=True).clamp_min(1e-8)
static_feat = MLPFeatures(
    context=torch.randn(B, C, device=DEVICE),
    street=torch.zeros(B, dtype=torch.long, device=DEVICE),
    to_act=torch.zeros(B, dtype=torch.long, device=DEVICE),
    board=torch.randint(0, 52, (B, 5), device=DEVICE),
    beliefs=raw,
)

# ── CUDA event wall-clock timer ──────────────────────────────────────────────
def cuda_time_ms(fn, warmup=5, runs=RUNS) -> float:
    with torch.no_grad():
        for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    with torch.no_grad():
        for _ in range(runs): fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / runs

# ══════════════════════════════════════════════════════════════════════════════
# 1.  EAGER step-by-step (same instrumented approach as before)
# ══════════════════════════════════════════════════════════════════════════════
step_events: dict[str, list] = {}

class Step:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self._s = torch.cuda.Event(enable_timing=True); self._s.record(); return self
    def __exit__(self, *_):
        e = torch.cuda.Event(enable_timing=True); e.record()
        step_events.setdefault(self.name, []).append((self._s, e))

def _fwd_base_inst(self, features, static_base):
    pb = features.beliefs.view(-1, self.num_players, NUM_HANDS)
    with Step("board_ctx"):
        bctx = self._board_context(features.board) if (
            self.board_conditioned_hand_embedding_dim > 0 or
            self.belief_low_rank_board_conditioned) else None
    with Step("hand_emb"):
        he = self._hand_embedding(bctx)
    with Step("belief_moments"):
        ppb, ppv = self._belief_moments(pb, he, bctx)
    with Step("belief_proj"):
        bf = self.belief_proj(self._belief_projection_input(ppb, ppv))
    flat = static_base + bf
    with Step("range_ctx_delta"):
        d = self._range_context_delta(features.context, pb)
        if d is not None: flat = flat + d
    with Step("cross_range"):
        c = self._cross_range_interaction(ppb)
        if c is not None: flat = flat + c
    with Step("board_interaction"):
        bs = self._board_stats(features.board, pb.dtype)
        inter = self._belief_board_interaction(pb, bs)
        if inter is not None: flat = flat + inter
    with Step("trunk"):
        x = (self._postflop_trunk_output(static_base, ppb)
             if self.postflop_multi_token_trunk
             else self._postflop_trunk_output(flat, ppb))
    return pb, flat, x, he, bs

def _fvh_inst(self, features, head, static_base_features=None, apply_zero_sum=True):
    with Step("static_prefix"):
        if static_base_features is None:
            static_base_features = self.static_feature_base(features)
    pb, flat, x, he, _ = _fwd_base_inst(self, features, static_base_features)
    with Step("value_head"):
        return self._value_tensor_from_base(pb, x, he, head, features,
                                            apply_zero_sum=apply_zero_sum)

model._forward_value_head = types.MethodType(_fvh_inst, model)

# Warmup
with torch.no_grad():
    for _ in range(5): model.forward_pre(static_feat)
torch.cuda.synchronize(); step_events.clear()

# Timed eager runs
with torch.no_grad():
    for _ in range(RUNS): model.forward_pre(static_feat)
torch.cuda.synchronize()

print(f"=== EAGER step timing (B={B}, {RUNS} runs) ===")
print(f"\n{'Step':<22} {'Mean μs':>10}  {'% total':>8}")
print("─" * 46)
step_us_eager = {}
for name, pairs in step_events.items():
    ms = sum(s.elapsed_time(e) for s, e in pairs) / len(pairs)
    step_us_eager[name] = ms * 1000
total_eager = sum(step_us_eager.values())
for name, us in sorted(step_us_eager.items(), key=lambda x: -x[1]):
    print(f"  {name:<20} {us:>9.1f}μs  {us/total_eager*100:>7.1f}%")
print(f"  {'─'*42}\n  {'TOTAL':<20} {total_eager:>9.1f}μs")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  CUDA GRAPH: per-step mini-graphs
#     Capture each step in isolation; replay to get GPU time without launch OH
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n=== CUDA GRAPH per-step mini-graphs ===")

# Pre-compute intermediate tensors that each step receives as input.
with torch.no_grad():
    static_base = model.static_feature_base(static_feat)
    pb = static_feat.beliefs.view(-1, P, NUM_HANDS)
    bctx = None  # board_conditioned_hand_embedding_dim = 0
    he = model._hand_embedding(bctx)
    ppb, ppv = model._belief_moments(pb, he, bctx)
    bs = model._board_stats(static_feat.board, pb.dtype)
    # Run through trunk to get x
    bf = model.belief_proj(model._belief_projection_input(ppb, ppv))
    inter = model._belief_board_interaction(pb, bs)
    flat = static_base + bf + inter
    x_state = model._postflop_trunk_output(flat, ppb)
torch.cuda.synchronize()

# Helper: capture a lambda into a CUDA graph and time it
def capture_and_time(name, fn, warmup_stream=None):
    # Warmup on a side stream so cuBLAS state is initialized
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        with torch.no_grad():
            for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad(): fn()

    for _ in range(5): g.replay()
    torch.cuda.synchronize()

    t_s = torch.cuda.Event(enable_timing=True)
    t_e = torch.cuda.Event(enable_timing=True)
    t_s.record()
    for _ in range(RUNS): g.replay()
    t_e.record()
    torch.cuda.synchronize()
    return t_s.elapsed_time(t_e) / RUNS * 1000  # μs

step_us_graph = {}
with torch.no_grad():
    step_us_graph["static_prefix"]   = capture_and_time("static_prefix",
        lambda: model.static_feature_base(static_feat))
    step_us_graph["hand_emb"]        = capture_and_time("hand_emb",
        lambda: model._hand_embedding(None))
    step_us_graph["belief_moments"]  = capture_and_time("belief_moments",
        lambda: model._belief_moments(pb, he, None))
    step_us_graph["belief_proj"]     = capture_and_time("belief_proj",
        lambda: model.belief_proj(model._belief_projection_input(ppb, ppv)))
    step_us_graph["board_interaction"] = capture_and_time("board_interaction",
        lambda: model._belief_board_interaction(pb, bs))
    step_us_graph["trunk"]           = capture_and_time("trunk",
        lambda: model._postflop_trunk_output(flat, ppb))
    # value_head = sequential of (num_value_layers FFN blocks) + output_projection
    # Split: tower = all but last; out_proj = last module
    val_tower   = torch.nn.Sequential(*list(model.pre_value_head.children())[:-1])
    val_out_proj = list(model.pre_value_head.children())[-1]

    step_us_graph["value_head"]      = capture_and_time("value_head",
        lambda: model._value_tensor_from_base(pb, x_state, he,
                                              model.pre_value_head, static_feat,
                                              apply_zero_sum=False))

    # Pre-compute tower output for the out_proj step
    with torch.no_grad():
        tower_out = val_tower(x_state)
    torch.cuda.synchronize()

    step_us_graph["value_tower_ffn"] = capture_and_time("value_tower_ffn",
        lambda: val_tower(x_state))
    step_us_graph["value_out_proj"]  = capture_and_time("value_out_proj",
        lambda: val_out_proj(tower_out))

print(f"\n{'Step':<24} {'Eager μs':>10}  {'Graph μs':>10}  {'Speedup':>9}")
print("─" * 62)
all_names = sorted(step_us_eager, key=lambda k: -step_us_eager[k])
# Insert value head sub-breakdown right after value_head
expanded = []
for n in all_names:
    expanded.append(n)
    if n == "value_head":
        expanded += ["  value_tower_ffn", "  value_out_proj"]
for name in expanded:
    key = name.strip()
    indent = "  " if name.startswith("  ") else ""
    e = step_us_eager.get(key, float('nan'))
    g = step_us_graph.get(key, float('nan'))
    e_str = f"{e:>9.1f}μs" if e == e else f"{'—':>10}"
    g_str = f"{g:>9.1f}μs" if g == g else f"{'—':>10}"
    sp = f"{e/g:.2f}×" if (e == e and g == g and g > 0) else "—"
    print(f"  {indent}{key:<22} {e_str}  {g_str}  {sp:>9}")

# ══════════════════════════════════════════════════════════════════════════════
# 3.  OVERALL wall-clock: eager vs full cuda graph vs torch.compile
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n=== Overall wall-clock (uninstrumented, {RUNS} runs) ===")

model2 = make_model()

eager_ms = cuda_time_ms(lambda: model2.forward_pre(static_feat))

# Full CUDA graph
s3 = torch.cuda.Stream()
with torch.cuda.stream(s3):
    with torch.no_grad():
        for _ in range(3): model2.forward_pre(static_feat)
torch.cuda.current_stream().wait_stream(s3)
g3 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g3):
    with torch.no_grad(): model2.forward_pre(static_feat)
for _ in range(5): g3.replay()
graph_ms = cuda_time_ms(g3.replay)

# torch.compile reduce-overhead
compiled = torch.compile(model2, mode="reduce-overhead")
with torch.no_grad():
    for _ in range(15): compiled.forward_pre(static_feat)
compile_ms = cuda_time_ms(lambda: compiled.forward_pre(static_feat))

print(f"\n  {'Mode':<28} {'ms/fwd':>8}  {'speedup':>9}")
print(f"  {'─'*50}")
print(f"  {'eager':<28} {eager_ms:>8.3f}  {'1.00×':>9}")
print(f"  {'cuda graph (full)':<28} {graph_ms:>8.3f}  {eager_ms/graph_ms:>8.2f}×")
print(f"  {'torch.compile reduce-overhead':<28} {compile_ms:>8.3f}  {eager_ms/compile_ms:>8.2f}×")
