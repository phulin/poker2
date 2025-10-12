from __future__ import annotations

import argparse
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.search.cfr_manager import CFRManager, SearchConfig


class DummyModel(nn.Module):
    def __init__(self, num_bins: int):
        super().__init__()
        self.num_bins = num_bins

    def forward(self, embedding_data):
        N = embedding_data.token_ids.shape[0]

        class Out:
            pass

        out = Out()
        out.policy_logits = torch.zeros(
            N,
            self.num_bins,
            dtype=torch.float32,
            device=embedding_data.token_ids.device,
        )
        out.value = torch.zeros(
            N, 1, dtype=torch.float32, device=embedding_data.token_ids.device
        )
        return out


def collapse_policy_full_to_4(prob_full: torch.Tensor) -> torch.Tensor:
    N, B = prob_full.shape
    out = torch.zeros(N, 4, dtype=prob_full.dtype, device=prob_full.device)
    out[:, 0] = prob_full[:, 0]
    out[:, 1] = prob_full[:, 1]
    if B > 3:
        out[:, 2] = prob_full[:, 2:-1].sum(dim=1)
    out[:, 3] = prob_full[:, -1]
    s = out.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return out / s


def collapse_legal_full_to_4(mask_full: torch.Tensor) -> torch.Tensor:
    N, B = mask_full.shape
    m = torch.zeros(N, 4, dtype=torch.bool, device=mask_full.device)
    m[:, 0] = mask_full[:, 0]
    m[:, 1] = mask_full[:, 1]
    if B > 3:
        m[:, 2] = mask_full[:, 2:-1].any(dim=1)
    m[:, 3] = mask_full[:, -1]
    return m


def dcfr_iterate(
    logits_full: torch.Tensor,
    legal_mask_full: torch.Tensor,
    init_values: torch.Tensor,
    depth_offsets: List[int],
    depth: int,
    iterations: int,
):
    device = logits_full.device
    M, B = logits_full.shape
    # Random initial collapsed policy (masked)
    torch.manual_seed(0)
    legal4 = collapse_legal_full_to_4(legal_mask_full)
    prior_raw = torch.randn(M, 4, device=device)
    prior4 = torch.where(legal4, prior_raw, torch.full_like(prior_raw, float("-inf")))
    prior4 = F.softmax(prior4, dim=1)

    regrets = torch.zeros(M, 4, device=device, dtype=logits_full.dtype)
    policy = torch.where(legal4, prior4, torch.zeros_like(prior4))
    policy = policy / policy.sum(dim=1, keepdim=True).clamp_min(1e-12)

    node_values_hist = []
    root_policy_hist = []

    # Base random leaf values and conditional strength
    leaf_sl = slice(depth_offsets[depth], depth_offsets[depth + 1])
    base_leaf = torch.randn(leaf_sl.stop - leaf_sl.start, device=device) * 0.01
    gamma = 0.1

    for it in range(iterations):
        node_values = torch.zeros(M, dtype=init_values.dtype, device=device)
        # deepest level uses conditional values based on current parent policy
        par_sl = slice(depth_offsets[depth - 1], depth_offsets[depth])
        num_par = par_sl.stop - par_sl.start
        parent_pol = policy[par_sl]  # [num_par,4]
        cond = (parent_pol.reshape(-1) - 0.25) * gamma  # [num_par*4]
        node_values[leaf_sl] = base_leaf + cond
        for d in range(depth - 1, -1, -1):
            par_sl = slice(depth_offsets[d], depth_offsets[d + 1])
            chi_sl = slice(depth_offsets[d + 1], depth_offsets[d + 2])
            num_par = depth_offsets[d + 1] - depth_offsets[d]
            child_vals = node_values[chi_sl].view(num_par, 4)
            par_pol = policy[par_sl]
            par_leg = legal4[par_sl]
            masked_pol = torch.where(par_leg, par_pol, torch.zeros_like(par_pol))
            masked_pol = masked_pol / masked_pol.sum(dim=1, keepdim=True).clamp_min(
                1e-12
            )
            v = (masked_pol * child_vals).sum(dim=1)
            node_values[par_sl] = v
            inst_reg = torch.where(
                par_leg, child_vals - v.unsqueeze(1), torch.zeros_like(child_vals)
            )
            regrets[par_sl] += inst_reg
            rpos = torch.clamp(regrets[par_sl], min=0.0)
            new_pol = torch.where(
                rpos.sum(dim=1, keepdim=True) > 0,
                rpos / rpos.sum(dim=1, keepdim=True),
                torch.full_like(rpos, 0.25),
            )
            new_pol = torch.where(par_leg, new_pol, torch.zeros_like(new_pol))
            policy[par_sl] = new_pol / new_pol.sum(dim=1, keepdim=True).clamp_min(1e-12)

        node_values_hist.append(node_values.detach().cpu())
        root_policy_hist.append(
            policy[depth_offsets[0] : depth_offsets[1]].detach().cpu()
        )

        def fmt_row(t: torch.Tensor, prec: int = 4) -> str:
            return "[" + ", ".join(f"{v:.{prec}f}" for v in t.tolist()) + "]"

        print(f"Iteration {it+1} root policy (fold,call,bet,allin):")
        rp = policy[depth_offsets[0] : depth_offsets[1]].detach().cpu()
        for i, row in enumerate(rp):
            print(f"  root[{i}]: {fmt_row(row)}")

        # ASCII tree rendering per root
        def child_indices(par_idx: int, d: int) -> List[int]:
            # returns 4 child indices for parent index par_idx at depth d
            par_start = depth_offsets[d]
            chi_start = depth_offsets[d + 1]
            local = par_idx - par_start
            base = chi_start + 4 * local
            return [base + k for k in range(4)]

        labels = ["F", "C", "B", "A"]

        def render_node(idx: int, d: int, prefix: str = ""):
            val = node_values[idx].item()
            pol = policy[idx].tolist()
            pol_str = ", ".join(f"{p:.3f}" for p in pol)
            print(f"{prefix}- d{d} idx{idx} v={val:.5f} pi=[{pol_str}]")
            if d < depth:
                kids = child_indices(idx, d)
                for k, child in enumerate(kids):
                    render_node(child, d + 1, prefix + f"  {labels[k]} ")

        for root_local in range(len(rp)):
            root_idx = depth_offsets[0] + root_local
            print(f"  Tree for root[{root_local}]")
            render_node(root_idx, 0, prefix="  ")
        for d in range(depth + 1):
            sl = slice(depth_offsets[d], depth_offsets[d + 1])
            vals = node_values[sl].detach().cpu()
            print(
                f"  Depth {d} nodes {sl.start}:{sl.stop} values: {fmt_row(vals, prec=5)}"
            )

    return node_values_hist, root_policy_hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cpu")
    rng = torch.Generator(device=device)
    base_env = HUNLTensorEnv(
        num_envs=max(8, args.batch * 8),
        starting_stack=20000,
        sb=50,
        bb=100,
        device=device,
        rng=rng,
        float_dtype=torch.float32,
        debug_step_table=False,
        flop_showdown=False,
    )
    base_env.reset()

    bet_bins = [1.0]
    mgr = CFRManager(
        batch_size=args.batch,
        env_proto=base_env,
        bet_bins=bet_bins,
        sequence_length=8,
        device=device,
        float_dtype=torch.float32,
        cfg=SearchConfig(depth=args.depth, iterations=args.iters, branching=4),
    )
    roots = mgr.seed_roots(base_env, torch.arange(args.batch, device=device))

    dummy = DummyModel(num_bins=len(bet_bins) + 3)
    logits_full, legal_full, values, _ = mgr.build_tree_tensors(dummy)

    dcfr_iterate(
        logits_full=logits_full,
        legal_mask_full=legal_full,
        init_values=values,
        depth_offsets=mgr.depth_offsets,
        depth=args.depth,
        iterations=args.iters,
    )


if __name__ == "__main__":
    main()
