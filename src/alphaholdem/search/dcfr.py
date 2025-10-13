from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

import torch
import torch.nn.functional as F


@dataclass
class DCFRResult:
    root_policy_collapsed: torch.Tensor  # [B, 4]
    # Optional diagnostics
    regrets: torch.Tensor | None = None


def collapse_policy_full_to_4(prob_full: torch.Tensor) -> torch.Tensor:
    """Collapse full-bin probs [N, B] to 4-action probs [N, 4]."""
    N, B = prob_full.shape
    out = torch.zeros(N, 4, dtype=prob_full.dtype, device=prob_full.device)
    out[:, 0] = prob_full[:, 0]
    out[:, 1] = prob_full[:, 1]
    if B > 3:
        out[:, 2] = prob_full[:, 2:-1].sum(dim=1)
    out[:, 3] = prob_full[:, -1]
    # normalize just in case of numerical underflow
    s = out.sum(dim=1, keepdim=True).clamp_min(1e-12)
    out = out / s
    return out


def collapse_legal_full_to_4(mask_full: torch.Tensor) -> torch.Tensor:
    N, B = mask_full.shape
    m = torch.zeros(N, 4, dtype=torch.bool, device=mask_full.device)
    m[:, 0] = mask_full[:, 0]
    m[:, 1] = mask_full[:, 1]
    if B > 3:
        m[:, 2] = mask_full[:, 2:-1].any(dim=1)
    m[:, 3] = mask_full[:, -1]
    return m


def regret_matching(regrets: torch.Tensor) -> torch.Tensor:
    rpos = torch.clamp(regrets, min=0.0)
    z = rpos.sum(dim=1, keepdim=True)
    # if all zero, use uniform
    return torch.where(z > 0, rpos / z, torch.full_like(rpos, 1.0 / rpos.shape[1]))


def run_dcfr(
    logits_full: torch.Tensor,
    legal_mask_full: torch.Tensor,
    values: torch.Tensor,
    to_act: torch.Tensor,
    depth_offsets: List[int],
    depth: int,
    iterations: int,
) -> DCFRResult:
    """Vectorized DCFR-like updates over a fixed 4-ary tree.

    Inputs are per-node tensors over all depths (concatenated):
      - logits_full: [M, B]
      - legal_mask_full: [M, B]
      - values: [M] (terminal rewards or value net at deepest depth)
    """
    device = logits_full.device
    M, B = logits_full.shape

    # Prior policy and legal mask collapsed to 4 actions
    prob_full = F.softmax(
        logits_full.masked_fill(~legal_mask_full, float("-inf")), dim=1
    )
    prior4 = collapse_policy_full_to_4(prob_full)
    legal4 = collapse_legal_full_to_4(legal_mask_full)

    # Initialize regrets and policy (masked and renormalized)
    regrets = torch.zeros(M, 4, device=device, dtype=logits_full.dtype)
    policy = torch.where(legal4, prior4, torch.zeros_like(prior4))
    z = policy.sum(dim=1, keepdim=True).clamp_min(1e-12)
    policy = policy / z

    # Buffers
    node_values = torch.zeros(M, dtype=values.dtype, device=device)

    for _ in range(iterations):
        # Set leaf values (deepest depth)
        leaf_sl = slice(depth_offsets[depth], depth_offsets[depth + 1])
        node_values[leaf_sl] = values[leaf_sl]

        # Bottom-up backup and regret updates
        for d in range(depth - 1, -1, -1):
            par_sl = slice(depth_offsets[d], depth_offsets[d + 1])
            chi_sl = slice(depth_offsets[d + 1], depth_offsets[d + 2])
            num_par = depth_offsets[d + 1] - depth_offsets[d]
            # children values shaped [num_par, 4]
            child_vals = node_values[chi_sl].view(num_par, 4)
            par_pol = policy[par_sl]
            par_leg = legal4[par_sl]
            par_to_act = to_act[par_sl]
            # For opponent nodes, compute regrets from opponent perspective (-value for p0)
            # Adjust child values by sign for actor perspective
            child_vals_adj = child_vals.clone()
            if (par_to_act == 1).any():
                opp_rows = torch.where(par_to_act == 1)[0]
                if opp_rows.numel() > 0:
                    child_vals_adj[opp_rows] = -child_vals_adj[opp_rows]
            # masked policy renorm
            masked_pol = torch.where(par_leg, par_pol, torch.zeros_like(par_pol))
            z = masked_pol.sum(dim=1, keepdim=True).clamp_min(1e-12)
            masked_pol = masked_pol / z
            # expected value
            v_actor = (masked_pol * child_vals_adj).sum(dim=1)
            # store values from player 0 perspective so parents always read p0-oriented values
            v_p0 = torch.where(par_to_act == 0, v_actor, -v_actor)
            node_values[par_sl] = v_p0
            # instantaneous regrets Q - V
            inst_reg = child_vals_adj - v_actor.unsqueeze(1)
            # zero out illegal actions
            inst_reg = torch.where(par_leg, inst_reg, torch.zeros_like(inst_reg))
            regrets[par_sl] += inst_reg
            # Update policy by regret matching
            rpos = torch.clamp(regrets[par_sl], min=0.0)
            z = rpos.sum(dim=1, keepdim=True)
            new_pol = torch.where(z > 0, rpos / z, torch.full_like(rpos, 0.25))
            # ensure illegal actions stay zero
            new_pol = torch.where(par_leg, new_pol, torch.zeros_like(new_pol))
            z = new_pol.sum(dim=1, keepdim=True).clamp_min(1e-12)
            policy[par_sl] = new_pol / z

    root_sl = slice(depth_offsets[0], depth_offsets[1])
    return DCFRResult(root_policy_collapsed=policy[root_sl], regrets=regrets)
