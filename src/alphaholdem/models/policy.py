from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from alphaholdem.core.interfaces import Policy


class CategoricalPolicyV1(Policy):
    def action(
        self,
        logits: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[int, float]:
        if legal_mask is not None:
            # mask out illegal actions
            logits = torch.where(legal_mask == 0, -1e9, logits)
        # Fast sampling: log_softmax -> exp -> multinomial
        log_probs_vec = F.log_softmax(logits.float(), dim=-1)
        probs_vec = log_probs_vec.exp()
        a = torch.multinomial(probs_vec, num_samples=1, generator=rng)
        action_idx = int(a.item())
        logp = log_probs_vec[action_idx]
        return action_idx, float(logp.item())

    def action_batch(
        self,
        logits: torch.Tensor,
        legal_masks: Optional[torch.Tensor] = None,
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample actions for a batch of environments.

        Args:
            logits: [N, B] tensor of action logits
            legal_masks: [N, B] tensor of legal action masks

        Returns:
            Tuple of (action_indices [N], log_probs [N, B])
        """
        if legal_masks is not None:
            # Debug: detect environments with no legal actions before softmax
            no_legal_mask = legal_masks.sum(dim=-1) == 0
            if no_legal_mask.any():
                bad_indices = torch.where(no_legal_mask)[0]
                print(
                    "[CategoricalPolicyV1] No legal actions detected",
                    {
                        "indices": bad_indices.tolist(),
                        "logits": logits[bad_indices].detach().cpu().tolist(),
                        "legal_masks": legal_masks[bad_indices].detach().cpu().tolist(),
                    },
                )

            mask_value = torch.tensor(-1e4, dtype=logits.dtype, device=logits.device)
            safe_logits = torch.where(legal_masks, logits, mask_value)
            max_logits = safe_logits.max(dim=-1, keepdim=True).values
            # Clamp max to zero if it became -inf due to masking (shouldn't happen, but safe)
            max_logits = torch.where(
                torch.isfinite(max_logits), max_logits, torch.zeros_like(max_logits)
            )

            shifted = safe_logits - max_logits
            exp_logits = torch.exp(shifted)
            exp_logits = exp_logits * legal_masks.to(exp_logits.dtype)
            denom = exp_logits.sum(dim=-1, keepdim=True)
            denom = denom.clamp_min(1e-12)
            probs = exp_logits / denom

            log_probs = torch.where(
                legal_masks,
                torch.log(probs.clamp_min(1e-12)),
                torch.full_like(probs, float("-inf")),
            )

            if (
                torch.isnan(probs).any()
                or torch.isinf(probs).any()
                or (probs < 0).any()
            ):
                invalid_rows = torch.where(
                    torch.isnan(probs).any(dim=-1)
                    | torch.isinf(probs).any(dim=-1)
                    | (probs < 0).any(dim=-1)
                )[0]
                print(
                    "[CategoricalPolicyV1] Invalid probs detected",
                    {
                        "indices": invalid_rows.tolist(),
                        "logits": logits[invalid_rows].detach().cpu().tolist(),
                        "safe_logits": safe_logits[invalid_rows]
                        .detach()
                        .cpu()
                        .tolist(),
                        "shifted": shifted[invalid_rows].detach().cpu().tolist(),
                        "exp_logits": exp_logits[invalid_rows].detach().cpu().tolist(),
                        "denom": denom[invalid_rows].detach().cpu().tolist(),
                        "probs": probs[invalid_rows].detach().cpu().tolist(),
                    },
                )
                legal_counts = legal_masks.sum(dim=-1, keepdim=True).clamp_min(1)
                probs = legal_masks.float() / legal_counts.float()
                log_probs = torch.where(
                    legal_masks,
                    torch.log(probs.clamp_min(1e-12)),
                    torch.full_like(probs, float("-inf")),
                )
        else:
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()

        action_indices = torch.multinomial(probs, num_samples=1, generator=rng).squeeze(
            1
        )

        return action_indices, log_probs
