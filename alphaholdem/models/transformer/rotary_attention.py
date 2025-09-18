"""Rotary self-attention utilities and module."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.profiling import profile

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func

    _FLASH_ATTN_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    flash_attn_varlen_kvpacked_func = None
    _FLASH_ATTN_AVAILABLE = False


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate last dimension by half for RoPE application."""

    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query/key pair."""

    cos = cos.to(dtype=q.dtype)
    sin = sin.to(dtype=q.dtype)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class RotarySelfAttention(nn.Module):
    """Multi-head self-attention with RoPE positional encoding."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert (
            self.d_head % 2 == 0
        ), "Rotary embeddings require an even projected head dimension"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    @profile
    def forward(
        self,
        x_new: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
        cache: dict[str, torch.Tensor] | None,
        capture_cache: bool,
        past_lengths: torch.Tensor,
        valid_new_mask: torch.Tensor,
        new_token_counts: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict[str, torch.Tensor] | None, torch.Tensor]:
        """Compute self-attention on the new tokens only."""

        batch_size, max_t_new, _ = x_new.shape
        device = x_new.device

        past_k = cache.get("k") if cache is not None else None
        past_v = cache.get("v") if cache is not None else None

        q = self.q_proj(x_new).view(batch_size, max_t_new, self.n_heads, self.d_head)
        k_new = self.k_proj(x_new).view(
            batch_size, max_t_new, self.n_heads, self.d_head
        )
        v_new = self.v_proj(x_new).view(
            batch_size, max_t_new, self.n_heads, self.d_head
        )

        q = q.permute(0, 2, 1, 3)
        k_new = k_new.permute(0, 2, 1, 3)
        v_new = v_new.permute(0, 2, 1, 3)

        q, k_new = apply_rotary_pos_emb(q, k_new, rotary_cos, rotary_sin)

        valid_new = valid_new_mask.unsqueeze(1).unsqueeze(-1).to(dtype=q.dtype)
        q = q * valid_new
        k_new = k_new * valid_new
        v_new = v_new * valid_new

        trimmed_past = past_lengths
        if past_k is not None and past_v is not None:
            past_k = past_k.to(device=device, dtype=q.dtype)
            past_v = past_v.to(device=device, dtype=q.dtype)
            max_cached = past_k.shape[2]
            trimmed_past = torch.clamp(trimmed_past, max=max_cached)
            max_past = int(trimmed_past.max().item()) if trimmed_past.numel() > 0 else 0
            if max_past > 0:
                past_k = past_k[:, :, :max_past]
                past_v = past_v[:, :, :max_past]
                past_idx = torch.arange(max_past, device=device)
                past_mask = past_idx.unsqueeze(0) < trimmed_past.unsqueeze(1)
                past_mask = past_mask.unsqueeze(1).unsqueeze(-1).to(past_k.dtype)
                past_k = past_k * past_mask
                past_v = past_v * past_mask
            else:
                past_k = None
                past_v = None
        else:
            trimmed_past = torch.zeros_like(past_lengths)

        if max_t_new > 0:
            new_mask = valid_new_mask.unsqueeze(1).unsqueeze(-1).to(k_new.dtype)
            k_new = k_new * new_mask
            v_new = v_new * new_mask

        total_lengths = trimmed_past + new_token_counts
        max_total_len = (
            int(total_lengths.max().item()) if total_lengths.numel() > 0 else 0
        )

        if past_k is not None:
            k_total = torch.cat([past_k, k_new], dim=2)
            v_total = torch.cat([past_v, v_new], dim=2)
        else:
            k_total = k_new
            v_total = v_new

        if k_total.shape[2] > max_total_len:
            k_total = k_total[:, :, :max_total_len]
            v_total = v_total[:, :, :max_total_len]

        attn_output = None
        attn_dropout = self.attn_dropout.p if self.training else 0.0
        if (
            _FLASH_ATTN_AVAILABLE
            and q.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16)
            and max_total_len > 0
            and new_token_counts.sum() > 0
        ):
            try:
                attn_output = self._flash_attention(
                    q,
                    k_total,
                    v_total,
                    new_token_counts,
                    total_lengths,
                    max_total_len,
                    attn_dropout,
                )
            except RuntimeError:
                attn_output = None

        if attn_output is None:
            total_positions = torch.arange(max_total_len, device=device).unsqueeze(0)
            valid_keys = total_positions < total_lengths.unsqueeze(1)
            attn_mask = (~valid_keys).unsqueeze(1).unsqueeze(1)

            attn_output = F.scaled_dot_product_attention(
                q,
                k_total,
                v_total,
                attn_mask=attn_mask,
                dropout_p=attn_dropout,
                is_causal=False,
            )

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, max_t_new, self.d_model)
        attn_output = self.out_proj(attn_output)
        attn_output = attn_output * valid_new_mask.unsqueeze(-1).to(attn_output.dtype)

        new_cache: dict[str, torch.Tensor] | None = None
        if capture_cache:
            new_cache = {
                "k": k_total.detach(),
                "v": v_total.detach(),
                "lengths": total_lengths.detach(),
            }

        return attn_output, new_cache, total_lengths

    def _flash_attention(
        self,
        q: torch.Tensor,
        k_total: torch.Tensor,
        v_total: torch.Tensor,
        new_token_counts: torch.Tensor,
        total_lengths: torch.Tensor,
        max_total_len: int,
        dropout_p: float,
    ) -> torch.Tensor:
        """Compute attention using FlashAttention v2 when available."""

        if not _FLASH_ATTN_AVAILABLE or flash_attn_varlen_kvpacked_func is None:
            raise RuntimeError("FlashAttention is not available")

        batch_size, n_heads, max_t_new, d_head = q.shape

        q_lengths = new_token_counts.to(dtype=torch.int32)
        kv_lengths = total_lengths.to(dtype=torch.int32)

        total_q = int(q_lengths.sum().item())
        if total_q == 0:
            return torch.zeros_like(q)

        max_kv_len = int(kv_lengths.max().item()) if kv_lengths.numel() > 0 else 0
        if max_kv_len == 0:
            return torch.zeros_like(q)

        q_for_flash = q.permute(0, 2, 1, 3).contiguous()  # [B, Tq, H, Dh]
        k_for_flash = k_total.permute(0, 2, 1, 3).contiguous()  # [B, Tk, H, Dh]
        v_for_flash = v_total.permute(0, 2, 1, 3).contiguous()

        q_offsets = torch.zeros(batch_size + 1, device=q.device, dtype=torch.int32)
        kv_offsets = torch.zeros(batch_size + 1, device=q.device, dtype=torch.int32)
        q_offsets[1:] = torch.cumsum(q_lengths, dim=0)
        kv_offsets[1:] = torch.cumsum(kv_lengths, dim=0)

        q_positions = torch.arange(max_t_new, device=q.device).unsqueeze(0)
        q_mask = q_positions < new_token_counts.unsqueeze(1)
        kv_positions = torch.arange(max_total_len, device=q.device).unsqueeze(0)
        kv_mask = kv_positions < total_lengths.unsqueeze(1)

        packed_q = q_for_flash.view(-1, n_heads, d_head)[q_mask.view(-1)]
        packed_k = k_for_flash.view(-1, n_heads, d_head)[kv_mask.view(-1)]
        packed_v = v_for_flash.view(-1, n_heads, d_head)[kv_mask.view(-1)]
        packed_kv = torch.stack((packed_k, packed_v), dim=1)

        attn_out_packed = flash_attn_varlen_kvpacked_func(
            packed_q,
            packed_kv,
            q_offsets,
            kv_offsets,
            int(max_t_new),
            max_kv_len,
            dropout_p,
            None,
            False,
        )

        attn_output = q_for_flash.new_zeros((batch_size, max_t_new, n_heads, d_head))
        attn_output_flat = attn_output.view(-1, n_heads, d_head)
        attn_output_flat[q_mask.view(-1)] = attn_out_packed
        return attn_output.permute(0, 2, 1, 3).contiguous()
