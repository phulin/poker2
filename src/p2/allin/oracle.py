from __future__ import annotations

import io
from dataclasses import fields
from functools import lru_cache
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional CUDA fast path
    triton = None
    tl = None

from p2.allin.model import PreflopAllIn169EquityModel
from p2.allin.train import AllInTrainConfig, _build_model
from p2.env.card_utils import (
    PREFLOP_HANDS,
    combo_lookup_tensor,
    combo_to_onehot_tensor,
    combo_to_preflop_class_tensor,
    hand_combos_tensor,
    preflop_class_compatibility_counts_tensor,
    preflop_class_multiplicity_tensor,
)

_U16_SCALE = float(torch.iinfo(torch.uint16).max)
_QUANTIZED_FORMAT = "p2.allin.preflop_allin_169.u16.v1"
_SHARE3_PACKED_TRITON_MIN_ROWS = 256
_SHARE3_PACKED_CHUNK_SIZE = 1024
_SHARE3_PACKED_BLOCK_M = 8
_SHARE3_PACKED_BLOCK_P = 256
_SHARE3_TRANSPOSED_MIN_ROWS = 2048
_SHARE3_TRANSPOSED_SMALL_ROWS = 4096
_SHARE3_TRANSPOSED_SMALL_CHUNK_SIZE = 4096
_SHARE3_TRANSPOSED_CHUNK_SIZE = 8192
_SHARE3_TRANSPOSED_BLOCK_M = 4


if triton is not None:

    @triton.jit
    def _packed_outer_kernel(
        opp0_ptr,
        opp1_ptr,
        tri_i_ptr,
        tri_j_ptr,
        out_ptr,
        rows: tl.constexpr,
        packed_count: tl.constexpr,
        hand_count: tl.constexpr,
        stride_out: tl.constexpr,
        block_m: tl.constexpr,
        block_p: tl.constexpr,
    ) -> None:
        row_start = tl.program_id(0) * block_m
        col_start = tl.program_id(1) * block_p
        row = row_start + tl.arange(0, block_m)
        col = col_start + tl.arange(0, block_p)
        row_mask = row < rows
        col_mask = col < packed_count
        i = tl.load(tri_i_ptr + col, mask=col_mask, other=0)
        j = tl.load(tri_j_ptr + col, mask=col_mask, other=0)
        p0_i = tl.load(
            opp0_ptr + row[:, None] * hand_count + i[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        p1_j = tl.load(
            opp1_ptr + row[:, None] * hand_count + j[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        p0_j = tl.load(
            opp0_ptr + row[:, None] * hand_count + j[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        p1_i = tl.load(
            opp1_ptr + row[:, None] * hand_count + i[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        offdiag = i != j
        value = p0_i * p1_j + tl.where(offdiag[None, :], p0_j * p1_i, 0.0)
        tl.store(
            out_ptr + row[:, None] * stride_out + col[None, :],
            value,
            mask=row_mask[:, None] & col_mask[None, :],
        )

    @triton.jit
    def _denom_card_formula_with_same_kernel(
        q0_ptr,
        q1_ptr,
        qsame_ptr,
        base0_ptr,
        base1_ptr,
        base_same_ptr,
        total0_ptr,
        total1_ptr,
        total_same_ptr,
        hero0_ptr,
        hero1_ptr,
        pair0_ptr,
        pair1_ptr,
        out_ptr,
        rows: tl.constexpr,
        hand_count: tl.constexpr,
        card_count: tl.constexpr,
        stride_out: tl.constexpr,
        block_m: tl.constexpr,
        block_h: tl.constexpr,
    ) -> None:
        row_start = tl.program_id(0) * block_m
        h_start = tl.program_id(1) * block_h
        row = row_start + tl.arange(0, block_m)
        h = h_start + tl.arange(0, block_h)
        row_mask = row < rows
        h_mask = h < hand_count

        h0 = tl.load(hero0_ptr + h, mask=h_mask, other=0)
        h1 = tl.load(hero1_ptr + h, mask=h_mask, other=0)
        total0 = tl.load(total0_ptr + row, mask=row_mask, other=0.0)
        total1 = tl.load(total1_ptr + row, mask=row_mask, other=0.0)
        base0_h0 = tl.load(
            base0_ptr + row[:, None] * card_count + h0[None, :],
            mask=row_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        base0_h1 = tl.load(
            base0_ptr + row[:, None] * card_count + h1[None, :],
            mask=row_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        base1_h0 = tl.load(
            base1_ptr + row[:, None] * card_count + h0[None, :],
            mask=row_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        base1_h1 = tl.load(
            base1_ptr + row[:, None] * card_count + h1[None, :],
            mask=row_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        q0_h = tl.load(
            q0_ptr + row[:, None] * hand_count + h[None, :],
            mask=row_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        q1_h = tl.load(
            q1_ptr + row[:, None] * hand_count + h[None, :],
            mask=row_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        allowed0 = total0[:, None] - base0_h0 - base0_h1 + q0_h
        allowed1 = total1[:, None] - base1_h0 - base1_h1 + q1_h

        collision = tl.zeros((block_m, block_h), dtype=tl.float32)
        for card in range(0, card_count):
            card_is_live = (card != h0) & (card != h1)
            cls0 = tl.load(pair0_ptr + h * card_count + card, mask=h_mask, other=0)
            cls1 = tl.load(pair1_ptr + h * card_count + card, mask=h_mask, other=0)
            b0 = tl.load(base0_ptr + row * card_count + card, mask=row_mask, other=0.0)
            b1 = tl.load(base1_ptr + row * card_count + card, mask=row_mask, other=0.0)
            sub00 = tl.load(
                q0_ptr + row[:, None] * hand_count + cls0[None, :],
                mask=row_mask[:, None] & h_mask[None, :],
                other=0.0,
            )
            sub01 = tl.load(
                q0_ptr + row[:, None] * hand_count + cls1[None, :],
                mask=row_mask[:, None] & h_mask[None, :],
                other=0.0,
            )
            sub10 = tl.load(
                q1_ptr + row[:, None] * hand_count + cls0[None, :],
                mask=row_mask[:, None] & h_mask[None, :],
                other=0.0,
            )
            sub11 = tl.load(
                q1_ptr + row[:, None] * hand_count + cls1[None, :],
                mask=row_mask[:, None] & h_mask[None, :],
                other=0.0,
            )
            c0 = tl.where(card_is_live[None, :], b0[:, None] - sub00 - sub01, 0.0)
            c1 = tl.where(card_is_live[None, :], b1[:, None] - sub10 - sub11, 0.0)
            collision += c0 * c1

        total_same = tl.load(total_same_ptr + row, mask=row_mask, other=0.0)
        base_same_h0 = tl.load(
            base_same_ptr + row[:, None] * card_count + h0[None, :],
            mask=row_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        base_same_h1 = tl.load(
            base_same_ptr + row[:, None] * card_count + h1[None, :],
            mask=row_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        qsame_h = tl.load(
            qsame_ptr + row[:, None] * hand_count + h[None, :],
            mask=row_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        same_correction = total_same[:, None] - base_same_h0 - base_same_h1 + qsame_h
        denom = allowed0 * allowed1 - collision + same_correction
        tl.store(
            out_ptr + row[:, None] * stride_out + h[None, :],
            denom,
            mask=row_mask[:, None] & h_mask[None, :],
        )

    @triton.jit
    def _fill_allin_folded_value_kernel(
        out_ptr,
        node_idx_ptr,
        starting_ptr,
        stacks_after_ptr,
        scale_ptr,
        rows: tl.constexpr,
        players: tl.constexpr,
        hands: tl.constexpr,
        out_stride0: tl.constexpr,
        out_stride1: tl.constexpr,
        out_stride2: tl.constexpr,
        env_stride0: tl.constexpr,
        env_stride1: tl.constexpr,
        block: tl.constexpr,
    ) -> None:
        offs = tl.program_id(0) * block + tl.arange(0, block)
        total = rows * players * hands
        mask = offs < total
        hand = offs % hands
        tmp = offs // hands
        player = tmp % players
        row = tmp // players
        node = tl.load(node_idx_ptr + row, mask=mask, other=0)
        stack_after = tl.load(
            stacks_after_ptr + row * env_stride0 + player * env_stride1,
            mask=mask,
            other=0.0,
        )
        starting = tl.load(
            starting_ptr + row * env_stride0 + player * env_stride1,
            mask=mask,
            other=0.0,
        )
        scale = tl.load(scale_ptr + row, mask=mask, other=1.0)
        value = (stack_after - starting) / tl.maximum(scale, 1.0)
        tl.store(
            out_ptr
            + node * out_stride0
            + player * out_stride1
            + hand * out_stride2,
            value,
            mask=mask,
        )

    @triton.jit
    def _write_allin_entry_share_values_kernel(
        out_ptr,
        node_idx_ptr,
        entry_rows_ptr,
        hero_players_ptr,
        share_ptr,
        starting_ptr,
        stacks_after_ptr,
        max_eligible_ptr,
        scale_ptr,
        entries: tl.constexpr,
        hands: tl.constexpr,
        out_stride0: tl.constexpr,
        out_stride1: tl.constexpr,
        out_stride2: tl.constexpr,
        env_stride0: tl.constexpr,
        env_stride1: tl.constexpr,
        max_stride0: tl.constexpr,
        max_stride1: tl.constexpr,
        share_stride0: tl.constexpr,
        share_stride1: tl.constexpr,
        block: tl.constexpr,
    ) -> None:
        offs = tl.program_id(0) * block + tl.arange(0, block)
        total = entries * hands
        mask = offs < total
        hand = offs % hands
        entry = offs // hands
        row = tl.load(entry_rows_ptr + entry, mask=mask, other=0)
        player = tl.load(hero_players_ptr + entry, mask=mask, other=0)
        node = tl.load(node_idx_ptr + row, mask=mask, other=0)
        share = tl.load(
            share_ptr + entry * share_stride0 + hand * share_stride1,
            mask=mask,
            other=0.0,
        )
        stack_after = tl.load(
            stacks_after_ptr + row * env_stride0 + player * env_stride1,
            mask=mask,
            other=0.0,
        )
        starting = tl.load(
            starting_ptr + row * env_stride0 + player * env_stride1,
            mask=mask,
            other=0.0,
        )
        max_eligible = tl.load(
            max_eligible_ptr + row * max_stride0 + player * max_stride1,
            mask=mask,
            other=0.0,
        )
        scale = tl.load(scale_ptr + row, mask=mask, other=1.0)
        value = (stack_after + share * max_eligible - starting) / tl.maximum(scale, 1.0)
        tl.store(
            out_ptr
            + node * out_stride0
            + player * out_stride1
            + hand * out_stride2,
            value,
            mask=mask,
        )


def _device_cache_key(device: torch.device) -> tuple[str, int | None]:
    if device.type == "cuda" and device.index is None:
        return device.type, torch.cuda.current_device()
    return device.type, device.index


def _load_torch_payload(path: str | Path) -> dict[str, object]:
    path = Path(path).expanduser()
    if path.suffix == ".zst":
        import zstandard as zstd

        data = zstd.ZstdDecompressor().decompress(path.read_bytes())
        return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    return torch.load(path, map_location="cpu", weights_only=False)


def _dequant_share(payload: dict[str, object], key: str, device: torch.device) -> torch.Tensor:
    value = payload.get(key)
    if not torch.is_tensor(value):
        raise ValueError(f"all-in 169 cache is missing tensor {key!r}")
    return value.to(device=device, dtype=torch.float32).div(_U16_SCALE).contiguous()


@lru_cache(maxsize=4)
def _preflop_3p_compatibility_counts_cached(
    device_type: str,
    device_index: int | None,
) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    class_ids = combo_to_preflop_class_tensor(device=torch.device("cpu"))
    combo_cards = combo_to_onehot_tensor(device=torch.device("cpu")).to(torch.float32)
    combo_compatible = (combo_cards @ combo_cards.T) < 0.5
    class_onehot = torch.zeros(
        combo_cards.shape[0],
        PREFLOP_HANDS,
        dtype=torch.float32,
    )
    class_onehot.scatter_(1, class_ids[:, None], 1.0)
    representative = torch.empty(PREFLOP_HANDS, dtype=torch.long)
    for class_id in range(PREFLOP_HANDS):
        representative[class_id] = torch.nonzero(
            class_ids == class_id,
            as_tuple=False,
        )[0, 0]

    counts = torch.empty(
        PREFLOP_HANDS,
        PREFLOP_HANDS,
        PREFLOP_HANDS,
        dtype=torch.float32,
    )
    for hero_class in range(PREFLOP_HANDS):
        allowed = combo_compatible[representative[hero_class]]
        pair_allowed = combo_compatible & allowed[:, None] & allowed[None, :]
        counts[hero_class] = class_onehot.T @ pair_allowed.to(torch.float32) @ class_onehot
    return counts.to(device=device, non_blocking=True).contiguous()


def _preflop_3p_compatibility_counts(device: torch.device) -> torch.Tensor:
    return _preflop_3p_compatibility_counts_cached(*_device_cache_key(device))


@lru_cache(maxsize=4)
def _preflop_3p_card_counts_cached(
    device_type: str,
    device_index: int | None,
) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    class_ids = combo_to_preflop_class_tensor(device=torch.device("cpu"))
    combo_cards = combo_to_onehot_tensor(device=torch.device("cpu")).to(torch.float32)
    combo_compatible = (combo_cards @ combo_cards.T) < 0.5
    class_onehot = torch.zeros(
        combo_cards.shape[0],
        PREFLOP_HANDS,
        dtype=torch.float32,
    )
    class_onehot.scatter_(1, class_ids[:, None], 1.0)
    representative = torch.empty(PREFLOP_HANDS, dtype=torch.long)
    for class_id in range(PREFLOP_HANDS):
        representative[class_id] = torch.nonzero(
            class_ids == class_id,
            as_tuple=False,
        )[0, 0]

    counts = torch.empty(PREFLOP_HANDS, PREFLOP_HANDS, 52, dtype=torch.float32)
    for hero_class in range(PREFLOP_HANDS):
        allowed = combo_compatible[representative[hero_class]].to(torch.float32)
        counts[hero_class] = class_onehot.T @ (combo_cards * allowed[:, None])
    return counts.to(device=device, non_blocking=True).contiguous()


def _preflop_3p_card_counts(device: torch.device) -> torch.Tensor:
    return _preflop_3p_card_counts_cached(*_device_cache_key(device))


@lru_cache(maxsize=4)
def _preflop_3p_fast_denom_metadata_cached(
    device_type: str,
    device_index: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device(device_type, device_index)
    cpu = torch.device("cpu")
    class_ids_cpu = combo_to_preflop_class_tensor(device=cpu)
    combos_cpu = hand_combos_tensor(device=cpu)
    representative = torch.empty(PREFLOP_HANDS, dtype=torch.long)
    for class_id in range(PREFLOP_HANDS):
        representative[class_id] = torch.nonzero(
            class_ids_cpu == class_id,
            as_tuple=False,
        )[0, 0]
    hero_cards = combos_cpu[representative].to(device=device)
    hero0 = hero_cards[:, 0].to(torch.int32).contiguous()
    hero1 = hero_cards[:, 1].to(torch.int32).contiguous()

    combo_lookup = combo_lookup_tensor(device=cpu)
    pair_class_cpu = torch.full((52, 52), 0, dtype=torch.long)
    valid = combo_lookup >= 0
    pair_class_cpu[valid] = class_ids_cpu[combo_lookup[valid]]
    pair_class = pair_class_cpu.to(device=device)
    cards = torch.arange(52, device=device)
    pair0 = pair_class[hero_cards[:, 0, None], cards[None, :]].to(torch.int32).contiguous()
    pair1 = pair_class[hero_cards[:, 1, None], cards[None, :]].to(torch.int32).contiguous()

    combo_cards = combo_to_onehot_tensor(device=device).to(torch.float32)
    class_ids = combo_to_preflop_class_tensor(device=device)
    class_card_counts = torch.zeros(
        PREFLOP_HANDS,
        52,
        dtype=torch.float32,
        device=device,
    )
    class_card_counts.scatter_add_(
        0,
        class_ids[:, None].expand(-1, 52),
        combo_cards,
    )
    return (
        class_card_counts.contiguous(),
        hero0,
        hero1,
        pair0,
        pair1,
    )


def _preflop_3p_fast_denom_metadata(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _preflop_3p_fast_denom_metadata_cached(*_device_cache_key(device))


def _max_eligible_to_win(
    committed: torch.Tensor,
    folded_mask: torch.Tensor,
) -> torch.Tensor:
    return PreflopAllIn169EquityModel._max_eligible_to_win(committed, folded_mask)


def eligible_pot_share_to_net_values_169(
    share: torch.Tensor,
    *,
    starting_stacks: torch.Tensor,
    committed: torch.Tensor,
    stacks_after: torch.Tensor,
    folded_mask: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    max_eligible = _max_eligible_to_win(committed, folded_mask).to(share.dtype)
    scale_f = scale[:, None, None].to(share.dtype).clamp_min(1.0)
    values = (
        stacks_after[:, :, None].to(share.dtype)
        + share * max_eligible[:, :, None]
        - starting_stacks[:, :, None].to(share.dtype)
    ) / scale_f
    folded_value = (
        stacks_after - starting_stacks
    )[:, :, None].to(share.dtype) / scale_f
    return torch.where(folded_mask[:, :, None], folded_value, values)


class PreflopAllIn169Oracle:
    """Native 169-class preflop all-in resolver.

    Exact cached tensors are used for 2/3 live players. A trained native 169
    model is used for 4+ live players.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        exact_cache_path: str | Path | None = None,
        model_checkpoint_path: str | Path | None = None,
        compile_model: bool = False,
    ) -> None:
        self.device = device
        self.exact_cache_path = None if exact_cache_path is None else Path(exact_cache_path)
        self.model_checkpoint_path = (
            None if model_checkpoint_path is None else Path(model_checkpoint_path)
        )
        self.compile_model = bool(compile_model)
        self._exact_loaded = False
        self._share2: torch.Tensor | None = None
        self._share3: torch.Tensor | None = None
        self._share3_weighted_flat: torch.Tensor | None = None
        self._compat3_flat: torch.Tensor | None = None
        self._share3_combined_flat: torch.Tensor | None = None
        self._share3_card_counts_a_hc: torch.Tensor | None = None
        self._share3_tri_i: torch.Tensor | None = None
        self._share3_tri_j: torch.Tensor | None = None
        self._share3_weighted_packed: torch.Tensor | None = None
        self._share3_weighted_packed_t: torch.Tensor | None = None
        self._share3_class_card_counts: torch.Tensor | None = None
        self._share3_hero0: torch.Tensor | None = None
        self._share3_hero1: torch.Tensor | None = None
        self._share3_pair0: torch.Tensor | None = None
        self._share3_pair1: torch.Tensor | None = None
        self._model: torch.nn.Module | None = None
        self._model_target_mode = "eligible_pot_share"
        self._multiplicity = preflop_class_multiplicity_tensor(device=device).to(
            dtype=torch.float32
        )
        self._compat2 = preflop_class_compatibility_counts_tensor(device=device).to(
            dtype=torch.float32
        )
        self._compat3: torch.Tensor | None = None

    def _ensure_exact(self) -> None:
        if self._exact_loaded:
            return
        if self.exact_cache_path is None:
            raise RuntimeError("preflop 169 all-in exact cache path is required")
        payload = _load_torch_payload(self.exact_cache_path)
        fmt = payload.get("format")
        if fmt != _QUANTIZED_FORMAT:
            raise ValueError(
                f"unsupported preflop 169 all-in cache format {fmt!r}; "
                f"expected {_QUANTIZED_FORMAT!r}"
            )
        self._share2 = _dequant_share(payload, "allin_2p_share0_u16", self.device)
        self._share3 = _dequant_share(payload, "allin_3p_share0_u16", self.device)
        if self._share2.shape != (PREFLOP_HANDS, PREFLOP_HANDS):
            raise ValueError(f"bad allin_2p_share0 shape {tuple(self._share2.shape)}")
        if self._share3.shape != (PREFLOP_HANDS, PREFLOP_HANDS, PREFLOP_HANDS):
            raise ValueError(f"bad allin_3p_share0 shape {tuple(self._share3.shape)}")
        self._exact_loaded = True

    def _ensure_model(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model
        if self.model_checkpoint_path is None:
            raise RuntimeError("preflop 169 all-in model checkpoint path is required")
        checkpoint = torch.load(
            self.model_checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        config = checkpoint.get("config")
        if not isinstance(config, dict):
            raise ValueError("all-in model checkpoint does not contain config dict")
        allowed = {field.name for field in fields(AllInTrainConfig)}
        cfg = AllInTrainConfig(**{k: v for k, v in config.items() if k in allowed})
        if cfg.range_hand_dim != PREFLOP_HANDS:
            raise ValueError("all-in oracle model checkpoint is not native 169")
        model = _build_model(cfg).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if self.device.type == "cuda":
            model.to(dtype=torch.bfloat16)
        model.eval()
        if self.compile_model:
            model = torch.compile(model, dynamic=True)
        self._model = model
        self._model_target_mode = str(config.get("target_mode", "net_value"))
        return model

    def _share2_values(
        self,
        hero: torch.Tensor,
        opp: torch.Tensor,
    ) -> torch.Tensor:
        self._ensure_exact()
        assert self._share2 is not None
        opp_per_combo = opp.to(torch.float32) / self._multiplicity
        weighted = self._compat2 * self._share2
        numer = opp_per_combo @ weighted.T
        denom = (opp_per_combo @ self._compat2.T).clamp_min(1.0e-8)
        return (numer / denom).to(hero.dtype)

    def _share3_values(
        self,
        hero: torch.Tensor,
        opp0: torch.Tensor,
        opp1: torch.Tensor,
    ) -> torch.Tensor:
        self._ensure_exact()
        assert self._share3 is not None
        if self._compat3 is None:
            self._compat3 = _preflop_3p_compatibility_counts(self.device)
        weighted_for_packed: torch.Tensor | None = None
        if self._share3_weighted_flat is None:
            weighted_for_packed = self._compat3 * self._share3
            self._share3_weighted_flat = (
                weighted_for_packed.reshape(
                    PREFLOP_HANDS,
                    PREFLOP_HANDS * PREFLOP_HANDS,
                )
                .T.contiguous()
            )
        if self._compat3_flat is None:
            self._compat3_flat = (
                self._compat3.reshape(PREFLOP_HANDS, PREFLOP_HANDS * PREFLOP_HANDS)
                .T.contiguous()
            )
        if self._share3_combined_flat is None:
            self._share3_combined_flat = torch.cat(
                [self._share3_weighted_flat, self._compat3_flat],
                dim=-1,
            ).contiguous()
        if self._share3_card_counts_a_hc is None:
            card_counts = _preflop_3p_card_counts(self.device)
            self._share3_card_counts_a_hc = (
                card_counts.permute(1, 0, 2)
                .reshape(PREFLOP_HANDS, PREFLOP_HANDS * 52)
                .contiguous()
            )
        if self._share3_weighted_packed is None and triton is not None:
            if weighted_for_packed is None:
                weighted_for_packed = self._compat3 * self._share3
            tri = torch.triu_indices(
                PREFLOP_HANDS,
                PREFLOP_HANDS,
                device=self.device,
            )
            self._share3_weighted_packed = weighted_for_packed[
                :,
                tri[0],
                tri[1],
            ].T.contiguous()
            self._share3_weighted_packed_t = self._share3_weighted_packed.T.contiguous()
            self._share3_tri_i = tri[0].to(torch.int32).contiguous()
            self._share3_tri_j = tri[1].to(torch.int32).contiguous()
        if self._share3_class_card_counts is None and triton is not None:
            (
                self._share3_class_card_counts,
                self._share3_hero0,
                self._share3_hero1,
                self._share3_pair0,
                self._share3_pair1,
            ) = _preflop_3p_fast_denom_metadata(self.device)
        opp0_per_combo = opp0.to(torch.float32) / self._multiplicity
        opp1_per_combo = opp1.to(torch.float32) / self._multiplicity
        out = torch.empty(hero.shape[0], PREFLOP_HANDS, dtype=torch.float32, device=hero.device)
        weighted_flat = self._share3_weighted_flat
        compat_flat = self._compat3_flat
        combined_flat = self._share3_combined_flat
        card_counts_a_hc = self._share3_card_counts_a_hc
        tri_i = self._share3_tri_i
        tri_j = self._share3_tri_j
        weighted_packed = self._share3_weighted_packed
        weighted_packed_t = self._share3_weighted_packed_t
        class_card_counts = self._share3_class_card_counts
        hero0 = self._share3_hero0
        hero1 = self._share3_hero1
        pair0 = self._share3_pair0
        pair1 = self._share3_pair1
        assert weighted_flat is not None
        assert compat_flat is not None
        assert combined_flat is not None
        assert card_counts_a_hc is not None
        use_packed_triton = (
            triton is not None
            and hero.device.type == "cuda"
            and hero.shape[0] >= _SHARE3_PACKED_TRITON_MIN_ROWS
            and tri_i is not None
            and tri_j is not None
            and weighted_packed is not None
            and class_card_counts is not None
            and hero0 is not None
            and hero1 is not None
            and pair0 is not None
            and pair1 is not None
        )
        if use_packed_triton:
            packed_count = weighted_packed.shape[0]
            use_transposed_gemm = (
                weighted_packed_t is not None
                and hero.shape[0] >= _SHARE3_TRANSPOSED_MIN_ROWS
            )
            chunk_size = (
                _SHARE3_TRANSPOSED_SMALL_CHUNK_SIZE
                if use_transposed_gemm and hero.shape[0] <= _SHARE3_TRANSPOSED_SMALL_ROWS
                else (
                    _SHARE3_TRANSPOSED_CHUNK_SIZE
                    if use_transposed_gemm
                    else _SHARE3_PACKED_CHUNK_SIZE
                )
            )
            block_m = (
                _SHARE3_TRANSPOSED_BLOCK_M
                if use_transposed_gemm
                else _SHARE3_PACKED_BLOCK_M
            )
            packed = torch.empty(
                min(chunk_size, hero.shape[0]),
                packed_count,
                dtype=torch.float32,
                device=hero.device,
            )
            for start in range(0, hero.shape[0], chunk_size):
                end = min(start + chunk_size, hero.shape[0])
                rows = end - start
                packed_chunk = packed[:rows]
                grid = (
                    triton.cdiv(rows, block_m),
                    triton.cdiv(packed_count, _SHARE3_PACKED_BLOCK_P),
                )
                _packed_outer_kernel[grid](
                    opp0_per_combo[start:end],
                    opp1_per_combo[start:end],
                    tri_i,
                    tri_j,
                    packed_chunk,
                    rows,
                    packed_count,
                    PREFLOP_HANDS,
                    packed_chunk.stride(0),
                    block_m,
                    _SHARE3_PACKED_BLOCK_P,
                    num_warps=8,
                )
                numer = (
                    (weighted_packed_t @ packed_chunk.T).T.contiguous()
                    if use_transposed_gemm
                    else packed_chunk @ weighted_packed
                )
                o0 = opp0_per_combo[start:end]
                o1 = opp1_per_combo[start:end]
                same = o0 * o1
                combined_base = torch.cat((o0, o1, same), dim=0) @ class_card_counts
                base0, base1, base_same = combined_base.split(rows, dim=0)
                total0 = base0.sum(dim=-1).mul_(0.5).contiguous()
                total1 = base1.sum(dim=-1).mul_(0.5).contiguous()
                denom = torch.empty(
                    rows,
                    PREFLOP_HANDS,
                    dtype=torch.float32,
                    device=hero.device,
                )
                denom_grid = (
                    triton.cdiv(rows, block_m),
                    triton.cdiv(PREFLOP_HANDS, 32),
                )
                total_same = base_same.sum(dim=-1).mul_(0.5).contiguous()
                _denom_card_formula_with_same_kernel[denom_grid](
                    o0,
                    o1,
                    same.contiguous(),
                    base0,
                    base1,
                    base_same.contiguous(),
                    total0,
                    total1,
                    total_same,
                    hero0,
                    hero1,
                    pair0,
                    pair1,
                    denom,
                    rows,
                    PREFLOP_HANDS,
                    52,
                    denom.stride(0),
                    block_m,
                    32,
                    num_warps=4,
                )
                denom.clamp_min_(1.0e-8)
                out[start:end] = numer / denom
        elif hero.shape[0] < 512:
            for start in range(0, hero.shape[0], 256):
                end = min(start + 256, hero.shape[0])
                o0 = opp0_per_combo[start:end]
                o1 = opp1_per_combo[start:end]
                outer = (o0[:, :, None] * o1[:, None, :]).flatten(1)
                combined = outer @ combined_flat
                numer, denom = combined.split(PREFLOP_HANDS, dim=-1)
                out[start:end] = numer / denom.clamp_min(1.0e-8)
        else:
            for start in range(0, hero.shape[0], 512):
                end = min(start + 512, hero.shape[0])
                o0 = opp0_per_combo[start:end]
                o1 = opp1_per_combo[start:end]
                outer = (o0[:, :, None] * o1[:, None, :]).flatten(1)
                numer = outer @ weighted_flat
                o0_allowed = o0 @ self._compat2.T
                o1_allowed = o1 @ self._compat2.T
                o0_cards = (o0 @ card_counts_a_hc).view(-1, PREFLOP_HANDS, 52)
                o1_cards = (o1 @ card_counts_a_hc).view(-1, PREFLOP_HANDS, 52)
                card_collisions = (o0_cards * o1_cards).sum(dim=-1)
                same_combo_correction = (o0 * o1) @ self._compat2.T
                denom = (
                    o0_allowed * o1_allowed
                    - card_collisions
                    + same_combo_correction
                ).clamp_min(1.0e-8)
                out[start:end] = numer / denom
        return out.to(hero.dtype)

    def _exact_shares(
        self,
        beliefs: torch.Tensor,
        folded_mask: torch.Tensor,
        *,
        live_players: int,
    ) -> torch.Tensor:
        batch_size, players, _ = beliefs.shape
        live = ~folded_mask
        share = torch.zeros_like(beliefs)
        player_ids = torch.arange(players, device=beliefs.device)
        if live_players == 3 and players > live_players:
            rows, hero_players = torch.nonzero(live, as_tuple=True)
            if rows.numel() == 0:
                return share
            opp_mask = live[rows].clone()
            entry = torch.arange(rows.shape[0], device=beliefs.device)
            opp_mask[entry, hero_players] = False
            opps = torch.nonzero(opp_mask, as_tuple=False)[:, 1].reshape(-1, 2)
            values = self._share3_values(
                beliefs[rows, hero_players],
                beliefs[rows, opps[:, 0]],
                beliefs[rows, opps[:, 1]],
            )
            share[rows, hero_players] = values
            return share

        for player in range(players):
            other_live = live & (player_ids[None, :] != player)
            if live_players == 2:
                opp = other_live.to(torch.long).argmax(dim=1)
                opp_beliefs = beliefs.gather(
                    1,
                    opp[:, None, None].expand(batch_size, 1, PREFLOP_HANDS),
                ).squeeze(1)
                values = self._share2_values(
                    beliefs[:, player],
                    opp_beliefs,
                )
                share[:, player] = torch.where(live[:, player, None], values, 0.0)
            elif live_players == 3:
                opps = other_live.to(torch.long).topk(k=2, dim=1).indices
                opp0_beliefs = beliefs.gather(
                    1,
                    opps[:, 0, None, None].expand(batch_size, 1, PREFLOP_HANDS),
                ).squeeze(1)
                opp1_beliefs = beliefs.gather(
                    1,
                    opps[:, 1, None, None].expand(batch_size, 1, PREFLOP_HANDS),
                ).squeeze(1)
                values = self._share3_values(
                    beliefs[:, player],
                    opp0_beliefs,
                    opp1_beliefs,
                )
                share[:, player] = torch.where(live[:, player, None], values, 0.0)
            else:
                raise ValueError("exact all-in resolver only handles 2/3 live players")
        return share

    @torch.no_grad()
    def values_for_live2_entries(
        self,
        *,
        beliefs: torch.Tensor,
        starting_stacks: torch.Tensor,
        committed: torch.Tensor,
        stacks_after: torch.Tensor,
        folded_mask: torch.Tensor,
        scale: torch.Tensor,
        live_entry_rows: torch.Tensor,
        hero_players: torch.Tensor,
        opp_players: torch.Tensor,
    ) -> torch.Tensor:
        if beliefs.shape[-1] != PREFLOP_HANDS:
            raise ValueError(f"expected 169-class beliefs, got {beliefs.shape}")
        share = torch.zeros_like(beliefs)
        if live_entry_rows.numel() > 0:
            values = self._share2_values(
                beliefs[live_entry_rows, hero_players],
                beliefs[live_entry_rows, opp_players],
            )
            share[live_entry_rows, hero_players] = values
        return eligible_pot_share_to_net_values_169(
            share,
            starting_stacks=starting_stacks,
            committed=committed,
            stacks_after=stacks_after,
            folded_mask=folded_mask,
            scale=scale,
        ).to(beliefs.dtype)

    @torch.no_grad()
    def values_for_live3_entries(
        self,
        *,
        beliefs: torch.Tensor,
        starting_stacks: torch.Tensor,
        committed: torch.Tensor,
        stacks_after: torch.Tensor,
        folded_mask: torch.Tensor,
        scale: torch.Tensor,
        live_entry_rows: torch.Tensor,
        hero_players: torch.Tensor,
        opp0_players: torch.Tensor,
        opp1_players: torch.Tensor,
    ) -> torch.Tensor:
        if beliefs.shape[-1] != PREFLOP_HANDS:
            raise ValueError(f"expected 169-class beliefs, got {beliefs.shape}")
        share = torch.zeros_like(beliefs)
        if live_entry_rows.numel() > 0:
            values = self._share3_values(
                beliefs[live_entry_rows, hero_players],
                beliefs[live_entry_rows, opp0_players],
                beliefs[live_entry_rows, opp1_players],
            )
            share[live_entry_rows, hero_players] = values
        return eligible_pot_share_to_net_values_169(
            share,
            starting_stacks=starting_stacks,
            committed=committed,
            stacks_after=stacks_after,
            folded_mask=folded_mask,
            scale=scale,
        ).to(beliefs.dtype)

    def _write_entry_values_fallback_(
        self,
        *,
        output: torch.Tensor,
        node_indices: torch.Tensor,
        share_values: torch.Tensor,
        starting_stacks: torch.Tensor,
        committed: torch.Tensor,
        stacks_after: torch.Tensor,
        folded_mask: torch.Tensor,
        scale: torch.Tensor,
        live_entry_rows: torch.Tensor,
        hero_players: torch.Tensor,
    ) -> None:
        share = torch.zeros(
            starting_stacks.shape[0],
            starting_stacks.shape[1],
            PREFLOP_HANDS,
            dtype=share_values.dtype,
            device=share_values.device,
        )
        if live_entry_rows.numel() > 0:
            share[live_entry_rows, hero_players] = share_values
        values = eligible_pot_share_to_net_values_169(
            share,
            starting_stacks=starting_stacks,
            committed=committed,
            stacks_after=stacks_after,
            folded_mask=folded_mask,
            scale=scale,
        )
        output[node_indices] = values.to(output.dtype)

    def _write_entry_values_triton_(
        self,
        *,
        output: torch.Tensor,
        node_indices: torch.Tensor,
        share_values: torch.Tensor,
        starting_stacks: torch.Tensor,
        committed: torch.Tensor,
        stacks_after: torch.Tensor,
        folded_mask: torch.Tensor,
        scale: torch.Tensor,
        live_entry_rows: torch.Tensor,
        hero_players: torch.Tensor,
    ) -> None:
        if (
            triton is None
            or output.device.type != "cuda"
            or live_entry_rows.numel() == 0
        ):
            self._write_entry_values_fallback_(
                output=output,
                node_indices=node_indices,
                share_values=share_values,
                starting_stacks=starting_stacks,
                committed=committed,
                stacks_after=stacks_after,
                folded_mask=folded_mask,
                scale=scale,
                live_entry_rows=live_entry_rows,
                hero_players=hero_players,
            )
            return
        rows, players = starting_stacks.shape
        hands = share_values.shape[-1]
        max_eligible = _max_eligible_to_win(committed, folded_mask).to(torch.float32)
        block = 256
        total_default = rows * players * hands
        _fill_allin_folded_value_kernel[(triton.cdiv(total_default, block),)](
            output,
            node_indices,
            starting_stacks,
            stacks_after,
            scale,
            rows,
            players,
            hands,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            starting_stacks.stride(0),
            starting_stacks.stride(1),
            block,
            num_warps=4,
        )
        entries = live_entry_rows.numel()
        total_entries = entries * hands
        _write_allin_entry_share_values_kernel[(triton.cdiv(total_entries, block),)](
            output,
            node_indices,
            live_entry_rows,
            hero_players,
            share_values,
            starting_stacks,
            stacks_after,
            max_eligible,
            scale,
            entries,
            hands,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            starting_stacks.stride(0),
            starting_stacks.stride(1),
            max_eligible.stride(0),
            max_eligible.stride(1),
            share_values.stride(0),
            share_values.stride(1),
            block,
            num_warps=4,
        )

    @torch.no_grad()
    def write_live2_entry_values_(
        self,
        *,
        output: torch.Tensor,
        node_indices: torch.Tensor,
        beliefs: torch.Tensor,
        starting_stacks: torch.Tensor,
        committed: torch.Tensor,
        stacks_after: torch.Tensor,
        folded_mask: torch.Tensor,
        scale: torch.Tensor,
        live_entry_rows: torch.Tensor,
        hero_players: torch.Tensor,
        opp_players: torch.Tensor,
    ) -> None:
        if beliefs.shape[-1] != PREFLOP_HANDS:
            raise ValueError(f"expected 169-class beliefs, got {beliefs.shape}")
        share_values = self._share2_values(
            beliefs[live_entry_rows, hero_players],
            beliefs[live_entry_rows, opp_players],
        )
        self._write_entry_values_triton_(
            output=output,
            node_indices=node_indices,
            share_values=share_values,
            starting_stacks=starting_stacks,
            committed=committed,
            stacks_after=stacks_after,
            folded_mask=folded_mask,
            scale=scale,
            live_entry_rows=live_entry_rows,
            hero_players=hero_players,
        )

    @torch.no_grad()
    def write_live2_entry_values_from_beliefs_(
        self,
        *,
        output: torch.Tensor,
        node_indices: torch.Tensor,
        hero_beliefs: torch.Tensor,
        opp_beliefs: torch.Tensor,
        starting_stacks: torch.Tensor,
        committed: torch.Tensor,
        stacks_after: torch.Tensor,
        folded_mask: torch.Tensor,
        scale: torch.Tensor,
        live_entry_rows: torch.Tensor,
        hero_players: torch.Tensor,
    ) -> None:
        if hero_beliefs.shape[-1] != PREFLOP_HANDS:
            raise ValueError(f"expected 169-class beliefs, got {hero_beliefs.shape}")
        share_values = self._share2_values(hero_beliefs, opp_beliefs)
        self._write_entry_values_triton_(
            output=output,
            node_indices=node_indices,
            share_values=share_values,
            starting_stacks=starting_stacks,
            committed=committed,
            stacks_after=stacks_after,
            folded_mask=folded_mask,
            scale=scale,
            live_entry_rows=live_entry_rows,
            hero_players=hero_players,
        )

    @torch.no_grad()
    def write_live3_entry_values_(
        self,
        *,
        output: torch.Tensor,
        node_indices: torch.Tensor,
        beliefs: torch.Tensor,
        starting_stacks: torch.Tensor,
        committed: torch.Tensor,
        stacks_after: torch.Tensor,
        folded_mask: torch.Tensor,
        scale: torch.Tensor,
        live_entry_rows: torch.Tensor,
        hero_players: torch.Tensor,
        opp0_players: torch.Tensor,
        opp1_players: torch.Tensor,
    ) -> None:
        if beliefs.shape[-1] != PREFLOP_HANDS:
            raise ValueError(f"expected 169-class beliefs, got {beliefs.shape}")
        share_values = self._share3_values(
            beliefs[live_entry_rows, hero_players],
            beliefs[live_entry_rows, opp0_players],
            beliefs[live_entry_rows, opp1_players],
        )
        self._write_entry_values_triton_(
            output=output,
            node_indices=node_indices,
            share_values=share_values,
            starting_stacks=starting_stacks,
            committed=committed,
            stacks_after=stacks_after,
            folded_mask=folded_mask,
            scale=scale,
            live_entry_rows=live_entry_rows,
            hero_players=hero_players,
        )

    @torch.no_grad()
    def write_live3_entry_values_from_beliefs_(
        self,
        *,
        output: torch.Tensor,
        node_indices: torch.Tensor,
        hero_beliefs: torch.Tensor,
        opp0_beliefs: torch.Tensor,
        opp1_beliefs: torch.Tensor,
        starting_stacks: torch.Tensor,
        committed: torch.Tensor,
        stacks_after: torch.Tensor,
        folded_mask: torch.Tensor,
        scale: torch.Tensor,
        live_entry_rows: torch.Tensor,
        hero_players: torch.Tensor,
    ) -> None:
        if hero_beliefs.shape[-1] != PREFLOP_HANDS:
            raise ValueError(f"expected 169-class beliefs, got {hero_beliefs.shape}")
        share_values = self._share3_values(hero_beliefs, opp0_beliefs, opp1_beliefs)
        self._write_entry_values_triton_(
            output=output,
            node_indices=node_indices,
            share_values=share_values,
            starting_stacks=starting_stacks,
            committed=committed,
            stacks_after=stacks_after,
            folded_mask=folded_mask,
            scale=scale,
            live_entry_rows=live_entry_rows,
            hero_players=hero_players,
        )

    @torch.no_grad()
    def values(
        self,
        *,
        beliefs: torch.Tensor,
        starting_stacks: torch.Tensor,
        committed: torch.Tensor,
        stacks_after: torch.Tensor,
        allin_mask: torch.Tensor,
        folded_mask: torch.Tensor,
        scale: torch.Tensor,
        live_players: int,
    ) -> torch.Tensor:
        if beliefs.shape[-1] != PREFLOP_HANDS:
            raise ValueError(f"expected 169-class beliefs, got {beliefs.shape}")
        if live_players <= 3:
            share = self._exact_shares(
                beliefs,
                folded_mask,
                live_players=live_players,
            )
            return eligible_pot_share_to_net_values_169(
                share,
                starting_stacks=starting_stacks,
                committed=committed,
                stacks_after=stacks_after,
                folded_mask=folded_mask,
                scale=scale,
            ).to(beliefs.dtype)

        model = self._ensure_model()
        base_model = getattr(model, "_orig_mod", model)
        model_dtype = next(base_model.parameters()).dtype
        pred_hand_major = model(
            beliefs.to(model_dtype).transpose(1, 2).contiguous(),
            starting_stacks.to(model_dtype),
            committed.to(model_dtype),
            stacks_after.to(model_dtype),
            allin_mask,
            folded_mask,
            hardcode_folded_values=False,
        )
        pred = pred_hand_major.transpose(1, 2).contiguous()
        if self._model_target_mode == "net_value":
            folded_value = (
                stacks_after - starting_stacks
            )[:, :, None].to(pred.dtype) / scale[:, None, None].to(pred.dtype).clamp_min(
                1.0
            )
            return torch.where(folded_mask[:, :, None], folded_value, pred).to(
                beliefs.dtype
            )
        if self._model_target_mode != "eligible_pot_share":
            raise ValueError(f"unsupported all-in model target_mode {self._model_target_mode!r}")
        return eligible_pot_share_to_net_values_169(
            pred,
            starting_stacks=starting_stacks,
            committed=committed,
            stacks_after=stacks_after,
            folded_mask=folded_mask,
            scale=scale,
        ).to(beliefs.dtype)
