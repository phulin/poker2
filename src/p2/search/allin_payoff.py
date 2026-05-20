from __future__ import annotations

import itertools
import hashlib
import os
from functools import lru_cache
from pathlib import Path

import torch

from p2.env.card_utils import (
    NUM_HANDS,
    board_allowed_hands,
    calculate_unblocked_mass,
    combo_compatible_tensor,
    combo_suit_permutation_tensor,
    hand_combos_tensor,
    suit_permutations_tensor,
)
from p2.env.rules import rank_hands as rank_hands_torch
from p2.env.rules_triton import rank_7_cards_single_batch_triton, triton_is_available

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional dependency
    triton = None
    tl = None


FLOP_I8_SCALE = 127.0
I16_SCALE = 32768.0
_CACHE_VERSION = 1
_FLOP_CACHE_NAME = f"canonical_flop_i8_v{_CACHE_VERSION}.pt"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _rank_hands(board: torch.Tensor) -> torch.Tensor:
    if board.device.type == "cuda" and triton_is_available():
        from p2.env.rules_triton import rank_hands_triton

        ranks, _ = rank_hands_triton(board.int())
        return ranks.contiguous()
    ranks, _ = rank_hands_torch(board.int())
    return ranks.to(torch.int32).contiguous()


def _runout_boards(board: torch.Tensor) -> torch.Tensor:
    board_cpu = [int(c) for c in board.detach().cpu().tolist() if int(c) >= 0]
    if len(board_cpu) not in (3, 4):
        raise ValueError(f"expected flop or turn board, got {board_cpu!r}")
    remaining = [c for c in range(52) if c not in set(board_cpu)]
    missing = 5 - len(board_cpu)
    runouts = torch.tensor(
        list(itertools.combinations(remaining, missing)),
        dtype=torch.long,
        device=board.device,
    )
    prefix = torch.tensor(board_cpu, dtype=torch.long, device=board.device).view(1, -1)
    return torch.cat([prefix.expand(runouts.shape[0], len(board_cpu)), runouts], dim=1)


def compute_postflop_payoff_reference(
    board: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Exact postflop all-in payoff matrix for one flop/turn board.

    Returns ``[1326, 1326]`` where rows are hero hands, columns are opponent
    hands, and values are ``E[win - loss]`` over future board completions valid
    for that exact hand pair. Blocked hand pairs are zero.
    """
    board = board.to(dtype=torch.long)
    device = board.device
    full_boards = _runout_boards(board)
    ranks = _rank_hands(full_boards)
    combos = hand_combos_tensor(device=device)
    compatible = combo_compatible_tensor(device=device)
    hand_ok = board_allowed_hands(board.view(1, -1)).squeeze(0)
    runouts = full_boards[:, board.numel() :]

    score = torch.zeros(NUM_HANDS, NUM_HANDS, dtype=torch.float32, device=device)
    counts = torch.zeros_like(score)
    pair_ok = compatible & hand_ok[:, None] & hand_ok[None, :]
    for t in range(full_boards.shape[0]):
        runout = runouts[t]
        runout_ok = torch.ones(NUM_HANDS, dtype=torch.bool, device=device)
        for card in runout:
            runout_ok &= ~(combos == card).any(dim=1)
        valid = pair_ok & runout_ok[:, None] & runout_ok[None, :]
        cmp = ranks[t, :, None] - ranks[t, None, :]
        score += torch.where(cmp > 0, 1.0, torch.where(cmp < 0, -1.0, 0.0)) * valid
        counts += valid

    payoff = torch.where(counts > 0, score / counts.clamp_min(1), torch.zeros_like(score))
    return payoff.to(dtype=dtype)


def compute_postflop_payoff_quantized(
    board: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    payoff = compute_postflop_payoff_reference(board, dtype=torch.float32)
    if dtype == torch.int8:
        return (payoff * FLOP_I8_SCALE).round().clamp(-128, 127).to(torch.int8)
    if dtype == torch.int16:
        info = torch.iinfo(torch.int16)
        return (payoff * I16_SCALE).round().clamp(info.min, info.max).to(torch.int16)
    raise ValueError(f"unsupported quantized dtype {dtype}")


def allin_values_from_payoff_reference(
    payoff: torch.Tensor,
    beliefs: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Compute both-player all-in leaf values from a payoff matrix.

    ``payoff`` is row-hero/column-opponent EV in ``[-1, 1]``. ``beliefs`` is
    ``[M, 2, 1326]``.
    """
    payoff_f = payoff.to(device=beliefs.device, dtype=torch.float32) / float(scale)
    out = torch.empty_like(beliefs)
    denom_p0 = calculate_unblocked_mass(beliefs[:, 1]).clamp_min(1e-8)
    denom_p1 = calculate_unblocked_mass(beliefs[:, 0]).clamp_min(1e-8)
    out[:, 0] = beliefs[:, 1].to(torch.float32) @ payoff_f.T
    out[:, 0] /= denom_p0
    out[:, 1] = beliefs[:, 0].to(torch.float32) @ payoff_f.T
    out[:, 1] /= denom_p1
    return out.to(dtype=beliefs.dtype)


def allin_values_from_payoff_batch(
    payoff: torch.Tensor,
    beliefs: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Batched table matvec for all-in EV.

    ``payoff`` is ``[M, 1326, 1326]`` and may be int8/int16/fp32.
    """
    payoff_f = payoff.to(device=beliefs.device, dtype=torch.float32) / float(scale)
    out = torch.empty_like(beliefs)
    denom_p0 = calculate_unblocked_mass(beliefs[:, 1]).clamp_min(1e-8)
    denom_p1 = calculate_unblocked_mass(beliefs[:, 0]).clamp_min(1e-8)
    out[:, 0] = torch.bmm(payoff_f, beliefs[:, 1, :, None].to(torch.float32)).squeeze(-1)
    out[:, 0] /= denom_p0
    out[:, 1] = torch.bmm(payoff_f, beliefs[:, 0, :, None].to(torch.float32)).squeeze(-1)
    out[:, 1] /= denom_p1
    return out.to(dtype=beliefs.dtype)


@lru_cache(maxsize=1)
def canonical_flop_data_cpu() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return canonical flop data on CPU.

    Returns:
      canonical_flops: ``[1755, 3]`` sorted card IDs.
      actual_to_canonical: ``[22100]`` canonical row id for lexicographic actual flop id.
      actual_perm: ``[22100]`` suit permutation index mapping actual -> canonical.
    """
    perms = suit_permutations_tensor()
    canonical: dict[tuple[int, int, int], int] = {}
    actual_to_canonical: list[int] = []
    actual_perm: list[int] = []
    canonical_flops: list[tuple[int, int, int]] = []

    for flop in itertools.combinations(range(52), 3):
        cards = torch.tensor(flop, dtype=torch.long)
        ranks = cards % 13
        suits = cards // 13
        best_key: tuple[int, int, int] | None = None
        best_perm = 0
        for perm_idx, perm in enumerate(perms):
            mapped = (perm[suits] * 13 + ranks).sort().values
            key = tuple(int(x) for x in mapped.tolist())
            if best_key is None or key < best_key:
                best_key = key
                best_perm = perm_idx
        assert best_key is not None
        row = canonical.get(best_key)
        if row is None:
            row = len(canonical_flops)
            canonical[best_key] = row
            canonical_flops.append(best_key)
        actual_to_canonical.append(row)
        actual_perm.append(best_perm)

    return (
        torch.tensor(canonical_flops, dtype=torch.long),
        torch.tensor(actual_to_canonical, dtype=torch.long),
        torch.tensor(actual_perm, dtype=torch.long),
    )


def precompute_canonical_flop_tables_i8(device: torch.device) -> torch.Tensor:
    """Compute all canonical flop exact all-in payoff tables into GPU int8 VRAM."""
    canonical_flops, _, _ = canonical_flop_data_cpu()
    out = torch.empty(
        canonical_flops.shape[0],
        NUM_HANDS,
        NUM_HANDS,
        dtype=torch.int8,
        device=device,
    )
    for i, flop_cpu in enumerate(canonical_flops):
        out[i] = compute_postflop_payoff_quantized_triton(
            flop_cpu.to(device),
            dtype=torch.int8,
        )
    return out


def _allin_cache_dir() -> Path:
    root = os.environ.get("P2_ALLIN_CACHE_DIR")
    if root:
        return Path(root).expanduser()
    return _REPO_ROOT / "outputs" / "allin_cache"


def _atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _device_cache_key(device: torch.device) -> str:
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        return f"cuda:{index}"
    return str(device)


def _validate_payoff_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if tensor.dtype != dtype or tuple(tensor.shape) != shape:
        raise ValueError(
            f"{name} cache tensor must be shape={shape} dtype={dtype}; "
            f"got shape={tuple(tensor.shape)} dtype={tensor.dtype}"
        )
    return tensor.contiguous()


_FLOP_I8_CACHE: dict[int, torch.Tensor] = {}


def canonical_flop_tables_i8(device: torch.device) -> torch.Tensor | None:
    """Process-local canonical flop payoff table cache.

    CUDA evaluators always use the precomputed canonical flop table. CPU
    evaluators keep the existing per-board eager fallback to avoid forcing a
    multi-GB host allocation.
    """
    if device.type != "cuda":
        return None
    key = device.index if device.index is not None else torch.cuda.current_device()
    table = _FLOP_I8_CACHE.get(key)
    if table is None:
        cache_path = _allin_cache_dir() / _FLOP_CACHE_NAME
        if cache_path.exists():
            payload = torch.load(cache_path, map_location=device, weights_only=False)
            table = _validate_payoff_tensor(
                payload["flop_i8"],
                name=str(cache_path),
                dtype=torch.int8,
                shape=(canonical_flop_data_cpu()[0].shape[0], NUM_HANDS, NUM_HANDS),
            )
            _FLOP_I8_CACHE[key] = table
            return table
        table = precompute_canonical_flop_tables_i8(device)
        _atomic_torch_save(
            {
                "flop_i8": table.cpu(),
                "metadata": {
                    "version": _CACHE_VERSION,
                    "scale": FLOP_I8_SCALE,
                    "num_hands": NUM_HANDS,
                    "canonical_flops": canonical_flop_data_cpu()[0],
                },
            },
            cache_path,
        )
        _FLOP_I8_CACHE[key] = table
    return table


_PREFLOP_I16_CACHE: dict[tuple[str, int, int, str], torch.Tensor] = {}


def _preflop_cache_path(path: Path, size: int, mtime_ns: int) -> Path:
    source = str(path.expanduser().resolve())
    digest = hashlib.sha256(f"{source}:{size}:{mtime_ns}".encode()).hexdigest()[:16]
    return _allin_cache_dir() / f"preflop_i16_v{_CACHE_VERSION}_{digest}.pt"


def _load_preflop_source_cpu(path: Path) -> torch.Tensor:
    if path.suffix == ".zst":
        import io

        import zstandard as zstd

        data = zstd.ZstdDecompressor().decompress(path.read_bytes())
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if "payoff_i16" not in payload:
        raise KeyError(f"{path} does not contain payoff_i16")
    return _validate_payoff_tensor(
        payload["payoff_i16"],
        name=str(path),
        dtype=torch.int16,
        shape=(NUM_HANDS, NUM_HANDS),
    )


def load_preflop_payoff_i16(path: str | Path, device: torch.device) -> torch.Tensor:
    path = Path(path).expanduser()
    stat = path.stat()
    device_key = _device_cache_key(device)
    source_key = str(path.resolve())
    key = (source_key, int(stat.st_size), int(stat.st_mtime_ns), device_key)
    cached = _PREFLOP_I16_CACHE.get(key)
    if cached is not None:
        return cached

    cache_path = _preflop_cache_path(path, int(stat.st_size), int(stat.st_mtime_ns))
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        table_cpu = _validate_payoff_tensor(
            payload["payoff_i16"],
            name=str(cache_path),
            dtype=torch.int16,
            shape=(NUM_HANDS, NUM_HANDS),
        )
    else:
        table_cpu = _load_preflop_source_cpu(path)
        _atomic_torch_save(
            {
                "payoff_i16": table_cpu,
                "metadata": {
                    "version": _CACHE_VERSION,
                    "source": source_key,
                    "source_size": int(stat.st_size),
                    "source_mtime_ns": int(stat.st_mtime_ns),
                    "scale": I16_SCALE,
                    "num_hands": NUM_HANDS,
                },
            },
            cache_path,
        )

    table = table_cpu.to(device=device, dtype=torch.int16).contiguous()
    _PREFLOP_I16_CACHE[key] = table
    return table


class AllInPayoffResolver:
    """Shared all-in-call terminal payoff helper."""

    def __init__(
        self,
        *,
        device: torch.device,
        preflop_table_path: str | Path | None = None,
    ) -> None:
        self.device = device
        self.preflop_table_path = preflop_table_path
        self._preflop_i16: torch.Tensor | None = None
        self._flop_i8: torch.Tensor | None = canonical_flop_tables_i8(device)
        self._turn_cache: dict[tuple[int, ...], torch.Tensor] = {}
        self._flop_cache: dict[tuple[int, int, int], torch.Tensor] = {}
        self._actual_to_canon_device: torch.Tensor | None = None
        self._actual_perm_device: torch.Tensor | None = None
        self._combo_perms_device: torch.Tensor | None = None

    def flop_lookup_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._actual_to_canon_device is None or self._actual_perm_device is None:
            _, actual_to_canon, actual_perm = canonical_flop_data_cpu()
            self._actual_to_canon_device = actual_to_canon.to(self.device)
            self._actual_perm_device = actual_perm.to(self.device)
        if self._combo_perms_device is None:
            self._combo_perms_device = combo_suit_permutation_tensor(device=self.device)
        return (
            self._actual_to_canon_device,
            self._actual_perm_device,
            self._combo_perms_device,
        )

    def payoff_for_board(self, board: torch.Tensor, street: int) -> tuple[torch.Tensor, float]:
        if street == 0:
            if self._preflop_i16 is None:
                if self.preflop_table_path is None:
                    raise RuntimeError(
                        "preflop all-in payoff table path is required for preflop all-in abstraction"
                    )
                self._preflop_i16 = load_preflop_payoff_i16(self.preflop_table_path, self.device)
            return self._preflop_i16, I16_SCALE

        cards = tuple(int(c) for c in board[: 3 if street == 1 else 4].tolist())
        if street == 1:
            flop_key = tuple(sorted(cards))
            if self._flop_i8 is not None:
                canon, actual_to_canon, actual_perm = canonical_flop_data_cpu()
                # Lexicographic combination rank for actual sorted flop.
                actual_idx = _flop_combination_index(flop_key)
                canon_idx = int(actual_to_canon[actual_idx].item())
                perm_idx = int(actual_perm[actual_idx].item())
                table = self._flop_i8[canon_idx]
                if perm_idx == 0:
                    return table, FLOP_I8_SCALE
                perm = combo_suit_permutation_tensor(device=self.device)[perm_idx]
                # Materialize a permuted view for the rare non-canonical lookup path.
                return table.index_select(0, perm).index_select(1, perm), FLOP_I8_SCALE
            table = self._flop_cache.get(flop_key)
            if table is None:
                table = compute_postflop_payoff_quantized_triton(
                    torch.tensor(flop_key, device=self.device),
                    dtype=torch.int8,
                )
                self._flop_cache[flop_key] = table
            return table, FLOP_I8_SCALE

        if street == 2:
            turn_key = tuple(cards)
            table = self._turn_cache.get(turn_key)
            if table is None:
                table = compute_postflop_payoff_quantized_triton(
                    torch.tensor(turn_key, device=self.device),
                    dtype=torch.int16,
                )
                self._turn_cache[turn_key] = table
            return table, I16_SCALE

        raise ValueError(f"all-in payoff resolver only handles preflop/flop/turn, got street={street}")

    def values_for_boards(
        self,
        *,
        street: int,
        boards: torch.Tensor,
        beliefs: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized all-in values for one street and many boards."""
        if beliefs.numel() == 0:
            return torch.empty_like(beliefs)
        if street == 0:
            table, scale = self.payoff_for_board(boards.new_empty(0), 0)
            return allin_values_from_payoff_reference(table, beliefs, scale=scale)
        if street == 1:
            if self._flop_i8 is not None:
                flop = boards[:, :3].long().sort(dim=1).values
                actual_idx = _flop_combination_index_tensor(flop)
                _, actual_to_canon, actual_perm = canonical_flop_data_cpu()
                canon_ids = actual_to_canon.to(self.device)[actual_idx]
                perm_ids = actual_perm.to(self.device)[actual_idx]
                tables = self._flop_i8.index_select(0, canon_ids)
                perms = combo_suit_permutation_tensor(device=self.device).index_select(
                    0, perm_ids
                )
                tables = tables.gather(
                    1,
                    perms[:, :, None].expand(-1, -1, NUM_HANDS),
                ).gather(
                    2,
                    perms[:, None, :].expand(-1, NUM_HANDS, -1),
                )
                return allin_values_from_payoff_batch(tables, beliefs, scale=FLOP_I8_SCALE)
            tables = compute_postflop_payoff_quantized_triton_batched(
                boards[:, :3],
                dtype=torch.int8,
            )
            return allin_values_from_payoff_batch(tables, beliefs, scale=FLOP_I8_SCALE)
        if street == 2:
            tables = compute_postflop_payoff_quantized_triton_batched(
                boards[:, :4],
                dtype=torch.int16,
            )
            return allin_values_from_payoff_batch(tables, beliefs, scale=I16_SCALE)
        raise ValueError(f"unsupported all-in street {street}")


def _flop_combination_index(flop: tuple[int, int, int]) -> int:
    a, b, c = flop
    total = 52 * 51 * 50 // 6
    prefix_a = total - (52 - a) * (51 - a) * (50 - a) // 6
    prefix_b = (51 - a) * (50 - a) // 2 - (52 - b) * (51 - b) // 2
    return int(prefix_a + prefix_b + c - b - 1)


def _flop_combination_index_tensor(flop: torch.Tensor) -> torch.Tensor:
    a = flop[:, 0]
    b = flop[:, 1]
    c = flop[:, 2]
    total = 52 * 51 * 50 // 6
    rem_a = (52 - a) * (51 - a) * (50 - a) // 6
    prefix_a = total - rem_a
    rem_after_a = (51 - a) * (50 - a) // 2
    rem_after_b = (52 - b) * (51 - b) // 2
    prefix_b = rem_after_a - rem_after_b
    return (prefix_a + prefix_b + c - b - 1).long()


if triton is not None:

    @triton.jit
    def _allin_table_values_both_players_kernel(
        table,
        beliefs,
        node_indices,
        latest_values,
        stacks,
        pot,
        starting_stacks,
        env_scale,
        compatible,
        canon_ids,
        perm_ids,
        combo_perms,
        M: tl.constexpr,
        H: tl.constexpr,
        TABLE_SCALE: tl.constexpr,
        BH: tl.constexpr,
        BK: tl.constexpr,
        HAS_BATCH_TABLE: tl.constexpr,
        HAS_CANON_TABLE: tl.constexpr,
        HAS_PERM: tl.constexpr,
    ):
        m = tl.program_id(0)
        bh = tl.program_id(1)
        h = bh * BH + tl.arange(0, BH)
        hmask = h < H
        node = tl.load(node_indices + m)

        denom0 = tl.zeros((BH,), tl.float32)
        numer0 = tl.zeros((BH,), tl.float32)
        denom1 = tl.zeros((BH,), tl.float32)
        numer1 = tl.zeros((BH,), tl.float32)
        hero_table = h
        perm_id = tl.full((), 0, tl.int64)
        if HAS_PERM:
            perm_id = tl.load(perm_ids + m)
            hero_table = tl.load(combo_perms + perm_id * H + h, mask=hmask, other=0)
        table_base = tl.full((), 0, tl.int64)
        if HAS_BATCH_TABLE:
            table_base = m * H * H
        if HAS_CANON_TABLE:
            canon_id = tl.load(canon_ids + m)
            table_base = canon_id * H * H

        for k0 in range(0, H, BK):
            k = k0 + tl.arange(0, BK)
            kmask = k < H
            belief0 = tl.load(
                beliefs + (node * 2) * H + k,
                mask=kmask,
                other=0.0,
            ).to(tl.float32)
            belief1 = tl.load(
                beliefs + (node * 2 + 1) * H + k,
                mask=kmask,
                other=0.0,
            ).to(tl.float32)
            opp_table = k
            if HAS_PERM:
                opp_table = tl.load(combo_perms + perm_id * H + k, mask=kmask, other=0)
            q = tl.load(
                table + table_base + hero_table[:, None] * H + opp_table[None, :],
                mask=hmask[:, None] & kmask[None, :],
                other=0,
            ).to(tl.float32)
            q = q / TABLE_SCALE
            pair_ok = tl.load(
                compatible + h[:, None] * H + k[None, :],
                mask=hmask[:, None] & kmask[None, :],
                other=0,
            ).to(tl.int1)
            b0 = tl.where(pair_ok, belief0[None, :], 0.0)
            b1 = tl.where(pair_ok, belief1[None, :], 0.0)
            denom0 += tl.sum(b1, axis=1)
            numer0 += tl.sum(q * b1, axis=1)
            denom1 += tl.sum(b0, axis=1)
            numer1 += tl.sum(q * b0, axis=1)

        ev0 = tl.where(denom0 > 1.0e-8, numer0 / denom0, 0.0)
        ev1 = tl.where(denom1 > 1.0e-8, numer1 / denom1, 0.0)
        p = tl.load(pot + node).to(tl.float32)
        sc = tl.load(env_scale + node).to(tl.float32)
        stack0 = tl.load(stacks + node * 2).to(tl.float32)
        stack1 = tl.load(stacks + node * 2 + 1).to(tl.float32)
        start0 = tl.load(starting_stacks + node * 2).to(tl.float32)
        start1 = tl.load(starting_stacks + node * 2 + 1).to(tl.float32)
        ev0 *= (stack0 + p - start0) / tl.maximum(sc, 1.0e-8)
        ev1 *= (stack1 + p - start1) / tl.maximum(sc, 1.0e-8)
        tl.store(latest_values + (node * 2) * H + h, ev0, mask=hmask)
        tl.store(latest_values + (node * 2 + 1) * H + h, ev1, mask=hmask)


    @triton.jit
    def _turn_allin_values_kernel(
        ranks,
        river_valid,
        beliefs,
        node_indices,
        latest_values,
        stacks,
        pot,
        starting_stacks,
        env_scale,
        combos,
        hand_ok,
        compatible,
        M: tl.constexpr,
        T: tl.constexpr,
        H: tl.constexpr,
        BH: tl.constexpr,
        BK: tl.constexpr,
    ):
        m = tl.program_id(0)
        player = tl.program_id(1)
        bh = tl.program_id(2)
        h = bh * BH + tl.arange(0, BH)
        hmask = h < H
        node = tl.load(node_indices + m)
        opp = 1 - player

        hi0 = tl.load(combos + h * 2, mask=hmask, other=-99)
        hi1 = tl.load(combos + h * 2 + 1, mask=hmask, other=-98)
        hok = tl.load(hand_ok + m * H + h, mask=hmask, other=0).to(tl.int1)
        denom = tl.zeros((BH,), tl.float32)
        numer = tl.zeros((BH,), tl.float32)

        for k0 in range(0, H, BK):
            k = k0 + tl.arange(0, BK)
            kmask = k < H
            kj0 = tl.load(combos + k * 2, mask=kmask, other=-97)
            kj1 = tl.load(combos + k * 2 + 1, mask=kmask, other=-96)
            belief = tl.load(
                beliefs + (node * 2 + opp) * H + k,
                mask=kmask,
                other=0.0,
            ).to(tl.float32)
            pair_ok = (
                hok[:, None]
                & tl.load(hand_ok + m * H + k, mask=kmask, other=0)[None, :].to(tl.int1)
                & tl.load(
                    compatible + h[:, None] * H + k[None, :],
                    mask=hmask[:, None] & kmask[None, :],
                    other=0,
                ).to(tl.int1)
            )
            score = tl.zeros((BH, BK), tl.float32)
            count = tl.zeros((BH, BK), tl.float32)
            for t in range(0, T):
                rv_ok = tl.load(river_valid + m * T + t).to(tl.int1)
                river = t
                valid = (
                    rv_ok
                    & pair_ok
                    & (river != hi0[:, None])
                    & (river != hi1[:, None])
                    & (river != kj0[None, :])
                    & (river != kj1[None, :])
                )
                rh = tl.load(
                    ranks + (m * T + t) * H + h,
                    mask=hmask,
                    other=0,
                )
                rk = tl.load(
                    ranks + (m * T + t) * H + k,
                    mask=kmask,
                    other=0,
                )
                cmp = tl.where(
                    rh[:, None] > rk[None, :],
                    1.0,
                    tl.where(rh[:, None] < rk[None, :], -1.0, 0.0),
                )
                score += tl.where(valid, cmp, 0.0)
                count += tl.where(valid, 1.0, 0.0)

            payoff = tl.where(count > 0.0, score / tl.maximum(count, 1.0), 0.0)
            b = tl.where(pair_ok & (count > 0.0), belief[None, :], 0.0)
            denom += tl.sum(b, axis=1)
            numer += tl.sum(payoff * b, axis=1)

        ev = tl.where(denom > 1.0e-8, numer / denom, 0.0)
        stack = tl.load(stacks + node * 2 + player).to(tl.float32)
        p = tl.load(pot + node).to(tl.float32)
        start = tl.load(starting_stacks + node * 2 + player).to(tl.float32)
        sc = tl.load(env_scale + node).to(tl.float32)
        ev *= (stack + p - start) / tl.maximum(sc, 1.0e-8)
        tl.store(latest_values + (node * 2 + player) * H + h, ev, mask=hmask)

    @triton.jit
    def _postflop_payoff_tile_kernel(
        ranks,
        runouts,
        combos,
        hand_ok,
        compatible,
        score_out,
        count_out,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        BT: tl.constexpr,
        BI: tl.constexpr,
        BJ: tl.constexpr,
    ):
        pi = tl.program_id(0)
        pj = tl.program_id(1)
        pt = tl.program_id(2)
        oi = pi * BI + tl.arange(0, BI)
        oj = pj * BJ + tl.arange(0, BJ)
        ot = pt * BT + tl.arange(0, BT)
        mi = oi < H
        mj = oj < H
        mt = ot < T

        ri = tl.load(ranks + ot[:, None] * H + oi[None, :], mask=mt[:, None] & mi[None, :], other=0)
        rj = tl.load(ranks + ot[:, None] * H + oj[None, :], mask=mt[:, None] & mj[None, :], other=0)

        hi0 = tl.load(combos + oi * 2, mask=mi, other=-99)
        hi1 = tl.load(combos + oi * 2 + 1, mask=mi, other=-98)
        hj0 = tl.load(combos + oj * 2, mask=mj, other=-97)
        hj1 = tl.load(combos + oj * 2 + 1, mask=mj, other=-96)
        ok_i = tl.load(hand_ok + oi, mask=mi, other=0).to(tl.int1)
        ok_j = tl.load(hand_ok + oj, mask=mj, other=0).to(tl.int1)
        pair_ok = (
            ok_i[:, None]
            & ok_j[None, :]
            & tl.load(compatible + oi[:, None] * H + oj[None, :], mask=mi[:, None] & mj[None, :], other=0).to(tl.int1)
        )

        r0 = tl.load(runouts + ot * R, mask=mt, other=-1)
        valid_t_i = (r0[:, None] != hi0[None, :]) & (r0[:, None] != hi1[None, :])
        valid_t_j = (r0[:, None] != hj0[None, :]) & (r0[:, None] != hj1[None, :])
        if R == 2:
            r1 = tl.load(runouts + ot * R + 1, mask=mt, other=-1)
            valid_t_i = valid_t_i & (r1[:, None] != hi0[None, :]) & (r1[:, None] != hi1[None, :])
            valid_t_j = valid_t_j & (r1[:, None] != hj0[None, :]) & (r1[:, None] != hj1[None, :])

        valid = mt[:, None, None] & pair_ok[None, :, :] & valid_t_i[:, :, None] & valid_t_j[:, None, :]
        cmp = tl.where(ri[:, :, None] > rj[:, None, :], 1, tl.where(ri[:, :, None] < rj[:, None, :], -1, 0))
        score = tl.sum(tl.where(valid, cmp, 0), axis=0)
        count = tl.sum(tl.where(valid, 1, 0), axis=0)
        offsets = oi[:, None] * H + oj[None, :]
        mask = mi[:, None] & mj[None, :]
        tl.atomic_add(score_out + offsets, score, sem="relaxed", mask=mask)
        tl.atomic_add(count_out + offsets, count, sem="relaxed", mask=mask)


    @triton.jit
    def _postflop_payoff_batched_tile_kernel(
        ranks,
        runouts,
        runout_valid,
        combos,
        hand_ok,
        compatible,
        score_out,
        count_out,
        B: tl.constexpr,
        T: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        TBLOCKS: tl.constexpr,
        BT: tl.constexpr,
        BI: tl.constexpr,
        BJ: tl.constexpr,
    ):
        pbt = tl.program_id(0)
        pb = pbt // TBLOCKS
        pt = pbt - pb * TBLOCKS
        pi = tl.program_id(1)
        pj = tl.program_id(2)
        oi = pi * BI + tl.arange(0, BI)
        oj = pj * BJ + tl.arange(0, BJ)
        ot = pt * BT + tl.arange(0, BT)
        mi = oi < H
        mj = oj < H
        mt = ot < T

        rank_base = pb * T * H
        ri = tl.load(
            ranks + rank_base + ot[:, None] * H + oi[None, :],
            mask=mt[:, None] & mi[None, :],
            other=0,
        )
        rj = tl.load(
            ranks + rank_base + ot[:, None] * H + oj[None, :],
            mask=mt[:, None] & mj[None, :],
            other=0,
        )

        hi0 = tl.load(combos + oi * 2, mask=mi, other=-99)
        hi1 = tl.load(combos + oi * 2 + 1, mask=mi, other=-98)
        hj0 = tl.load(combos + oj * 2, mask=mj, other=-97)
        hj1 = tl.load(combos + oj * 2 + 1, mask=mj, other=-96)
        ok_i = tl.load(hand_ok + pb * H + oi, mask=mi, other=0).to(tl.int1)
        ok_j = tl.load(hand_ok + pb * H + oj, mask=mj, other=0).to(tl.int1)
        pair_ok = (
            ok_i[:, None]
            & ok_j[None, :]
            & tl.load(
                compatible + oi[:, None] * H + oj[None, :],
                mask=mi[:, None] & mj[None, :],
                other=0,
            ).to(tl.int1)
        )

        runout_ok = tl.load(runout_valid + pb * T + ot, mask=mt, other=0).to(tl.int1)
        r0 = tl.load(runouts + (pb * T + ot) * R, mask=mt, other=-1)
        valid_t_i = (r0[:, None] != hi0[None, :]) & (r0[:, None] != hi1[None, :])
        valid_t_j = (r0[:, None] != hj0[None, :]) & (r0[:, None] != hj1[None, :])
        if R == 2:
            r1 = tl.load(runouts + (pb * T + ot) * R + 1, mask=mt, other=-1)
            valid_t_i = valid_t_i & (r1[:, None] != hi0[None, :]) & (r1[:, None] != hi1[None, :])
            valid_t_j = valid_t_j & (r1[:, None] != hj0[None, :]) & (r1[:, None] != hj1[None, :])

        valid = (
            mt[:, None, None]
            & runout_ok[:, None, None]
            & pair_ok[None, :, :]
            & valid_t_i[:, :, None]
            & valid_t_j[:, None, :]
        )
        cmp = tl.where(
            ri[:, :, None] > rj[:, None, :],
            1,
            tl.where(ri[:, :, None] < rj[:, None, :], -1, 0),
        )
        score = tl.sum(tl.where(valid, cmp, 0), axis=0)
        count = tl.sum(tl.where(valid, 1, 0), axis=0)
        offsets = pb * H * H + oi[:, None] * H + oj[None, :]
        mask = mi[:, None] & mj[None, :]
        tl.atomic_add(score_out + offsets, score, sem="relaxed", mask=mask)
        tl.atomic_add(count_out + offsets, count, sem="relaxed", mask=mask)


def compute_postflop_payoff_quantized_triton(
    board: torch.Tensor,
    *,
    dtype: torch.dtype,
    bt: int = 32,
    bi: int = 16,
    bj: int = 16,
) -> torch.Tensor:
    if triton is None or board.device.type != "cuda":
        return compute_postflop_payoff_quantized(board, dtype=dtype)
    board = board.to(device=board.device, dtype=torch.long)
    full_boards = _runout_boards(board)
    runouts = full_boards[:, board.numel() :].to(torch.int16).contiguous()
    combos = hand_combos_tensor(device=board.device).to(torch.int16).contiguous()
    cards = torch.cat(
        [
            combos.view(1, NUM_HANDS, 2).expand(full_boards.shape[0], NUM_HANDS, 2),
            full_boards.to(torch.int16).view(full_boards.shape[0], 1, 5).expand(
                full_boards.shape[0], NUM_HANDS, 5
            ),
        ],
        dim=-1,
    ).reshape(-1, 7).contiguous()
    ranks = rank_7_cards_single_batch_triton(cards).view(full_boards.shape[0], NUM_HANDS).contiguous()
    hand_ok = board_allowed_hands(board.view(1, -1)).squeeze(0).contiguous()
    compatible = combo_compatible_tensor(device=board.device).contiguous()
    score = torch.zeros(NUM_HANDS, NUM_HANDS, dtype=torch.int32, device=board.device)
    count = torch.zeros_like(score)
    grid = (
        triton.cdiv(NUM_HANDS, bi),
        triton.cdiv(NUM_HANDS, bj),
        triton.cdiv(full_boards.shape[0], bt),
    )
    _postflop_payoff_tile_kernel[grid](
        ranks,
        runouts,
        combos,
        hand_ok,
        compatible,
        score,
        count,
        full_boards.shape[0],
        NUM_HANDS,
        runouts.shape[1],
        BT=bt,
        BI=bi,
        BJ=bj,
        num_warps=8,
    )
    payoff = torch.where(count > 0, score.float() / count.clamp_min(1).float(), torch.zeros_like(score, dtype=torch.float32))
    if dtype == torch.int8:
        return (payoff * FLOP_I8_SCALE).round().clamp(-128, 127).to(torch.int8)
    if dtype == torch.int16:
        info = torch.iinfo(torch.int16)
        return (payoff * I16_SCALE).round().clamp(info.min, info.max).to(torch.int16)
    raise ValueError(f"unsupported quantized dtype {dtype}")


def compute_postflop_payoff_quantized_triton_batched(
    boards: torch.Tensor,
    *,
    dtype: torch.dtype,
    bt: int = 32,
    bi: int = 16,
    bj: int = 16,
) -> torch.Tensor:
    if boards.dim() != 2 or boards.shape[1] not in (3, 4):
        raise ValueError(f"boards must be [B,3] or [B,4], got {tuple(boards.shape)}")
    if triton is None or boards.device.type != "cuda":
        raise RuntimeError("batched postflop all-in payoff generation requires CUDA + Triton")
    device = boards.device
    boards = boards.long().contiguous()
    batch = boards.shape[0]
    missing = 5 - boards.shape[1]
    if missing == 2:
        runouts = hand_combos_tensor(device=device).to(torch.long)
        runout_valid = (
            runouts.view(1, runouts.shape[0], runouts.shape[1], 1)
            != boards.view(batch, 1, 1, boards.shape[1])
        ).all(dim=(2, 3))
    else:
        runouts = torch.arange(52, device=device, dtype=torch.long)[:, None]
        runout_valid = (
            runouts.view(1, runouts.shape[0], 1)
            != boards.view(batch, 1, boards.shape[1])
        ).all(dim=2)
    T = runouts.shape[0]
    R = runouts.shape[1]
    runouts_b = runouts.view(1, T, R).expand(batch, T, R).contiguous()
    full_boards = torch.cat(
        [
            boards[:, None, :].expand(batch, T, boards.shape[1]),
            runouts_b,
        ],
        dim=2,
    ).reshape(batch * T, 5)

    ranks = _rank_hands(full_boards).view(batch, T, NUM_HANDS).contiguous()
    combos = hand_combos_tensor(device=device).to(torch.int16).contiguous()
    hand_ok = board_allowed_hands(boards).contiguous()
    compatible = combo_compatible_tensor(device=device).contiguous()
    score = torch.zeros(batch, NUM_HANDS, NUM_HANDS, dtype=torch.int32, device=device)
    count = torch.zeros_like(score)
    tblocks = triton.cdiv(T, bt)
    grid = (
        batch * tblocks,
        triton.cdiv(NUM_HANDS, bi),
        triton.cdiv(NUM_HANDS, bj),
    )
    _postflop_payoff_batched_tile_kernel[grid](
        ranks,
        runouts_b.to(torch.int16),
        runout_valid.contiguous(),
        combos,
        hand_ok,
        compatible,
        score,
        count,
        batch,
        T,
        NUM_HANDS,
        R,
        tblocks,
        BT=bt,
        BI=bi,
        BJ=bj,
        num_warps=8,
    )
    payoff = torch.where(
        count > 0,
        score.float() / count.clamp_min(1).float(),
        torch.zeros_like(score, dtype=torch.float32),
    )
    if dtype == torch.int8:
        return (payoff * FLOP_I8_SCALE).round().clamp(-128, 127).to(torch.int8)
    if dtype == torch.int16:
        info = torch.iinfo(torch.int16)
        return (payoff * I16_SCALE).round().clamp(info.min, info.max).to(torch.int16)
    raise ValueError(f"unsupported quantized dtype {dtype}")


def write_allin_table_values_triton_(
    *,
    table: torch.Tensor,
    beliefs: torch.Tensor,
    node_indices: torch.Tensor,
    latest_values: torch.Tensor,
    stacks: torch.Tensor,
    pot: torch.Tensor,
    starting_stacks: torch.Tensor,
    env_scale: torch.Tensor,
    table_scale: float,
    canon_ids: torch.Tensor | None = None,
    perm_ids: torch.Tensor | None = None,
    combo_perms: torch.Tensor | None = None,
    batch_table: bool = False,
    block_h: int = 16,
    block_k: int = 64,
) -> None:
    if triton is None or beliefs.device.type != "cuda":
        raise RuntimeError("fused all-in table matvec requires CUDA + Triton")
    if node_indices.numel() == 0:
        return
    device = beliefs.device
    empty = torch.empty(0, dtype=torch.long, device=device)
    compatible = combo_compatible_tensor(device=device).contiguous()
    has_canon = canon_ids is not None
    has_perm = perm_ids is not None
    if canon_ids is None:
        canon_ids = empty
    if perm_ids is None:
        perm_ids = empty
    if combo_perms is None:
        combo_perms = empty
    grid = (node_indices.numel(), triton.cdiv(NUM_HANDS, block_h))
    _allin_table_values_both_players_kernel[grid](
        table.contiguous(),
        beliefs.contiguous(),
        node_indices.contiguous(),
        latest_values,
        stacks,
        pot,
        starting_stacks,
        env_scale,
        compatible,
        canon_ids.contiguous(),
        perm_ids.contiguous(),
        combo_perms.contiguous(),
        node_indices.numel(),
        NUM_HANDS,
        float(table_scale),
        BH=block_h,
        BK=block_k,
        HAS_BATCH_TABLE=batch_table,
        HAS_CANON_TABLE=has_canon,
        HAS_PERM=has_perm,
        num_warps=4,
    )


def write_turn_allin_values_triton_(
    *,
    boards: torch.Tensor,
    beliefs: torch.Tensor,
    node_indices: torch.Tensor,
    latest_values: torch.Tensor,
    stacks: torch.Tensor,
    pot: torch.Tensor,
    starting_stacks: torch.Tensor,
    env_scale: torch.Tensor,
    block_h: int = 8,
    block_k: int = 32,
) -> None:
    if triton is None or beliefs.device.type != "cuda":
        raise RuntimeError("fused turn all-in EV requires CUDA + Triton")
    if node_indices.numel() == 0:
        return
    device = beliefs.device
    turn_boards = boards[:, :4].long().contiguous()
    rivers = torch.arange(52, dtype=torch.long, device=device)
    river_valid = (rivers.view(1, 52, 1) != turn_boards[:, None, :]).all(dim=2)
    full_boards = torch.cat(
        [
            turn_boards[:, None, :].expand(turn_boards.shape[0], 52, 4),
            rivers.view(1, 52, 1).expand(turn_boards.shape[0], 52, 1),
        ],
        dim=2,
    ).reshape(-1, 5)
    ranks = _rank_hands(full_boards).view(turn_boards.shape[0], 52, NUM_HANDS)
    combos = hand_combos_tensor(device=device).to(torch.int16).contiguous()
    hand_ok = board_allowed_hands(turn_boards).contiguous()
    compatible = combo_compatible_tensor(device=device).contiguous()
    grid = (node_indices.numel(), 2, triton.cdiv(NUM_HANDS, block_h))
    _turn_allin_values_kernel[grid](
        ranks.contiguous(),
        river_valid,
        beliefs.contiguous(),
        node_indices.contiguous(),
        latest_values,
        stacks,
        pot,
        starting_stacks,
        env_scale,
        combos,
        hand_ok,
        compatible,
        node_indices.numel(),
        52,
        NUM_HANDS,
        BH=block_h,
        BK=block_k,
        num_warps=4,
    )
