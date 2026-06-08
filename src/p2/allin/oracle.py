from __future__ import annotations

import io
from dataclasses import fields
from functools import lru_cache
from pathlib import Path

import torch

from p2.allin.model import PreflopAllIn169EquityModel
from p2.allin.train import AllInTrainConfig, _build_model
from p2.env.card_utils import (
    PREFLOP_HANDS,
    combo_to_onehot_tensor,
    combo_to_preflop_class_tensor,
    preflop_class_compatibility_counts_tensor,
    preflop_class_multiplicity_tensor,
)

_U16_SCALE = float(torch.iinfo(torch.uint16).max)
_QUANTIZED_FORMAT = "p2.allin.preflop_allin_169.u16.v1"


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
        opp0_per_combo = opp0.to(torch.float32) / self._multiplicity
        opp1_per_combo = opp1.to(torch.float32) / self._multiplicity
        out = torch.empty(hero.shape[0], PREFLOP_HANDS, dtype=torch.float32, device=hero.device)
        weighted = self._compat3 * self._share3
        for start in range(0, hero.shape[0], 256):
            end = min(start + 256, hero.shape[0])
            o0 = opp0_per_combo[start:end]
            o1 = opp1_per_combo[start:end]
            numer = torch.einsum("ra,rb,hab->rh", o0, o1, weighted)
            denom = torch.einsum("ra,rb,hab->rh", o0, o1, self._compat3).clamp_min(1.0e-8)
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
        for player in range(players):
            rows = torch.nonzero(live[:, player], as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            if live_players == 2:
                opp_mask = live[rows].clone()
                opp_mask[:, player] = False
                opp = opp_mask.to(torch.long).argmax(dim=1)
                share[rows, player] = self._share2_values(
                    beliefs[rows, player],
                    beliefs[rows, opp],
                )
            elif live_players == 3:
                opp_mask = live[rows].clone()
                opp_mask[:, player] = False
                opps = torch.nonzero(opp_mask, as_tuple=False)[:, 1].reshape(-1, 2)
                share[rows, player] = self._share3_values(
                    beliefs[rows, player],
                    beliefs[rows, opps[:, 0]],
                    beliefs[rows, opps[:, 1]],
                )
            else:
                raise ValueError("exact all-in resolver only handles 2/3 live players")
        return share

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
        pred_hand_major = model(
            beliefs.transpose(1, 2).contiguous(),
            starting_stacks,
            committed,
            stacks_after,
            allin_mask,
            folded_mask,
            hardcode_folded_values=self._model_target_mode == "net_value",
        )
        pred = pred_hand_major.transpose(1, 2).contiguous()
        if self._model_target_mode == "net_value":
            return pred.to(beliefs.dtype)
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
