from __future__ import annotations

import torch

from p2.models.mlp.preflop_token_mixer_mpk import (
    PreflopTokenMixerMPKConfig,
    _validate_runtime_tensors,
    mirage_mpk_is_available,
)


def test_preflop_token_mixer_mpk_module_is_import_safe() -> None:
    assert isinstance(mirage_mpk_is_available(), bool)


def test_preflop_token_mixer_mpk_shape_validation_accepts_staged_contract() -> None:
    config = PreflopTokenMixerMPKConfig(batch_size=2, dim=16, device="cpu")
    x = torch.empty(2, 7, 16, dtype=torch.bfloat16)
    y = torch.empty_like(x)
    gate = torch.empty_like(x)
    w_in = torch.empty(28, 7, dtype=torch.bfloat16)
    w_out = torch.empty(7, 28, dtype=torch.bfloat16)

    _validate_runtime_tensors(x, y, gate, w_in, w_out, config)
