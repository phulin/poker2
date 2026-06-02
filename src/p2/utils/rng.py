from __future__ import annotations

import torch


def generator_state_for_set_state(state: torch.Tensor) -> torch.Tensor:
    """Return a CPU ByteTensor suitable for torch.Generator.set_state."""
    if not isinstance(state, torch.Tensor):
        raise TypeError(f"expected torch.Tensor RNG state, got {type(state).__name__}")
    if state.dtype != torch.uint8:
        raise TypeError(f"expected uint8 RNG state, got {state.dtype}")
    return state.detach().to(device=torch.device("cpu")).contiguous()
