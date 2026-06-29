from __future__ import annotations

import torch

from p2.models.mlp.better_ffn import (
    _PreflopGatedTokenMixerBlock,
    _preflop_token_mixer_gate_residual_triton,
)


def preflop_gated_token_mixer_token_path(
    block: _PreflopGatedTokenMixerBlock,
    x: torch.Tensor,
) -> torch.Tensor:
    """Run the current best eval/no-grad token path without the block FFN."""

    y = block.token_norm(x)
    gate = block.token_gate(y)
    return _preflop_token_mixer_gate_residual_triton(
        x,
        y,
        gate,
        block.token_mixer.linear_in.weight,
        block.token_mixer.linear_out.weight,
    )


class PreflopGatedTokenMixerCudaGraphRunner:
    """Fixed-shape CUDA Graph runner for the gated token-mixer token path.

    The runner owns static input and output buffers. ``copy_replay`` copies a
    new input into the static input buffer, replays the captured graph, and
    returns the static output buffer. That output is overwritten by the next
    replay. This explicit lifetime contract is why this is not hidden behind
    the normal module ``forward`` method.
    """

    def __init__(
        self,
        block: _PreflopGatedTokenMixerBlock,
        x_template: torch.Tensor,
        *,
        warmup: int = 8,
    ) -> None:
        if not x_template.is_cuda:
            raise ValueError("CUDA graph token mixer runner requires a CUDA input")
        if x_template.ndim != 3 or x_template.shape[1] != 7:
            raise ValueError("CUDA graph token mixer runner expects shape [B, 7, D]")
        if block.training:
            raise ValueError("CUDA graph token mixer runner requires an eval-mode block")
        if torch.is_grad_enabled():
            raise RuntimeError("construct graph runner under torch.no_grad()")

        self.block = block
        self.shape = tuple(x_template.shape)
        self.dtype = x_template.dtype
        self.device = x_template.device
        self.static_x = torch.empty_like(x_template)
        self.static_x.copy_(x_template)
        self.graph = torch.cuda.CUDAGraph()

        warmup_stream = torch.cuda.Stream(device=x_template.device)
        warmup_stream.wait_stream(torch.cuda.current_stream(x_template.device))
        with torch.cuda.stream(warmup_stream):
            for _ in range(warmup):
                out = preflop_gated_token_mixer_token_path(block, self.static_x)
        torch.cuda.current_stream(x_template.device).wait_stream(warmup_stream)
        del out
        torch.cuda.synchronize(x_template.device)

        with torch.cuda.graph(self.graph):
            self.static_out = preflop_gated_token_mixer_token_path(block, self.static_x)

    def _validate_input(self, x: torch.Tensor) -> None:
        if tuple(x.shape) != self.shape:
            raise ValueError(f"expected input shape {self.shape}, got {tuple(x.shape)}")
        if x.dtype != self.dtype:
            raise ValueError(f"expected input dtype {self.dtype}, got {x.dtype}")
        if x.device != self.device:
            raise ValueError(f"expected input device {self.device}, got {x.device}")

    def replay_static(self) -> torch.Tensor:
        self.graph.replay()
        return self.static_out

    def copy_replay(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        self.static_x.copy_(x)
        self.graph.replay()
        return self.static_out
