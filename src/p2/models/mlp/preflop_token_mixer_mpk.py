from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any

import torch


@dataclass(frozen=True)
class PreflopTokenMixerMPKConfig:
    """Launch-time shape and scheduling knobs for the MPK prototype.

    The custom MPK task is the staged counterpart to
    ``_preflop_token_mixer_gate_residual_persistent_triton``: callers compute
    RMSNorm and the gate projection with PyTorch/cuBLAS, then MPK runs the
    7 -> 28 -> 7 token mixer, sigmoid gate, and residual add.
    """

    batch_size: int
    dim: int
    dtype: torch.dtype = torch.bfloat16
    device: torch.device | str = "cuda"
    block_b: int = 8
    block_d: int = 32
    block_dim: tuple[int, int, int] = (128, 1, 1)
    num_workers: int | None = None
    num_local_schedulers: int | None = None
    num_remote_schedulers: int = 0
    trace_name: str = "preflop_token_mixer_gate_residual"
    use_cutlass_kernel: bool = False
    custom_task_name: str = "preflop_token_mixer_gate_residual"


def mirage_mpk_is_available() -> bool:
    return importlib.util.find_spec("mirage") is not None


def _require_mirage() -> ModuleType:
    try:
        mi = importlib.import_module("mirage")
    except ImportError as exc:
        raise RuntimeError(
            "Mirage/MPK is not installed. Install a Mirage checkout with MPK "
            "support before constructing PreflopTokenMixerGateResidualMPK."
        ) from exc
    if not hasattr(mi, "PersistentKernel"):
        raise RuntimeError(
            "The installed mirage package does not expose PersistentKernel; "
            "use the MPK branch/build of Mirage."
        )
    return mi


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _mirage_dtype(mi: ModuleType, dtype: torch.dtype) -> Any:
    if dtype == torch.bfloat16:
        return mi.bfloat16
    if dtype == torch.float16:
        return mi.float16
    if dtype == torch.float32:
        return mi.float32
    raise ValueError(f"unsupported MPK dtype {dtype}")


def _normalized_config(
    config: PreflopTokenMixerMPKConfig,
) -> PreflopTokenMixerMPKConfig:
    device = torch.device(config.device)
    if device.type != "cuda":
        raise ValueError("MPK preflop token mixer requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("MPK preflop token mixer requires CUDA")
    if device.index is None:
        device = torch.device(device.type, torch.cuda.current_device())
    if config.batch_size <= 0 or config.dim <= 0:
        raise ValueError("batch_size and dim must be positive")
    if config.block_b <= 0 or config.block_d <= 0:
        raise ValueError("block_b and block_d must be positive")
    if any(axis <= 0 for axis in config.block_dim):
        raise ValueError("block_dim axes must be positive")
    return replace(config, device=device)


def _validate_runtime_tensors(
    x: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor,
    w_in: torch.Tensor,
    w_out: torch.Tensor,
    config: PreflopTokenMixerMPKConfig,
) -> None:
    expected = (config.batch_size, 7, config.dim)
    device = torch.device(config.device)
    if x.shape != expected or y.shape != expected or gate.shape != expected:
        raise ValueError(f"x, y, and gate must all have shape {expected}")
    if w_in.shape != (28, 7) or w_out.shape != (7, 28):
        raise ValueError("token mixer weights must have shapes (28, 7) and (7, 28)")
    tensors = (x, y, gate, w_in, w_out)
    if any(tensor.device != device for tensor in tensors):
        raise ValueError("all MPK token mixer tensors must be on the configured device")
    if any(tensor.dtype != config.dtype for tensor in tensors):
        raise ValueError("all MPK token mixer tensors must use the configured dtype")


def _worker_layout(
    config: PreflopTokenMixerMPKConfig,
) -> tuple[int, int, int]:
    local_schedulers = (
        4 if config.num_local_schedulers is None else int(config.num_local_schedulers)
    )
    remote_schedulers = int(config.num_remote_schedulers)
    if local_schedulers < 0 or remote_schedulers < 0:
        raise ValueError("scheduler counts must be non-negative")
    total_schedulers = local_schedulers + remote_schedulers
    if total_schedulers % 4 != 0:
        raise ValueError("MPK scheduler counts must sum to a multiple of four")

    scheduler_sms = total_schedulers // 4
    if config.num_workers is not None:
        workers = int(config.num_workers)
    else:
        device = torch.device(config.device)
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        workers = max(1, sms - scheduler_sms)
    if workers <= 0:
        raise ValueError("num_workers must be positive")
    return workers, local_schedulers, remote_schedulers


def _make_persistent_kernel(
    mi: ModuleType,
    config: PreflopTokenMixerMPKConfig,
) -> Any:
    workers, local_schedulers, remote_schedulers = _worker_layout(config)
    params = mi.PersistentKernel.get_default_init_parameters()
    params.update(
        {
            "mode": "offline",
            "world_size": 1,
            "mpi_rank": 0,
            "num_workers": workers,
            "num_local_schedulers": local_schedulers,
            "num_remote_schedulers": remote_schedulers,
            "max_seq_length": 1,
            "max_num_batched_requests": 1,
            "max_num_batched_tokens": 1,
            "max_num_pages": 1,
            "page_size": 1,
            "meta_tensors": {},
            "profiler_tensor": None,
            "trace_name": config.trace_name,
            "spec_decode_config": None,
            "use_cutlass_kernel": config.use_cutlass_kernel,
            "test_mode": True,
        }
    )
    return mi.PersistentKernel(**params)


def _register_gate_residual_custom_task(
    mi: ModuleType,
    mpk: Any,
    x_dtensor: Any,
    y_dtensor: Any,
    gate_dtensor: Any,
    w_in_dtensor: Any,
    w_out_dtensor: Any,
    out_dtensor: Any,
    config: PreflopTokenMixerMPKConfig,
) -> None:
    batch_tiles = _ceil_div(config.batch_size, config.block_b)
    dim_tiles = _ceil_div(config.dim, config.block_d)
    grid_dim = (batch_tiles, dim_tiles, 1)

    tb_graph = mi.new_threadblock_graph(
        grid_dim=grid_dim,
        block_dim=config.block_dim,
        forloop_range=1,
        reduction_dimx=64,
    )
    tb_graph.new_input(x_dtensor, (0, 2, -1), -1, True)
    tb_graph.new_input(y_dtensor, (0, 2, -1), -1, True)
    tb_graph.new_input(gate_dtensor, (0, 2, -1), -1, True)
    tb_graph.new_input(w_in_dtensor, (-1, -1, -1), -1, True)
    tb_graph.new_input(w_out_dtensor, (-1, -1, -1), -1, True)
    tb_graph.new_input(out_dtensor, (0, 2, -1), -1, True)
    mpk.kn_graph.customized(
        [x_dtensor, y_dtensor, gate_dtensor, w_in_dtensor, w_out_dtensor, out_dtensor],
        tb_graph,
    )
    mpk.kn_graph.register_task(
        tb_graph,
        config.custom_task_name,
        [config.batch_size, config.dim, config.block_b, config.block_d],
    )


class PreflopTokenMixerGateResidualMPK:
    """Experimental MPK wrapper for the staged gated token mixer.

    This wrapper is intentionally not wired into model ``forward``. The current
    upstream MPK docs expose graph/task registration but do not include a built-in
    task for the LeakyReLU token mixer plus sigmoid-gated residual, so the Mirage
    checkout must provide a custom C++ MPK task named by ``custom_task_name``.
    """

    def __init__(
        self,
        w_in: torch.Tensor,
        w_out: torch.Tensor,
        config: PreflopTokenMixerMPKConfig,
        *,
        compile_now: bool = True,
        output_dir: str | Path | None = None,
    ) -> None:
        config = _normalized_config(config)
        self.config = config
        self._mi = _require_mirage()
        self._compiled = False

        self.x_buffer = torch.empty(
            (config.batch_size, 7, config.dim),
            device=config.device,
            dtype=config.dtype,
        )
        self.y_buffer = torch.empty_like(self.x_buffer)
        self.gate_buffer = torch.empty_like(self.x_buffer)
        self.out_buffer = torch.empty_like(self.x_buffer)
        self.w_in = w_in.detach().contiguous()
        self.w_out = w_out.detach().contiguous()
        _validate_runtime_tensors(
            self.x_buffer,
            self.y_buffer,
            self.gate_buffer,
            self.w_in,
            self.w_out,
            config,
        )

        self.mpk = _make_persistent_kernel(self._mi, config)
        dtype = _mirage_dtype(self._mi, config.dtype)
        x_dtensor = self.mpk.attach_input(self.x_buffer, "preflop_mixer_x")
        y_dtensor = self.mpk.attach_input(self.y_buffer, "preflop_mixer_y")
        gate_dtensor = self.mpk.attach_input(self.gate_buffer, "preflop_mixer_gate")
        w_in_dtensor = self.mpk.attach_input(self.w_in, "preflop_mixer_w_in")
        w_out_dtensor = self.mpk.attach_input(self.w_out, "preflop_mixer_w_out")
        out_dtensor = self.mpk.attach_input(self.out_buffer, "preflop_mixer_out")
        for dtensor in (
            x_dtensor,
            y_dtensor,
            gate_dtensor,
            w_in_dtensor,
            w_out_dtensor,
            out_dtensor,
        ):
            if dtensor.dtype != dtype:
                raise RuntimeError("Mirage attached an unexpected dtype")

        _register_gate_residual_custom_task(
            self._mi,
            self.mpk,
            x_dtensor,
            y_dtensor,
            gate_dtensor,
            w_in_dtensor,
            w_out_dtensor,
            out_dtensor,
            config,
        )
        if compile_now:
            self.compile(output_dir=output_dir)

    def compile(self, *, output_dir: str | Path | None = None) -> None:
        if self._compiled:
            return
        if output_dir is None:
            self.mpk.compile()
        else:
            self.mpk.compile(output_dir=str(output_dir))
        self._compiled = True

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        if not self._compiled:
            self.compile()
        _validate_runtime_tensors(x, y, gate, self.w_in, self.w_out, self.config)
        self.x_buffer.copy_(x)
        self.y_buffer.copy_(y)
        self.gate_buffer.copy_(gate)
        self.mpk(default_stream=torch.cuda.current_stream(self.x_buffer.device))
        return self.out_buffer

    def close(self) -> None:
        finalize = getattr(self.mpk, "finalize", None)
        if finalize is not None:
            finalize()
