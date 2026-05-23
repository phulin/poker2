"""
Optional profiling utilities.

This module provides optional profiling decorators that gracefully degrade
when line_profiler is not available.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Any

try:
    from line_profiler import profile

    PROFILE_AVAILABLE = True
except ImportError:
    # Create a no-op decorator when line_profiler is not available
    def profile(func):
        """No-op decorator when line_profiler is not available."""
        return func

    PROFILE_AVAILABLE = False

_TRITON_COMPILE_LOGGER_INSTALLED = False


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _short_hash(value: Any) -> str:
    return hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:10]


def _format_triton_constants(constants: dict[Any, Any], *, max_items: int = 12) -> str:
    if not constants:
        return "{}"
    items = list(constants.items())
    shown = ", ".join(f"{key}={value!r}" for key, value in items[:max_items])
    if len(items) > max_items:
        shown += f", ... +{len(items) - max_items}"
    return "{" + shown + "}"


def _triton_fn_name(fn: Any) -> str:
    module = getattr(fn, "module", "")
    module_name = getattr(module, "__name__", module)
    name = getattr(fn, "name", "<unknown>")
    return f"{module_name}.{name}" if module_name else str(name)


def install_triton_compile_logger_from_env() -> bool:
    """Install Triton JIT compile logging when ``P2_TRITON_COMPILE_LOG=1``.

    The hook reports in-process JIT cache misses and whether the compiler
    loaded artifacts from Triton's disk cache or ran the lowering pipeline.
    """
    global _TRITON_COMPILE_LOGGER_INSTALLED
    if _TRITON_COMPILE_LOGGER_INSTALLED or not _env_flag("P2_TRITON_COMPILE_LOG"):
        return _TRITON_COMPILE_LOGGER_INSTALLED

    try:
        from triton import knobs
    except ImportError:
        return False

    show_constants = not _env_flag("P2_TRITON_COMPILE_LOG_HIDE_CONSTS")
    pending: dict[Any, float] = {}
    previous_pre_hook = knobs.runtime.jit_cache_hook
    previous_post_hook = knobs.runtime.jit_post_compile_hook
    previous_listener = knobs.compilation.listener

    def pre_hook(**kwargs):
        if previous_pre_hook is not None:
            result = previous_pre_hook(**kwargs)
            if result:
                return result
        key = kwargs["key"]
        pending[key] = time.perf_counter()
        compile_info = kwargs["compile"]
        fn = kwargs["fn"]
        constants = compile_info["constants"]
        consts = (
            f" consts={_format_triton_constants(constants)}"
            if show_constants
            else ""
        )
        print(
            "[triton jit miss] "
            f"{_triton_fn_name(fn)} "
            f"key={_short_hash(key)} "
            f"warmup={compile_info['is_warmup']} "
            f"device={compile_info['device']}"
            f"{consts}",
            flush=True,
        )
        return None

    def post_hook(**kwargs) -> None:
        if previous_post_hook is not None:
            previous_post_hook(**kwargs)
        key = kwargs["key"]
        start = pending.pop(key, None)
        elapsed_ms = (time.perf_counter() - start) * 1000 if start is not None else -1.0
        print(
            "[triton jit ready] "
            f"{kwargs['fn'].name} key={_short_hash(key)} elapsed_ms={elapsed_ms:.1f}",
            flush=True,
        )

    def compiler_listener(*, src, metadata, metadata_group, times, cache_hit) -> None:
        if previous_listener is not None:
            previous_listener(
                src=src,
                metadata=metadata,
                metadata_group=metadata_group,
                times=times,
                cache_hit=cache_hit,
            )
        print(
            "[triton compiler] "
            f"{src.name} disk_cache_hit={cache_hit} total_us={times.total}",
            flush=True,
        )

    knobs.runtime.jit_cache_hook = pre_hook
    knobs.runtime.jit_post_compile_hook = post_hook
    knobs.compilation.listener = compiler_listener
    _TRITON_COMPILE_LOGGER_INSTALLED = True
    return True


# Export the profile decorator (either real or no-op)
__all__ = [
    "PROFILE_AVAILABLE",
    "install_triton_compile_logger_from_env",
    "profile",
]
