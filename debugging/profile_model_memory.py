import argparse
import os
import sys
import time
from typing import List, Optional

import torch


def _add_src_to_path() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


_add_src_to_path()

from alphaholdem.models.transformer.poker_transformer import (  # noqa: E402
    PokerTransformerV1,
)
from alphaholdem.models.transformer.structured_embedding_data import (  # noqa: E402
    StructuredEmbeddingData,
)
from alphaholdem.models.transformer.tokens import (  # noqa: E402
    CLS_INDEX,
    GAME_INDEX,
    Special,
)


def get_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif device.type == "mps":
        try:
            torch.mps.empty_cache()  # type: ignore[attr-defined]
        except Exception:
            pass


def reset_peak_stats(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    elif device.type == "mps":
        # No official reset API; rely on deltas per run
        pass


def current_allocated(device: torch.device) -> int:
    if device.type == "cuda":
        return int(torch.cuda.memory_allocated(device))
    if device.type == "mps":
        try:
            return int(torch.mps.current_allocated_memory())  # type: ignore[attr-defined]
        except Exception:
            return 0
    return 0


def max_allocated(device: torch.device) -> int:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    if device.type == "mps":
        try:
            return int(torch.mps.driver_allocated_memory())  # type: ignore[attr-defined]
        except Exception:
            return 0
    return 0


def bytes_to_mb(x: int) -> float:
    return x / (1024.0 * 1024.0)


def build_model(
    seq_len: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    num_bet_bins: int,
    dropout: float,
    value_head_type: str,
    device: torch.device,
) -> PokerTransformerV1:
    model = PokerTransformerV1(
        max_sequence_length=seq_len,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        num_bet_bins=num_bet_bins,
        dropout=dropout,
        use_gradient_checkpointing=False,
        value_head_type=value_head_type,
        value_head_num_quantiles=51 if value_head_type == "quantile" else 1,
    )
    model.eval()
    model.to(device)
    return model


def make_dummy_batch(
    batch_size: int,
    seq_len: int,
    num_bet_bins: int,
    device: torch.device,
) -> StructuredEmbeddingData:
    data = StructuredEmbeddingData.empty(
        batch_size=batch_size,
        seq_len=seq_len,
        num_bet_bins=num_bet_bins,
        device=device,
    )
    # Populate minimal valid tokens to engage full attention
    data.token_ids[:] = Special.CLS.value
    data.token_ids[:, CLS_INDEX] = Special.CLS.value
    data.token_ids[:, GAME_INDEX] = Special.GAME.value
    data.lengths[:] = seq_len
    return data


def profile_batches(
    batch_sizes: List[int],
    seq_len: int,
    num_bet_bins: int,
    device: torch.device,
    use_autocast: bool,
    d_model: int,
    n_layers: int,
    n_heads: int,
    dropout: float,
    value_head_type: str,
) -> None:
    model = build_model(
        seq_len=seq_len,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        num_bet_bins=num_bet_bins,
        dropout=dropout,
        value_head_type=value_head_type,
        device=device,
    )

    # Warmup
    warm_bs = min(8, max(1, batch_sizes[0]))
    with torch.no_grad():
        warm = make_dummy_batch(warm_bs, seq_len, num_bet_bins, device)
        _ = model(warm)
        del warm
    empty_cache(device)

    # Pretty header
    info = model.get_model_info()
    autocast_on = use_autocast and (device.type == "cuda" or device.type == "mps")
    print(
        f"Device={device.type}  Autocast={'on' if autocast_on else 'off'}  "
        f"Model(d={info['d_model']}, L={info['n_layers']}, H={info['n_heads']})  "
        f"SeqLen={seq_len}  BetBins={num_bet_bins}"
    )
    header = (
        f"{'BS':>6} {'SEQ':>5} {'PEAK(MB)':>10} {'CURR(MB)':>10} {'ELAPSED(ms)':>12}"
    )
    rule = f"{'-'*6} {'-'*5} {'-'*10} {'-'*10} {'-'*12}"
    print(header)
    print(rule)
    for bs in batch_sizes:
        empty_cache(device)
        reset_peak_stats(device)
        start_alloc = current_allocated(device)
        start_t = time.time()

        data = make_dummy_batch(bs, seq_len, num_bet_bins, device)
        with torch.no_grad():
            if autocast_on:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    _ = model(data)
            else:
                _ = model(data)

        # Ensure kernels finish
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            try:
                torch.mps.synchronize()  # type: ignore[attr-defined]
            except Exception:
                pass

        elapsed_ms = (time.time() - start_t) * 1000.0
        peak = max_allocated(device)
        curr = current_allocated(device)
        delta_peak = max(0, peak - start_alloc)
        delta_curr = max(0, curr - start_alloc)
        print(
            f"{bs:>6} {seq_len:>5} {bytes_to_mb(delta_peak):>10.1f} {bytes_to_mb(delta_curr):>10.1f} {elapsed_ms:>12.1f}"
        )

        del data
        empty_cache(device)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Profile transformer forward peak memory across batch sizes",
    )
    p.add_argument("--device", type=str, default=None, help="cuda|mps|cpu")
    p.add_argument("--seq_len", type=int, default=47)
    p.add_argument("--num_bet_bins", type=int, default=8)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument(
        "--value_head_type",
        type=str,
        default="scalar",
        choices=["scalar", "quantile"],
    )
    p.add_argument(
        "--batch_sizes",
        type=str,
        default="1024,2048,4096,8192,16384,32768",
        help="Comma-separated batch sizes",
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--autocast",
        dest="autocast",
        action="store_true",
        help="Use fp16 autocast on CUDA (default)",
    )
    group.add_argument(
        "--no-autocast",
        dest="autocast",
        action="store_false",
        help="Disable autocast",
    )
    p.set_defaults(autocast=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    profile_batches(
        batch_sizes=batch_sizes,
        seq_len=args.seq_len,
        num_bet_bins=args.num_bet_bins,
        device=device,
        use_autocast=args.autocast,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        value_head_type=args.value_head_type,
    )


if __name__ == "__main__":
    main()
