from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from p2.core.structured_config import Config, ModelType, NonlinearityType
from p2.env.card_utils import NUM_HANDS


DEFAULT_FORCE_DECK = [12, 25, 38, 51, 0, 13, 26, 1, 14]


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _action_labels(bet_bins: list[float]) -> list[str]:
    labels = ["fold", "check_call"]
    labels.extend(f"bet_{v:g}x_pot" for v in bet_bins)
    labels.append("all_in")
    return labels


def _normalize_export_state(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    clean_state = {
        name.removeprefix("_orig_mod."): tensor for name, tensor in state.items()
    }
    if not any(name.startswith("policy_model.") for name in clean_state):
        return clean_state

    export_state: dict[str, torch.Tensor] = {}
    for name, tensor in clean_state.items():
        if name.startswith("policy_model."):
            export_state[name] = tensor
        elif name.startswith("value_model.pre_value_head."):
            suffix = name.removeprefix("value_model.pre_value_head.")
            export_state[f"hand_value_head.{suffix}"] = tensor
        elif name.startswith("value_model.post_value_head."):
            continue
        elif name.startswith("value_model."):
            export_state[name.removeprefix("value_model.")] = tensor
    return export_state


def _load_checkpoint(snapshot: Path) -> tuple[Config, dict[str, torch.Tensor], int | None]:
    checkpoint = torch.load(snapshot, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(checkpoint["config"])
    raw_state = checkpoint["model"]
    state = _normalize_export_state(raw_state)
    if (
        cfg.model.name != ModelType.better_ffn
        and "street_embedding.weight" not in state
    ):
        raise ValueError(f"{snapshot} is not a BetterFFN checkpoint")
    cfg.model.name = ModelType.better_ffn
    cfg.model.num_actions = len(cfg.env.bet_bins) + 3
    step = checkpoint.get("step")
    if step is not None:
        step = int(step)
    return cfg, state, step


def _validate_supported(cfg: Config) -> None:
    if cfg.model.name != ModelType.better_ffn:
        raise ValueError(f"unsupported model type {cfg.model.name}")
    if _enum_value(cfg.model.nonlinearity) != NonlinearityType.leaky_relu.value:
        raise ValueError(
            "webgpu_cfr currently supports only BetterFFN checkpoints with leaky_relu"
        )
    if not cfg.model.shared_trunk:
        raise ValueError("webgpu_cfr currently requires shared_trunk=True")
    if cfg.model.num_actions != len(cfg.env.bet_bins) + 3:
        raise ValueError("num_actions must equal len(env.bet_bins) + 3")


def _tensor_bytes(tensor: torch.Tensor, storage_dtype: str) -> bytes:
    if not tensor.dtype.is_floating_point:
        raise ValueError(f"cannot export non-floating tensor with dtype {tensor.dtype}")
    if storage_dtype == "float16":
        arr = tensor.detach().cpu().to(torch.float16).contiguous().numpy()
        if arr.dtype.byteorder not in ("<", "="):
            arr = arr.astype("<f2", copy=False)
    elif storage_dtype == "float32":
        arr = tensor.detach().cpu().to(torch.float32).contiguous().numpy()
        if arr.dtype.byteorder not in ("<", "="):
            arr = arr.astype("<f4", copy=False)
    else:
        raise ValueError(f"unsupported storage dtype {storage_dtype!r}")
    return arr.tobytes(order="C")


def _scheduled_cfr_config(cfg: Config, step: int | None) -> dict[str, Any]:
    search = cfg.search
    if search.iterations_final is not None:
        if step is None:
            progress = 1.0
        else:
            progress = max(0.0, min(1.0, float(step) / max(1.0, float(cfg.num_steps))))
        iterations = int(
            round(
                search.iterations
                + (search.iterations_final - search.iterations) * progress
            )
        )
    else:
        progress = None
        iterations = int(search.iterations)

    warm_start_iterations = max(1, iterations // 20)
    dcfr_plus_delay = int(round(iterations * 0.4))
    iterations = max(warm_start_iterations + 1, iterations)

    return {
        "source": "checkpoint.config.search",
        "enabled": bool(search.enabled),
        "depth": int(search.depth),
        "iterations": iterations,
        "iterationsStart": int(search.iterations),
        "iterationsFinal": search.iterations_final,
        "scheduleProgress": progress,
        "warmStartIterations": warm_start_iterations,
        "warmStartType": _enum_value(search.warm_start_type),
        "warmStartMultiplier": float(search.warm_start_multiplier),
        "branching": int(search.branching),
        "beliefSamples": int(search.belief_samples),
        "sampleEpsilon": float(search.sample_epsilon),
        "dcfrAlpha": float(search.dcfr_alpha),
        "dcfrAlphaFinal": search.dcfr_alpha_final,
        "dcfrBeta": float(search.dcfr_beta),
        "dcfrBetaFinal": search.dcfr_beta_final,
        "dcfrGamma": float(search.dcfr_gamma),
        "dcfrGammaFinal": search.dcfr_gamma_final,
        "dcfrPlusDelay": dcfr_plus_delay,
        "dcfrPlusDelayConfigured": int(search.dcfr_plus_delay),
        "includeAveragePolicy": bool(search.include_average_policy),
        "cfrType": _enum_value(search.cfr_type),
        "cfrPlus": bool(search.cfr_plus),
        "cfrAvg": bool(search.cfr_avg),
        "sparse": bool(search.sparse),
        "sparseFused": bool(search.sparse_fused),
        "valueTargetsFromFinalPolicy": bool(search.value_targets_from_final_policy),
    }


def export_model(
    snapshot: Path,
    out: Path,
    weights_name: str = "weights.bin.gz",
    storage_dtype: str = "float16",
    compression: str = "gzip",
    allin_manifest: Path | None = None,
) -> dict[str, Any]:
    cfg, state, step = _load_checkpoint(snapshot)
    _validate_supported(cfg)

    out.mkdir(parents=True, exist_ok=True)
    weights_path = out / weights_name
    tensors: list[dict[str, Any]] = []
    offset = 0
    weights_hash = hashlib.sha256()
    raw_weights = bytearray()

    for name, tensor in state.items():
        clean_name = name.removeprefix("_orig_mod.")
        data = _tensor_bytes(tensor, storage_dtype)
        tensor_hash = hashlib.sha256(data).hexdigest()
        raw_weights.extend(data)
        weights_hash.update(data)
        tensors.append(
            {
                "name": clean_name,
                "shape": list(tensor.shape),
                "dtype": storage_dtype,
                "byteOffset": offset,
                "byteLength": len(data),
                "sha256": tensor_hash,
            }
        )
        offset += len(data)

    if compression == "gzip":
        payload = gzip.compress(bytes(raw_weights), compresslevel=9, mtime=0)
        compression_manifest: dict[str, Any] | None = {
            "format": "gzip",
            "byteLength": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    elif compression == "none":
        payload = bytes(raw_weights)
        compression_manifest = None
    else:
        raise ValueError(f"unsupported compression {compression!r}")

    with weights_path.open("wb") as fh:
        fh.write(payload)

    split_policy_value = any(name.startswith("policy_model.") for name in state)
    context_dim = int(state["context_encoder.norm.weight"].shape[0])
    manifest = {
        "schemaVersion": 1,
        "format": "p2.better_ffn.webgpu",
        "source": {
            "snapshot": str(snapshot),
            "exporter": "webgpu_cfr.python.export_model",
        },
        "architecture": {
            "numHands": NUM_HANDS,
            "numPlayers": 2,
            "numActions": cfg.model.num_actions,
            "hiddenDim": cfg.model.hidden_dim,
            "rangeHiddenDim": cfg.model.range_hidden_dim,
            "boardInteractionDim": cfg.model.board_interaction_dim,
            "ffnDim": cfg.model.ffn_dim,
            "numHiddenLayers": cfg.model.num_hidden_layers,
            "numPolicyLayers": cfg.model.num_policy_layers,
            "numValueLayers": cfg.model.num_value_layers,
            "sharedTrunk": bool(cfg.model.shared_trunk),
            "enforceZeroSum": bool(cfg.model.enforce_zero_sum),
            "nonlinearity": _enum_value(cfg.model.nonlinearity),
            "normalization": "rmsnorm",
            "contextDim": context_dim,
            "policyRank": int(getattr(cfg.model, "policy_rank", 64)),
            "policyHandBiasRank": int(
                getattr(cfg.model, "policy_hand_bias_rank", 32)
            ),
            "splitPolicyValue": split_policy_value,
        },
        "env": {
            "stack": cfg.env.stack,
            "sb": cfg.env.sb,
            "bb": cfg.env.bb,
            "betBins": list(cfg.env.bet_bins),
            "flopShowdown": bool(cfg.env.flop_showdown),
            "maxStackBb": cfg.env.max_stack_bb,
            "defaultButton": 1,
            "defaultForceDeck": DEFAULT_FORCE_DECK,
        },
        "cfr": _scheduled_cfr_config(cfg, step),
        "actionLabels": _action_labels(list(cfg.env.bet_bins)),
        "tensors": tensors,
        "weights": {
            "file": weights_name,
            "byteLength": offset,
            "sha256": weights_hash.hexdigest(),
            "storageDtype": storage_dtype,
        },
    }
    if compression_manifest is not None:
        manifest["weights"]["compression"] = compression_manifest
    if allin_manifest is not None:
        manifest["allIn"] = json.loads(allin_manifest.read_text())
    (out / "model.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weights-name", default="weights.bin.gz")
    parser.add_argument(
        "--storage-dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    parser.add_argument("--compression", choices=("gzip", "none"), default="gzip")
    parser.add_argument(
        "--allin-manifest",
        type=Path,
        default=None,
        help="Optional all-in asset manifest JSON to embed as model.json allIn.",
    )
    args = parser.parse_args()

    manifest = export_model(
        args.snapshot,
        args.out,
        args.weights_name,
        args.storage_dtype,
        args.compression,
        args.allin_manifest,
    )
    print(
        json.dumps(
            {
                "manifest": str(args.out / "model.json"),
                "weights": str(args.out / manifest["weights"]["file"]),
                "byteLength": manifest["weights"]["byteLength"],
                "tensors": len(manifest["tensors"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
