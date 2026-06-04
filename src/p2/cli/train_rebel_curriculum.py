#!/usr/bin/env python3
"""Curriculum orchestrator for staged postflop ReBeL training."""

from __future__ import annotations

import copy
import json
import os
import shutil
import time
from typing import Any

import hydra
import torch
import wandb
from omegaconf import DictConfig

from p2.cli.train_rebel import (
    _device_from_config,
    _init_wandb,
    _log_model_parameter_summary,
)
from p2.core.structured_config import (
    Config,
    CurriculumSubstepConfig,
    ModelScope,
    StreetValueHeads,
)
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.rl.rebel_loop import (
    cleanup_old_checkpoints,
    print_rebel_training_stats,
    run_training_loop,
)
from p2.search.chance_node_helper import ChanceNodeHelper
from p2.search.end_of_street_distillation import build_end_of_street_value_batch
from p2.search.postflop_spot_sampler import sample_end_of_street_chance_roots
from p2.utils.profiling import install_triton_compile_logger_from_env


def _stage_checkpoint_dir(cfg: Config, substep_name: str) -> str:
    return os.path.join(cfg.checkpoint_dir, substep_name)


def _promote_dir(cfg: Config) -> str:
    return cfg.curriculum.promote_dir or os.path.join(cfg.checkpoint_dir, "promoted")


def _promoted_checkpoint_path(cfg: Config, net: str) -> str:
    return os.path.join(_promote_dir(cfg), f"{net}.pt")


def _state_path(cfg: Config) -> str:
    return os.path.join(_promote_dir(cfg), "curriculum_state.json")


def _save_curriculum_state(cfg: Config, promoted: dict[str, str]) -> None:
    os.makedirs(_promote_dir(cfg), exist_ok=True)
    tmp_path = f"{_state_path(cfg)}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump({"promoted": promoted}, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp_path, _state_path(cfg))


def _load_curriculum_state(cfg: Config) -> dict[str, str]:
    path = _state_path(cfg)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        state = json.load(fh)
    promoted = state.get("promoted", {})
    if not isinstance(promoted, dict):
        return {}
    return {str(key): str(value) for key, value in promoted.items()}


def _promote_checkpoint(
    cfg: Config, substep: CurriculumSubstepConfig, final_path: str
) -> str:
    os.makedirs(_promote_dir(cfg), exist_ok=True)
    promoted_path = _promoted_checkpoint_path(cfg, substep.net)
    shutil.copy2(final_path, promoted_path)
    sidecar = os.path.splitext(final_path)[0] + "_replay_buffers.pt"
    if os.path.exists(sidecar):
        promoted_sidecar = os.path.splitext(promoted_path)[0] + "_replay_buffers.pt"
        shutil.copy2(sidecar, promoted_sidecar)
    return promoted_path


def _stage_wandb_name(cfg: Config, substep_name: str) -> str | None:
    if cfg.wandb_name:
        return f"{cfg.wandb_name}-{substep_name}"
    return None


def _should_print_preflop_analyzer(net: str) -> bool:
    del net
    return False


def _apply_overrides(target: object, overrides: dict[str, Any], *, label: str) -> None:
    for key, value in overrides.items():
        if not hasattr(target, key):
            raise ValueError(f"Unknown {label} override: {key}")
        setattr(target, key, value)


def _checkpoint_metadata(
    substep_name: str, substep: CurriculumSubstepConfig
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "curriculum_substep": substep_name,
        "curriculum_kind": substep.kind,
        "curriculum_net": substep.net,
    }
    if substep.from_net is not None:
        metadata["curriculum_from_net"] = substep.from_net
    if substep.closing_net is not None:
        metadata["curriculum_closing_net"] = substep.closing_net
    if substep.chance is not None:
        metadata["curriculum_chance"] = substep.chance
    return metadata


def _read_checkpoint_metadata(path: str, device: torch.device) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    checkpoint = torch.load(path, weights_only=False, map_location=device)
    metadata = checkpoint.get("metadata", {})
    return dict(metadata) if isinstance(metadata, dict) else {}


def _stage_names(cfg: Config) -> list[str]:
    names = list(cfg.curriculum.stages)
    if not names:
        names = list(cfg.curriculum.substeps.keys())
    if not names:
        raise ValueError("curriculum.stages must list at least one sub-step")
    missing = [name for name in names if name not in cfg.curriculum.substeps]
    if missing:
        raise ValueError(f"curriculum stages missing substep configs: {missing}")
    return names


def _resume_start_index(stage_names: list[str], metadata: dict[str, Any]) -> int:
    substep = metadata.get("curriculum_substep")
    if isinstance(substep, str) and substep in stage_names:
        return stage_names.index(substep)
    return 0


def _validate_resume_checkpoint(
    resume_from: str | None,
    stage_names: list[str],
    metadata: dict[str, Any],
) -> None:
    if not resume_from:
        return
    if not os.path.exists(resume_from):
        raise FileNotFoundError(f"resume checkpoint does not exist: {resume_from}")
    substep = metadata.get("curriculum_substep")
    if not isinstance(substep, str):
        raise ValueError(
            "Curriculum resume checkpoint is missing metadata field "
            "`curriculum_substep`"
        )
    if substep not in stage_names:
        raise ValueError(
            "Curriculum resume checkpoint substep is not in curriculum.stages: "
            f"{substep!r}"
        )


def _stage_config(
    cfg: Config,
    substep_name: str,
    substep: CurriculumSubstepConfig,
    *,
    resume_from: str | None,
    promoted: dict[str, str] | None = None,
) -> Config:
    stage_cfg = copy.deepcopy(cfg)
    stage_cfg.num_steps = int(substep.num_steps)
    stage_cfg.checkpoint_dir = substep.output_dir or _stage_checkpoint_dir(
        cfg, substep_name
    )
    stage_cfg.resume_from = resume_from
    stage_cfg.wandb_name = _stage_wandb_name(cfg, substep_name)
    stage_cfg.trueskill.enabled = False
    if substep.kind == "distill":
        stage_cfg.model.street_value_heads = StreetValueHeads.pre
    _apply_overrides(
        stage_cfg.data, substep.data_overrides, label=f"{substep_name}.data"
    )
    _apply_overrides(
        stage_cfg.train, substep.train_overrides, label=f"{substep_name}.train"
    )
    _apply_overrides(
        stage_cfg.search, substep.search_overrides, label=f"{substep_name}.search"
    )
    closing_checkpoint = substep.closing_checkpoint
    if closing_checkpoint is None and promoted is not None and substep.closing_net:
        closing_checkpoint = promoted.get(substep.closing_net)
    stage_cfg.search.closing_leaf_checkpoint = closing_checkpoint
    if (
        closing_checkpoint is not None
        and "model_scope" not in substep.search_overrides
    ):
        stage_cfg.search.model_scope = ModelScope.end_of_street
    return stage_cfg


def _source_checkpoint(
    substep: CurriculumSubstepConfig,
    promoted: dict[str, str],
    resume_metadata: dict[str, Any] | None = None,
) -> str:
    if substep.checkpoint is not None:
        return substep.checkpoint
    if substep.from_net is not None and substep.from_net in promoted:
        return promoted[substep.from_net]
    if resume_metadata is not None:
        source_checkpoint = resume_metadata.get("curriculum_source_checkpoint")
        if isinstance(source_checkpoint, str) and source_checkpoint:
            return source_checkpoint
    raise ValueError(
        "Distill substep requires either `checkpoint` or a promoted `from_net`; "
        f"got net={substep.net!r}, from_net={substep.from_net!r}"
    )


def _closed_street_for_end_net(net: str) -> int:
    mapping = {"E_preflop": 0, "E_flop": 1, "E_turn": 2}
    if net not in mapping:
        raise ValueError(
            f"Distill net must be one of E_preflop, E_flop, or E_turn; got {net!r}"
        )
    return mapping[net]


def _value_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "value_model", model)


def _policy_model(model: torch.nn.Module) -> torch.nn.Module | None:
    policy_model = getattr(model, "policy_model", None)
    return policy_model if isinstance(policy_model, torch.nn.Module) else None


def _value_initialization_checkpoint(
    stage_cfg: Config,
    substep: CurriculumSubstepConfig,
) -> str | None:
    if substep.value_checkpoint is not None:
        return substep.value_checkpoint
    if substep.closing_net is not None:
        return stage_cfg.search.closing_leaf_checkpoint
    return None


def _initialize_policy_from_checkpoint(
    trainer: RebelCFRTrainer,
    checkpoint_path: str,
    *,
    substep_name: str,
) -> None:
    target_policy = _policy_model(trainer.model)
    if target_policy is None:
        raise ValueError(
            f"Curriculum train substep {substep_name} cannot initialize policy: "
            "target model has no policy_model"
        )
    source_model = trainer._load_closing_leaf_model(checkpoint_path)
    source_policy = _policy_model(source_model)
    if source_policy is None:
        raise ValueError(
            f"Curriculum train substep {substep_name} cannot initialize policy "
            f"from value-only checkpoint: {checkpoint_path}"
        )
    target_policy.load_state_dict(
        source_policy.state_dict(), strict=trainer.cfg.strict_model_loading
    )
    trainer._sync_inference_model()
    print(f"Initialized policy for train substep {substep_name} from {checkpoint_path}")


def _copy_value_state_from_source(
    target_value: torch.nn.Module,
    source_value: torch.nn.Module,
) -> dict[str, int]:
    target_state = target_value.state_dict()
    source_state = source_value.state_dict()
    load_state: dict[str, torch.Tensor] = {}
    exact = 0
    pre_to_post = 0

    def add_tensor(target_key: str, source_tensor: torch.Tensor) -> bool:
        target_tensor = target_state.get(target_key)
        if target_tensor is None or target_tensor.shape != source_tensor.shape:
            return False
        if source_tensor.dtype.is_floating_point:
            load_state[target_key] = source_tensor.to(
                device=target_tensor.device,
                dtype=target_tensor.dtype,
            )
        else:
            load_state[target_key] = source_tensor.to(device=target_tensor.device)
        return True

    for key, value in source_state.items():
        if add_tensor(key, value):
            exact += 1
        if key.startswith("pre_value_head."):
            post_key = f"post_value_head.{key.removeprefix('pre_value_head.')}"
            if add_tensor(post_key, value):
                pre_to_post += 1

    if not load_state:
        raise ValueError(
            "Value initialization checkpoint had no compatible value-model tensors"
        )
    target_value.load_state_dict(load_state, strict=False)
    return {"exact": exact, "pre_to_post": pre_to_post, "total": len(load_state)}


def _initialize_value_from_checkpoint(
    trainer: RebelCFRTrainer,
    checkpoint_path: str,
    *,
    substep_name: str,
) -> dict[str, int]:
    target_value = _value_model(trainer.model)
    source_model = trainer._load_closing_leaf_model(checkpoint_path)
    source_value = _value_model(source_model)
    loaded = _copy_value_state_from_source(target_value, source_value)
    trainer._sync_inference_model()
    print(
        f"Initialized value model for train substep {substep_name} from "
        f"{checkpoint_path} "
        f"(exact={loaded['exact']}, pre_to_post={loaded['pre_to_post']})"
    )
    return loaded


def _run_train_substep(
    cfg: Config,
    substep_name: str,
    substep: CurriculumSubstepConfig,
    *,
    device: torch.device,
    resume_from: str | None,
    promoted: dict[str, str] | None = None,
) -> str:
    if cfg.data.mode not in {"live", "hybrid"}:
        raise NotImplementedError(
            "Curriculum train substeps currently support data.mode=live or "
            "data.mode=hybrid only; "
            "use train_rebel.py with data.mode=pregenerated for bounded HP sweeps."
        )

    stage_cfg = _stage_config(
        cfg,
        substep_name,
        substep,
        resume_from=resume_from,
        promoted=promoted,
    )
    if stage_cfg.search.closing_leaf_checkpoint is None and resume_from:
        resume_metadata = _read_checkpoint_metadata(resume_from, device)
        closing_checkpoint = resume_metadata.get("curriculum_closing_checkpoint")
        if isinstance(closing_checkpoint, str) and closing_checkpoint:
            stage_cfg.search.closing_leaf_checkpoint = closing_checkpoint
            if "model_scope" not in substep.search_overrides:
                stage_cfg.search.model_scope = ModelScope.end_of_street
    os.makedirs(stage_cfg.checkpoint_dir, exist_ok=True)
    metadata = _checkpoint_metadata(substep_name, substep)
    if stage_cfg.search.closing_leaf_checkpoint is not None:
        metadata["curriculum_closing_checkpoint"] = (
            stage_cfg.search.closing_leaf_checkpoint
        )
    value_checkpoint = _value_initialization_checkpoint(stage_cfg, substep)
    if value_checkpoint is not None:
        metadata["curriculum_value_checkpoint"] = value_checkpoint

    run_cm = _init_wandb(
        stage_cfg,
        device,
        group=cfg.curriculum.wandb_group,
        name=stage_cfg.wandb_name,
    )
    with run_cm as run:
        trainer = RebelCFRTrainer(cfg=stage_cfg, device=device)
        _log_model_parameter_summary(trainer.model, run)
        if isinstance(run, wandb.Run):
            run.watch(trainer.model, log_freq=100)

        start_step = 0
        if resume_from and os.path.exists(resume_from):
            print(f"Resuming curriculum substep {substep_name}: {resume_from}")
            start_step = trainer.load_checkpoint(resume_from) + 1
            print(f"Resumed {substep_name} at step {start_step}")
        else:
            if substep.from_net is not None or substep.checkpoint is not None:
                policy_source = _source_checkpoint(substep, promoted or {})
                _initialize_policy_from_checkpoint(
                    trainer, policy_source, substep_name=substep_name
                )
            if value_checkpoint is not None:
                _initialize_value_from_checkpoint(
                    trainer, value_checkpoint, substep_name=substep_name
                )

        run_training_loop(
            trainer,
            stage_cfg,
            run,
            start_step=start_step,
            stop_step=stage_cfg.num_steps,
            stage_tag=substep_name,
            checkpoint_metadata=metadata,
            print_preflop_analyzer=_should_print_preflop_analyzer(substep.net),
        )

    final_path = os.path.join(stage_cfg.checkpoint_dir, "rebel_final.pt")
    promoted_path = _promote_checkpoint(cfg, substep, final_path)
    print(f"Promoted train substep {substep_name}: {promoted_path}")
    return promoted_path


def _run_distill_substep(
    cfg: Config,
    substep_name: str,
    substep: CurriculumSubstepConfig,
    *,
    device: torch.device,
    resume_from: str | None,
    promoted: dict[str, str],
) -> str:
    stage_cfg = _stage_config(
        cfg,
        substep_name,
        substep,
        resume_from=resume_from,
        promoted=promoted,
    )
    os.makedirs(stage_cfg.checkpoint_dir, exist_ok=True)

    resume_metadata = (
        _read_checkpoint_metadata(resume_from, device) if resume_from else {}
    )
    source_checkpoint = _source_checkpoint(
        substep,
        promoted,
        resume_metadata=resume_metadata,
    )
    closed_street = _closed_street_for_end_net(substep.net)
    chance = substep.chance or "auto"
    metadata = _checkpoint_metadata(substep_name, substep)
    metadata["curriculum_source_checkpoint"] = source_checkpoint

    run_cm = _init_wandb(
        stage_cfg,
        device,
        group=cfg.curriculum.wandb_group,
        name=stage_cfg.wandb_name,
    )
    with run_cm as run:
        trainer = RebelCFRTrainer(cfg=stage_cfg, device=device)
        _log_model_parameter_summary(trainer.model, run)
        if isinstance(run, wandb.Run):
            run.watch(trainer.model, log_freq=100)

        start_step = 0
        if resume_from and os.path.exists(resume_from):
            print(f"Resuming curriculum distill substep {substep_name}: {resume_from}")
            start_step = trainer.load_checkpoint(resume_from) + 1
            print(f"Resumed {substep_name} at step {start_step}")

        source_model = trainer._load_closing_leaf_model(source_checkpoint)
        value_model = _value_model(trainer.model)
        chance_helper = ChanceNodeHelper(
            device=device,
            float_dtype=trainer.float_dtype,
            num_players=trainer.num_players,
            model=source_model,
            generator=trainer.rng,
        )

        loop_start = time.time()
        for step in range(start_step, stage_cfg.num_steps):
            step_start = time.time()
            sample = sample_end_of_street_chance_roots(
                trainer.env,
                batch_size=stage_cfg.train.batch_size,
                closed_street=closed_street,
                generator=trainer.rng,
            )
            value_encoder = value_model.create_feature_encoder(
                env=sample.pbs.env,
                device=device,
                dtype=trainer.float_dtype,
            )
            batch = build_end_of_street_value_batch(
                sample,
                value_encoder=value_encoder,
                target_model=source_model,
                chance_helper=chance_helper,
                chance=chance,
                float_dtype=trainer.float_dtype,
                generator=trainer.rng,
            )
            metrics = trainer.train_value_batch(
                batch, step, sync_inference_model=False
            )
            step_elapsed = time.time() - step_start
            total_elapsed = time.time() - loop_start
            print_rebel_training_stats(
                metrics, step, stage_cfg.num_steps, step_elapsed, total_elapsed
            )

            if stage_cfg.use_wandb:
                metrics["step_time_s"] = step_elapsed
                run.log(metrics, step=metrics["step"])

            if (
                stage_cfg.checkpoint_interval > 0
                and (step + 1) % stage_cfg.checkpoint_interval == 0
            ):
                ckpt_path = os.path.join(
                    stage_cfg.checkpoint_dir, f"rebel_step_{step + 1}.pt"
                )
                wandb_run_id = run.id if run else None
                trainer.save_value_checkpoint(
                    ckpt_path,
                    step,
                    wandb_run_id=wandb_run_id,
                    save_optimizer=False,
                    save_dtype=torch.bfloat16,
                    metadata=metadata,
                )
                trainer.save_value_checkpoint(
                    os.path.join(stage_cfg.checkpoint_dir, "rebel_latest.pt"),
                    step,
                    wandb_run_id=wandb_run_id,
                    save_optimizer=True,
                    save_dtype=None,
                    metadata=metadata,
                )
                if stage_cfg.economize_checkpoints:
                    cleanup_old_checkpoints(stage_cfg.checkpoint_dir, ckpt_path)
                print(f"Checkpoint saved at step {step + 1} -> {ckpt_path}")

        final_path = os.path.join(stage_cfg.checkpoint_dir, "rebel_final.pt")
        trainer.save_value_checkpoint(
            final_path,
            stage_cfg.num_steps,
            save_optimizer=False,
            save_dtype=None,
            metadata=metadata,
        )

    promoted_path = _promote_checkpoint(cfg, substep, final_path)
    print(f"Promoted distill substep {substep_name}: {promoted_path}")
    return promoted_path


def train_rebel_curriculum(cfg: Config) -> None:
    if install_triton_compile_logger_from_env():
        print("Triton compile logging enabled via P2_TRITON_COMPILE_LOG=1")

    device = _device_from_config(cfg)
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    torch._dynamo.config.recompile_limit = 16
    torch.manual_seed(cfg.seed)

    stage_names = _stage_names(cfg)
    resume_metadata = _read_checkpoint_metadata(cfg.resume_from or "", device)
    _validate_resume_checkpoint(cfg.resume_from, stage_names, resume_metadata)
    start_index = _resume_start_index(stage_names, resume_metadata)
    promoted: dict[str, str] = _load_curriculum_state(cfg)

    for index, substep_name in enumerate(stage_names[start_index:], start=start_index):
        substep = cfg.curriculum.substeps[substep_name]
        resume_from = cfg.resume_from if index == start_index else None
        if substep.kind == "train":
            promoted[substep.net] = _run_train_substep(
                cfg,
                substep_name,
                substep,
                device=device,
                resume_from=resume_from,
                promoted=promoted,
            )
            _save_curriculum_state(cfg, promoted)
        elif substep.kind == "distill":
            promoted[substep.net] = _run_distill_substep(
                cfg,
                substep_name,
                substep,
                device=device,
                resume_from=resume_from,
                promoted=promoted,
            )
            _save_curriculum_state(cfg, promoted)
        else:
            raise ValueError(
                f"Unsupported curriculum substep kind for {substep_name}: "
                f"{substep.kind!r}"
            )

    print(f"Curriculum complete for implemented substeps: {promoted}")


@hydra.main(
    version_base=None, config_path="../../../conf", config_name="config_rebel_cfr"
)
def main(dict_config: DictConfig) -> None:
    config = Config.from_dict_config(dict_config)
    train_rebel_curriculum(config)


if __name__ == "__main__":
    main()
