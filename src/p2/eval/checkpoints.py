"""Checkpoint -> :class:`~p2.eval.agents.SearchAgent` loading for real-hand evals.

A trained ReBeL checkpoint is only half of a playable agent: the other half is
the CFR evaluator that resolves a subgame every decision round. This module
rebuilds both from a run's *resolved* Hydra config (the same pattern used by
``scripts/evaluate_rebel_value_loss_from_resolved.py``) and hands back a
``SearchAgent`` that ``p2.eval.duplicate_match`` can seat directly.

Two things here exist specifically so evals stay comparable:

* **Search fidelity is pinned, not inherited.** Training schedules CFR
  iterations over the course of a run, so an agent evaluated mid-run would
  search at a different depth than the same agent evaluated later. Every agent
  built here gets an explicit :class:`SearchFidelity`, written into the config
  *before* the evaluator is constructed and re-asserted on the evaluator
  afterwards, and it is recorded per game by ``SearchAgent.search_fidelity``.
* **The env prototype comes from the config**, so both sides of a match play
  the same game (blinds, bet bins, stack distribution).

Checkpoints are opened read-only; nothing here writes to a run directory.
"""

from __future__ import annotations

import copy
import json
from dataclasses import MISSING, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from p2.core.structured_config import Config
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.eval.agents import AgentIdentity, SearchAgent

__all__ = [
    "SearchFidelity",
    "LoadedSearchAgent",
    "load_eval_config",
    "build_env_proto",
    "load_search_agent",
]


@dataclass(frozen=True)
class SearchFidelity:
    """Explicit CFR search settings, pinned for comparability across evals.

    The defaults are the terminal fidelity of the ``rebel-hu-context-v3``
    lineage (``search.iterations_final=300``, ``warm_start_iterations=10``,
    ``dcfr_plus_delay=80``). Lower them for cheap smoke runs, but never let an
    eval inherit a live training schedule.
    """

    cfr_iterations: int = 300
    warm_start_iterations: int = 10
    dcfr_delay: int = 80

    def to_dict(self) -> dict[str, int]:
        return {
            "cfr_iterations": int(self.cfr_iterations),
            "warm_start_iterations": int(self.warm_start_iterations),
            "dcfr_delay": int(self.dcfr_delay),
        }


@dataclass
class LoadedSearchAgent:
    """A playable agent plus the machinery it was built from.

    ``trainer`` is retained only because it owns the model and the evaluator;
    nothing here trains. ``env_proto`` is a single-env prototype matching the
    checkpoint's training env, suitable for ``play_duplicate_match``.
    """

    agent: SearchAgent
    trainer: Any
    cfg: Config
    env_proto: HUNLTensorEnv
    checkpoint_path: str
    step: int
    fidelity: SearchFidelity


# ------------------------------------------------------------------- config


def _filter_dataclass_fields(
    dataclass_type: type, container: dict[str, Any]
) -> dict[str, Any]:
    """Drop keys the current ``Config`` schema no longer has (configs age)."""
    clean: dict[str, Any] = {}
    for field_info in fields(dataclass_type):
        if field_info.name not in container:
            continue
        value = container[field_info.name]
        default_factory = getattr(field_info, "default_factory", MISSING)
        if isinstance(value, dict) and default_factory is not MISSING:
            try:
                default_value = default_factory()
            except TypeError:
                default_value = None
            if default_value is not None and is_dataclass(default_value):
                value = _filter_dataclass_fields(type(default_value), value)
        clean[field_info.name] = value
    return clean


def _config_from_container(container: dict[str, Any]) -> Config:
    return Config.from_dict(_filter_dataclass_fields(Config, container))


def load_eval_config(
    checkpoint: str | Path,
    resolved_config: str | Path | None = None,
    *,
    device: str | torch.device = "cuda",
    fidelity: SearchFidelity = SearchFidelity(),
    num_envs: int = 32,
) -> Config:
    """Build an eval-mode ``Config`` for ``checkpoint``.

    ``resolved_config`` (a run's ``resolved_config.json``) wins over the config
    embedded in the checkpoint, because the embedded copy can predate config
    schema changes made mid-run.
    """
    if resolved_config is not None:
        container = json.loads(Path(resolved_config).read_text())
    else:
        blob = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        container = blob.get("config")
        if not isinstance(container, dict):
            raise ValueError(
                f"no resolved_config given and {checkpoint} embeds no config dict"
            )
    cfg = copy.deepcopy(_config_from_container(container))

    # Eval mode: no training, no logging, no replay, no rating side-effects.
    cfg.device = str(device)
    cfg.resume_from = None
    cfg.use_wandb = False
    cfg.trueskill.enabled = False
    cfg.validation_set.enabled = False
    cfg.data.mode = "live"
    cfg.model.compile = "off"
    cfg.num_envs = int(num_envs)
    cfg.train.replay_buffer_device = "cpu"
    cfg.train.replay_buffer_batches = 1

    # Pin search fidelity. Both ends of the schedule are set so nothing can
    # interpolate back to a training value.
    cfg.search.iterations = int(fidelity.cfr_iterations)
    cfg.search.iterations_final = int(fidelity.cfr_iterations)
    cfg.search.warm_start_iterations = int(fidelity.warm_start_iterations)
    cfg.search.dcfr_plus_delay = int(fidelity.dcfr_delay)
    return cfg


def build_env_proto(
    cfg: Config, device: str | torch.device, num_envs: int = 1
) -> HUNLTensorEnv:
    """Single-env prototype matching the checkpoint's training environment."""
    return HUNLTensorEnv(
        num_envs=num_envs,
        starting_stack=cfg.env.stack,
        sb=cfg.env.sb,
        bb=cfg.env.bb,
        default_bet_bins=cfg.env.bet_bins,
        device=torch.device(device),
        float_dtype=torch.float32,
        flop_showdown=cfg.env.flop_showdown,
        randomize_stacks=cfg.env.randomize_stacks,
        stack_mode=cfg.env.stack_mode,
        min_stack_bb=cfg.env.min_stack_bb,
        mid_stack_bb=cfg.env.mid_stack_bb,
        max_stack_bb=cfg.env.max_stack_bb,
        high_stack_mass_ratio=cfg.env.high_stack_mass_ratio,
    )


# -------------------------------------------------------------------- weights


def _load_model_weights(trainer: Any, checkpoint_path: str) -> int:
    """Read-only load of ``checkpoint_path`` into ``trainer``'s model.

    Returns the checkpoint's training step. Mirrors the loader in
    ``scripts/evaluate_rebel_value_loss_from_resolved.py``, including the
    save-dtype widening and the value-only checkpoint case.
    """
    from p2.models.mlp.better_ffn import BetterSplitFFN

    checkpoint = torch.load(
        checkpoint_path, map_location=trainer.device, weights_only=False
    )
    model_state = checkpoint["model"]
    save_dtype = checkpoint.get("save_dtype")
    if save_dtype is not None and save_dtype != str(trainer.float_dtype):
        model_state = {
            key: (
                value.to(trainer.float_dtype)
                if value.dtype.is_floating_point
                else value
            )
            for key, value in model_state.items()
        }

    if checkpoint.get("model_component") == "value_model":
        if type(trainer.model) is not BetterSplitFFN:
            raise TypeError("value-only checkpoints require a BetterSplitFFN model")
        trainer.model.value_model.load_state_dict(
            model_state, strict=trainer.cfg.strict_model_loading
        )
    else:
        trainer.model.load_state_dict(
            model_state, strict=trainer.cfg.strict_model_loading
        )
    trainer._sync_inference_model()
    trainer.model.eval()
    return int(checkpoint.get("step", -1))


# --------------------------------------------------------------------- public


def load_search_agent(
    checkpoint: str | Path,
    *,
    resolved_config: str | Path | None = None,
    device: str | torch.device = "cuda",
    fidelity: SearchFidelity = SearchFidelity(),
    name: Optional[str] = None,
    num_envs: int = 32,
    cfg: Config | None = None,
) -> LoadedSearchAgent:
    """Load ``checkpoint`` into a ``SearchAgent`` ready to play real hands.

    Args:
        checkpoint: path to a ``rebel_step_*.pt`` / ``rebel_final.pt`` file.
            Opened read-only.
        resolved_config: the lineage's ``resolved_config.json``; falls back to
            the config embedded in the checkpoint.
        fidelity: pinned CFR search settings (see :class:`SearchFidelity`).
        num_envs: env batch the trainer allocates internally. Only bounds
            memory; the match player sizes its own env from ``num_pairs``.
        cfg: pre-built config, bypassing ``load_eval_config``.

    Returns:
        A :class:`LoadedSearchAgent`. Keep the whole object alive for the
        duration of a match -- the agent holds only the evaluator, but the
        evaluator's model lives on the trainer.
    """
    from p2.rl.cfr_trainer import RebelCFRTrainer
    from p2.runtime.training_run import device_from_config, setup_torch_runtime

    checkpoint = str(checkpoint)
    if cfg is None:
        cfg = load_eval_config(
            checkpoint,
            resolved_config,
            device=device,
            fidelity=fidelity,
            num_envs=num_envs,
        )

    torch_device = device_from_config(cfg)
    setup_torch_runtime(cfg, torch_device)
    trainer = RebelCFRTrainer(cfg=cfg, device=torch_device, pregeneration_only=True)
    step = _load_model_weights(trainer, checkpoint)

    evaluator = trainer.cfr_evaluator
    # Re-assert the pinned fidelity on the evaluator itself: nothing else in an
    # eval process advances the schedule, but this makes the guarantee local.
    evaluator.cfr_iterations = int(fidelity.cfr_iterations)
    evaluator.warm_start_iterations = int(fidelity.warm_start_iterations)
    if hasattr(evaluator, "dcfr_delay"):
        evaluator.dcfr_delay = int(fidelity.dcfr_delay)

    identity = AgentIdentity(
        name=name or Path(checkpoint).stem,
        kind="search",
        checkpoint=checkpoint,
        step=step,
        extra={"fidelity": fidelity.to_dict()},
    )
    agent = SearchAgent(evaluator, identity)
    return LoadedSearchAgent(
        agent=agent,
        trainer=trainer,
        cfg=cfg,
        env_proto=build_env_proto(cfg, torch_device),
        checkpoint_path=checkpoint,
        step=step,
        fidelity=fidelity,
    )
