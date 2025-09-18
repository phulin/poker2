"""Profile a full training step with line_profiler."""

from __future__ import annotations

import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphaholdem.core.structured_config import (
    Config,
    EnvConfig,
    ModelConfig,
    StateEncoderConfig,
    TrainingConfig,
)
from alphaholdem.rl.self_play import SelfPlayTrainer


def build_profiling_config(device: torch.device) -> Config:
    """Create a lightweight config for profiling."""

    train_cfg = TrainingConfig(
        learning_rate=3e-4,
        learning_rate_final=3e-4,
        lr_schedule="constant",
        batch_size=64,
        num_epochs=1,
        replay_buffer_batches=1,
        max_trajectory_length=12,
        gamma=0.99,
        gae_lambda=0.95,
        ppo_eps=0.2,
        ppo_delta1=3.0,
        value_coef=0.05,
        entropy_coef=0.01,
        entropy_coef_final=0.01,
        entropy_decay_portion=1.0,
        grad_clip=1.0,
        use_mixed_precision=False,
        auxiliary_loss_coef=0.0,
        warmup_steps=0,
        weight_decay=0.0,
    )

    model_kwargs = {
        "d_model": 128,
        "n_layers": 2,
        "n_heads": 4,
        "num_bet_bins": 8,
        "dropout": 0.1,
        "use_auxiliary_loss": False,
        "use_gradient_checkpointing": False,
    }
    model_cfg = ModelConfig(
        name="poker_transformer_v1",
        kwargs=model_kwargs,
        policy={"name": "categorical_v1", "kwargs": {}},
        use_gradient_checkpointing=False,
        use_torch_compile=True,
    )

    env_cfg = EnvConfig(
        stack=500,
        sb=5,
        bb=10,
        bet_bins=[0.5, 0.75, 1.0, 1.5, 2.0],
        debug_step_table=False,
        flop_showdown=False,
    )

    state_encoder_cfg = StateEncoderConfig(name="transformer", kwargs={})

    cfg = Config(
        num_steps=1,
        opponent_pool_type="k_best",
        k_best_pool_size=2,
        min_elo_diff=0.0,
        min_step_diff=0,
        k_factor=1.0,
        checkpoint_interval=10_000,
        eval_interval=10_000,
        checkpoint_dir="profiling_tmp",
        device=device.type,
        use_tensor_env=True,
        num_envs=32,
        use_wandb=False,
        wandb_project=None,
        wandb_name=None,
        wandb_tags=[],
        wandb_run_id=None,
        resume_from=None,
        seed=42,
        config=None,
        economize_checkpoints=False,
        train=train_cfg,
        model=model_cfg,
        env=env_cfg,
        state_encoder=state_encoder_cfg,
    )

    return cfg


def main() -> None:
    target_device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    cfg = build_profiling_config(target_device)
    trainer = SelfPlayTrainer(cfg=cfg, device=target_device)

    # Warm up replay buffer before profiling
    trainer._fill_replay_buffer(trainer.batch_size)

    trainer.train_step(1)


if __name__ == "__main__":
    main()
