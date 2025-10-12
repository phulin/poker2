import os

import torch
from omegaconf import OmegaConf

from alphaholdem.core.structured_config import (
    Config,
    EnvConfig,
    ExploiterConfig,
    ModelConfig,
    TrainingConfig,
)
from alphaholdem.rl.self_play import SelfPlayTrainer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_config(config_path: str) -> Config:
    hydra_cfg = OmegaConf.load(config_path)
    cfg_dict = OmegaConf.to_container(hydra_cfg, resolve=True)

    train_config = TrainingConfig(**cfg_dict.get("train", {}))
    model_config = ModelConfig(**cfg_dict.get("model", {}))
    env_config = EnvConfig(**cfg_dict.get("env", {}))
    exploiter_config = ExploiterConfig(**cfg_dict.get("exploiter", {}))

    return Config(
        train=train_config,
        model=model_config,
        env=env_config,
        exploiter=exploiter_config,
        **{
            k: v
            for k, v in cfg_dict.items()
            if k not in ["train", "model", "env", "state_encoder", "exploiter"]
        },
    )


def test_winner_reward_sign_and_replay_buffer_alignment():
    cfg = _load_config(os.path.join(PROJECT_ROOT, "conf", "config_transformer.yaml"))

    # Debug overrides suitable for CI
    cfg.num_envs = 100
    cfg.use_wandb = False
    cfg.use_tensor_env = True
    cfg.device = "cpu"
    cfg.train.use_mixed_precision = False
    cfg.train.use_kv_cache = False
    cfg.exploiter.enabled = False
    cfg.train.max_trajectory_length = max(8, int(cfg.train.max_trajectory_length))

    torch.manual_seed(cfg.seed)
    device = torch.device("cpu")

    trainer = SelfPlayTrainer(cfg=cfg, device=device)

    # Collect one completed round of trajectories; store into replay buffer
    per_episode_rewards = trainer.collect_tensor_trajectories(
        min_trajectories=1, add_to_replay_buffer=True
    )

    winners = trainer.tensor_env.winner
    rewards = per_episode_rewards

    EPS = 1e-6

    # Verify env-level winner sign (for episodes we actually recorded)
    for i in range(int(rewards.numel())):
        w = int(winners[i].item())
        r = float(rewards[i].item())
        if w == 0:
            assert r > 0, f"idx={i} expected positive reward for winner=0, got {r}"
        elif w == 1:
            assert r < 0, f"idx={i} expected negative reward for winner=1, got {r}"
        else:  # tie
            assert abs(r) < EPS, f"idx={i} expected ~0 reward for tie, got {r}"

    # Validate replay buffer final-step rewards match env rewards
    buf = trainer.replay_buffer
    valid_idxs = torch.where(buf.trajectory_lengths > 0)[0]

    # Take the last episode_count_raw valid trajectories (most recent)
    traj_lengths = buf.trajectory_lengths[valid_idxs]
    last_pos = traj_lengths - 1
    final_rewards = buf.rewards[valid_idxs, last_pos].cpu()

    # Filter out opp SB folds (+0.005) from both env rewards and buffer rewards in aligned order
    sb_mask = (rewards - 0.005).abs() > EPS
    winners_f = winners[sb_mask]
    rewards_f = rewards[sb_mask]
    final_rewards_f = final_rewards

    assert (
        rewards_f.numel() == final_rewards_f.numel()
    ), "Filtered env rewards and buffer rewards length mismatch"

    for i in range(int(rewards_f.numel())):
        w = int(winners_f[i].item())
        r_env = float(rewards_f[i].item())
        r_buf = float(final_rewards_f[i].item())
        # Rewards must match closely
        assert abs(r_env - r_buf) < EPS, f"idx={i} env={r_env} buf={r_buf}"
        # Winner sign must match
        if w == 0:
            assert (
                r_buf > 0
            ), f"idx={i} expected positive buf reward for winner=0, got {r_buf}"
        elif w == 1:
            assert (
                r_buf < 0
            ), f"idx={i} expected negative buf reward for winner=1, got {r_buf}"
        else:
            assert (
                abs(r_buf) < EPS
            ), f"idx={i} expected ~0 buf reward for tie, got {r_buf}"
