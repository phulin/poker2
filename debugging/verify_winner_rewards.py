#!/usr/bin/env python3
import os
import sys
from typing import Optional

import torch
from omegaconf import OmegaConf

from alphaholdem.core.structured_config import (
    Config,
    EnvConfig,
    ExploiterConfig,
    ModelConfig,
    StateEncoderConfig,
    TrainingConfig,
)
from alphaholdem.rl.self_play import SelfPlayTrainer

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_config(config_path: str) -> Config:
    hydra_cfg = OmegaConf.load(config_path)
    cfg_dict = OmegaConf.to_container(hydra_cfg, resolve=True)

    train_config = TrainingConfig(**cfg_dict.get("train", {}))
    model_config = ModelConfig(**cfg_dict.get("model", {}))
    env_config = EnvConfig(**cfg_dict.get("env", {}))
    state_encoder_config = StateEncoderConfig(**cfg_dict.get("state_encoder", {}))
    exploiter_config = ExploiterConfig(**cfg_dict.get("exploiter", {}))

    return Config(
        train=train_config,
        model=model_config,
        env=env_config,
        state_encoder=state_encoder_config,
        exploiter=exploiter_config,
        **{
            k: v
            for k, v in cfg_dict.items()
            if k not in ["train", "model", "env", "state_encoder", "exploiter"]
        },
    )


def main(
    config_path: str = os.path.join(PROJECT_ROOT, "conf", "config_transformer.yaml"),
    device_str: Optional[str] = None,
):
    cfg = load_config(config_path)

    # Device
    if device_str is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    else:
        device = torch.device(device_str)

    # Safe overrides for this debugging scenario
    cfg.num_envs = 100
    cfg.use_wandb = False
    cfg.use_tensor_env = True
    cfg.device = device.type
    # Disable features that aren't helpful for a quick debug run
    cfg.train.use_mixed_precision = False
    cfg.train.use_kv_cache = False
    cfg.exploiter.enabled = False
    # Keep trajectories short-ish
    cfg.train.max_trajectory_length = max(8, int(cfg.train.max_trajectory_length))

    # Seed for reproducibility
    torch.manual_seed(cfg.seed)

    trainer = SelfPlayTrainer(cfg=cfg, device=device, rng_seed=cfg.seed)

    # Collect exactly one trajectory per environment and add to replay buffer
    per_episode_rewards = trainer.collect_tensor_trajectories(
        min_trajectories=1,
        add_to_replay_buffer=True,
    )

    winners_all = trainer.tensor_env.winner.clone().cpu()
    rewards_all = per_episode_rewards.clone().cpu()

    # There may be zero-length (no-step) trajectories; per_episode_rewards only includes completed ones with our action
    # Filter to the first len(per_episode_rewards) winners, matched in collection order
    episode_count_raw = int(rewards_all.numel())
    winners = winners_all[:episode_count_raw]
    rewards = rewards_all

    episode_count = int(rewards.numel())

    # Verify sign matches winner label (env-level, for those with rewards emitted and not SB folds)
    EPS = 1e-6
    env_mismatches = []
    for i in range(episode_count):
        w = int(winners[i].item())
        r = float(rewards[i].item())
        if w == 0 and not (r > 0):
            env_mismatches.append((i, w, r))
        elif w == 1 and not (r < 0):
            env_mismatches.append((i, w, r))
        elif w == 2 and not (abs(r) < EPS):
            env_mismatches.append((i, w, r))

    # Verify replay buffer final-step rewards match winners and per_episode_rewards
    buf = trainer.replay_buffer
    # Find valid (length > 0) trajectories among the most recent window
    valid_idxs = torch.where(buf.trajectory_lengths > 0)[0]

    traj_lengths = buf.trajectory_lengths[valid_idxs]
    last_pos = traj_lengths - 1
    final_rewards_buffer = buf.rewards[valid_idxs, last_pos].cpu()

    # Remove cases where opponent folded on SB (assumed env reward +0.005)
    sb_fold_mask = (rewards - 0.005).abs() > EPS
    winners_without_sb_fold = winners[sb_fold_mask]
    rewards_without_sb_fold = rewards[sb_fold_mask]

    assert len(rewards_without_sb_fold) == len(final_rewards_buffer)

    # Compare signs against winners and per_episode_rewards (aligned by filtered order)
    buf_mismatches = []
    for i in range(len(rewards_without_sb_fold)):
        w = int(winners_without_sb_fold[i].item())
        r_env = float(rewards_without_sb_fold[i].item())
        r_buf = float(final_rewards_buffer[i].item())
        # Per-episode reward and buffer final reward should match closely
        if abs(r_env - r_buf) > EPS:
            buf_mismatches.append((i, w, r_env, r_buf))
        # Check winner sign vs buffer reward
        if w == 0 and not (r_buf > 0):
            buf_mismatches.append((i, w, r_env, r_buf))
        elif w == 1 and not (r_buf < 0):
            buf_mismatches.append((i, w, r_env, r_buf))
        elif w == 2 and not (abs(r_buf) < EPS):
            buf_mismatches.append((i, w, r_env, r_buf))

    total_reward = float(rewards_without_sb_fold.sum().item())
    print(
        f"Collected {episode_count} completed episodes (after filtering SB folds); total_reward={total_reward:.4f}"
    )
    if env_mismatches:
        print(
            f"Env winner/reward mismatches: {len(env_mismatches)} (showing up to 10):"
        )
        for i, w, r in env_mismatches[:10]:
            print(f"  idx={i} winner={w} reward={r:.6f}")
    else:
        print("All env winners match reward signs ✅")

    if buf_mismatches:
        print(f"Replay buffer mismatches: {len(buf_mismatches)} (showing up to 10):")
        for i, w, r_env, r_buf in buf_mismatches[:10]:
            print(f"  idx={i} winner={w} env_reward={r_env:.6f} buf_reward={r_buf:.6f}")
        raise SystemExit(1)
    else:
        print("Replay buffer final rewards match env rewards and winner signs ✅")


if __name__ == "__main__":
    config = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(PROJECT_ROOT, "conf", "config_transformer.yaml")
    )
    dev = sys.argv[2] if len(sys.argv) > 2 else None
    main(config, dev)
