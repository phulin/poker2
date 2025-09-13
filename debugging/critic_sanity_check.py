from __future__ import annotations

import time
from typing import Dict, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim

# Internal imports
from alphaholdem.rl.replay import compute_gae_returns, prepare_ppo_batch
from alphaholdem.rl.self_play import SelfPlayTrainer


def run_critic_sanity_check(
    trainer,
    num_samples: Optional[int] = None,
    lr: float = 1e-3,
    use_clipped_returns: bool = False,
    recompute_returns: bool = True,
    use_gae: bool = False,
) -> Dict[str, float]:
    """Run a single critic-only update on a fixed minibatch to verify sign/correctness.

    Args:
        trainer: An initialized SelfPlayTrainer with a populated replay buffer
        num_samples: Optional fixed minibatch size to subsample from the buffer
        lr: Learning rate for the temporary critic-only optimizer
        use_clipped_returns: If True, clip targets using per-sample δ2/δ3

    Returns:
        Dict with loss_before, loss_after, num_samples, and improved flag
    """
    if len(trainer.replay_buffer.trajectories) == 0:
        return {"error": "Replay buffer is empty. Collect trajectories first."}

    # Prepare a batch from current buffer. Optionally recompute returns on the fly
    if recompute_returns:
        observations = []
        returns_list = []
        delta2_list = []
        delta3_list = []
        for traj in trainer.replay_buffer.trajectories:
            # Gather rewards and (optionally) bootstrap with zeros
            rewards = [t.reward for t in traj.transitions]
            if use_gae:
                # Use GAE with lambda from trainer
                values_dummy = [0.0 for _ in rewards] + [0.0]
                adv, rets = compute_gae_returns(
                    rewards,
                    values_dummy,
                    gamma=trainer.gamma,
                    lambda_=trainer.gae_lambda,
                )
            else:
                # Pure Monte Carlo returns (no bootstrap)
                rets = []
                ret = 0.0
                for r in reversed(rewards):
                    ret = r + trainer.gamma * ret
                    rets.insert(0, ret)
            for i, t in enumerate(traj.transitions):
                observations.append(t.observation)
                returns_list.append(rets[i])
                delta2_list.append(t.delta2)
                delta3_list.append(t.delta3)
        batch = {
            "observations": torch.stack(observations),
            "returns": torch.tensor(returns_list, dtype=torch.float32),
            "delta2": torch.tensor(delta2_list, dtype=torch.float32),
            "delta3": torch.tensor(delta3_list, dtype=torch.float32),
        }
    else:
        batch = prepare_ppo_batch(list(trainer.replay_buffer.trajectories))

    # Move tensors to trainer device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(trainer.device)

    # Subsample if requested
    total = batch["returns"].shape[0]
    if num_samples is not None and num_samples < total:
        idx = torch.randperm(total, device=trainer.device)[:num_samples]

        def take(d):
            return d.index_select(0, idx) if isinstance(d, torch.Tensor) else d

        batch = {k: take(v) for k, v in batch.items()}
        total = num_samples

    # Reconstruct model inputs
    observations = batch["observations"]
    cards = observations[:, : (6 * 4 * 13)].reshape(-1, 6, 4, 13)
    actions_tensor = observations[:, (6 * 4 * 13) :].reshape(
        -1, 24, 4, trainer.num_bet_bins
    )

    # Targets: returns or clipped returns
    targets = batch["returns"]
    if use_clipped_returns:
        d2 = batch["delta2"]
        d3 = batch["delta3"]
        min_b = torch.minimum(d2, d3)
        max_b = torch.maximum(d2, d3)
        targets = torch.clamp(targets, min=min_b, max=max_b)

    # Compute loss before
    trainer.model.eval()
    with torch.no_grad():
        _, values_before = trainer.model(cards, actions_tensor)
        loss_before = F.mse_loss(values_before, targets).item()

    # Snapshot value head weights
    value_params = [p for p in trainer.model.value_head.parameters()]
    saved = [p.detach().clone() for p in value_params]

    # One critic-only optimizer step on the same minibatch
    critic_optim = optim.Adam(trainer.model.value_head.parameters(), lr=lr)
    critic_optim.zero_grad(set_to_none=True)
    _, values_mid = trainer.model(cards, actions_tensor)
    loss_mid = F.mse_loss(values_mid, targets)
    loss_mid.backward()
    critic_optim.step()

    # Measure after
    with torch.no_grad():
        _, values_after = trainer.model(cards, actions_tensor)
        loss_after = F.mse_loss(values_after, targets).item()

    # Restore original value head weights
    with torch.no_grad():
        for p, w in zip(value_params, saved):
            p.copy_(w)

    return {
        "num_samples": float(total),
        "loss_before": float(loss_before),
        "loss_after": float(loss_after),
        "improved": float(loss_after < loss_before),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Critic sanity check (single batch, critic-only step)."
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to YAML config"
    )
    parser.add_argument(
        "--warmup_batches",
        type=int,
        default=None,
        help="Number of batches to warm up replay (defaults to cfg.replay_buffer_batches)",
    )
    parser.add_argument(
        "--warmup_target_steps",
        type=int,
        default=2048,
        help="Cap warmup steps to avoid long runs (default: 2048)",
    )
    parser.add_argument(
        "--max_warmup_seconds",
        type=float,
        default=10.0,
        help="Time limit for warmup (seconds)",
    )
    parser.add_argument(
        "--max_warmup_trajectories",
        type=int,
        default=200,
        help="Trajectory cap for warmup loop",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1024,
        help="Fixed minibatch size for the check",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for temporary critic optimizer",
    )
    parser.add_argument(
        "--use_clipped_returns",
        action="store_true",
        help="Clip targets with per-sample deltas",
    )
    args = parser.parse_args()

    # Build trainer
    trainer = SelfPlayTrainer(config=args.config)

    # Determine target warmup steps
    cfg_batches = getattr(trainer.cfg, "replay_buffer_batches", 1)
    warmup_batches = (
        args.warmup_batches if args.warmup_batches is not None else cfg_batches
    )
    target_steps = max(1, warmup_batches) * trainer.batch_size
    # Keep warmup short by capping steps
    target_steps = min(target_steps, max(1, args.warmup_target_steps))

    # Warm up replay buffer with self-play trajectories (no updates)
    start_ts = time.time()
    collected = 0
    while trainer.replay_buffer.num_steps() < target_steps:
        traj, reward = trainer.collect_trajectory(opponent_snapshot=None)
        if len(traj.transitions) == 0:
            continue
        trainer.replay_buffer.add_trajectory(traj)
        collected += 1
        # Stop if limits exceeded
        if (time.time() - start_ts) > args.max_warmup_seconds:
            break
        if collected >= args.max_warmup_trajectories:
            break

    result = run_critic_sanity_check(
        trainer,
        num_samples=args.num_samples,
        lr=args.lr,
        use_clipped_returns=args.use_clipped_returns,
    )
    print(result)
