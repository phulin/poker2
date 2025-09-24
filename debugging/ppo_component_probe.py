#!/usr/bin/env python3
"""
PPO Component Probe

Roll out a single trajectory with the tensorized environment, then print detailed
per-step GAE and PPO components (advantages, returns, clipped returns, ratios,
clips, entropy) to verify signs and scaling.
"""

import argparse

import torch
import torch.nn.functional as F

from alphaholdem.rl.losses import trinal_clip_ppo_loss
from alphaholdem.rl.self_play import SelfPlayTrainer


def format_float(x: float) -> str:
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)


def planes_to_card_strs(planes: torch.Tensor) -> list[str]:
    """Convert a (4,13) one-hot plane to a list of short card strings (e.g., As, Kh)."""
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    suits = ["♠", "♥", "♦", "♣"]
    cards: list[str] = []
    # planes indexed as (suit, rank)
    for s in range(4):
        for r in range(13):
            if planes[s, r] > 0.5:
                cards.append(ranks[r] + suits[s])
    return cards


def probe_once(trainer: SelfPlayTrainer) -> None:
    # Clear the replay buffer first
    trainer.replay_buffer.clear()

    # Collect a single trajectory using the tensorized environment
    print("Collecting trajectory using tensorized environment...")
    total_reward, episode_count = trainer.collect_tensor_trajectories(
        min_steps=200,  # Need enough steps to get meaningful statistics
        all_opponent_snapshots=trainer.opponent_pool.snapshots,
    )

    # Check if we got any data
    if trainer.replay_buffer.num_steps() == 0:
        print("Failed to collect any trajectory data.")
        return

    print(
        f"Collected {trainer.replay_buffer.num_steps()} steps from {episode_count} episodes"
    )

    # Compute GAE for all stored trajectories
    trainer.replay_buffer.compute_gae_returns(
        gamma=trainer.gamma, lambda_=trainer.gae_lambda
    )

    # Prepare batch as in training (includes delta2/delta3 per-trajectory)
    longest_trajectory = trainer.replay_buffer.trajectory_lengths.argmax().item()

    # Re-run forward on stored observations to get current logits/values
    obs = trainer.replay_buffer.observations[longest_trajectory]  # shape (N, C_flat)
    actions = trainer.replay_buffer.actions[longest_trajectory]
    log_probs_old = trainer.replay_buffer.log_probs[longest_trajectory]
    adv = trainer.replay_buffer.advantages[longest_trajectory]
    returns = trainer.replay_buffer.returns[longest_trajectory]
    legal_masks = trainer.replay_buffer.legal_masks[longest_trajectory]
    d2 = trainer.replay_buffer.delta2[longest_trajectory]
    d3 = trainer.replay_buffer.delta3[longest_trajectory]

    N = obs.shape[0]
    cards = obs[:, : (6 * 4 * 13)].reshape(-1, 6, 4, 13)
    actions_tensor = obs[:, (6 * 4 * 13) :].reshape(-1, 24, 4, trainer.num_bet_bins)

    with torch.no_grad():
        logits, values = trainer.model(cards, actions_tensor)

    # Mask illegal actions and compute per-sample quantities
    masked_logits = logits.clone()
    masked_logits[legal_masks == 0] = -1e9
    log_probs_new = F.log_softmax(masked_logits, dim=-1)
    probs_new = log_probs_new.exp()
    a_logp_new = log_probs_new.gather(1, actions.unsqueeze(1)).squeeze(1)
    ratio = torch.exp(a_logp_new - log_probs_old)

    # Standard PPO clip and Trinal clip (match losses.py behavior)
    first_clipped = torch.clamp(ratio, 1.0 - trainer.epsilon, 1.0 + trainer.epsilon)
    negative = adv < 0
    second_clipped = torch.clamp(ratio, 0.0, trainer.delta1)
    final_clipped = torch.where(
        negative, torch.minimum(first_clipped, second_clipped), first_clipped
    )

    # Value clipping
    clipped_returns = torch.clamp(returns, d2, d3)

    # Entropy per sample
    ent = -(probs_new * log_probs_new).sum(dim=-1)

    # Prepare short action labels
    bin_labels = [f"x{m:g}" for m in trainer.cfg.bet_bins]
    action_names = ["fold", "check", *bin_labels, "allin"]

    print("\nPer-step (showing first 10 steps):")
    steps_to_show = min(10, N)
    for i in range(steps_to_show):
        # Decode hole cards and board cards from planes
        hole = planes_to_card_strs(cards[i, 0])  # channel 0 = hole
        if len(hole) == 0:
            break

        # Debug: show what's in each channel
        public = planes_to_card_strs(cards[i, 4])  # channel 4 = public (all board)

        # Use public channel for now to see what's actually there
        board_str = "".join(public) if public else "--"

        act_idx = int(actions[i].item())
        act_name = (
            action_names[act_idx] if act_idx < len(action_names) else str(act_idx)
        )
        # Policy surrogate term
        pi_term = float(
            -(
                min(
                    ratio[i].item() * adv[i].item(),
                    final_clipped[i].item() * adv[i].item(),
                )
            )
        )
        # Short line
        print(
            f"{i} {''.join(hole) or '----'} {board_str:<10} "
            f"{act_name:<5} r={returns[i].item():6.3f} rc={clipped_returns[i].item():6.3f} "
            f"d2={d2[i].item():6.3f} d3={d3[i].item():6.3f} "
            f"V={values[i].item():6.3f} A={adv[i].item():6.3f} "
            f"ratio={ratio[i].item():.3f} "
            f"pi={pi_term:6.3f}"
        )

    # Show cards from the sampled trajectory (not from current env state)
    print(f"\nSampled trajectory cards:")
    if N > 0:
        # Show cards from the first step of the sampled trajectory
        first_hole = planes_to_card_strs(cards[0, 0])
        first_flop = planes_to_card_strs(cards[0, 1])
        first_turn = planes_to_card_strs(cards[0, 2])
        first_river = planes_to_card_strs(cards[0, 3])
        first_board = (
            "".join(first_flop + first_turn + first_river)
            if (first_flop + first_turn + first_river)
            else "--"
        )
        print(f"  First step: hole={''.join(first_hole) or '----'} board={first_board}")

        # Show cards from the last step of the sampled trajectory
        last_hole = planes_to_card_strs(cards[-1, 0])
        last_flop = planes_to_card_strs(cards[-1, 1])
        last_turn = planes_to_card_strs(cards[-1, 2])
        last_river = planes_to_card_strs(cards[-1, 3])
        last_board = (
            "".join(last_flop + last_turn + last_river)
            if (last_flop + last_turn + last_river)
            else "--"
        )
        print(f"  Last step:  hole={''.join(last_hole) or '----'} board={last_board}")

    # Also compute the loss dictionary for consistency
    loss_dict = trinal_clip_ppo_loss(
        logits=logits,
        values=(
            values.squeeze(-1) if values.dim() == 2 and values.size(1) == 1 else values
        ),
        actions=actions,
        log_probs_old=log_probs_old,
        advantages=adv,
        returns=returns,
        legal_masks=legal_masks,
        epsilon=trainer.epsilon,
        delta1=trainer.delta1,
        delta2=d2,
        delta3=d3,
        value_coef=trainer.value_coef,
        entropy_coef=trainer.entropy_coef,
        value_loss_type=trainer.cfg.value_loss_type,
        huber_delta=trainer.cfg.huber_delta,
    )

    # Summary insights
    print(f"\nSummary:")
    print(f"  Total steps collected: {trainer.replay_buffer.num_steps()}")
    print(f"  Episodes completed: {episode_count}")
    print(
        f"  Average steps per episode: {trainer.replay_buffer.num_steps() / episode_count:.1f}"
    )
    print(f"  Policy loss: {loss_dict['policy_loss'].item():.3f}")
    print(f"  Value loss: {loss_dict['value_loss'].item():.3f}")
    print(f"  Entropy: {loss_dict['entropy'].item():.3f}")
    print(f"  Average advantage: {loss_dict['advantage_mean'].item():.3f}")
    print(f"  Advantage std: {loss_dict['advantage_std'].item():.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Probe PPO components for a single trajectory using tensorized environment."
    )
    parser.add_argument(
        "checkpoint", nargs="?", default=None, help="Optional checkpoint path"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps"],
        default="mps",
        help="Device to use for computation (default: mps)",
    )
    args = parser.parse_args()

    # Convert device string to torch.device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    trainer = SelfPlayTrainer(
        use_tensor_env=True,
        num_envs=4,  # Use small number of environments for debugging
        batch_size=8,
        k_best_pool_size=2,
        min_elo_diff=20.0,
        device=device,
    )
    if args.checkpoint:
        try:
            step = trainer.load_checkpoint(args.checkpoint)
            print(f"Loaded checkpoint at step {step}")
        except Exception as e:
            print(f"Warning: failed to load checkpoint: {e}")

    trainer.model.eval()
    with torch.no_grad():
        probe_once(trainer)


if __name__ == "__main__":
    main()
