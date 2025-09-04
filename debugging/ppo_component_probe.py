#!/usr/bin/env python3
"""
PPO Component Probe

Roll out a single trajectory with the batched collector, then print detailed
per-step GAE and PPO components (advantages, returns, clipped returns, ratios,
clips, entropy) to verify signs and scaling.
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from alphaholdem.rl.self_play import SelfPlayTrainer
from alphaholdem.rl.replay import compute_gae_returns, prepare_ppo_batch
from alphaholdem.rl.losses import trinal_clip_ppo_loss
from alphaholdem.env import rules


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
    # Collect a single non-empty trajectory
    max_tries = 1000
    traj = None
    for _ in range(max_tries):
        t, _ = trainer.collect_trajectory()
        if len(t.transitions) > 0:
            traj = t
            break
    if traj is None:
        print("Failed to collect a non-empty trajectory after retries.")
        return

    # Compute GAE using the same gamma/lambda as trainer
    rewards = [tr.reward for tr in traj.transitions]
    values = [tr.value for tr in traj.transitions]
    values.append(0.0)  # bootstrap at terminal
    advs, rets = compute_gae_returns(rewards, values, trainer.gamma, trainer.gae_lambda)

    for i, tr in enumerate(traj.transitions):
        tr.advantage = advs[i]
        tr.return_ = rets[i]

    # Prepare batch as in training (includes delta2/delta3 per-trajectory)
    batch = prepare_ppo_batch([traj])

    # Move to device
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(trainer.device)

    # Re-run forward on stored observations to get current logits/values
    obs = batch["observations"]  # shape (N, C_flat)
    actions = batch["actions"]
    log_probs_old = batch["log_probs_old"]
    adv = batch["advantages"]
    returns = batch["returns"]
    legal_masks = batch["legal_masks"]
    d2 = batch["delta2"]
    d3 = batch["delta3"]

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

    print("Per-step:")
    for i in range(N):
        # Decode hole/public cards from planes
        hole = planes_to_card_strs(cards[i, 0])  # channel 0 = hole
        public = planes_to_card_strs(cards[i, 4])  # channel 4 = public
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
            f"{i} hole={''.join(hole) or '----'} {''.join(public) or '--':<10} "
            f"{act_name:<5} r={rewards[i] if i < len(rewards) else 0.0:9.3f} "
            f"V={values[i].item():9.3f} A={adv[i].item():9.3f} "
            f"retc={clipped_returns[i].item():9.3f} ratio={ratio[i].item():.3f} "
            f"pi={pi_term:9.3f}"
        )

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
    )

    print("\nBatch metrics:")
    for k in [
        "total_loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "ratio_mean",
        "ratio_std",
        "advantage_mean",
        "advantage_std",
        "clipped_ratio_mean",
        "clipped_ratio_std",
    ]:
        v = loss_dict.get(k)
        print(f"  {k:>18}: {float(v.item()) if hasattr(v, 'item') else v}")

    # Show both players' hole cards and hand evaluations at the end if available
    print("\nFinal state summary:")
    try:
        s = getattr(trainer, "env", None).state  # type: ignore[attr-defined]
    except Exception:
        s = None
    if s is None:
        print("  (final GameState unavailable)")
        return

    # Render cards
    def cards_ints_to_strs(card_ints: list[int]) -> str:
        ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
        suits = ["♠", "♥", "♦", "♣"]
        out: list[str] = []
        for c in card_ints:
            out.append(ranks[rules.rank(c)] + suits[rules.suit(c)])
        return "".join(out) if out else "--"

    p0 = s.players[0]
    p1 = s.players[1]
    board = s.board
    print(
        f"  P0: {cards_ints_to_strs(p0.hole_cards)}  P1: {cards_ints_to_strs(p1.hole_cards)}  Board: {cards_ints_to_strs(board)}"
    )

    # Hand evaluations only make sense at showdown when board has 5 cards and no fold
    if (
        s.street == "showdown"
        and len(board) == 5
        and not p0.has_folded
        and not p1.has_folded
    ):
        v0 = rules.evaluate_7(p0.hole_cards + board)
        v1 = rules.evaluate_7(p1.hole_cards + board)
        print(
            f"  Eval P0: {v0}  Eval P1: {v1}  Winner: {s.winner if s.winner is not None else 'split'}"
        )
    else:
        if p0.has_folded or p1.has_folded:
            print(f"  Hand ended by fold. Winner: {s.winner}")
        else:
            print("  No showdown evaluation (board incomplete)")


def main():
    parser = argparse.ArgumentParser(
        description="Probe PPO components for a single trajectory."
    )
    parser.add_argument(
        "checkpoint", nargs="?", default=None, help="Optional checkpoint path"
    )
    args = parser.parse_args()

    trainer = SelfPlayTrainer()
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
