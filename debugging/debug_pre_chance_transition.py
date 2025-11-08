#!/usr/bin/env python3
"""
Report model predictions at a street transition: compare the pre-chance estimate at the
end of a betting round to the model outputs on every determinized post-chance node
(i.e., each canonical single-card extension, typically 48 for flop->turn).

For each determinized node we print the mean value (across all hands for the acting
player) and the values on five representative hands (first hand in each equity group).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import hydra
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from alphaholdem.core.structured_config import Config
from alphaholdem.env.aggression_analyzer import build_hand_to_group_mapping
from alphaholdem.env.card_utils import (
    IDX_TO_RANK,
    NUM_HANDS,
    hand_combos_tensor,
)
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.mlp.mlp_features import MLPFeatures
from alphaholdem.models.mlp.rebel_ffn import RebelFFN
from alphaholdem.models.model_output import ModelOutput
from alphaholdem.rl.cfr_trainer import RebelCFRTrainer
from alphaholdem.search.rebel_cfr_evaluator import RebelCFREvaluator


def advance_to_street_with_history(
    env: HUNLTensorEnv,
    street: str,
    button: int = 1,
) -> None:
    """Reset env and play deterministic actions to reach a target street."""

    assert env.N == 1, "advance_to_street_with_history expects a single-env instance"

    env.reset(force_button=torch.tensor([button], device=env.device))

    bet_bin_index = min(3, len(env.default_bet_bins) + 2)
    call_action = torch.ones(env.N, dtype=torch.long, device=env.device)

    # Preflop: SB raise, BB call
    env.step_bins(
        torch.full((env.N,), bet_bin_index, dtype=torch.long, device=env.device)
    )
    env.step_bins(call_action)

    if street == "flop":
        return

    # Flop: check, check
    env.step_bins(call_action)
    env.step_bins(call_action)

    if street == "turn":
        return

    # Turn: SB check; BB bets 1.5x pot; SB call
    env.step_bins(call_action)
    env.step_bins(
        torch.full((env.N,), bet_bin_index, dtype=torch.long, device=env.device)
    )
    env.step_bins(call_action)

    if street == "river":
        return

    raise ValueError("Unknown street target for advance helper")


@dataclass
class ScriptArgs:
    checkpoint: Optional[str] = None
    street: str = "flop"  # street we are LEAVING (flop->turn or turn->river)
    verbose: bool = False


cs = ConfigStore.instance()
cs.store(group="", name="debug_pre_chance_transition_schema", node=ScriptArgs)


def _card_index_to_str(card_idx: int) -> str:
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    suits = ["♠", "♥", "♦", "♣"]
    if card_idx < 0:
        return "--"
    suit = card_idx // 13
    rank = card_idx % 13
    return f"{ranks[rank]}{suits[suit]}"


def _combo_to_name(combo_idx: int, combos: torch.Tensor) -> str:
    c1, c2 = combos[combo_idx]
    r1, r2 = c1 % 13, c2 % 13
    s1, s2 = c1 // 13, c2 // 13
    if r1 == r2:
        rank_char = IDX_TO_RANK[int(r1)]
        return f"{rank_char}{rank_char}"
    higher = max(r1, r2)
    lower = min(r1, r2)
    suited = s1 == s2
    return (
        f"{IDX_TO_RANK[int(higher)]}{IDX_TO_RANK[int(lower)]}{'s' if suited else 'o'}"
    )


def _load_model(cfg: Config, checkpoint_path: Optional[str], device: torch.device):
    if checkpoint_path is None:
        model = RebelFFN(
            input_dim=cfg.model.input_dim,
            num_actions=len(cfg.env.bet_bins) + 3,
            hidden_dim=cfg.model.hidden_dim,
            num_hidden_layers=cfg.model.num_hidden_layers,
            detach_value_head=cfg.model.detach_value_head,
            num_players=2,
        )
        rng = torch.Generator(device="cpu")
        rng.manual_seed(0)
        model.init_weights(rng)
        model.to(device)
        model.eval()
        return model

    ckpt_path = to_absolute_path(checkpoint_path)
    trainer = RebelCFRTrainer(cfg=cfg, device=device)
    trainer.load_checkpoint(ckpt_path)
    trainer.model.eval()
    return trainer.model


def _target_env_street(street: str) -> str:
    mapping = {"flop": "turn", "turn": "river"}
    if street not in mapping:
        raise ValueError("street must be either 'flop' or 'turn'")
    return mapping[street]


def _build_hand_representatives(device: torch.device) -> list[tuple[int, str]]:
    chunks = build_hand_to_group_mapping(device=device)
    combos = hand_combos_tensor(device=device)
    reps: list[tuple[int, str]] = []
    seen_names = set()
    for chunk in chunks:
        idx = int(chunk[0].item())
        name = _combo_to_name(idx, combos)
        # Guard in case of duplicates due to ordering artifacts.
        offset = 1
        while name in seen_names and offset < len(chunk):
            idx = int(chunk[offset].item())
            name = _combo_to_name(idx, combos)
            offset += 1
        seen_names.add(name)
        reps.append((idx, name))
    return reps


def _enumerate_single_card_nodes(
    evaluator: RebelCFREvaluator,
    root_idx: int,
    features_pre: MLPFeatures,
    pre_chance_beliefs: torch.Tensor,
    board_prev: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Clone of ChanceNodeHelper.single_card_chance_values that returns per-sample values.
    """
    helper = evaluator.chance_helper
    device = evaluator.device
    dtype = evaluator.float_dtype

    pre_beliefs = pre_chance_beliefs[root_idx : root_idx + 1].to(dtype=dtype)
    board_prev_root = board_prev[root_idx : root_idx + 1].clone()
    context_root = features_pre.context[root_idx : root_idx + 1].clone()
    street_root = features_pre.street[root_idx : root_idx + 1].clone()
    to_act_root = features_pre.to_act[root_idx : root_idx + 1].clone()

    board_prev_root = board_prev_root.to(device)

    available_mask = torch.ones(1, 52, dtype=torch.bool, device=device)
    for slot in range(board_prev_root.shape[1]):
        cards = board_prev_root[:, slot]
        valid = cards >= 0
        if valid.any():
            available_mask[valid, cards[valid]] = False

    cards = torch.arange(52, device=device, dtype=torch.long)
    flat_mask = available_mask.view(-1)
    flat_indices = torch.nonzero(flat_mask, as_tuple=False).squeeze(1)

    if flat_indices.numel() == 0:
        return {
            "cards": torch.empty(0, dtype=torch.long, device=device),
            "board_samples": torch.empty(
                0, board_prev_root.shape[1], dtype=torch.long, device=device
            ),
            "hand_values": torch.empty(
                0, evaluator.num_players, NUM_HANDS, device=device, dtype=dtype
            ),
        }

    card_values = cards[flat_mask]
    num_samples = card_values.numel()

    board_samples = board_prev_root.repeat(num_samples, 1)
    num_cards = (board_samples >= 0).sum(dim=1)
    board_samples[torch.arange(num_samples, device=device), num_cards] = card_values

    board_onehot = torch.zeros(num_samples, 52, dtype=torch.bool, device=device)
    valid_mask = board_samples >= 0
    idx_sample, idx_slot = torch.nonzero(valid_mask, as_tuple=True)
    board_onehot[idx_sample, board_samples[idx_sample, idx_slot]] = True

    allowed_mask = (
        helper.combo_onehot_float @ board_onehot.T.float() < 0.5
    ).T  # [num_samples, 1326]

    post_beliefs = pre_beliefs.repeat(num_samples, 1, 1)
    post_beliefs.masked_fill_(~allowed_mask.unsqueeze(1), 0.0)
    sums = post_beliefs.sum(dim=-1, keepdim=True)
    uniform = allowed_mask.unsqueeze(1).float()
    uniform_sum = uniform.sum(dim=-1, keepdim=True).clamp(min=1.0)
    uniform = uniform / uniform_sum
    post_beliefs = torch.where(
        sums > 1e-12, post_beliefs / sums.clamp(min=1e-12), uniform
    )

    features = MLPFeatures(
        context=context_root.repeat(num_samples, 1),
        street=street_root.repeat(num_samples),
        to_act=to_act_root.repeat(num_samples),
        board=board_samples,
        beliefs=post_beliefs.reshape(num_samples, -1),
    )

    evaluator.model.eval()
    with torch.no_grad():
        outputs: ModelOutput = evaluator.model(features)
        hand_values = outputs.hand_values.to(dtype=dtype)

    return {
        "cards": card_values,
        "board_samples": board_samples,
        "hand_values": hand_values,
    }


def _format_board(board_tensor: torch.Tensor, new_card: int) -> str:
    cards = [c for c in board_tensor.tolist() if c >= 0]
    pieces = [_card_index_to_str(c) for c in cards]
    return (
        " ".join(pieces[:-1] + [f"[{pieces[-1]}]"])
        if pieces
        else _card_index_to_str(new_card)
    )


def debug_transition(cfg: Config, args: ScriptArgs) -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    cfg.device = str(device)
    cfg.num_envs = 1

    env = HUNLTensorEnv(
        num_envs=1,
        starting_stack=cfg.env.stack,
        sb=cfg.env.sb,
        bb=cfg.env.bb,
        default_bet_bins=cfg.env.bet_bins,
        device=device,
        float_dtype=torch.float32,
    )
    env.reset()
    target_env_street = _target_env_street(args.street)
    advance_to_street_with_history(env, street=target_env_street, button=1)

    model = _load_model(cfg, args.checkpoint, device=device)

    evaluator = RebelCFREvaluator(
        search_batch_size=1,
        env_proto=env,
        model=model,
        bet_bins=cfg.env.bet_bins,
        max_depth=cfg.search.depth,
        cfr_iterations=cfg.search.iterations,
        device=device,
        float_dtype=torch.float32,
        warm_start_iterations=0,
        cfr_type=cfg.search.cfr_type,
        cfr_avg=cfg.search.cfr_avg,
        dcfr_delay=cfg.search.dcfr_plus_delay,
    )

    roots = torch.tensor([0], device=device)
    initial_beliefs = torch.full((1, 2, NUM_HANDS), 1.0 / NUM_HANDS, device=device)
    evaluator.initialize_search(env, roots, initial_beliefs)

    features_pre = evaluator.feature_encoder.encode(
        evaluator.beliefs, pre_chance_node=True
    )
    root_idx = 0
    actor = int(evaluator.env.to_act[root_idx].item())

    with torch.no_grad():
        pre_output = evaluator.model(
            MLPFeatures(
                context=features_pre.context[root_idx : root_idx + 1],
                street=features_pre.street[root_idx : root_idx + 1],
                to_act=features_pre.to_act[root_idx : root_idx + 1],
                board=features_pre.board[root_idx : root_idx + 1],
                beliefs=features_pre.beliefs[root_idx : root_idx + 1],
            )
        )
        pre_hand_values = pre_output.hand_values[0, actor]

    board_prev = evaluator.env.last_board_indices
    samples = _enumerate_single_card_nodes(
        evaluator,
        root_idx=root_idx,
        features_pre=features_pre,
        pre_chance_beliefs=evaluator.root_pre_chance_beliefs,
        board_prev=board_prev,
    )

    cards = samples["cards"]
    board_samples = samples["board_samples"]
    hand_values = samples["hand_values"][:, actor, :]

    board_means = hand_values.mean(dim=-1)
    reps = _build_hand_representatives(device=device)

    print("=== Pre-Chance vs Post-Chance Value Report ===")
    print(
        f"Street analyzed: {args.street.upper()} (transition to {target_env_street.upper()})"
    )
    print(f"Checkpoint: {args.checkpoint or 'random init'}")
    print(f"Player to act: P{actor}")
    print("")
    print(
        f"Pre-chance mean (across hands): {pre_hand_values.mean().item():+.6f} "
        f"| std={pre_hand_values.std().item():.6f}"
    )
    print(
        f"Mean of determinized nodes:     {board_means.mean().item():+.6f} "
        f"| std={board_means.std().item():.6f}"
    )
    print("")
    header = ["Idx", "Board", "Mean"] + [name for _, name in reps]
    col_widths = [11, 20, 10] + [10] * len(reps)

    def _fmt_row(values: list[str]) -> str:
        return " ".join(value.rjust(width) for value, width in zip(values, col_widths))

    print(_fmt_row(header))
    print("-" * (sum(col_widths) + len(col_widths) - 1))

    for idx, (card, board, mean_value) in enumerate(
        zip(cards.tolist(), board_samples.tolist(), board_means.tolist())
    ):
        board_display = " ".join([_card_index_to_str(c) for c in board if c >= 0])
        row_vals = [
            str(idx),
            board_display,
            f"{mean_value:+.5f}",
        ]
        for combo_idx, _ in reps:
            val = hand_values[idx, combo_idx].item()
            row_vals.append(f"{val:+.5f}")
        print(_fmt_row(row_vals))

    # Calculate and print mean across all nodes
    mean_of_means = board_means.mean().item()
    mean_hand_values = [
        hand_values[:, combo_idx].mean().item() for combo_idx, _ in reps
    ]

    print("-" * (sum(col_widths) + len(col_widths) - 1))
    footer_vals = [
        "MEAN",
        "",
        f"{mean_of_means:+.5f}",
    ]
    for mean_val in mean_hand_values:
        footer_vals.append(f"{mean_val:+.5f}")
    print(_fmt_row(footer_vals))

    # Show pre-chance estimate values (labeled as POST-CHANCE per user request)
    pre_mean = pre_hand_values.mean().item()
    pre_hand_vals = [pre_hand_values[combo_idx].item() for combo_idx, _ in reps]
    post_chance_vals = [
        "POST-CHANCE",
        "",
        f"{pre_mean:+.5f}",
    ]
    for pre_val in pre_hand_vals:
        post_chance_vals.append(f"{pre_val:+.5f}")
    print(_fmt_row(post_chance_vals))

    print("\nRepresentative hands:", ", ".join(name for _, name in reps))


@hydra.main(version_base=None, config_path="../conf", config_name="config_rebel_cfr")
def main(dict_config: DictConfig) -> None:
    container = OmegaConf.to_container(dict_config, resolve=True)
    assert isinstance(container, dict)
    checkpoint = container.pop("checkpoint", None)
    street = container.pop("street", "flop")
    verbose = bool(container.pop("verbose", False))

    cfg = Config.from_dict(container)
    args = ScriptArgs(checkpoint=checkpoint, street=street, verbose=verbose)
    debug_transition(cfg, args)


if __name__ == "__main__":
    main()
