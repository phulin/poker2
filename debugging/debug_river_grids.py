#!/usr/bin/env python3
"""
Debugging script to print range % grids and value grids for river spots,
similar to rebelpreflopanalyzer.

Usage:
  source venv/bin/activate
  python debugging/debug_river_grids.py \
    --checkpoint /path/to/checkpoint.pt \
    [--board "As Kh Qd Jc Tc"] \
    [--actions "call,call,call,call,call,call"] \
    [--device cuda]
"""

import argparse
import os
from typing import List, Optional, Tuple

import torch

from alphaholdem.core.structured_config import Config
from alphaholdem.env.analyze_tensor_env import _create_169_grid
from alphaholdem.env.card_utils import (
    NUM_HANDS,
    combo_lookup_tensor,
    hand_combos_tensor,
)
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.mlp.rebel_feature_encoder import RebelFeatureEncoder
from alphaholdem.rl.cfr_trainer import RebelCFRTrainer
from alphaholdem.search.rebel_cfr_evaluator import (
    PublicBeliefState,
    RebelCFREvaluator,
)
from alphaholdem.utils.training_utils import print_combined_tables


def parse_card(card_str: str) -> int:
    """Parse card string like 'As' to card index."""
    card_str = card_str.strip().upper()
    if len(card_str) != 2:
        raise ValueError(f"Invalid card format: {card_str}")

    rank_char = card_str[0]
    suit_char = card_str[1].lower()

    # Convert rank
    if rank_char == "T":
        rank = 8  # 10
    elif rank_char == "J":
        rank = 9
    elif rank_char == "Q":
        rank = 10
    elif rank_char == "K":
        rank = 11
    elif rank_char == "A":
        rank = 12
    else:
        rank = int(rank_char) - 2

    # Convert suit
    suit_map = {"s": 0, "h": 1, "d": 2, "c": 3}
    if suit_char not in suit_map:
        raise ValueError(f"Invalid suit: {suit_char}")
    suit = suit_map[suit_char]

    # Card index = suit * 13 + rank
    return suit * 13 + rank


def parse_board(board_str: str) -> List[int]:
    """Parse board string like 'As Kh Qd Jc Tc' to list of card indices."""
    cards = board_str.split()
    if len(cards) != 5:
        raise ValueError(f"Board must have exactly 5 cards, got {len(cards)}")
    return [parse_card(c) for c in cards]


def parse_action_sequence(actions_str: str, bet_bins: List[float]) -> List[int]:
    """Parse action sequence like 'call,call,bet' to list of action bin indices."""
    action_map = {
        "fold": 0,
        "call": 1,
        "check": 1,  # check is same as call
    }
    actions = []
    for action_str in actions_str.split(","):
        action_str = action_str.strip().lower()
        if action_str in action_map:
            actions.append(action_map[action_str])
        elif action_str.startswith("bet"):
            # Parse bet size like "bet0.5" or "bet1.0"
            bet_size_str = action_str[3:]
            try:
                bet_size = float(bet_size_str)
                if bet_size not in bet_bins:
                    raise ValueError(f"Bet size {bet_size} not in bet_bins {bet_bins}")
                bin_idx = bet_bins.index(bet_size) + 2  # +2 for fold/call
                actions.append(bin_idx)
            except ValueError:
                raise ValueError(f"Invalid bet action: {action_str}")
        elif action_str == "allin":
            actions.append(len(bet_bins) + 2)  # all-in is last action
        else:
            raise ValueError(f"Unknown action: {action_str}")
    return actions


def create_random_board(
    rng: Optional[torch.Generator] = None, used_cards: Optional[List[int]] = None
) -> List[int]:
    """Create a random 5-card board."""
    all_cards = list(range(52))
    if used_cards:
        all_cards = [c for c in all_cards if c not in used_cards]

    # Sample 5 cards
    # torch.randperm doesn't support MPS generators, so always use CPU for this
    cpu_rng = torch.Generator(device="cpu")
    if rng is not None:
        # Try to preserve seed from the original generator
        try:
            # Get the seed from the generator state if possible
            state = rng.get_state()
            cpu_rng.set_state(state.cpu())
        except:
            # Fallback to manual seed
            cpu_rng.manual_seed(42)
    else:
        cpu_rng.manual_seed(42)

    indices = torch.randperm(len(all_cards), generator=cpu_rng)[:5].tolist()
    return [all_cards[i] for i in indices]


class RiverAnalyzer:
    """Analyzer for river spots with range and value grids."""

    def __init__(
        self,
        model: torch.nn.Module,
        button: int = 0,
        starting_stack: int = 1000,
        sb: int = 5,
        bb: int = 10,
        bet_bins: List[float] = None,
        device: torch.device = None,
        rng: torch.Generator = None,
        flop_showdown: bool = False,
        popart_normalizer=None,
        cfr_iterations: int = 100,
        max_depth: int = 2,
    ):
        """Initialize the river analyzer."""
        if bet_bins is None:
            bet_bins = [0.5, 0.75, 1.0, 1.5, 2.0]

        if device is None:
            device = torch.device("cpu")

        if rng is None:
            rng = torch.Generator(device=device)

        self.model = model
        self.device = device
        self.rng = rng
        self.popart_normalizer = popart_normalizer
        self.combo_lookup = combo_lookup_tensor(device=self.device)
        self.bet_bins = bet_bins
        self.num_bet_bins = len(bet_bins) + 3  # fold, call, bet bins, all-in

        # Create environment with 1326 hands (all possible hole card combos)
        self.env = HUNLTensorEnv(
            num_envs=1326,
            starting_stack=starting_stack,
            sb=sb,
            bb=bb,
            default_bet_bins=bet_bins,
            device=device,
            rng=rng,
            flop_showdown=flop_showdown,
        )

        # Create state encoder
        self.state_encoder = RebelFeatureEncoder(env=self.env, device=device)

        # All possible hand combinations
        self.all_hands = hand_combos_tensor(device=device)

        # Create CFR evaluator for single environment
        self.cfr_env = HUNLTensorEnv(
            num_envs=1,
            starting_stack=starting_stack,
            sb=sb,
            bb=bb,
            default_bet_bins=bet_bins,
            device=device,
            rng=rng,
            flop_showdown=flop_showdown,
        )

        self.cfr_evaluator = RebelCFREvaluator(
            search_batch_size=1,
            env_proto=self.cfr_env,
            model=self.model,
            bet_bins=bet_bins,
            max_depth=max_depth,
            cfr_iterations=cfr_iterations,
            device=device,
            float_dtype=torch.float32,
            generator=rng,
        )
        self.current_board: Optional[List[int]] = None

    def setup_river_state(
        self,
        board: List[int],
        action_sequence: List[int],
        button: int = 0,
    ) -> None:
        """Set up river state with given board and action sequence."""
        # Reset CFR environment
        self.cfr_env.reset(
            force_button=torch.tensor([button], dtype=torch.long, device=self.device)
        )

        # Apply action sequence to advance through streets
        # This will automatically advance streets when betting rounds close
        for action_bin in action_sequence:
            legal_masks = self.cfr_env.legal_bins_mask()
            if not legal_masks[0, action_bin]:
                # Action not legal, use call/check instead
                action_bin = 1
            bin_amounts, legal_masks = self.cfr_env.legal_bins_amounts_and_mask()
            self.cfr_env.step_bins(
                torch.tensor([action_bin], dtype=torch.long, device=self.device),
                bin_amounts=bin_amounts,
                legal_masks=legal_masks,
            )
            # If we've reached river, stop
            if self.cfr_env.street[0].item() == 3:
                break

        # Force board to be set manually
        # Deck order: board cards first, then hole cards
        # We'll use placeholder hole cards for now
        board_tensor = torch.tensor(board, device=self.device, dtype=torch.long)
        placeholder_hole = torch.tensor(
            [0, 1, 2, 3], device=self.device, dtype=torch.long
        )
        deck_cards = torch.cat([board_tensor, placeholder_hole])

        self.cfr_env.deck[0, :9] = deck_cards[:9]
        self.cfr_env.deck_pos[0] = 9

        # Set board cards manually
        for i, card in enumerate(board):
            self.cfr_env.board_indices[0, i] = card
            rank = card % 13
            suit = card // 13
            self.cfr_env.board_onehot[0, i, suit, rank] = 1.0

        # Force street to river
        self.cfr_env.street[0] = 3

        # Store board for later use
        self.current_board = board

    def get_probabilities_from_cfr(
        self, seat: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get probabilities and values by running CFR search."""
        # Create uniform beliefs
        uniform_beliefs = torch.full(
            (1, 2, NUM_HANDS),
            1.0 / NUM_HANDS,
            device=self.device,
            dtype=torch.float32,
        )

        pbs = PublicBeliefState.from_proto(
            env_proto=self.cfr_env,
            beliefs=uniform_beliefs,
        )
        root_idx = torch.tensor([0], dtype=torch.long, device=self.device)
        pbs.env.copy_state_from(
            self.cfr_env,
            root_idx,
            root_idx,
            copy_deck=True,
        )

        # Run CFR search
        self.cfr_evaluator.initialize_search(
            pbs.env,
            root_idx,
            pbs.beliefs,
        )
        self.cfr_evaluator.evaluate_cfr(training_mode=False)

        # Get root policy
        actions_slice = slice(1, 1 + self.cfr_evaluator.num_actions)
        root_policy = self.cfr_evaluator.policy_probs_avg[actions_slice]
        root_policy /= root_policy.sum(dim=0, keepdim=True).clamp_min(1e-12)

        # Get legal actions
        legal_masks = self.cfr_evaluator.valid_mask[actions_slice]
        legal_masks = legal_masks[None, :].expand(NUM_HANDS, -1)

        # Get values
        root_values = self.cfr_evaluator.values_avg[0, seat].clone()

        if self.popart_normalizer is not None:
            root_values = self.popart_normalizer.denormalize_value(root_values)

        return root_policy.T, root_values, legal_masks

    def convert_1326_to_169_tensor(
        self, values_1326: torch.Tensor, board_cards: Optional[List[int]] = None
    ) -> torch.Tensor:
        """Convert 1326-hand values to 169-hand-bucket tensor.

        Args:
            values_1326: Tensor of 1326 values for each hand
            board_cards: Optional list of board cards to filter out invalid hands
        """
        grid_sums = torch.zeros(13, 13, dtype=values_1326.dtype, device=self.device)
        grid_counts = torch.zeros(13, 13, dtype=torch.long, device=self.device)

        # Filter out hands that contain board cards if board is provided
        valid_mask = torch.ones(
            values_1326.shape[0], dtype=torch.bool, device=self.device
        )
        if board_cards is not None:
            board_set = set(board_cards)
            for i in range(len(self.all_hands)):
                card1 = self.all_hands[i, 0].item()
                card2 = self.all_hands[i, 1].item()
                if card1 in board_set or card2 in board_set:
                    valid_mask[i] = False

        # Only process valid hands
        valid_hands = self.all_hands[valid_mask]
        valid_values = values_1326[valid_mask]

        suits = valid_hands // 13
        ranks = valid_hands % 13
        ranks_sorted = torch.sort(ranks, dim=1).values
        ranks_flat = torch.where(
            suits[:, 0] == suits[:, 1],
            ranks_sorted[:, 1] * 13 + ranks_sorted[:, 0],
            ranks_sorted[:, 0] * 13 + ranks_sorted[:, 1],
        )
        grid_counts.flatten().scatter_add_(0, ranks_flat, torch.ones_like(ranks_flat))
        grid_sums.flatten().scatter_add_(0, ranks_flat, valid_values)

        averaged = torch.where(
            grid_counts > 0,
            grid_sums / grid_counts.clamp(min=1),
            torch.zeros_like(grid_sums),
        )
        return averaged

    def make_range_grid(
        self,
        values_1326: torch.Tensor,
        value_type: str = "probability",
        board_cards: Optional[List[int]] = None,
    ) -> str:
        """Convert 1326-hand values to 169-hand-bucket grid."""
        values_169 = self.convert_1326_to_169_tensor(values_1326, board_cards)
        return _create_169_grid(values_169, value_type)

    def get_river_grids(self, board_cards: Optional[List[int]] = None) -> dict:
        """Get river range and value grids."""
        probs, values, _ = self.get_probabilities_from_cfr(0)

        # Get acting player and top hands
        to_act = self.cfr_evaluator.env.to_act[0].item()
        beliefs = self.cfr_evaluator.beliefs_avg[
            0, to_act
        ]  # Beliefs about acting player

        # Get top hands by belief
        top_k = min(10, NUM_HANDS)
        top_hands = torch.topk(beliefs, top_k)
        top_hand_indices = top_hands.indices.cpu().tolist()
        top_hand_probs = top_hands.values.cpu().tolist()

        return {
            "ranges": [
                self.make_range_grid(probs[:, bin_index], "probability", board_cards)
                for bin_index in range(probs.shape[1])
            ],
            "betting": self.make_range_grid(
                probs[:, 2 : probs.shape[1] - 1].sum(dim=1), "probability", board_cards
            ),
            "value": self.make_range_grid(values, "value", board_cards),
            "to_act": to_act,
            "top_hands": list(zip(top_hand_indices, top_hand_probs)),
        }


def load_trainer_from_checkpoint(
    checkpoint_path: str, cfg: Config, device: torch.device
) -> Tuple[RebelCFRTrainer, int]:
    """Load trainer from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = checkpoint["model"]

    # Detect BetterFFN
    is_better_ffn = "street_embedding.weight" in model_state

    # Infer bet bins from model
    policy_w = model_state.get("policy_head.weight")
    if policy_w is not None:
        out_dim = int(policy_w.shape[0])
        num_actions = max(3, out_dim // 1326)
        k = max(0, num_actions - 3)
        if k > 0:
            cfg.env.bet_bins = [0.5 * (i + 1) for i in range(k)]

    if is_better_ffn:
        if "post_norm.weight" in model_state:
            cfg.model.hidden_dim = int(model_state["post_norm.weight"].shape[0])
            cfg.model.range_hidden_dim = 128
            cfg.model.ffn_dim = int(cfg.model.hidden_dim * 2)
        trunk_layers: set[int] = set()
        for kname in model_state.keys():
            if kname.startswith("trunk."):
                parts = kname.split(".")
                if len(parts) >= 2 and parts[1].isdigit():
                    trunk_layers.add(int(parts[1]))
        if trunk_layers:
            cfg.model.num_hidden_layers = max(trunk_layers) + 1

    trainer = RebelCFRTrainer(cfg=cfg, device=device)
    loaded_step = trainer.load_checkpoint(checkpoint_path)
    return trainer, int(loaded_step)


def main():
    parser = argparse.ArgumentParser(
        description="Print range % grids and value grids for river spots"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--board",
        type=str,
        default=None,
        help="Board cards (e.g., 'As Kh Qd Jc Tc'), or random if not specified",
    )
    parser.add_argument(
        "--actions",
        type=str,
        default="call,call,call,call,call,call",
        help="Action sequence (e.g., 'call,call,bet0.5'), defaults to 6 calls",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda, mps)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="conf/config_rebel_cfr.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Load config
    from omegaconf import OmegaConf

    from alphaholdem.core.structured_config import (
        EnvConfig,
        ModelConfig,
        ModelType,
        SearchConfig,
        TrainingConfig,
    )

    if os.path.exists(args.config):
        dict_config = OmegaConf.load(args.config)
        container = OmegaConf.to_container(dict_config, resolve=True)
        # Remove any keys that Config doesn't expect (like 'defaults' from Hydra)
        container.pop("defaults", None)
        cfg = Config.from_dict(container)
    else:
        # Create default config
        cfg = Config()
        cfg.train = TrainingConfig()
        cfg.model = ModelConfig()
        cfg.model.name = ModelType.rebel_ffn
        cfg.env = EnvConfig()
        cfg.env.stack = 1000
        cfg.env.sb = 5
        cfg.env.bb = 10
        cfg.env.bet_bins = [0.5, 0.75, 1.0, 1.5, 2.0]
        cfg.search = SearchConfig()
        cfg.search.iterations = 100
        cfg.search.depth = 2

    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Set seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Load trainer
    trainer, loaded_step = load_trainer_from_checkpoint(args.checkpoint, cfg, device)
    trainer.model.eval()

    # Set seed on trainer's RNG
    trainer.rng.manual_seed(seed)

    # Parse board
    if args.board:
        board = parse_board(args.board)
    else:
        cpu_rng = torch.Generator(device="cpu")
        cpu_rng.manual_seed(seed)
        board = create_random_board(cpu_rng)

    # Parse action sequence
    action_sequence = parse_action_sequence(args.actions, cfg.env.bet_bins)

    # Create analyzer with seeded RNG
    analyzer_rng = torch.Generator(device=device)
    analyzer_rng.manual_seed(seed)

    analyzer = RiverAnalyzer(
        model=trainer.model,
        button=0,
        starting_stack=cfg.env.stack,
        sb=cfg.env.sb,
        bb=cfg.env.bb,
        bet_bins=cfg.env.bet_bins,
        device=device,
        rng=analyzer_rng,
        flop_showdown=getattr(cfg.env, "flop_showdown", False),
        popart_normalizer=getattr(trainer, "popart_normalizer", None),
        cfr_iterations=cfg.search.iterations,
        max_depth=cfg.search.depth,
    )

    # Set up river state
    analyzer.setup_river_state(board, action_sequence)

    # Get grids
    print(f"\n--- River Range & Value Grids ---")
    print(f"Seed: {seed}")

    # Format board display
    ranks = "23456789TJQKA"
    suits = "shdc"
    board_str = " ".join([f"{ranks[c % 13]}{suits[c // 13]}" for c in board])
    print(f"Board: {board_str}")
    print(f"Action sequence: {args.actions}")
    print()

    grids = analyzer.get_river_grids(board_cards=board)
    fold_grid = grids["ranges"][0].splitlines()
    call_grid = grids["ranges"][1].splitlines()
    allin_grid = grids["ranges"][len(cfg.env.bet_bins) + 2].splitlines()
    betting_grid = grids["betting"].splitlines()
    value_grid = grids["value"]

    # Get acting player and top hands
    to_act = grids["to_act"]
    top_hands = grids["top_hands"]

    # Display acting player and top hands
    print(f"Acting player: P{to_act}")
    print(f"Top {len(top_hands)} most likely hands for P{to_act}:")

    # Helper function to convert hand index to string
    def hand_idx_to_string(hand_idx: int) -> str:
        """Convert hand index to string like 'As Kh'."""
        combos = hand_combos_tensor(device=torch.device("cpu"))
        c1, c2 = combos[hand_idx].tolist()
        ranks = "23456789TJQKA"
        suits = "shdc"
        card1_str = f"{ranks[c1 % 13]}{suits[c1 // 13]}"
        card2_str = f"{ranks[c2 % 13]}{suits[c2 // 13]}"
        return f"{card1_str} {card2_str}"

    for i, (hand_idx, prob) in enumerate(top_hands, 1):
        hand_str = hand_idx_to_string(hand_idx)
        print(f"  {i:2d}. {hand_str:8s} (belief: {prob:.4f})")
    print()

    # First row: Fold | Call
    print_combined_tables(
        [
            (fold_grid, "Small blind (first) - fold (%)"),
            (call_grid, "Small blind (first) - call (%)"),
        ],
    )

    # Second row: Betting | All-in
    print_combined_tables(
        [
            (betting_grid, "Small blind (first) - betting (%)"),
            (allin_grid, "Small blind (first) - all-in (%)"),
        ],
    )

    # Print value estimates grid
    print("--- River Value Estimates ---")
    print("Small blind (first) - value estimates (×1000)")
    print(value_grid)
    print()


if __name__ == "__main__":
    main()
