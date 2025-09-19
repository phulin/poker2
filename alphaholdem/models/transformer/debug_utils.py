"""Debugging utilities for transformer model state reconstruction."""

from __future__ import annotations

import torch
from typing import Dict, List

from ...env.hunl_tensor_env import HUNLTensorEnv
from .structured_embedding_data import StructuredEmbeddingData
from .state_encoder import TransformerStateEncoder


class TransformerStateDebugger:
    """Debug utility to reconstruct and validate transformer input data."""

    def __init__(self, tensor_env: HUNLTensorEnv):
        self.tensor_env = tensor_env
        self.device = tensor_env.device

        # Card mapping: 0-51 -> (rank, suit)
        self.card_to_rank_suit = {}
        for card_idx in range(52):
            rank = card_idx % 13
            suit = card_idx // 13
            self.card_to_rank_suit[card_idx] = (rank, suit)

    def reconstruct_cards_from_tokens(
        self, token_ids: torch.Tensor, card_streets: torch.Tensor
    ) -> Dict[str, List]:
        """Reconstruct card information from token IDs and street information."""
        batch_size, seq_len = token_ids.shape
        reconstructed = {
            "hole_cards": [],
            "flop_cards": [],
            "turn_card": [],
            "river_card": [],
        }

        for batch_idx in range(batch_size):
            batch_hole = []
            batch_flop = []
            batch_turn = []
            batch_river = []

            for seq_idx in range(seq_len):
                token_id = token_ids[batch_idx, seq_idx].item()
                street = card_streets[batch_idx, seq_idx].item()

                # Skip padding tokens
                if token_id == -1:
                    continue

                # Skip non-card tokens (special tokens like CLS)
                if token_id >= 52:
                    continue

                rank, suit = self.card_to_rank_suit[token_id]

                if street == 0:  # Hole cards
                    batch_hole.append((rank, suit))
                elif street == 1:  # Flop
                    batch_flop.append((rank, suit))
                elif street == 2:  # Turn
                    batch_turn.append((rank, suit))
                elif street == 3:  # River
                    batch_river.append((rank, suit))

            reconstructed["hole_cards"].append(batch_hole)
            reconstructed["flop_cards"].append(batch_flop)
            reconstructed["turn_card"].append(batch_turn)
            reconstructed["river_card"].append(batch_river)

        return reconstructed

    def reconstruct_actions_from_tokens(
        self, action_actors: torch.Tensor, action_streets: torch.Tensor
    ) -> Dict[str, List]:
        """Reconstruct action information from actor and street data."""
        batch_size, seq_len = action_actors.shape
        reconstructed = {
            "preflop_actions": [],
            "flop_actions": [],
            "turn_actions": [],
            "river_actions": [],
        }

        for batch_idx in range(batch_size):
            batch_preflop = []
            batch_flop = []
            batch_turn = []
            batch_river = []

            for seq_idx in range(seq_len):
                actor = action_actors[batch_idx, seq_idx].item()
                street = action_streets[batch_idx, seq_idx].item()

                # Skip padding
                if actor == -1:
                    continue

                if street == 0:  # Preflop
                    batch_preflop.append(actor)
                elif street == 1:  # Flop
                    batch_flop.append(actor)
                elif street == 2:  # Turn
                    batch_turn.append(actor)
                elif street == 3:  # River
                    batch_river.append(actor)

            reconstructed["preflop_actions"].append(batch_preflop)
            reconstructed["flop_actions"].append(batch_flop)
            reconstructed["turn_actions"].append(batch_turn)
            reconstructed["river_actions"].append(batch_river)

        return reconstructed

    def reconstruct_context_from_features(
        self, context_features: torch.Tensor
    ) -> Dict[str, List]:
        """Reconstruct context information from context features tensor."""
        batch_size, seq_len, feature_dim = context_features.shape
        reconstructed = {
            "pot_sizes": [],
            "stack_sizes": [],
            "committed_amounts": [],
            "to_call_amounts": [],
            "street_info": [],
            "button_positions": [],
            "min_raise_amounts": [],
            "is_allin_flags": [],
            "has_folded_flags": [],
            "action_counts": [],
        }

        for batch_idx in range(batch_size):
            batch_pot = []
            batch_stack = []
            batch_committed = []
            batch_to_call = []
            batch_street = []
            batch_button = []
            batch_min_raise = []
            batch_allin = []
            batch_folded = []
            batch_action_count = []

            for seq_idx in range(seq_len):
                features = context_features[batch_idx, seq_idx]

                # Context features mapping (from state_encoder.py):
                # [pot, stack_p0, stack_p1, committed_p0, committed_p1, position, street, actions_this_round, min_raise, bet_to_call]
                if feature_dim >= 10:
                    pot = features[0].item()
                    stack_p0 = features[1].item()
                    stack_p1 = features[2].item()
                    committed_p0 = features[3].item()
                    committed_p1 = features[4].item()
                    position = features[
                        5
                    ].item()  # Player 0's position (0=button, 1=big blind)
                    street = features[6].item()
                    actions_this_round = features[7].item()
                    min_raise = features[8].item()
                    bet_to_call = features[9].item()

                    batch_pot.append(pot)
                    batch_stack.append((stack_p0, stack_p1))
                    batch_committed.append((committed_p0, committed_p1))
                    batch_to_call.append(bet_to_call)
                    batch_street.append(street)
                    batch_button.append(position)  # Position is stored here
                    batch_min_raise.append(min_raise)
                    batch_action_count.append(actions_this_round)

            reconstructed["pot_sizes"].append(batch_pot)
            reconstructed["stack_sizes"].append(batch_stack)
            reconstructed["committed_amounts"].append(batch_committed)
            reconstructed["to_call_amounts"].append(batch_to_call)
            reconstructed["street_info"].append(batch_street)
            reconstructed["button_positions"].append(batch_button)
            reconstructed["min_raise_amounts"].append(batch_min_raise)
            reconstructed["action_counts"].append(batch_action_count)

        return reconstructed

    def get_env_state_for_comparison(self, env_idx: int) -> Dict[str, any]:
        """Get the actual environment state for comparison."""
        env_state = {
            "hole_cards": [],
            "board_cards": [],
            "stacks": [],
            "committed": [],
            "pot": [],
            "street": [],
            "to_act": [],
            "button": [],
            "min_raise": [],
            "actions_this_round": [],
            "has_folded": [],
            "is_allin": [],
        }

        # Get hole cards
        hole_indices = self.tensor_env.hole_indices[env_idx]
        for player in range(2):
            player_cards = []
            for card_pos in range(2):
                card_idx = hole_indices[player, card_pos].item()
                if card_idx >= 0:
                    rank, suit = self.card_to_rank_suit[card_idx]
                    player_cards.append((rank, suit))
            env_state["hole_cards"].append(player_cards)

        # Get board cards
        board_indices = self.tensor_env.board_indices[env_idx]
        board_cards = []
        for street in range(5):  # 5 board positions
            card_idx = board_indices[street].item()
            if card_idx >= 0:
                rank, suit = self.card_to_rank_suit[card_idx]
                board_cards.append((rank, suit))
        env_state["board_cards"] = board_cards

        # Get other state information
        env_state["stacks"] = self.tensor_env.stacks[env_idx].tolist()
        env_state["committed"] = self.tensor_env.committed[env_idx].tolist()
        env_state["pot"] = self.tensor_env.pot[env_idx].item()
        env_state["street"] = self.tensor_env.street[env_idx].item()
        env_state["to_act"] = self.tensor_env.to_act[env_idx].item()
        env_state["button"] = self.tensor_env.button[env_idx].item()
        env_state["min_raise"] = self.tensor_env.min_raise[env_idx].item()
        env_state["actions_this_round"] = self.tensor_env.actions_this_round[
            env_idx
        ].item()
        env_state["has_folded"] = self.tensor_env.has_folded[env_idx].tolist()
        env_state["is_allin"] = self.tensor_env.is_allin[env_idx].tolist()

        return env_state

    def compare_reconstructed_vs_env(
        self, embedding_data: StructuredEmbeddingData, env_idx: int
    ) -> Dict[str, bool]:
        """Compare reconstructed data with actual environment state."""
        # Reconstruct data from embeddings
        reconstructed_cards = self.reconstruct_cards_from_tokens(
            embedding_data.token_ids, embedding_data.card_streets
        )
        reconstructed_actions = self.reconstruct_actions_from_tokens(
            embedding_data.action_actors, embedding_data.action_streets
        )
        reconstructed_context = self.reconstruct_context_from_features(
            embedding_data.context_features
        )

        # Get actual environment state
        env_state = self.get_env_state_for_comparison(env_idx)

        # Compare
        comparison = {
            "hole_cards_match": False,
            "board_cards_match": False,
            "stacks_match": False,
            "committed_match": False,
            "pot_match": False,
            "street_match": False,
            "min_raise_match": False,
            "actions_this_round_match": False,
            "position_match": False,  # Player 0's position (not button)
        }

        # Compare hole cards
        if len(reconstructed_cards["hole_cards"]) > env_idx:
            reconstructed_hole = reconstructed_cards["hole_cards"][env_idx]
            env_hole = env_state["hole_cards"]
            comparison["hole_cards_match"] = len(reconstructed_hole) == len(
                env_hole
            ) and all(
                len(rh) == len(eh) for rh, eh in zip(reconstructed_hole, env_hole)
            )

        # Compare board cards
        if len(reconstructed_cards["flop_cards"]) > env_idx:
            reconstructed_board = (
                reconstructed_cards["flop_cards"][env_idx]
                + reconstructed_cards["turn_card"][env_idx]
                + reconstructed_cards["river_card"][env_idx]
            )
            env_board = env_state["board_cards"]
            comparison["board_cards_match"] = reconstructed_board == env_board

        # Compare other state
        if len(reconstructed_context["stack_sizes"]) > env_idx:
            reconstructed_stacks_list = reconstructed_context["stack_sizes"][env_idx]
            # Find the last non-zero stack values
            non_zero_stacks = [
                (i, stacks)
                for i, stacks in enumerate(reconstructed_stacks_list)
                if stacks != (0, 0)
            ]
            reconstructed_stacks = non_zero_stacks[-1][1] if non_zero_stacks else [0, 0]
            env_stacks = env_state["stacks"]
            comparison["stacks_match"] = list(reconstructed_stacks) == env_stacks

        if len(reconstructed_context["committed_amounts"]) > env_idx:
            reconstructed_committed_list = reconstructed_context["committed_amounts"][
                env_idx
            ]
            # Find the last non-zero committed values
            non_zero_committed = [
                (i, committed)
                for i, committed in enumerate(reconstructed_committed_list)
                if committed != (0, 0)
            ]
            reconstructed_committed = (
                non_zero_committed[-1][1] if non_zero_committed else [0, 0]
            )
            env_committed = env_state["committed"]
            comparison["committed_match"] = (
                list(reconstructed_committed) == env_committed
            )

        if len(reconstructed_context["pot_sizes"]) > env_idx:
            reconstructed_pot_list = reconstructed_context["pot_sizes"][env_idx]
            # Find the last non-zero pot value
            non_zero_pots = [
                (i, pot) for i, pot in enumerate(reconstructed_pot_list) if pot != 0
            ]
            reconstructed_pot = non_zero_pots[-1][1] if non_zero_pots else 0
            env_pot = env_state["pot"]
            comparison["pot_match"] = abs(reconstructed_pot - env_pot) < 1e-6

        if len(reconstructed_context["street_info"]) > env_idx:
            reconstructed_street_list = reconstructed_context["street_info"][env_idx]
            # Find the last non-zero street value
            non_zero_streets = [
                (i, street)
                for i, street in enumerate(reconstructed_street_list)
                if street != 0
            ]
            reconstructed_street = non_zero_streets[-1][1] if non_zero_streets else 0
            env_street = env_state["street"]
            comparison["street_match"] = reconstructed_street == env_street

        # Compare min_raise (stored in context features)
        if len(reconstructed_context["min_raise_amounts"]) > env_idx:
            reconstructed_min_raise_list = reconstructed_context["min_raise_amounts"][
                env_idx
            ]
            # Find the last non-zero min_raise value
            non_zero_min_raises = [
                (i, min_raise)
                for i, min_raise in enumerate(reconstructed_min_raise_list)
                if min_raise != 0
            ]
            reconstructed_min_raise = (
                non_zero_min_raises[-1][1] if non_zero_min_raises else 0
            )
            env_min_raise = env_state["min_raise"]
            comparison["min_raise_match"] = reconstructed_min_raise == env_min_raise

        # Compare actions_this_round (stored in context features)
        if len(reconstructed_context["action_counts"]) > env_idx:
            reconstructed_action_count_list = reconstructed_context["action_counts"][
                env_idx
            ]
            # Find the last non-zero action count value
            non_zero_action_counts = [
                (i, action_count)
                for i, action_count in enumerate(reconstructed_action_count_list)
                if action_count != 0
            ]
            reconstructed_action_count = (
                non_zero_action_counts[-1][1] if non_zero_action_counts else 0
            )
            env_action_count = env_state["actions_this_round"]
            comparison["actions_this_round_match"] = (
                reconstructed_action_count == env_action_count
            )

        # Compare position (Player 0's position, not button)
        # Note: This is conceptually different from button position
        if len(reconstructed_context["button_positions"]) > env_idx:
            reconstructed_position_list = reconstructed_context["button_positions"][
                env_idx
            ]
            # Find the last non-zero position value
            non_zero_positions = [
                (i, position)
                for i, position in enumerate(reconstructed_position_list)
                if position != 0
            ]
            reconstructed_position = (
                non_zero_positions[-1][1] if non_zero_positions else 0
            )
            # Player 0's position: 0=button/SB, 1=big blind
            # Environment button: 0=button, 1=big blind
            # So they should match directly
            env_position = env_state["button"]
            comparison["position_match"] = reconstructed_position == env_position

        return comparison

    def analyze_context_features(
        self, context_features: torch.Tensor, env_idx: int = 0
    ):
        """Analyze what the context features actually contain."""
        print(f"\n=== Context Features Analysis (Env {env_idx}) ===")

        batch_size, seq_len, feature_dim = context_features.shape
        print(f"Context features shape: {context_features.shape}")

        if batch_size > env_idx:
            features = context_features[env_idx]  # [seq_len, feature_dim]

            print(f"\nFeature values over sequence length:")
            for seq_idx in range(min(seq_len, 10)):  # Show first 10 sequence positions
                feat_vals = features[seq_idx]
                print(f"  Seq {seq_idx:2d}: {feat_vals.tolist()}")

            if seq_len > 10:
                print(f"  ... (showing first 10 of {seq_len} sequence positions)")

            # Look for patterns in the features
            print(f"\nFeature analysis:")
            for feat_idx in range(feature_dim):
                feat_values = features[:, feat_idx]
                non_zero_count = (feat_values != 0).sum().item()
                unique_values = torch.unique(feat_values).tolist()
                print(
                    f"  Feature {feat_idx}: {non_zero_count}/{seq_len} non-zero, unique values: {unique_values[:10]}{'...' if len(unique_values) > 10 else ''}"
                )

        print("=" * 50)

    def print_debug_comparison(
        self, embedding_data: StructuredEmbeddingData, env_idx: int = 0
    ):
        """Print a detailed comparison between reconstructed and actual environment state."""
        print(f"\n=== Transformer State Debug Comparison (Env {env_idx}) ===")

        # Reconstruct data
        reconstructed_cards = self.reconstruct_cards_from_tokens(
            embedding_data.token_ids, embedding_data.card_streets
        )
        reconstructed_actions = self.reconstruct_actions_from_tokens(
            embedding_data.action_actors, embedding_data.action_streets
        )
        reconstructed_context = self.reconstruct_context_from_features(
            embedding_data.context_features
        )

        # Get actual environment state
        env_state = self.get_env_state_for_comparison(env_idx)

        # Print comparison
        print(f"\n--- Cards ---")
        print(
            f"Reconstructed hole cards: {reconstructed_cards['hole_cards'][env_idx] if len(reconstructed_cards['hole_cards']) > env_idx else 'N/A'}"
        )
        print(f"Actual hole cards: {env_state['hole_cards']}")

        print(
            f"Reconstructed board: {reconstructed_cards['flop_cards'][env_idx] if len(reconstructed_cards['flop_cards']) > env_idx else 'N/A'}"
        )
        print(f"Actual board: {env_state['board_cards']}")

        print(f"\n--- Game State ---")
        if len(reconstructed_context["stack_sizes"]) > env_idx:
            reconstructed_stacks_list = reconstructed_context["stack_sizes"][env_idx]
            non_zero_stacks = [
                (i, stacks)
                for i, stacks in enumerate(reconstructed_stacks_list)
                if stacks != (0, 0)
            ]
            reconstructed_stacks = non_zero_stacks[-1][1] if non_zero_stacks else [0, 0]
            print(f"Reconstructed stacks: {reconstructed_stacks}")
        print(f"Actual stacks: {env_state['stacks']}")

        if len(reconstructed_context["committed_amounts"]) > env_idx:
            reconstructed_committed_list = reconstructed_context["committed_amounts"][
                env_idx
            ]
            non_zero_committed = [
                (i, committed)
                for i, committed in enumerate(reconstructed_committed_list)
                if committed != (0, 0)
            ]
            reconstructed_committed = (
                non_zero_committed[-1][1] if non_zero_committed else [0, 0]
            )
            print(f"Reconstructed committed: {reconstructed_committed}")
        print(f"Actual committed: {env_state['committed']}")

        if len(reconstructed_context["pot_sizes"]) > env_idx:
            reconstructed_pot_list = reconstructed_context["pot_sizes"][env_idx]
            non_zero_pots = [
                (i, pot) for i, pot in enumerate(reconstructed_pot_list) if pot != 0
            ]
            reconstructed_pot = non_zero_pots[-1][1] if non_zero_pots else 0
            print(f"Reconstructed pot: {reconstructed_pot}")
        print(f"Actual pot: {env_state['pot']}")

        if len(reconstructed_context["street_info"]) > env_idx:
            reconstructed_street_list = reconstructed_context["street_info"][env_idx]
            non_zero_streets = [
                (i, street)
                for i, street in enumerate(reconstructed_street_list)
                if street != 0
            ]
            reconstructed_street = non_zero_streets[-1][1] if non_zero_streets else 0
            print(f"Reconstructed street: {reconstructed_street}")
        print(f"Actual street: {env_state['street']}")

        print(f"\n--- Actions ---")
        if len(reconstructed_actions["preflop_actions"]) > env_idx:
            print(
                f"Reconstructed preflop actions: {reconstructed_actions['preflop_actions'][env_idx]}"
            )
        if len(reconstructed_actions["flop_actions"]) > env_idx:
            print(
                f"Reconstructed flop actions: {reconstructed_actions['flop_actions'][env_idx]}"
            )

        # Print comparison results
        comparison = self.compare_reconstructed_vs_env(embedding_data, env_idx)
        print(f"\n--- Comparison Results ---")
        for key, matches in comparison.items():
            status = "✓" if matches else "✗"
            print(f"{status} {key}: {matches}")

        print("=" * 50)


def debug_transformer_state(
    embedding_data: StructuredEmbeddingData,
    tensor_env: HUNLTensorEnv,
    env_idx: int = 0,
    analyze_context: bool = True,
):
    """Convenience function to debug transformer state."""
    debugger = TransformerStateDebugger(tensor_env)
    debugger.print_debug_comparison(embedding_data, env_idx)

    if analyze_context:
        debugger.analyze_context_features(embedding_data.context_features, env_idx)

    return debugger.compare_reconstructed_vs_env(embedding_data, env_idx)


def create_debugger_with_encoder(tensor_env: HUNLTensorEnv, device: torch.device):
    """Create both a state encoder and debugger for convenience."""
    encoder = TransformerStateEncoder(tensor_env, device)
    debugger = TransformerStateDebugger(tensor_env)
    return encoder, debugger
