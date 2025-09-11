"""Tokenizer for converting poker game states to token sequences."""

from __future__ import annotations

from typing import Tuple, List, Dict, Any
import torch
import torch.nn as nn

from ...env.hunl_tensor_env import HUNLTensorEnv


class PokerTokenizer:
    """Tokenizer for converting poker game states to token sequences.

    Converts game states into sequences of tokens suitable for transformer processing.
    Token types:
    - Card tokens (0-51): Individual cards
    - Action tokens (52-67): Historical actions
    - Context tokens (68-75): Numeric game state
    - Special tokens (76-79): CLS, SEP, MASK, PAD
    """

    def __init__(self, max_sequence_length: int = 50):
        self.max_sequence_length = max_sequence_length

        # Special tokens
        self.special_tokens = {"CLS": 0, "SEP": 1, "MASK": 2, "PAD": 3}

        # Token type offsets
        self.card_offset = 4
        self.action_offset = 56  # 4 + 52 cards
        self.context_offset = 72  # 4 + 52 + 16 actions

        # Action type mapping
        self.action_type_map = {
            "fold": 0,
            "check": 1,
            "call": 2,
            "bet": 3,
            "raise": 4,
            "allin": 5,
        }

        # Context token types
        self.context_types = {
            "pot": 0,
            "stack_p0": 1,
            "stack_p1": 2,
            "position": 3,
            "street": 4,
            "actions_round": 5,
            "min_raise": 6,
            "bet_to_call": 7,
        }

    def _tokenize_action_history(
        self, tensor_env: HUNLTensorEnv, env_idx: int
    ) -> List[int]:
        """Tokenize action history for the environment."""
        tokens = []

        # Get action history
        action_history = tensor_env.get_action_history()[env_idx]  # [4, 6, 4, num_bins]
        # action_history structure: [street, slot, row, bin]
        # row 0: player 0 actions, row 1: player 1 actions, row 2: sum, row 3: legal mask

        # Process each street
        for street_idx in range(4):
            street_actions = action_history[street_idx]  # [6, 4, num_bins]

            for slot_idx in range(6):
                slot_actions = street_actions[slot_idx]  # [4, num_bins]

                # Check if there are any actions in this slot (sum row)
                if slot_actions[2].sum() > 0:  # row 2 is the sum
                    # Find which player acted and what action
                    for player_idx in range(2):  # Only check player rows (0, 1)
                        player_actions = slot_actions[player_idx]  # [num_bins]
                        if player_actions.sum() > 0:
                            # Find the action bin
                            action_bin = player_actions.float().argmax().item()

                            # Create action token with proper encoding
                            # Format: action_offset + player * num_action_types + action_bin
                            action_token = (
                                self.action_offset
                                + player_idx
                                * len(self.action_type_map)  # 6 actions per player
                                + action_bin
                            )
                            tokens.append(action_token)
                            break  # Only one player acts per slot

        return tokens

    def tokenize_state_structured(
        self, tensor_env: HUNLTensorEnv, env_idx: int, player: int
    ) -> Dict[str, torch.Tensor]:
        """Convert game state to structured embedding components.

        Args:
            tensor_env: HUNLTensorEnv instance
            env_idx: Environment index
            player: Player index (0 or 1)

        Returns:
            Dictionary containing structured embedding components
        """
        seq_len = self.max_sequence_length

        # Initialize tensors for all components
        card_indices = torch.full((seq_len,), -1, dtype=torch.long)  # -1 for padding
        card_stages = torch.zeros(seq_len, dtype=torch.long)
        card_visibility = torch.zeros(seq_len, dtype=torch.long)
        card_order = torch.zeros(seq_len, dtype=torch.long)

        action_actors = torch.zeros(seq_len, dtype=torch.long)
        action_types = torch.zeros(seq_len, dtype=torch.long)
        action_streets = torch.zeros(seq_len, dtype=torch.long)
        action_size_bins = torch.zeros(seq_len, dtype=torch.long)
        action_size_features = torch.zeros(seq_len, 3, dtype=torch.float)

        context_pot_sizes = torch.zeros(seq_len, 1, dtype=torch.float)
        context_stack_sizes = torch.zeros(seq_len, 2, dtype=torch.float)
        context_positions = torch.zeros(seq_len, dtype=torch.long)
        context_street_context = torch.zeros(seq_len, 4, dtype=torch.float)

        pos = 0

        # Add CLS token (special token at position 0)
        # For CLS token, we'll use a special card index that won't cause embedding issues
        # We'll use card index 52 (out of range) to represent CLS, and handle it specially in embeddings
        card_indices[pos] = 52  # Special CLS token index
        card_stages[pos] = 0
        card_visibility[pos] = 0
        card_order[pos] = 0
        pos += 1

        # Add visible cards
        visible_cards = tensor_env.get_visible_card_indices(env_idx, player)
        for i, card_idx in enumerate(visible_cards):
            if pos >= seq_len:
                break
            card_indices[pos] = card_idx.item()
            card_stages[pos] = self._get_card_stage(
                card_idx.item(), tensor_env, env_idx
            )
            card_visibility[pos] = self._get_card_visibility(
                card_idx.item(), tensor_env, env_idx, player
            )
            card_order[pos] = i
            pos += 1

        # Add action history
        action_pos = pos
        action_tokens = self._tokenize_action_history_structured(tensor_env, env_idx)
        for i, action_data in enumerate(action_tokens):
            if pos >= seq_len:
                break
            action_actors[pos] = action_data["actor"]
            action_types[pos] = action_data["action_type"]
            action_streets[pos] = action_data["street"]
            action_size_bins[pos] = action_data["size_bin"]
            action_size_features[pos] = torch.tensor(
                action_data["size_features"], dtype=torch.float
            )
            pos += 1

        # Add context information
        context_data = self._tokenize_context_structured(tensor_env, env_idx, player)
        context_pot_sizes[0] = torch.tensor(
            [context_data["pot_size"]], dtype=torch.float
        )
        context_stack_sizes[0] = torch.tensor(
            context_data["stack_sizes"], dtype=torch.float
        )
        context_positions[0] = context_data["position"]
        context_street_context[0] = torch.tensor(
            context_data["street_context"], dtype=torch.float
        )

        return {
            "card_indices": card_indices,
            "card_stages": card_stages,
            "card_visibility": card_visibility,
            "card_order": card_order,
            "action_actors": action_actors,
            "action_types": action_types,
            "action_streets": action_streets,
            "action_size_bins": action_size_bins,
            "action_size_features": action_size_features,
            "context_pot_sizes": context_pot_sizes,
            "context_stack_sizes": context_stack_sizes,
            "context_positions": context_positions,
            "context_street_context": context_street_context,
        }

    def _get_card_stage(
        self, card_idx: int, tensor_env: HUNLTensorEnv, env_idx: int
    ) -> int:
        """Get the stage (hole/flop/turn/river) for a card."""
        # Check if it's a hole card
        hole_cards = tensor_env.hole_indices[env_idx].flatten()
        if card_idx in hole_cards:
            return 0  # hole

        # Check board cards
        board_cards = tensor_env.board_indices[env_idx]
        for i, board_card in enumerate(board_cards):
            if board_card == card_idx:
                return i + 1  # flop=1, turn=2, river=3

        return 0  # default to hole

    def _get_card_visibility(
        self, card_idx: int, tensor_env: HUNLTensorEnv, env_idx: int, player: int
    ) -> int:
        """Get visibility type (self/opponent/public) for a card."""
        # Check if it's a hole card
        hole_cards = tensor_env.hole_indices[env_idx, player]
        if card_idx in hole_cards:
            return 0  # self

        # Check opponent hole cards
        opp_hole_cards = tensor_env.hole_indices[env_idx, 1 - player]
        if card_idx in opp_hole_cards:
            return 1  # opponent

        # Must be a board card
        return 2  # public

    def _tokenize_action_history_structured(
        self, tensor_env: HUNLTensorEnv, env_idx: int
    ) -> List[Dict[str, Any]]:
        """Tokenize action history into structured components."""
        actions = []

        # Get action history
        action_history = tensor_env.get_action_history()[env_idx]  # [4, 6, 4, num_bins]

        # Process each street
        for street_idx in range(4):
            street_actions = action_history[street_idx]  # [6, 4, num_bins]

            for slot_idx in range(6):
                slot_actions = street_actions[slot_idx]  # [4, num_bins]

                # Check if there are any actions in this slot (sum row)
                if slot_actions[2].sum() > 0:  # row 2 is the sum
                    # Find which player acted and what action
                    for player_idx in range(2):  # Only check player rows (0, 1)
                        player_actions = slot_actions[player_idx]  # [num_bins]
                        if player_actions.sum() > 0:
                            # Find the action bin
                            action_bin = player_actions.float().argmax().item()

                            # Compute actual bet size features
                            # Get the bet amount from the action bin
                            # We need to get the legal bins amounts for this environment
                            bin_amounts, _ = tensor_env.legal_bins_amounts_and_mask()
                            bet_amount = bin_amounts[env_idx, action_bin].item()
                            pot_size = tensor_env.pot[env_idx].item()
                            stack_size = tensor_env.stacks[env_idx, player_idx].item()

                            # Compute size features
                            fraction_of_pot = (
                                bet_amount / max(pot_size, 1.0) if pot_size > 0 else 0.0
                            )
                            fraction_of_stack = (
                                bet_amount / max(stack_size, 1.0)
                                if stack_size > 0
                                else 0.0
                            )
                            log_chips = torch.log(
                                torch.tensor(max(bet_amount, 1.0))
                            ).item()

                            # Map action_bin to a more meaningful size bin (0-19)
                            # This is a simplified mapping - in practice you'd want more sophisticated binning
                            size_bin = min(
                                action_bin % 20, 19
                            )  # Ensure it's in range [0, 19]

                            # Create structured action data
                            action_data = {
                                "actor": player_idx,
                                "action_type": action_bin,
                                "street": street_idx,
                                "size_bin": size_bin,
                                "size_features": [
                                    fraction_of_pot,
                                    fraction_of_stack,
                                    log_chips,
                                ],
                            }
                            actions.append(action_data)
                            break  # Only one player acts per slot

        return actions

    def _tokenize_context_structured(
        self, tensor_env: HUNLTensorEnv, env_idx: int, player: int
    ) -> Dict[str, Any]:
        """Tokenize context information into structured components."""
        # Pot size (normalized)
        pot_size = tensor_env.pot[env_idx].item() / 1000.0  # Normalize

        # Stack sizes (normalized)
        stack_p0 = tensor_env.stacks[env_idx, 0].item() / 1000.0
        stack_p1 = tensor_env.stacks[env_idx, 1].item() / 1000.0

        # Position
        position = player

        # Street context
        street = tensor_env.street[env_idx].item()
        actions_round = tensor_env.actions_this_round[env_idx].item()
        min_raise = tensor_env.min_raise[env_idx].item() / 1000.0

        # Compute bet_to_call (amount needed to call)
        # Get current player's committed amount and opponent's committed amount
        current_player_committed = tensor_env.chips_placed[env_idx, player].item()
        opponent_player = 1 - player
        opponent_committed = tensor_env.chips_placed[env_idx, opponent_player].item()
        bet_to_call = max(0.0, (opponent_committed - current_player_committed) / 1000.0)

        return {
            "pot_size": pot_size,
            "stack_sizes": [stack_p0, stack_p1],
            "position": position,
            "street_context": [street, actions_round, min_raise, bet_to_call],
        }

    def _tokenize_context(
        self, tensor_env: HUNLTensorEnv, env_idx: int, player: int
    ) -> List[int]:
        """Tokenize context information."""
        tokens = []

        # Pot size (normalized)
        pot_size = tensor_env.pot[env_idx].item()
        pot_token = self.context_offset + self.context_types["pot"]
        tokens.append(pot_token)

        # Stack sizes (normalized)
        stack_p0 = tensor_env.stacks[env_idx, 0].item()
        stack_p1 = tensor_env.stacks[env_idx, 1].item()
        stack_p0_token = self.context_offset + self.context_types["stack_p0"]
        stack_p1_token = self.context_offset + self.context_types["stack_p1"]
        tokens.extend([stack_p0_token, stack_p1_token])

        # Position
        position_token = self.context_offset + self.context_types["position"] + player
        tokens.append(position_token)

        # Street
        street = tensor_env.street[env_idx].item()
        street_token = self.context_offset + self.context_types["street"] + street
        tokens.append(street_token)

        # Actions this round
        actions_round = tensor_env.actions_this_round[env_idx].item()
        actions_token = (
            self.context_offset
            + self.context_types["actions_round"]
            + min(actions_round, 7)
        )
        tokens.append(actions_token)

        return tokens

    def get_vocab_size(self) -> int:
        """Get total vocabulary size."""
        return self.context_offset + len(self.context_types)

    def decode_tokens(self, token_ids: torch.Tensor) -> List[str]:
        """Decode token IDs back to readable strings (for debugging)."""
        decoded = []

        for token_id in token_ids:
            token_id = token_id.item()

            if token_id < self.card_offset:
                # Special token
                for name, id_val in self.special_tokens.items():
                    if token_id == id_val:
                        decoded.append(f"[{name}]")
                        break
            elif token_id < self.action_offset:
                # Card token
                card_id = token_id - self.card_offset
                decoded.append(f"[CARD_{card_id}]")
            elif token_id < self.context_offset:
                # Action token
                action_id = token_id - self.action_offset
                decoded.append(f"[ACTION_{action_id}]")
            else:
                # Context token
                context_id = token_id - self.context_offset
                decoded.append(f"[CONTEXT_{context_id}]")

        return decoded
