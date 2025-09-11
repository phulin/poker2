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

    def tokenize_state(
        self, tensor_env: HUNLTensorEnv, env_idx: int, player: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert game state to token sequence.

        Args:
            tensor_env: HUNLTensorEnv instance
            env_idx: Environment index
            player: Player index (0 or 1)

        Returns:
            Tuple of (token_ids, attention_mask)
        """
        tokens = []

        # Add CLS token
        tokens.append(self.special_tokens["CLS"])

        # Add visible cards
        visible_cards = tensor_env.get_visible_card_indices(env_idx, player)
        for card_idx in visible_cards:
            token_id = self.card_offset + card_idx.item()
            tokens.append(token_id)

        # Add SEP token between cards and actions
        tokens.append(self.special_tokens["SEP"])

        # Add action history
        action_tokens = self._tokenize_action_history(tensor_env, env_idx)
        tokens.extend(action_tokens)

        # Add SEP token between actions and context
        tokens.append(self.special_tokens["SEP"])

        # Add context tokens
        context_tokens = self._tokenize_context(tensor_env, env_idx, player)
        tokens.extend(context_tokens)

        # Pad to max length
        while len(tokens) < self.max_sequence_length:
            tokens.append(self.special_tokens["PAD"])

        # Truncate if too long
        tokens = tokens[: self.max_sequence_length]

        # Create attention mask (1 = attend, 0 = ignore)
        attention_mask = torch.ones(len(tokens), dtype=torch.bool)
        attention_mask[tokens == self.special_tokens["PAD"]] = False

        return torch.tensor(tokens, dtype=torch.long), attention_mask

    def _tokenize_action_history(
        self, tensor_env: HUNLTensorEnv, env_idx: int
    ) -> List[int]:
        """Tokenize action history for the environment."""
        tokens = []

        # Get action history
        action_history = tensor_env.get_action_history()[env_idx]  # [4, 6, 4, num_bins]

        # Process each street
        for street_idx in range(4):
            street_actions = action_history[street_idx]  # [6, 4, num_bins]

            for slot_idx in range(6):
                slot_actions = street_actions[slot_idx]  # [4, num_bins]

                # Check if there are any actions in this slot
                if slot_actions.sum() > 0:
                    # Find the action
                    action_bin = slot_actions.argmax(dim=-1)  # [4]

                    # Get player action (first non-zero)
                    for player_idx in range(2):
                        if action_bin[player_idx] > 0:
                            # Create action token
                            action_token = (
                                self.action_offset
                                + player_idx * 8  # 8 actions per player
                                + street_idx * 2  # 2 streets per action type
                                + action_bin[player_idx].item()
                            )
                            tokens.append(action_token)
                            break

        return tokens

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
