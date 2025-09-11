"""State encoder for transformer-based poker models."""

from __future__ import annotations

from typing import Dict, Tuple, Optional
import torch

from ...env.hunl_tensor_env import HUNLTensorEnv
from ...env.hunl_env import HUNLEnv
from .tokenizer import PokerTokenizer


class TransformerStateEncoder:
    """State encoder for transformer models using token-based representation.

    This encoder converts poker game states into token sequences suitable for
    transformer processing, leveraging the 0-51 card indices from HUNLTensorEnv.
    """

    def __init__(self, device: torch.device, max_sequence_length: int = 50):
        self.device = device
        self.max_sequence_length = max_sequence_length
        self.tokenizer = PokerTokenizer(max_sequence_length)

    def encode_tensor_states(
        self, tensor_env: HUNLTensorEnv, num_envs: int, player: int = None
    ) -> Dict[str, torch.Tensor]:
        """Encode states for all environments in the tensorized environment.

        Args:
            tensor_env: HUNLTensorEnv instance
            num_envs: Number of environments
            player: Player index (0 or 1). If None, uses tensor_env.to_act for each env.

        Returns:
            Dictionary containing structured embedding components
        """
        batch_size = num_envs
        return self._encode_structured_states(tensor_env, batch_size, player)

    def _encode_structured_states(
        self, tensor_env: HUNLTensorEnv, batch_size: int, player: int
    ) -> Dict[str, torch.Tensor]:
        """Encode states using structured embeddings."""
        seq_len = self.max_sequence_length

        # Initialize all structured embedding tensors
        card_indices = torch.full(
            (batch_size, seq_len), -1, dtype=torch.long, device=self.device
        )
        card_stages = torch.zeros(
            batch_size, seq_len, dtype=torch.long, device=self.device
        )
        card_visibility = torch.zeros(
            batch_size, seq_len, dtype=torch.long, device=self.device
        )
        card_order = torch.zeros(
            batch_size, seq_len, dtype=torch.long, device=self.device
        )

        action_actors = torch.zeros(
            batch_size, seq_len, dtype=torch.long, device=self.device
        )
        action_types = torch.zeros(
            batch_size, seq_len, dtype=torch.long, device=self.device
        )
        action_streets = torch.zeros(
            batch_size, seq_len, dtype=torch.long, device=self.device
        )
        action_size_bins = torch.zeros(
            batch_size, seq_len, dtype=torch.long, device=self.device
        )
        action_size_features = torch.zeros(
            batch_size, seq_len, 3, dtype=torch.float, device=self.device
        )

        context_pot_sizes = torch.zeros(
            batch_size, seq_len, 1, dtype=torch.float, device=self.device
        )
        context_stack_sizes = torch.zeros(
            batch_size, seq_len, 2, dtype=torch.float, device=self.device
        )
        context_positions = torch.zeros(
            batch_size, seq_len, dtype=torch.long, device=self.device
        )
        context_street_context = torch.zeros(
            batch_size, seq_len, 4, dtype=torch.float, device=self.device
        )

        # Encode each environment
        for env_idx in range(batch_size):
            # Use per-environment player if player is None, otherwise use provided player
            current_player = (
                tensor_env.to_act[env_idx].item() if player is None else player
            )
            structured_data = self.tokenizer.tokenize_state_structured(
                tensor_env, env_idx, current_player
            )

            # Copy structured data to batch tensors
            card_indices[env_idx] = structured_data["card_indices"].to(self.device)
            card_stages[env_idx] = structured_data["card_stages"].to(self.device)
            card_visibility[env_idx] = structured_data["card_visibility"].to(
                self.device
            )
            card_order[env_idx] = structured_data["card_order"].to(self.device)

            action_actors[env_idx] = structured_data["action_actors"].to(self.device)
            action_types[env_idx] = structured_data["action_types"].to(self.device)
            action_streets[env_idx] = structured_data["action_streets"].to(self.device)
            action_size_bins[env_idx] = structured_data["action_size_bins"].to(
                self.device
            )
            action_size_features[env_idx] = structured_data["action_size_features"].to(
                self.device
            )

            context_pot_sizes[env_idx] = structured_data["context_pot_sizes"].to(
                self.device
            )
            context_stack_sizes[env_idx] = structured_data["context_stack_sizes"].to(
                self.device
            )
            context_positions[env_idx] = structured_data["context_positions"].to(
                self.device
            )
            context_street_context[env_idx] = structured_data[
                "context_street_context"
            ].to(self.device)

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

    def encode_single_state(
        self, game_state: HUNLEnv, player: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a single game state for a specific player.

        Args:
            game_state: HUNLEnv instance
            player: Player index (0 or 1)

        Returns:
            Tuple of (token_ids, attention_mask)
        """
        # Convert HUNLEnv to token sequence
        # This is a simplified version - in practice you'd need to extract
        # the card indices and action history from the HUNLEnv
        tokens, mask = self._encode_hunl_env_state(game_state, player)

        return tokens.to(self.device), mask.to(self.device)

    def _encode_hunl_env_state(
        self, game_state: HUNLEnv, player: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode HUNLEnv state to tokens (simplified implementation).

        This is a placeholder implementation. In practice, you'd need to:
        1. Extract card indices from game_state
        2. Extract action history
        3. Extract context information
        4. Convert to token sequence

        For now, we'll create a minimal token sequence.
        """
        tokens = []

        # Add CLS token
        tokens.append(self.tokenizer.special_tokens["CLS"])

        # Add visible cards (simplified - just add some placeholder cards)
        # In practice, you'd extract actual card indices from game_state
        visible_cards = [0, 1, 2, 3, 4]  # Placeholder
        for card_idx in visible_cards:
            token_id = self.tokenizer.card_offset + card_idx
            tokens.append(token_id)

        # Add SEP token
        tokens.append(self.tokenizer.special_tokens["SEP"])

        # Add context tokens (simplified)
        pot_token = self.tokenizer.context_offset + self.tokenizer.context_types["pot"]
        tokens.append(pot_token)

        # Pad to max length
        while len(tokens) < self.max_sequence_length:
            tokens.append(self.tokenizer.special_tokens["PAD"])

        # Truncate if too long
        tokens = tokens[: self.max_sequence_length]

        # Create attention mask
        attention_mask = torch.ones(len(tokens), dtype=torch.bool)
        attention_mask[tokens == self.tokenizer.special_tokens["PAD"]] = False

        return torch.tensor(tokens, dtype=torch.long), attention_mask

    def get_sequence_length(self) -> int:
        """Get maximum sequence length."""
        return self.max_sequence_length

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.tokenizer.get_vocab_size()

    def decode_tokens(self, token_ids: torch.Tensor) -> list[list[str]]:
        """Decode token IDs back to readable strings (for debugging).

        Args:
            token_ids: Token IDs tensor [batch_size, seq_len]

        Returns:
            List of decoded token sequences
        """
        decoded_sequences = []

        for i in range(token_ids.shape[0]):
            sequence = self.tokenizer.decode_tokens(token_ids[i])
            decoded_sequences.append(sequence)

        return decoded_sequences
