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
        self, tensor_env: HUNLTensorEnv, num_envs: int, player: int = 0
    ) -> Dict[str, torch.Tensor]:
        """Encode states for all environments in the tensorized environment.

        Args:
            tensor_env: HUNLTensorEnv instance
            num_envs: Number of environments
            player: Player index (0 or 1)

        Returns:
            Dictionary containing:
                - token_ids: Token sequences [num_envs, max_seq_len]
                - attention_masks: Attention masks [num_envs, max_seq_len]
        """
        batch_size = num_envs

        # Initialize output tensors
        token_ids = torch.zeros(
            batch_size, self.max_sequence_length, dtype=torch.long, device=self.device
        )
        attention_masks = torch.zeros(
            batch_size, self.max_sequence_length, dtype=torch.bool, device=self.device
        )

        # Encode each environment
        for env_idx in range(batch_size):
            tokens, mask = self.tokenizer.tokenize_state(tensor_env, env_idx, player)

            # Truncate if necessary
            seq_len = min(len(tokens), self.max_sequence_length)
            token_ids[env_idx, :seq_len] = tokens[:seq_len].to(self.device)
            attention_masks[env_idx, :seq_len] = mask[:seq_len].to(self.device)

        return {
            "token_ids": token_ids,
            "attention_masks": attention_masks,
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
