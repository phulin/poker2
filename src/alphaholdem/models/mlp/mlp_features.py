import torch
from dataclasses import dataclass

from alphaholdem.env.card_utils import (
    combo_lookup_tensor,
    hand_combos_tensor,
    NUM_HANDS,
)


@dataclass
class MLPFeatures:
    """Generic MLP features that can hold either structured or flat features."""

    context: torch.Tensor
    street: torch.Tensor
    board: torch.Tensor
    beliefs: torch.Tensor

    def __len__(self) -> int:
        """Get batch size."""
        return self.context.shape[0]

    def __getitem__(self, index: torch.Tensor | slice | int) -> "MLPFeatures":
        """Index into all features."""
        return MLPFeatures(
            context=self.context[index],
            street=self.street[index],
            board=self.board[index],
            beliefs=self.beliefs[index],
        )

    def __setitem__(
        self, index: torch.Tensor | slice | int, value: "MLPFeatures"
    ) -> None:
        """Set features at index."""
        self.context[index] = value.context
        self.street[index] = value.street
        self.board[index] = value.board
        self.beliefs[index] = value.beliefs

    def to(self, device: torch.device) -> "MLPFeatures":
        return MLPFeatures(
            context=self.context.to(device),
            street=self.street.to(device),
            board=self.board.to(device),
            beliefs=self.beliefs.to(device),
        )

    def clone(self) -> "MLPFeatures":
        return MLPFeatures(
            context=self.context.clone(),
            street=self.street.clone(),
            board=self.board.clone(),
            beliefs=self.beliefs.clone(),
        )

    def permute_suits(self, generator: torch.Generator | None = None) -> None:
        """Permute the suits of board cards and beliefs.

        Args:
            generator: Random generator for reproducibility.
        """
        batch_size = self.__len__()
        device = self.context.device

        # Generate suit permutations: [batch_size, 4]
        rands = torch.rand((batch_size, 4), device=device, generator=generator)
        suit_permutations = torch.argsort(rands, dim=-1).to(
            torch.long
        )  # [batch_size, 4]

        # Permute board cards
        # board is [batch_size, 5] with card indices or -1 for unflopped cards
        board_valid = self.board >= 0

        ranks = (self.board % 13).clamp(0, 12)  # [batch_size, 5]
        suits = (self.board // 13).clamp(0, 3)  # [batch_size, 5]

        # Apply suit permutation: for each card, look up its new suit
        new_suits = torch.gather(
            suit_permutations.unsqueeze(1).expand(-1, 5, -1),  # [batch_size, 5, 4]
            dim=2,
            index=suits.unsqueeze(2).to(torch.long),  # [batch_size, 5, 1]
        ).squeeze(2)

        # Reconstruct card indices: only update valid cards
        new_board = torch.where(board_valid, new_suits * 13 + ranks, self.board)
        self.board[:] = new_board

        # Permute beliefs
        # beliefs might be [batch_size, 1326] or [batch_size, 2 * NUM_HANDS]
        if self.beliefs.numel() > 0:
            # Get hand combos: [1326, 2] mapping hand index to two cards
            hand_combos = hand_combos_tensor(device=device)
            combo_lookup = combo_lookup_tensor(device=device)

            # Extract ranks and suits for all hands [1326, 2]
            all_ranks = hand_combos % 13
            all_suits = hand_combos // 13

            # Use the first permutation for simplicity (all batches use same permutation)
            suit_perm = suit_permutations[0]  # [4]

            # Vectorized suit permutation: [1326, 2]
            new_suits = suit_perm[all_suits]

            # Reconstruct card indices: [1326, 2]
            new_cards = all_ranks + new_suits * 13

            # Get min and max cards for each hand
            min_cards = torch.min(new_cards, dim=1).values  # [1326]
            max_cards = torch.max(new_cards, dim=1).values  # [1326]

            # Look up new hand indices using combo_lookup
            # combo_lookup is [52, 52], we index it with min_cards and max_cards
            # Note: we need to handle the indexing carefully
            # combo_lookup[min, max] gives the hand index
            remap = combo_lookup[min_cards, max_cards].to(torch.long)

            # Remap beliefs based on shape
            if self.beliefs.shape[1] == 1326:
                # [batch_size, 1326] - remap directly
                self.beliefs[:] = self.beliefs[:, remap]
            elif self.beliefs.shape[1] == 2 * NUM_HANDS:
                # [batch_size, 2 * NUM_HANDS] - remap each player's beliefs separately
                p0_beliefs = self.beliefs[:, :NUM_HANDS]
                p1_beliefs = self.beliefs[:, NUM_HANDS:]
                p0_remapped = p0_beliefs[:, remap]
                p1_remapped = p1_beliefs[:, remap]
                self.beliefs[:] = torch.cat([p0_remapped, p1_remapped], dim=1)
