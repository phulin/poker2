from dataclasses import dataclass

import torch

from alphaholdem.env.card_utils import (
    NUM_HANDS,
    combo_lookup_tensor,
    hand_combos_tensor,
)


@dataclass
class MLPFeatures:
    """Generic MLP features that can hold either structured or flat features."""

    context: torch.Tensor
    street: torch.Tensor
    to_act: torch.Tensor
    board: torch.Tensor
    beliefs: torch.Tensor

    def __post_init__(self) -> None:
        N = self.context.shape[0]
        assert self.street.shape == (N,)
        assert self.to_act.shape == (N,)
        assert self.board.shape == (N, 5)
        assert self.beliefs.shape == (N, 2 * NUM_HANDS)

    def __len__(self) -> int:
        """Get batch size."""
        return self.context.shape[0]

    def __getitem__(self, index: torch.Tensor | slice | int) -> "MLPFeatures":
        """Index into all features."""
        return MLPFeatures(
            context=self.context[index],
            street=self.street[index],
            to_act=self.to_act[index],
            board=self.board[index],
            beliefs=self.beliefs[index],
        )

    def __setitem__(
        self, index: torch.Tensor | slice | int, value: "MLPFeatures"
    ) -> None:
        """Set features at index."""
        self.context[index] = value.context
        self.street[index] = value.street
        self.to_act[index] = value.to_act
        self.board[index] = value.board
        self.beliefs[index] = value.beliefs

    def to(self, device: torch.device) -> "MLPFeatures":
        return MLPFeatures(
            context=self.context.to(device),
            street=self.street.to(device),
            to_act=self.to_act.to(device),
            board=self.board.to(device),
            beliefs=self.beliefs.to(device),
        )

    def clone(self) -> "MLPFeatures":
        return MLPFeatures(
            context=self.context.clone(),
            street=self.street.clone(),
            to_act=self.to_act.clone(),
            board=self.board.clone(),
            beliefs=self.beliefs.clone(),
        )

    def permute_suits(
        self,
        suit_permutations: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        """Permute the suits of board cards and beliefs.

        Args:
            generator: Random generator for reproducibility.
        """
        batch_size = self.__len__()
        device = self.context.device

        # Generate suit permutations: [batch_size, 4]
        if suit_permutations is None:
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
        # Get hand combos and lookup helpers.
        hand_combos = hand_combos_tensor(device=device)  # [1326, 2]
        combo_lookup = combo_lookup_tensor(device=device)  # [52, 52]

        # Extract ranks and suits for all hands [1326, 2] and expand across batch.
        all_ranks = (hand_combos % 13).unsqueeze(0).expand(batch_size, -1, -1)
        all_suits = (hand_combos // 13).unsqueeze(0).expand(batch_size, -1, -1)

        # Gather new suits per sample using that sample's permutation.
        suit_perm_expanded = suit_permutations[:, None, :].expand(-1, NUM_HANDS, -1)
        new_suits = torch.gather(suit_perm_expanded, 2, all_suits)

        # Reconstruct card indices per hand per sample.
        new_cards = all_ranks + new_suits * 13
        min_cards = torch.min(new_cards, dim=2).values
        max_cards = torch.max(new_cards, dim=2).values

        # Look up remapped hand indices for each sample.
        remap = combo_lookup[min_cards, max_cards].to(torch.long)
        inverse_remap = torch.argsort(remap, dim=1)

        # Remap beliefs based on shape.
        p0_beliefs = self.beliefs[:, :NUM_HANDS]
        p1_beliefs = self.beliefs[:, NUM_HANDS:]
        p0_remapped = torch.gather(p0_beliefs, 1, inverse_remap)
        p1_remapped = torch.gather(p1_beliefs, 1, inverse_remap)
        self.beliefs[:] = torch.cat([p0_remapped, p1_remapped], dim=1)
