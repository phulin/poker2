from dataclasses import dataclass

import torch

from p2.env.card_utils import (
    NUM_HANDS,
    PREFLOP_HANDS,
    combo_lookup_tensor,
    hand_combos_tensor,
)


# Keep feature indexing eager. These gathers are not on the expensive CFR path,
# and compiling this small helper specializes on singleton ragged batches.
def _index_all(
    context: torch.Tensor,
    street: torch.Tensor,
    to_act: torch.Tensor,
    board: torch.Tensor,
    beliefs: torch.Tensor,
    index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        context[index],
        street[index],
        to_act[index],
        board[index],
        beliefs[index],
    )


@dataclass
class MLPFeatures:
    """Generic MLP features that can hold either structured or flat features."""

    context: torch.Tensor
    street: torch.Tensor
    to_act: torch.Tensor
    board: torch.Tensor
    beliefs: torch.Tensor
    hand_dim: int = NUM_HANDS
    source_rows: torch.Tensor | None = None

    def __post_init__(self) -> None:
        N = self.context.shape[0]
        if self.hand_dim not in (NUM_HANDS, PREFLOP_HANDS):
            raise ValueError(
                f"hand_dim must be one of {NUM_HANDS} or {PREFLOP_HANDS}; "
                f"got {self.hand_dim}"
            )
        assert self.street.shape == (N,)
        assert self.to_act.shape == (N,)
        assert self.board.shape == (N, 5)
        assert self.beliefs.dim() == 2
        assert self.beliefs.shape[0] == N
        assert self.beliefs.shape[1] % self.hand_dim == 0

    @property
    def num_players(self) -> int:
        return self.beliefs.shape[1] // self.hand_dim

    def __len__(self) -> int:
        """Get batch size."""
        return self.context.shape[0]

    def __getitem__(self, index: torch.Tensor | slice | int) -> "MLPFeatures":
        """Index into all features."""
        if isinstance(index, torch.Tensor):
            if index.dtype == torch.bool:
                return MLPFeatures(
                    context=self.context[index],
                    street=self.street[index],
                    to_act=self.to_act[index],
                    board=self.board[index],
                    beliefs=self.beliefs[index],
                    hand_dim=self.hand_dim,
                    source_rows=(
                        None
                        if self.source_rows is None
                        else self.source_rows[index].contiguous()
                    ),
                )
            ctx, street, to_act, board, beliefs = _index_all(
                self.context, self.street, self.to_act, self.board, self.beliefs, index
            )
            return MLPFeatures(
                context=ctx,
                street=street,
                to_act=to_act,
                board=board,
                beliefs=beliefs,
                hand_dim=self.hand_dim,
                source_rows=(
                    None
                    if self.source_rows is None
                    else self.source_rows[index].contiguous()
                ),
            )
        return MLPFeatures(
            context=self.context[index],
            street=self.street[index],
            to_act=self.to_act[index],
            board=self.board[index],
            beliefs=self.beliefs[index],
            hand_dim=self.hand_dim,
            source_rows=(
                None if self.source_rows is None else self.source_rows[index]
            ),
        )

    def __setitem__(
        self, index: torch.Tensor | slice | int, value: "MLPFeatures"
    ) -> None:
        """Set features at index."""
        self.context[index] = value.context
        self.street[index] = value.street.to(self.street.dtype)
        self.to_act[index] = value.to_act.to(self.to_act.dtype)
        self.board[index] = value.board
        self.beliefs[index] = value.beliefs

    def to(self, device: torch.device) -> "MLPFeatures":
        return MLPFeatures(
            context=self.context.to(device),
            street=self.street.to(device),
            to_act=self.to_act.to(device),
            board=self.board.to(device),
            beliefs=self.beliefs.to(device),
            hand_dim=self.hand_dim,
            source_rows=None if self.source_rows is None else self.source_rows.to(device),
        )

    def clone(self) -> "MLPFeatures":
        return MLPFeatures(
            context=self.context.clone(),
            street=self.street.clone(),
            to_act=self.to_act.clone(),
            board=self.board.clone(),
            beliefs=self.beliefs.clone(),
            hand_dim=self.hand_dim,
            source_rows=None if self.source_rows is None else self.source_rows.clone(),
        )

    @classmethod
    def cat(cls, features_list: list["MLPFeatures"]) -> "MLPFeatures":
        """Concatenate a list of MLPFeatures objects."""
        if not features_list:
            raise ValueError("Cannot concatenate an empty list of features.")
        hand_dim = features_list[0].hand_dim
        if any(f.hand_dim != hand_dim for f in features_list):
            raise ValueError("Cannot concatenate features with different hand_dim.")

        return cls(
            context=torch.cat([f.context for f in features_list], dim=0),
            street=torch.cat([f.street for f in features_list], dim=0),
            to_act=torch.cat([f.to_act for f in features_list], dim=0),
            board=torch.cat([f.board for f in features_list], dim=0),
            beliefs=torch.cat([f.beliefs for f in features_list], dim=0),
            hand_dim=hand_dim,
            source_rows=(
                None
                if any(f.source_rows is None for f in features_list)
                else torch.cat([f.source_rows for f in features_list], dim=0)
            ),
        )

    def permute_suits(
        self,
        suit_permutations: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Permute the suits of board cards and beliefs.

        Args:
            generator: Random generator for reproducibility.

        Returns:
            Suit permutation indices.
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

        if self.hand_dim == PREFLOP_HANDS:
            return suit_permutations

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

        player_beliefs = self.beliefs.view(batch_size, self.num_players, NUM_HANDS)
        remapped = torch.gather(
            player_beliefs,
            2,
            inverse_remap[:, None, :].expand(-1, self.num_players, -1),
        )
        self.beliefs[:] = remapped.reshape(batch_size, -1)

        return suit_permutations
