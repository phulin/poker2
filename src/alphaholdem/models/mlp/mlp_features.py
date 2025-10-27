import torch
from dataclasses import dataclass


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
