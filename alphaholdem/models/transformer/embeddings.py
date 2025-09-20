"""Embedding modules for variable-length transformer poker model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .tokens import (
    Context,
    Special,
    get_action_token_id_offset,
    get_card_token_id_offset,
    get_special_token_id_offset,
)

from ...utils.profiling import profile

if TYPE_CHECKING:  # pragma: no cover - import guarded for type checkers only
    from .structured_embedding_data import StructuredEmbeddingData


class PokerFusedEmbedding(nn.Module):
    """Single fused embedding module covering all transformer token types."""

    def __init__(self, num_bet_bins: int, d_model: int = 256) -> None:
        super().__init__()
        self.num_bet_bins = num_bet_bins
        self.d_model = d_model

        self.vocab_size = get_action_token_id_offset() + num_bet_bins
        self.padding_idx = self.vocab_size  # sentinel row for padded tokens

        self.base_embedding = nn.Embedding(
            self.vocab_size + 1, d_model, padding_idx=self.padding_idx
        )
        self.street_emb = nn.Embedding(4, d_model)

        # Card components
        self.card_rank_emb = nn.Embedding(13, d_model)
        self.card_suit_emb = nn.Embedding(4, d_model)

        # Action components
        self.action_actor_emb = nn.Embedding(2, d_model)
        self.action_type_emb = nn.Embedding(num_bet_bins, d_model)
        self.legal_mask_mlp = nn.Sequential(
            nn.Linear(num_bet_bins, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Context components for CLS and dynamic context tokens
        self.cls_mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(Context.NUM_CONTEXT.value, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.register_buffer(
            "_padding_idx_tensor",
            torch.tensor(self.padding_idx, dtype=torch.long),
            persistent=False,
        )

    @profile
    def forward(self, data: "StructuredEmbeddingData") -> torch.Tensor:
        """Return fused embeddings for all tokens in the batch."""

        token_ids = data.token_ids

        # Map padded positions to the dedicated padding row in the embedding table.
        padding_mask = token_ids < 0
        padded_ids = torch.where(
            padding_mask,
            self._padding_idx_tensor.expand_as(token_ids),
            token_ids,
        )

        embeddings = self.base_embedding(padded_ids) + self.street_emb(
            data.token_streets
        )
        dtype = embeddings.dtype

        special_offset = get_special_token_id_offset()
        card_offset = get_card_token_id_offset()
        action_offset = get_action_token_id_offset()

        # Card token contributions (rank + suit + street)
        card_mask = (padded_ids >= card_offset) & (padded_ids < card_offset + 52)
        if card_mask.any():
            rows, cols = torch.where(card_mask)
            ranks = data.card_ranks[rows, cols].clamp(min=0, max=12)
            suits = data.card_suits[rows, cols].clamp(min=0, max=3)
            card_embed = (self.card_rank_emb(ranks) + self.card_suit_emb(suits)).to(
                dtype
            )
            embeddings[rows, cols] += card_embed

        # Action token contributions (actor + street + action id + legal mask projection)
        action_mask = (padded_ids >= action_offset) & (
            padded_ids < action_offset + self.num_bet_bins
        )
        if action_mask.any():
            rows, cols = torch.where(action_mask)
            actors = data.action_actors[rows, cols].clamp(min=0, max=1)
            action_ids = padded_ids[rows, cols] - action_offset
            legal_masks = data.action_legal_masks[rows, cols].to(embeddings.dtype)
            action_embed = (
                self.action_actor_emb(actors)
                + self.action_type_emb(action_ids)
                + self.legal_mask_mlp(legal_masks)
            ).to(dtype)
            embeddings[rows, cols] += action_embed

        # Special tokens: CLS + dynamic context augmentations
        # CLS token always at index 0
        embeddings[:, 0] += self.cls_mlp(data.context_features[:, 0, :3])

        context_id = special_offset + Special.CONTEXT.value
        context_mask = padded_ids == context_id
        if context_mask.any():
            ctx_features = data.context_features[context_mask]
            context_embed = self.context_mlp(ctx_features).to(dtype)
            embeddings[context_mask] += context_embed

        # Ensure explicit zeros for padded tokens after augmentations.
        if padding_mask.any():
            embeddings.masked_fill_(padding_mask.unsqueeze(-1), 0)

        return embeddings


def combine_embeddings(
    fused_embedding: PokerFusedEmbedding,
    *_unused,
    data,
) -> torch.Tensor:
    """Backward-compatible helper to obtain fused embeddings."""

    return fused_embedding(data)


__all__ = ["PokerFusedEmbedding", "combine_embeddings"]
