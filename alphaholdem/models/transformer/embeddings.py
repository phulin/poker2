"""Embedding modules for variable-length transformer poker model."""

from __future__ import annotations

import torch
import torch.nn as nn

from alphaholdem.models.transformer.state_encoder import TransformerStateEncoder
from alphaholdem.models.transformer.tokens import Context, Special

from ...utils.profiling import profile


class CardEmbedding(nn.Module):
    """Produce embeddings for card tokens."""

    def __init__(self, num_bet_bins: int, d_model: int = 256) -> None:
        super().__init__()
        self.num_bet_bins = num_bet_bins
        self.d_model = d_model

        self.rank_emb = nn.Embedding(13, d_model)
        self.suit_emb = nn.Embedding(4, d_model)
        self.street_emb = nn.Embedding(len(TransformerStateEncoder.STREETS), d_model)

    @profile
    def forward(
        self,
        token_ids: torch.Tensor,
        card_ranks: torch.Tensor,
        card_suits: torch.Tensor,
        card_streets: torch.Tensor,
    ) -> torch.Tensor:
        """Return embeddings for card positions with zeros elsewhere."""

        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        dtype = self.rank_emb.weight.dtype
        embeddings = torch.zeros(
            batch_size, seq_len, self.d_model, device=device, dtype=dtype
        )
        card_offset = TransformerStateEncoder.get_card_token_offset(self.num_bet_bins)
        card_mask = (token_ids >= card_offset) & (token_ids < card_offset + 52)

        if not card_mask.any():
            return embeddings

        rows, cols = torch.where(card_mask)
        ranks = card_ranks[rows, cols]
        suits = card_suits[rows, cols]
        streets = card_streets[rows, cols]

        card_embed = (
            self.rank_emb(ranks) + self.suit_emb(suits) + self.street_emb(streets)
        ).to(embeddings.dtype)
        embeddings[rows, cols] = card_embed
        return embeddings


class ActionEmbedding(nn.Module):
    """Produce embeddings describing historical actions."""

    def __init__(self, num_bet_bins: int, d_model: int = 256) -> None:
        super().__init__()
        self.num_bet_bins = num_bet_bins
        self.d_model = d_model

        self.actor_emb = nn.Embedding(2, d_model)
        self.street_emb = nn.Embedding(len(TransformerStateEncoder.STREETS), d_model)
        self.action_type_emb = nn.Embedding(num_bet_bins, d_model)
        self.legal_mask_mlp = nn.Sequential(
            nn.Linear(num_bet_bins, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    @profile
    def forward(
        self,
        token_ids: torch.Tensor,
        action_actors: torch.Tensor,
        action_streets: torch.Tensor,
        action_legal_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Return embeddings for variable-length action history."""

        batch_size, seq_len = token_ids.shape
        device = token_ids.device
        dtype = self.actor_emb.weight.dtype
        embeddings = torch.zeros(
            batch_size, seq_len, self.d_model, device=device, dtype=dtype
        )

        action_offset = TransformerStateEncoder.get_action_token_offset(
            self.num_bet_bins
        )
        action_mask = (token_ids >= action_offset) & (
            token_ids < action_offset + self.num_bet_bins
        )

        if not action_mask.any():
            return embeddings

        rows, cols = torch.where(action_mask)
        actors = action_actors[rows, cols].long().clamp(min=0, max=1)
        streets = action_streets[rows, cols].long()
        action_ids = token_ids[rows, cols] - action_offset
        legal_masks = action_legal_masks[rows, cols]

        action_embed = (
            self.actor_emb(actors)
            + self.street_emb(streets)
            + self.action_type_emb(action_ids)
            + self.legal_mask_mlp(legal_masks)
        ).to(embeddings.dtype)
        embeddings[rows, cols] = action_embed
        return embeddings


class ContextEmbedding(nn.Module):
    """Embed CLS/context tokens and street markers."""

    def __init__(self, num_bet_bins: int, d_model: int = 256) -> None:
        super().__init__()
        self.num_bet_bins = num_bet_bins
        self.d_model = d_model

        self.special_embedding = nn.Embedding(Special.NUM_SPECIAL.value, d_model)
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

    @profile
    def forward(
        self, token_ids: torch.Tensor, context_features: torch.Tensor
    ) -> torch.Tensor:
        """Return embeddings for all special tokens in the sequence."""

        batch_size, seq_len = token_ids.shape
        device = token_ids.device
        dtype = self.special_embedding.weight.dtype
        embeddings = torch.zeros(
            batch_size, seq_len, self.d_model, device=device, dtype=dtype
        )

        special_offset = TransformerStateEncoder.get_special_token_offset(
            self.num_bet_bins
        )
        special_ids = token_ids - special_offset
        special_mask = (token_ids >= special_offset) & (
            token_ids < special_offset + Special.NUM_SPECIAL.value
        )

        if special_mask.any():
            special_embed = self.special_embedding(special_ids[special_mask]).to(
                embeddings.dtype
            )
            embeddings[special_mask] = special_embed

        # CLS token augments: blinds and hero seat flag live in the first 3 dims.
        cls_id = special_offset + Special.CLS.value
        cls_mask = token_ids == cls_id
        if cls_mask.any():
            cls_features = context_features[cls_mask][:, :3]
            cls_embed = self.cls_mlp(cls_features).to(embeddings.dtype)
            embeddings[cls_mask] += cls_embed

        # Dynamic context token uses the standard 10-feature MLP.
        context_id = special_offset + Special.CONTEXT.value
        context_mask = token_ids == context_id
        if context_mask.any():
            ctx_features = context_features[context_mask]
            context_embed = self.context_mlp(ctx_features).to(embeddings.dtype)
            embeddings[context_mask] += context_embed

        return embeddings


def combine_embeddings(
    card_embedding: CardEmbedding,
    action_embedding: ActionEmbedding,
    context_embedding: ContextEmbedding,
    data,
) -> torch.Tensor:
    """Utility to gather and sum embedding components."""

    cards = card_embedding(
        data.token_ids,
        data.card_ranks,
        data.card_suits,
        data.card_streets,
    )
    actions = action_embedding(
        data.token_ids, data.action_actors, data.action_streets, data.action_legal_masks
    )
    context = context_embedding(data.token_ids, data.context_features)
    return cards + actions + context
