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

        base_dtype = self.rank_emb.weight.dtype
        embeddings = torch.zeros(
            batch_size, seq_len, self.d_model, device=device, dtype=base_dtype
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
        ).to(base_dtype)
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
        base_dtype = self.actor_emb.weight.dtype
        embeddings = torch.zeros(
            batch_size, seq_len, self.d_model, device=device, dtype=base_dtype
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
        ).to(base_dtype)
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
        self._cls_cache: dict[
            tuple[str, torch.dtype, torch.dtype],
            dict[tuple[float, float, float], torch.Tensor],
        ] = {}

    @profile
    def forward(
        self, token_ids: torch.Tensor, context_features: torch.Tensor
    ) -> torch.Tensor:
        """Return embeddings for all special tokens in the sequence."""

        batch_size, seq_len = token_ids.shape
        device = token_ids.device
        base_dtype = self.special_embedding.weight.dtype
        embeddings = torch.zeros(
            batch_size, seq_len, self.d_model, device=device, dtype=base_dtype
        )

        special_offset = TransformerStateEncoder.get_special_token_offset(
            self.num_bet_bins
        )
        special_ids = token_ids - special_offset
        special_mask = (token_ids >= special_offset) & (
            token_ids < special_offset + Special.NUM_SPECIAL.value
        )

        if special_mask.any():
            embeddings[special_mask] = self.special_embedding(
                special_ids[special_mask]
            ).to(base_dtype)

        # CLS token augments: blinds and hero seat flag live in the first 3 dims.
        cls_id = special_offset + Special.CLS.value
        cls_mask = token_ids == cls_id
        if cls_mask.any():
            cls_features = context_features[cls_mask][:, :3]
            cache_key = (
                cls_features.device.type,
                cls_features.dtype,
                base_dtype,
            )
            device_cache = self._cls_cache.setdefault(cache_key, {})
            cached_outputs = []
            for feat in cls_features:
                key = tuple(float(x) for x in feat.tolist())
                cached = device_cache.get(key)
                if cached is None:
                    cached = self.cls_mlp(feat.unsqueeze(0)).to(base_dtype).squeeze(0)
                    device_cache[key] = cached
                cached_outputs.append(cached)
            cls_embeddings = torch.stack(cached_outputs, dim=0)
            embeddings[cls_mask] += cls_embeddings

        # Dynamic context token uses the standard 10-feature MLP.
        context_id = special_offset + Special.CONTEXT.value
        context_mask = token_ids == context_id
        if context_mask.any():
            ctx_features = context_features[context_mask]
            embeddings[context_mask] += self.context_mlp(ctx_features).to(base_dtype)

        return embeddings


def combine_embeddings(
    card_embedding: CardEmbedding,
    action_embedding: ActionEmbedding,
    context_embedding: ContextEmbedding,
    data,
    gather_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Utility to gather and sum embedding components."""

    if gather_indices is None:
        token_ids = data.token_ids
        card_ranks = data.card_ranks
        card_suits = data.card_suits
        card_streets = data.card_streets
        action_actors = data.action_actors
        action_streets = data.action_streets
        action_legal_masks = data.action_legal_masks
        context_features = data.context_features
    else:
        gather_indices = gather_indices.long()
        token_ids = torch.gather(data.token_ids, 1, gather_indices)
        card_ranks = torch.gather(data.card_ranks, 1, gather_indices)
        card_suits = torch.gather(data.card_suits, 1, gather_indices)
        card_streets = torch.gather(data.card_streets, 1, gather_indices)
        action_actors = torch.gather(data.action_actors, 1, gather_indices)
        action_streets = torch.gather(data.action_streets, 1, gather_indices)
        action_legal_masks = torch.gather(
            data.action_legal_masks,
            1,
            gather_indices.unsqueeze(-1).expand(
                -1, -1, data.action_legal_masks.size(-1)
            ),
        )
        context_features = torch.gather(
            data.context_features,
            1,
            gather_indices.unsqueeze(-1).expand(-1, -1, data.context_features.size(-1)),
        )

    cards = card_embedding(token_ids, card_ranks, card_suits, card_streets)
    actions = action_embedding(
        token_ids, action_actors, action_streets, action_legal_masks
    )
    context = context_embedding(token_ids, context_features)
    return cards + actions + context
