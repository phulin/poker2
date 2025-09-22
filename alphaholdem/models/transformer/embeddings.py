"""Embedding modules for variable-length transformer poker model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from ...utils.profiling import profile
from .tokens import (
    GAME_INDEX,
    Context,
    Game,
    Special,
    get_action_token_id_offset,
    get_card_token_id_offset,
    get_special_token_id_offset,
)

if TYPE_CHECKING:  # pragma: no cover - import guarded for type checkers only
    from .structured_embedding_data import StructuredEmbeddingData

FOURIER_FEATURES = 5 * 2


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
        self.game_mlp = nn.Sequential(
            nn.Linear(5, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.register_buffer(
            "context_fourier",
            torch.pi * (2 ** torch.arange(0, FOURIER_FEATURES / 2)),
            persistent=False,
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(Context.NUM_CONTEXT.value * (1 + FOURIER_FEATURES), d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    @profile
    def forward(self, data: StructuredEmbeddingData) -> torch.Tensor:
        """Return fused embeddings for all tokens in the batch."""

        N = data.token_ids.shape[0]
        token_ids = data.token_ids

        # Map padded positions to the dedicated padding row in the embedding table.
        padding_mask = token_ids < 0
        padded_ids = torch.where(
            padding_mask,
            self.padding_idx,
            token_ids,
        )

        embeddings = self.base_embedding(padded_ids.int()) + self.street_emb(
            data.token_streets.int()
        )

        special_offset = get_special_token_id_offset()
        card_offset = get_card_token_id_offset()
        action_offset = get_action_token_id_offset()

        # Card token contributions (rank + suit + street)
        card_mask = (padded_ids >= card_offset) & (padded_ids < card_offset + 52)
        if card_mask.any():
            rows, cols = torch.where(card_mask)
            ranks = data.card_ranks[rows, cols].clamp(min=0, max=12)
            suits = data.card_suits[rows, cols].clamp(min=0, max=3)
            embeddings[rows, cols] += self.card_rank_emb(
                ranks.int()
            ) + self.card_suit_emb(suits.int())

        # Action token contributions (actor + street + action id + legal mask projection)
        action_mask = (padded_ids >= action_offset) & (
            padded_ids < action_offset + self.num_bet_bins
        )
        if action_mask.any():
            rows, cols = torch.where(action_mask)
            actors = data.action_actors[rows, cols].clamp(min=0, max=1)
            action_ids = padded_ids[rows, cols] - action_offset
            legal_masks = data.action_legal_masks[rows, cols]
            action_embed = (
                self.action_actor_emb(actors.int())
                + self.action_type_emb(action_ids.int())
                + self.legal_mask_mlp(legal_masks.float())
            )
            embeddings[rows, cols] += action_embed

        # GAME token always at index 1 - process raw features
        raw_game_features_float = data.context_features[
            :, GAME_INDEX, : Game.NUM_RAW_GAME.value
        ].float()  # Convert int16 to float

        # Extract raw values
        sb_value = raw_game_features_float[:, Game.SB.value]
        bb_value = raw_game_features_float[:, Game.BB.value]
        hero_position = raw_game_features_float[:, Game.HERO_POSITION.value]
        scale_value = 100.0 * bb_value.float()

        # Create game features tensor with scaled values
        game_features = torch.zeros(
            raw_game_features_float.shape[0],
            Game.NUM_GAME.value,
            device=raw_game_features_float.device,
            dtype=raw_game_features_float.dtype,
        )
        game_features[:, Game.SB.value] = sb_value
        game_features[:, Game.BB.value] = bb_value
        game_features[:, Game.HERO_POSITION.value] = hero_position

        # Check for division by zero and add small epsilon
        scale_value_safe = torch.where(scale_value == 0, 1e-8, scale_value)
        game_features[:, Game.SCALED_BB.value] = bb_value / scale_value_safe
        game_features[:, Game.SCALED_SB.value] = sb_value / scale_value_safe

        embeddings[:, GAME_INDEX] += self.game_mlp(game_features)

        context_id = special_offset + Special.CONTEXT.value
        context_mask = padded_ids == context_id
        if context_mask.any():
            # Process raw context features: convert to float and apply scaling
            raw_ctx_features_float = data.context_features[context_mask].float()

            # Get the batch indices for the context tokens
            context_batch_indices = torch.where(context_mask)[0]

            # Extract scaling values for the specific context tokens
            context_bb_value = bb_value[context_batch_indices]
            context_scale_value = scale_value[context_batch_indices]

            # Apply scaling factors and compute derived features
            scaled_ctx_features = self._process_context_features(
                raw_ctx_features_float, context_bb_value, context_scale_value
            )

            # Expand context features to broadcast to FOURIER_FEATURES features
            ctx_features = scaled_ctx_features.view(-1, Context.NUM_CONTEXT.value, 1)
            # 2^k * pi * x
            ctx_features_fourier = ctx_features * self.context_fourier.view(
                1, 1, FOURIER_FEATURES // 2
            )
            # sin/cos(2^k * pi * x)
            ctx_features_fourier_sin = torch.sin(ctx_features_fourier)
            ctx_features_fourier_cos = torch.cos(ctx_features_fourier)

            # Concatenate original features with Fourier features
            ctx_features_all = torch.cat(
                [ctx_features, ctx_features_fourier_sin, ctx_features_fourier_cos],
                dim=-1,
            )
            context_embed = self.context_mlp(
                ctx_features_all.view(
                    -1, Context.NUM_CONTEXT.value * (1 + FOURIER_FEATURES)
                )
            )
            embeddings[context_mask] += context_embed

        # Ensure explicit zeros for padded tokens after augmentations.
        if padding_mask.any():
            embeddings.masked_fill_(padding_mask.unsqueeze(-1), 0)

        return embeddings

    def _process_context_features(
        self,
        raw_features: torch.Tensor,
        bb_value: torch.Tensor,
        scale_value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process raw context features by applying scaling and computing derived features.

        Args:
            raw_features: Raw context features [batch_size, NUM_RAW_CONTEXT] as float
            bb_value: Big blind value tensor [batch_size]
            scale_value: Scale value tensor [batch_size] (100 * bb_value)

        Returns:
            Processed context features [batch_size, NUM_CONTEXT] with scaling and derived features
        """
        batch_size = raw_features.shape[0]
        processed_features = torch.zeros(
            batch_size,
            Context.NUM_CONTEXT.value,
            device=raw_features.device,
            dtype=raw_features.dtype,
        )

        # Extract raw features
        pot = raw_features[:, Context.POT.value]
        stack_p0 = raw_features[:, Context.STACK_P0.value]
        stack_p1 = raw_features[:, Context.STACK_P1.value]
        committed_p0 = raw_features[:, Context.COMMITTED_P0.value]
        committed_p1 = raw_features[:, Context.COMMITTED_P1.value]
        position = raw_features[:, Context.POSITION.value]
        actions_round = raw_features[:, Context.ACTIONS_ROUND.value]
        min_raise = raw_features[:, Context.MIN_RAISE.value]
        bet_to_call = raw_features[:, Context.BET_TO_CALL.value]

        # no scaling on these 2 (integers)
        processed_features[:, Context.POSITION.value] = position
        processed_features[:, Context.ACTIONS_ROUND.value] = actions_round

        # Store scaled raw features - add safety checks for division by zero
        scale_value_safe = torch.where(
            scale_value == 0, torch.ones_like(scale_value), scale_value
        )
        processed_features[:, Context.POT.value] = pot / scale_value_safe
        processed_features[:, Context.STACK_P0.value] = stack_p0 / scale_value_safe
        processed_features[:, Context.STACK_P1.value] = stack_p1 / scale_value_safe
        processed_features[:, Context.COMMITTED_P0.value] = (
            committed_p0 / scale_value_safe
        )
        processed_features[:, Context.COMMITTED_P1.value] = (
            committed_p1 / scale_value_safe
        )
        processed_features[:, Context.MIN_RAISE.value] = min_raise / scale_value_safe
        processed_features[:, Context.BET_TO_CALL.value] = (
            bet_to_call / scale_value_safe
        )

        # Compute derived features using the same BB value - add safety checks
        bb_value_safe = torch.where(bb_value == 0, torch.ones_like(bb_value), bb_value)
        pot_safe = torch.where(pot == 0, torch.ones_like(pot), pot)
        processed_features[:, Context.EFFECTIVE_STACK_P0.value] = (
            stack_p0 / bb_value_safe
        )
        processed_features[:, Context.EFFECTIVE_STACK_P1.value] = (
            stack_p1 / bb_value_safe
        )
        processed_features[:, Context.SPR_P0.value] = stack_p0 / pot_safe
        processed_features[:, Context.SPR_P1.value] = stack_p1 / pot_safe

        return processed_features


def combine_embeddings(
    fused_embedding: PokerFusedEmbedding,
    *_unused,
    data,
) -> torch.Tensor:
    """Backward-compatible helper to obtain fused embeddings."""

    return fused_embedding(data)


__all__ = ["PokerFusedEmbedding", "combine_embeddings"]
