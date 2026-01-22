"""Tests for variable-length transformer model and embeddings."""

import torch

from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.models.transformer.embeddings import (
    PokerFusedEmbedding,
    combine_embeddings,
)
from p2.models.transformer.poker_transformer import PokerTransformerV1
from p2.models.transformer.structured_embedding_data import (
    StructuredEmbeddingData,
)
from p2.models.transformer.token_sequence_builder import TokenSequenceBuilder
from p2.models.transformer.tokens import (
    Context,
    Special,
    get_action_token_id_offset,
    get_card_token_id_offset,
    get_special_token_id_offset,
)

BET_BINS = [0.5, 1.0, 1.5, 2.0]
NUM_BET_BINS = len(BET_BINS) + 3


def _build_env(device: torch.device) -> HUNLTensorEnv:
    return HUNLTensorEnv(
        num_envs=2,
        starting_stack=20000,
        sb=50,
        bb=100,
        default_bet_bins=BET_BINS,
        device=device,
    )


class TestTransformerStateEncoder:
    def test_sequence_layout_and_lengths(self):
        device = torch.device("cpu")
        env = _build_env(device)
        env.reset()
        encoder = TokenSequenceBuilder(
            tensor_env=env,
            sequence_length=100,
            bet_bins=BET_BINS,
            device=device,
            float_dtype=torch.float32,
        )
        idxs = torch.arange(env.N, device=device)

        encoder.add_game(idxs)
        encoder.add_context(idxs)
        encoder.add_street(idxs, torch.zeros_like(idxs))
        encoder.add_card(idxs, torch.zeros_like(idxs))
        encoder.add_card(idxs, torch.zeros_like(idxs))
        encoder.add_action(
            idxs,
            torch.zeros_like(idxs),
            torch.zeros_like(idxs),
            torch.zeros(
                2, NUM_BET_BINS, dtype=torch.bool
            ),  # [batch_size, num_bet_bins]
            torch.zeros_like(idxs),
            torch.zeros_like(idxs),  # token_streets
        )

        data = encoder.encode_tensor_states(player=0, idxs=idxs)

        # Length metadata must match number of valid tokens.
        valid_counts = (data.token_ids >= 0).sum(dim=1)
        assert torch.equal(data.lengths, valid_counts)

        special_offset = get_special_token_id_offset()
        # First non-negative token is GAME; position can vary if CLS at 0
        first_valid = (data.token_ids >= 0).float().argmax(dim=1)
        for i in range(data.token_ids.shape[0]):
            pos = int(first_valid[i].item())
            # Allow either CLS at 0 followed by GAME, or GAME first
            assert data.token_ids[i, pos] in (
                special_offset + Special.GAME.value,
                special_offset + Special.CLS.value,
            )
        # Ensure we emit at least one street marker (preflop) per state.
        preflop_token = special_offset + Special.STREET_PREFLOP.value
        assert torch.any(data.token_ids == preflop_token)


class TestEmbeddings:
    def setup_method(self):
        self.device = torch.device("cpu")
        self.env = _build_env(self.device)
        self.env.reset()
        self.encoder = TokenSequenceBuilder(
            tensor_env=self.env,
            sequence_length=100,
            bet_bins=BET_BINS,
            device=self.device,
            float_dtype=torch.float32,
        )
        idxs = torch.arange(self.env.N, device=self.device)

        # Build a proper token sequence
        self.encoder.add_game(idxs)
        self.encoder.add_context(idxs)
        self.encoder.add_street(idxs, torch.zeros_like(idxs))

        self.data = self.encoder.encode_tensor_states(player=0, idxs=idxs)
        self.num_bet_bins = NUM_BET_BINS
        self.d_model = 64
        self.embedding = PokerFusedEmbedding(self.num_bet_bins, d_model=self.d_model)
        self.embedding.eval()  # disable dropout for deterministic assertions

    def _clone_data(self) -> StructuredEmbeddingData:
        return StructuredEmbeddingData(
            token_ids=self.data.token_ids.clone(),
            token_streets=self.data.token_streets.clone(),
            card_ranks=self.data.card_ranks.clone(),
            card_suits=self.data.card_suits.clone(),
            action_actors=self.data.action_actors.clone(),
            action_legal_masks=self.data.action_legal_masks.clone(),
            action_amounts=self.data.action_amounts.clone(),
            context_features=self.data.context_features.clone(),
            lengths=self.data.lengths.clone(),
        )

    def test_fused_embedding_respects_padding(self):
        result = self.embedding(self.data)
        assert result.shape == (
            self.data.batch_size,
            self.data.seq_len,
            self.d_model,
        )
        padding_mask = self.data.token_ids < 0
        assert torch.count_nonzero(result[padding_mask]) == 0
        assert torch.count_nonzero(result[~padding_mask]) > 0

    def test_context_features_affect_embeddings(self):
        special_offset = get_special_token_id_offset()
        context_id = special_offset + Special.CONTEXT.value
        context_mask = self.data.token_ids == context_id
        assert torch.any(context_mask)

        base = self.embedding(self.data).detach()

        modified = self._clone_data()
        modified.context_features[context_mask] += 1

        updated = self.embedding(modified)
        assert not torch.allclose(base[context_mask], updated[context_mask])

    def test_combine_embeddings_wrapper_calls_fused_module(self):
        combined = combine_embeddings(self.embedding, data=self.data)
        direct = self.embedding(self.data)
        assert torch.allclose(combined, direct)


class TestPokerTransformerV1:
    def test_forward_with_real_environment(self):
        device = torch.device("cpu")
        env = _build_env(device)
        env.reset()
        encoder = TokenSequenceBuilder(
            tensor_env=env,
            sequence_length=100,
            bet_bins=BET_BINS,
            device=device,
            float_dtype=torch.float32,
        )
        idxs = torch.arange(env.N, device=device)

        # Build a proper token sequence first
        encoder.add_game(idxs)
        encoder.add_street(idxs, torch.zeros_like(idxs))
        encoder.add_card(idxs, torch.zeros_like(idxs))
        encoder.add_card(idxs, torch.zeros_like(idxs))
        encoder.add_context(idxs)

        structured = encoder.encode_tensor_states(player=0, idxs=idxs)

        model = PokerTransformerV1(
            d_model=128,
            n_layers=2,
            n_heads=4,
            num_bet_bins=NUM_BET_BINS,
            max_sequence_length=100,
            dropout=0.1,
            use_gradient_checkpointing=False,
        )

        model.eval()
        with torch.no_grad():
            outputs = model(structured)

        assert outputs.policy_logits.shape == (
            structured.batch_size,
            7,
        )
        assert outputs.value.shape == (structured.batch_size,)

    def test_factory_creates_state_encoder_and_model(self):
        device = torch.device("cpu")
        env = _build_env(device)
        encoder = TokenSequenceBuilder(
            tensor_env=env,
            sequence_length=100,
            bet_bins=BET_BINS,
            device=device,
            float_dtype=torch.float32,
        )
        assert isinstance(encoder, TokenSequenceBuilder)

        model = PokerTransformerV1(
            d_model=64,
            n_layers=1,
            n_heads=4,
            num_bet_bins=NUM_BET_BINS,
            dropout=0.1,
            max_sequence_length=100,
            use_gradient_checkpointing=False,
        )
        inner_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        assert isinstance(inner_model, PokerTransformerV1)

    def test_attention_mask_respects_variable_lengths(self):
        batch_size, seq_len = 3, 12
        num_bet_bins = NUM_BET_BINS
        special_offset = get_special_token_id_offset()
        card_offset = get_card_token_id_offset()
        action_offset = get_action_token_id_offset()

        token_ids = torch.full((batch_size, seq_len), -1)
        # Build sequences of different lengths
        raw_lengths = torch.tensor([5, 7, 3])
        for i, length in enumerate(raw_lengths):
            token_ids[i, 0] = special_offset + Special.GAME.value
            token_ids[i, 1] = special_offset + Special.CONTEXT.value
            if length > 2:
                token_ids[i, 2] = special_offset + Special.STREET_PREFLOP.value
            if length > 3:
                token_ids[i, 3] = card_offset
            if length > 4:
                token_ids[i, 4] = card_offset + 1
            if length > 5:
                token_ids[i, 5] = action_offset

        zeros = torch.zeros_like(token_ids)
        legal = torch.zeros(batch_size, seq_len, num_bet_bins, dtype=torch.bool)
        ctx = torch.zeros(
            batch_size, seq_len, Context.NUM_RAW_CONTEXT.value, dtype=torch.int16
        )
        lengths = (token_ids >= 0).sum(dim=1)
        data = StructuredEmbeddingData(
            token_ids=token_ids,
            token_streets=zeros,
            card_ranks=zeros,
            card_suits=zeros,
            action_actors=zeros,
            action_legal_masks=legal,
            action_amounts=zeros.to(torch.int32),
            context_features=ctx,
            lengths=lengths,
        )

        mask = data.attention_mask
        expected = data.token_ids >= 0  # True = allow attention (SDPA semantics)
        assert torch.equal(mask, expected)
        for i, length in enumerate(lengths):
            assert mask[i].sum().item() == int(
                length.item()
            )  # Sum should equal number of valid tokens
