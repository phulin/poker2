"""Tests for variable-length transformer model and embeddings."""

import torch

from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.factory import ModelFactory
from alphaholdem.models.transformer.embedding_data import StructuredEmbeddingData
from alphaholdem.models.transformer.embeddings import (
    ActionEmbedding,
    CardEmbedding,
    ContextEmbedding,
    combine_embeddings,
)
from alphaholdem.models.transformer.poker_transformer import PokerTransformerV1
from alphaholdem.models.transformer.state_encoder import TransformerStateEncoder
from alphaholdem.models.transformer.tokens import Special


def _build_env(device: torch.device) -> HUNLTensorEnv:
    return HUNLTensorEnv(
        num_envs=2,
        starting_stack=20000,
        sb=50,
        bb=100,
        bet_bins=[0.5, 1.0, 1.5, 2.0],
        device=device,
    )


class TestTransformerStateEncoder:
    def test_sequence_layout_and_lengths(self):
        device = torch.device("cpu")
        env = _build_env(device)
        env.reset()
        encoder = TransformerStateEncoder(env, device)
        idxs = torch.arange(env.N, device=device)

        data = encoder.encode_tensor_states(player=0, idxs=idxs)

        # Length metadata must match number of valid tokens.
        valid_counts = (data.token_ids >= 0).sum(dim=1)
        assert torch.equal(data.lengths, valid_counts)

        special_offset = encoder.get_special_token_offset(env.num_bet_bins)
        assert torch.all(data.token_ids[:, 0] == special_offset + Special.CLS.value)
        assert torch.all(data.token_ids[:, 1] == special_offset + Special.CONTEXT.value)

        # Ensure we emit at least one street marker (preflop) per state.
        preflop_token = special_offset + Special.STREET_PREFLOP.value
        assert torch.any(data.token_ids == preflop_token)


class TestEmbeddings:
    def setup_method(self):
        self.device = torch.device("cpu")
        self.env = _build_env(self.device)
        self.env.reset()
        self.encoder = TransformerStateEncoder(self.env, self.device)
        idxs = torch.arange(self.env.N, device=self.device)
        self.data = self.encoder.encode_tensor_states(player=0, idxs=idxs)
        self.num_bet_bins = self.env.num_bet_bins
        self.d_model = 64

    def test_card_embedding_masks_cards(self):
        embedding = CardEmbedding(self.num_bet_bins, d_model=self.d_model)
        result = embedding(
            self.data.token_ids,
            self.data.card_ranks,
            self.data.card_suits,
            self.data.card_streets,
        )
        assert result.shape == (
            self.data.batch_size,
            self.data.seq_len,
            self.d_model,
        )
        card_offset = TransformerStateEncoder.get_card_token_offset(self.num_bet_bins)
        card_mask = (self.data.token_ids >= card_offset) & (
            self.data.token_ids < card_offset + 52
        )
        assert torch.count_nonzero(result[card_mask]) > 0
        assert torch.count_nonzero(result[~card_mask]) == 0

    def test_action_embedding_nonzero_only_for_actions(self):
        embedding = ActionEmbedding(self.num_bet_bins, d_model=self.d_model)
        result = embedding(
            self.data.token_ids,
            self.data.action_actors,
            self.data.action_streets,
            self.data.action_legal_masks,
        )
        assert result.shape == (
            self.data.batch_size,
            self.data.seq_len,
            self.d_model,
        )
        action_offset = TransformerStateEncoder.get_action_token_offset(
            self.num_bet_bins
        )
        action_mask = (self.data.token_ids >= action_offset) & (
            self.data.token_ids < action_offset + self.num_bet_bins
        )
        if torch.any(action_mask):
            assert torch.count_nonzero(result[action_mask]) > 0
        assert torch.count_nonzero(result[~action_mask]) == 0

    def test_context_embedding_handles_special_tokens(self):
        embedding = ContextEmbedding(self.num_bet_bins, d_model=self.d_model)
        result = embedding(self.data.token_ids, self.data.context_features)
        assert result.shape == (
            self.data.batch_size,
            self.data.seq_len,
            self.d_model,
        )
        special_offset = TransformerStateEncoder.get_special_token_offset(
            self.num_bet_bins
        )
        special_mask = (self.data.token_ids >= special_offset) & (
            self.data.token_ids < special_offset + Special.NUM_SPECIAL.value
        )
        assert torch.count_nonzero(result[special_mask]) > 0

    def test_combine_embeddings_matches_sequence_shape(self):
        combined = combine_embeddings(
            CardEmbedding(self.num_bet_bins, d_model=self.d_model),
            ActionEmbedding(self.num_bet_bins, d_model=self.d_model),
            ContextEmbedding(self.num_bet_bins, d_model=self.d_model),
            self.data,
        )
        assert combined.shape == (
            self.data.batch_size,
            self.data.seq_len,
            self.d_model,
        )


class TestPokerTransformerV1:
    def test_forward_with_real_environment(self):
        device = torch.device("cpu")
        env = _build_env(device)
        env.reset()
        encoder = TransformerStateEncoder(env, device)
        idxs = torch.arange(env.N, device=device)
        structured = encoder.encode_tensor_states(player=0, idxs=idxs)

        model = PokerTransformerV1(
            d_model=128,
            n_layers=2,
            n_heads=4,
            num_bet_bins=env.num_bet_bins,
            dropout=0.1,
        )

        model.eval()
        with torch.no_grad():
            outputs = model(structured)

        assert outputs["policy_logits"].shape == (
            structured.batch_size,
            env.num_bet_bins,
        )
        assert outputs["value"].shape == (structured.batch_size,)

    def test_factory_creates_state_encoder_and_model(self):
        device = torch.device("cpu")
        env = _build_env(device)
        encoder = ModelFactory.create_state_encoder(
            "transformer", device, tensor_env=env
        )
        assert isinstance(encoder, TransformerStateEncoder)

        model_config = {
            "d_model": 64,
            "n_layers": 1,
            "n_heads": 4,
            "num_bet_bins": env.num_bet_bins,
            "dropout": 0.1,
            "use_auxiliary_loss": False,
            "use_gradient_checkpointing": False,
        }
        model = ModelFactory.create_model("transformer", model_config, device=device)
        inner_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        assert isinstance(inner_model, PokerTransformerV1)

    def test_attention_mask_respects_variable_lengths(self):
        device = torch.device("cpu")
        batch_size, seq_len = 3, 12
        num_bet_bins = 8
        special_offset = TransformerStateEncoder.get_special_token_offset(num_bet_bins)
        card_offset = TransformerStateEncoder.get_card_token_offset(num_bet_bins)
        action_offset = TransformerStateEncoder.get_action_token_offset(num_bet_bins)

        token_ids = torch.full((batch_size, seq_len), -1)
        # Build sequences of different lengths
        raw_lengths = torch.tensor([5, 7, 3])
        for i, length in enumerate(raw_lengths):
            token_ids[i, 0] = special_offset + Special.CLS.value
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
        legal = torch.zeros(batch_size, seq_len, num_bet_bins)
        ctx = torch.zeros(batch_size, seq_len, 10)
        lengths = (token_ids >= 0).sum(dim=1)
        data = StructuredEmbeddingData(
            token_ids=token_ids,
            card_ranks=zeros,
            card_suits=zeros,
            card_streets=zeros,
            action_actors=zeros,
            action_streets=zeros,
            action_legal_masks=legal,
            context_features=ctx,
            lengths=lengths,
        )

        mask = data.attention_mask
        expected = data.token_ids >= 0
        assert torch.equal(mask, expected)
        for i, length in enumerate(lengths):
            assert mask[i].sum().item() == int(length.item())
