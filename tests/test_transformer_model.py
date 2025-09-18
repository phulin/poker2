"""Tests for variable-length transformer model and embeddings."""

import torch
import pytest

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
from alphaholdem.models.transformer.rotary_attention import RotarySelfAttention
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

        for row in range(data.token_ids.shape[0]):
            length = int(data.lengths[row].item())
            last_token = int(data.token_ids[row, length - 1].item())
            assert (
                last_token == special_offset + Special.CONTEXT.value
            ), "Final token should be context snapshot"

        action_offset = TransformerStateEncoder.get_action_token_offset(
            env.num_bet_bins
        )
        action_positions = torch.nonzero(
            (data.token_ids >= action_offset)
            & (data.token_ids < action_offset + env.num_bet_bins)
        )
        for batch_idx, pos in action_positions:
            batch_idx = int(batch_idx)
            pos = int(pos)
            assert (
                data.token_ids[batch_idx, pos - 1]
                == special_offset + Special.CONTEXT.value
            )


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

    def test_rope_cache_reuse_and_expand(self):
        device = torch.device("cpu")
        model = PokerTransformerV1(
            d_model=64,
            n_layers=1,
            n_heads=4,
            num_bet_bins=8,
            dropout=0.0,
        ).to(device)

        dtype = torch.float32
        cos8, sin8 = model._get_rope_cache(8, device, dtype)
        key = (device.type, device.index, dtype)
        cache_entry = model._rope_cache[key]
        assert cache_entry["cos"].shape[0] >= 8

        base_ptr = cache_entry["cos"].untyped_storage().data_ptr()

        cos4, sin4 = model._get_rope_cache(4, device, dtype)
        cache_entry_after_shrink = model._rope_cache[key]
        assert cache_entry_after_shrink["cos"].untyped_storage().data_ptr() == base_ptr
        assert torch.equal(cos4, cache_entry_after_shrink["cos"][:4])
        assert torch.equal(sin4, cache_entry_after_shrink["sin"][:4])

        cos16, sin16 = model._get_rope_cache(16, device, dtype)
        cache_entry_extended = model._rope_cache[key]
        assert cache_entry_extended["cos"].shape[0] >= 16
        assert torch.equal(cos16[:8], cos8)
        assert torch.equal(sin16[:8], sin8)

    def test_rotary_attention_incremental_matches_full(self):
        device = torch.device("cpu")
        d_model = 32
        n_heads = 4
        attn_full = RotarySelfAttention(d_model, n_heads, dropout=0.0).to(device)
        attn_full.eval()

        torch.manual_seed(0)
        x_full = torch.randn(1, 7, d_model, device=device)
        valid_mask_full = torch.ones(1, 7, dtype=torch.bool, device=device)

        model = PokerTransformerV1(
            d_model=d_model,
            n_layers=1,
            n_heads=n_heads,
            num_bet_bins=8,
            dropout=0.0,
        ).to(device)

        cos_table, sin_table = model._get_rope_cache(7, device, x_full.dtype)
        head_dim = d_model // n_heads

        def gather_rotary(start: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
            if length == 0:
                zero = x_full.new_zeros(1, 1, 0, head_dim)
                return zero, zero
            positions = torch.arange(start, start + length, device=device).unsqueeze(0)
            positions = positions.to(torch.long)
            cos_slice = cos_table.index_select(0, positions.view(-1)).view(
                1, length, head_dim
            )
            sin_slice = sin_table.index_select(0, positions.view(-1)).view(
                1, length, head_dim
            )
            return cos_slice.unsqueeze(1), sin_slice.unsqueeze(1)

        rotary_cos_full, rotary_sin_full = gather_rotary(0, 7)
        new_counts_full = torch.tensor([7], device=device)

        with torch.no_grad():
            full_outputs, cache_full, total_lengths_full = attn_full(
                x_full,
                rotary_cos_full,
                rotary_sin_full,
                cache=None,
                capture_cache=True,
                past_lengths=torch.zeros(1, dtype=torch.long, device=device),
                valid_new_mask=valid_mask_full,
                new_token_counts=new_counts_full,
            )

        attn_incremental = RotarySelfAttention(d_model, n_heads, dropout=0.0).to(device)
        attn_incremental.load_state_dict(attn_full.state_dict())
        attn_incremental.eval()

        x_past = x_full[:, :5]
        x_new = x_full[:, 5:]
        valid_past = valid_mask_full[:, :5]
        valid_new = valid_mask_full[:, 5:]

        rotary_cos_past, rotary_sin_past = gather_rotary(0, 5)

        with torch.no_grad():
            past_outputs, cache_past, total_lengths_past = attn_incremental(
                x_past,
                rotary_cos_past,
                rotary_sin_past,
                cache=None,
                capture_cache=True,
                past_lengths=torch.zeros(1, dtype=torch.long, device=device),
                valid_new_mask=valid_past,
                new_token_counts=torch.tensor([5], device=device),
            )

        rotary_cos_new, rotary_sin_new = gather_rotary(
            int(total_lengths_past.item()), x_new.size(1)
        )

        with torch.no_grad():
            new_outputs, cache_new, total_lengths_new = attn_incremental(
                x_new,
                rotary_cos_new,
                rotary_sin_new,
                cache_past,
                capture_cache=True,
                past_lengths=total_lengths_past,
                valid_new_mask=valid_new,
                new_token_counts=torch.tensor([2], device=device),
            )

        combined_outputs = torch.cat([past_outputs, new_outputs], dim=1)

        torch.testing.assert_close(full_outputs, combined_outputs, atol=1e-5, rtol=1e-5)
        assert cache_new is not None
        assert total_lengths_new.tolist() == [7]

    def test_torch_compile_optional(self):
        if not hasattr(torch, "compile"):
            pytest.skip("torch.compile unavailable")

        device = torch.device("cpu")
        env = _build_env(device)
        env.reset()
        encoder = TransformerStateEncoder(env, device)
        idxs = torch.arange(env.N, device=device)
        structured = encoder.encode_tensor_states(player=0, idxs=idxs)

        model = PokerTransformerV1(
            d_model=64,
            n_layers=1,
            n_heads=4,
            num_bet_bins=env.num_bet_bins,
            dropout=0.0,
            use_torch_compile=True,
        ).to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(structured)

        assert outputs["policy_logits"].shape[0] == structured.batch_size
        assert outputs["value"].shape[0] == structured.batch_size

        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()

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

    def test_kv_cache_extension_matches_full_forward(self):
        device = torch.device("cpu")
        env = _build_env(device)
        env.reset()
        encoder = TransformerStateEncoder(env, device)
        idxs = torch.tensor([0], device=device)

        model = PokerTransformerV1(
            d_model=64,
            n_layers=2,
            n_heads=4,
            num_bet_bins=env.num_bet_bins,
            dropout=0.1,
        )
        model.eval()

        first_state = encoder.encode_tensor_states(player=0, idxs=idxs)
        with torch.no_grad():
            cached_outputs = model(first_state)
        kv_cache = cached_outputs.get("kv_cache")
        assert kv_cache is not None

        # Advance environment with a single action to extend the sequence
        env.step_bins(torch.tensor([2, -1], device=device))
        second_state = encoder.encode_tensor_states(player=0, idxs=idxs)

        first_len = int(first_state.lengths[0].item())
        second_state.token_ids[0, :first_len] = first_state.token_ids[0, :first_len]
        second_state.context_features[0, :first_len] = first_state.context_features[
            0, :first_len
        ]

        with torch.no_grad():
            cached_result = model(
                second_state,
                kv_cache=kv_cache,
            )
            full_result = model(second_state)

        assert torch.allclose(
            cached_result["policy_logits"],
            full_result["policy_logits"],
            atol=3e-1,
            rtol=3e-1,
        )
        assert torch.allclose(
            cached_result["value"],
            full_result["value"],
            atol=3e-1,
            rtol=3e-1,
        )

    def test_action_context_alignment(self):
        device = torch.device("cpu")
        env = _build_env(device)
        env.reset()
        encoder = TransformerStateEncoder(env, device)

        env.step_bins(torch.tensor([2, -1], device=device))
        env.step_bins(torch.tensor([1, -1], device=device))

        idxs = torch.tensor([0], device=device)
        data_p0 = encoder.encode_tensor_states(player=0, idxs=idxs)
        data_p1 = encoder.encode_tensor_states(player=1, idxs=idxs)

        def gather_context_tokens(data):
            contexts = []
            tokens = data.token_ids[0]
            features = data.context_features[0]
            length = int(data.lengths[0].item())
            action_offset = encoder.get_action_token_offset(env.num_bet_bins)
            for pos in range(length):
                token = int(tokens[pos].item())
                if action_offset <= token < action_offset + env.num_bet_bins:
                    contexts.append(features[pos - 1].clone())
            return contexts

        observed_p0 = gather_context_tokens(data_p0)
        observed_p1 = gather_context_tokens(data_p1)

        action_context_world = env.action_context[0]
        action_history = env.get_action_history()[0]
        expected_world = []
        for street_idx in range(len(TransformerStateEncoder.STREETS)):
            slots_taken = action_history[street_idx, :, 2, :].any(dim=1)
            for slot_idx, taken in enumerate(slots_taken.tolist()):
                if taken:
                    expected_world.append(
                        action_context_world[street_idx, slot_idx].unsqueeze(0)
                    )

        expected_world = torch.cat(expected_world, dim=0).to(device)
        expected_p0 = encoder._to_hero_context(expected_world, player=0)
        expected_p1 = encoder._to_hero_context(expected_world, player=1)

        assert len(observed_p0) == expected_p0.shape[0]
        assert len(observed_p1) == expected_p1.shape[0]

        for obs, exp in zip(observed_p0, expected_p0):
            torch.testing.assert_close(obs, exp, atol=1e-5, rtol=1e-5)

        for obs, exp in zip(observed_p1, expected_p1):
            torch.testing.assert_close(obs, exp, atol=1e-5, rtol=1e-5)

        # Final context token should reflect current state
        final_token = int(data_p0.token_ids[0, data_p0.lengths[0] - 1].item())
        special_offset = TransformerStateEncoder.get_special_token_offset(
            env.num_bet_bins
        )
        assert final_token == special_offset + Special.CONTEXT.value

        final_context_world = encoder._gather_env_context(idxs)
        expected_final_p0 = encoder._to_hero_context(final_context_world, player=0)
        torch.testing.assert_close(
            data_p0.context_features[0, data_p0.lengths[0] - 1],
            expected_final_p0[0],
            atol=1e-5,
            rtol=1e-5,
        )
        expected_final_p1 = encoder._to_hero_context(final_context_world, player=1)
        torch.testing.assert_close(
            data_p1.context_features[0, data_p1.lengths[0] - 1],
            expected_final_p1[0],
            atol=1e-5,
            rtol=1e-5,
        )

    def test_invariance_to_hidden_cards_and_undealt_deck(self):
        device = torch.device("cpu")
        env = _build_env(device)
        env.reset()
        encoder = TransformerStateEncoder(env, device)
        idxs = torch.tensor([0], device=device)

        model = PokerTransformerV1(
            d_model=64,
            n_layers=2,
            n_heads=4,
            num_bet_bins=env.num_bet_bins,
            dropout=0.1,
        )
        model.eval()

        def run_model() -> torch.Tensor:
            structured = encoder.encode_tensor_states(player=0, idxs=idxs)
            structured = structured.to_device(device)
            with torch.no_grad():
                outputs = model(structured)
            return torch.cat(
                [
                    outputs["policy_logits"].float().view(-1),
                    outputs["value"].float().view(-1),
                ]
            )

        base_output = run_model()

        base_holes = env.hole_indices.clone()
        base_deck = env.deck.clone()

        hero_cards = env.hole_indices[0, 0].tolist()
        board_cards = env.board_indices[0].tolist()
        seen = set(card for card in hero_cards + board_cards if card >= 0)

        unused = [card for card in range(52) if card not in seen]
        new_opponent_cards = torch.tensor(unused[:2], device=device, dtype=torch.long)

        env.hole_indices[0, 1] = new_opponent_cards

        # Rebuild deck: hero cards first, then opponent cards, then remaining
        remaining_cards = unused[2 : 2 + (env.deck.shape[1] - 4)]
        new_deck = hero_cards + new_opponent_cards.tolist() + remaining_cards
        env.deck[0] = torch.tensor(new_deck, device=device, dtype=torch.long)

        output_opponent_swapped = run_model()
        torch.testing.assert_close(
            base_output, output_opponent_swapped, atol=1e-6, rtol=1e-6
        )

        # Restore base state before next modification
        env.hole_indices.copy_(base_holes)
        env.deck.copy_(base_deck)

        # Shuffle undealt portion of deck (indices >= 4)
        perm = torch.arange(env.deck.shape[1], device=device)
        perm[4:] = perm[4:].flip(0)
        env.deck[0] = env.deck[0, perm]

        output_deck_shuffled = run_model()
        torch.testing.assert_close(
            base_output, output_deck_shuffled, atol=1e-6, rtol=1e-6
        )
