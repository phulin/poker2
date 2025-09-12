#!/usr/bin/env python3
"""Integration test for transformer poker model pipeline."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from alphaholdem.models.transformer import PokerTransformerV1
from alphaholdem.models.transformer.state_encoder import TransformerStateEncoder
from alphaholdem.models.transformer.embedding_data import StructuredEmbeddingData
from alphaholdem.models.factory import ModelFactory
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.rl.vectorized_replay import VectorizedReplayBuffer


def test_transformer_pipeline():
    """Test the complete transformer pipeline: env → encoder → model → replay → training."""
    print("🧪 Testing transformer poker pipeline...")

    # Setup
    device = torch.device("cpu")  # Use CPU for testing
    num_envs = 4
    batch_size = 8
    max_seq_len = 50

    # Create tensor environment
    tensor_env = HUNLTensorEnv(
        num_envs=num_envs,
        starting_stack=1000,
        sb=5,
        bb=10,
        bet_bins=[0.5, 0.75, 1.0, 1.5, 2.0],
        device=device,
        float_dtype=torch.float32,
    )

    # Create transformer model
    model_config = {
        "d_model": 128,  # Smaller for testing
        "n_layers": 2,
        "n_heads": 2,
        "vocab_size": 80,
        "num_actions": 8,
        "use_auxiliary_loss": True,
        "use_gradient_checkpointing": False,  # Disable for testing
    }

    model = ModelFactory.create_model("transformer", model_config, device)
    print(
        f"✅ Created transformer model with {sum(p.numel() for p in model.parameters())} parameters"
    )

    # Create tensor environment
    tensor_env = HUNLTensorEnv(
        num_envs=4,
        starting_stack=1000,
        sb=10,
        bb=20,
        bet_bins=[0.0, 0.5, 1.0, 2.0, 3.0],
        device=device,
    )

    # Create state encoder
    state_encoder = TransformerStateEncoder(tensor_env, device)
    print("✅ Created transformer state encoder")

    # Create replay buffer
    replay_buffer = VectorizedReplayBuffer(
        capacity=10,
        max_trajectory_length=20,
        num_bet_bins=8,
        is_transformer=True,
        sequence_length=42,  # Match state encoder sequence length
        device=device,
        float_dtype=torch.float32,
    )
    print("✅ Created replay buffer")

    # Test 1: Environment → State Encoder
    print("\n📝 Test 1: Environment → State Encoder")
    tensor_env.reset()

    # Take a few steps to generate some action history
    for _ in range(3):
        legal_bins_amounts, legal_bins_mask = tensor_env.legal_bins_amounts_and_mask()
        action_values = torch.randint(0, 8, (num_envs,), device=device)
        rewards, dones, _ = tensor_env.step_bins(
            action_values, legal_bins_amounts, legal_bins_mask
        )

        # Encode states
        idxs = torch.arange(num_envs, device=device)
        states = state_encoder.encode_tensor_states(player=0, idxs=idxs)
    print(f"✅ Encoded states: {list(states.to_dict().keys())}")

    # Verify state shapes
    for key, tensor in states.to_dict().items():
        print(f"   {key}: {tensor.shape}")

    # Test 2: State Encoder → Model
    print("\n📝 Test 2: State Encoder → Model")

    # Get active environments
    active_mask = ~tensor_env.done
    active_indices = torch.where(active_mask)[0]

    if len(active_indices) > 0:
        # Extract structured embeddings for active environments
        active_states_dict = {
            key: tensor[active_indices] for key, tensor in states.to_dict().items()
        }

        # Create structured data for active environments
        active_states = StructuredEmbeddingData.from_dict(active_states_dict)

        # Forward pass through model
        with torch.no_grad():
            outputs = model(structured_data=active_states)

        print(f"✅ Model forward pass successful")
        print(f"   policy_logits: {outputs['policy_logits'].shape}")
        print(f"   size_params: {outputs['size_params'].shape}")
        print(f"   value: {outputs['value'].shape}")
        if "hand_range_logits" in outputs:
            print(f"   hand_range_logits: {outputs['hand_range_logits'].shape}")

    # Test 3: Replay Buffer Integration
    print("\n📝 Test 3: Replay Buffer Integration")

    # Start adding trajectories
    replay_buffer.start_adding_trajectory_batches(num_envs)

    # Add a batch of transitions
    if len(active_indices) > 0:
        active_states_dict = {
            key: tensor[active_indices] for key, tensor in states.to_dict().items()
        }
        active_states = StructuredEmbeddingData.from_dict(active_states_dict)

        replay_buffer.add_batch(
            embedding_data=active_states,
            action_indices=torch.randint(0, 8, (len(active_indices),), device=device),
            log_probs=torch.randn(len(active_indices), device=device),
            rewards=torch.randn(len(active_indices), device=device),
            dones=torch.zeros(len(active_indices), dtype=torch.bool, device=device),
            legal_masks=torch.ones(
                len(active_indices), 8, dtype=torch.bool, device=device
            ),
            delta2=torch.randn(len(active_indices), device=device),
            delta3=torch.randn(len(active_indices), device=device),
            values=torch.randn(len(active_indices), device=device),
            trajectory_indices=torch.arange(len(active_indices), device=device),
        )
        print("✅ Added batch to replay buffer")

    # Finish adding trajectories
    trajectories_added, steps_added = replay_buffer.finish_adding_trajectory_batches()
    print(
        f"✅ Finished adding trajectories: {trajectories_added} trajectories, {steps_added} steps"
    )

    # Test 4: Sample from replay buffer
    print("\n📝 Test 4: Sample from Replay Buffer")

    if replay_buffer.num_steps() > 0:
        rng = torch.Generator(device=device)
        batch = replay_buffer.sample_batch(
            rng, min(batch_size, replay_buffer.num_steps())
        )

        print(f"✅ Sampled batch from replay buffer")
        print(f"   Batch size: {batch['action_indices'].shape[0]}")

        # Test model forward pass on sampled batch
        batch_structured = StructuredEmbeddingData.from_dict(batch)
        with torch.no_grad():
            outputs = model(structured_data=batch_structured)

        print(f"✅ Model forward pass on sampled batch successful")
        print(f"   policy_logits: {outputs['policy_logits'].shape}")
        print(f"   value: {outputs['value'].shape}")

    print("\n🎉 All transformer pipeline tests passed!")
    return True


def test_cls_pad_handling():
    """Test CLS/PAD handling in embeddings."""
    print("\n🧪 Testing CLS/PAD handling...")

    device = torch.device("cpu")
    model = PokerTransformerV1(
        d_model=64,
        n_layers=1,
        n_heads=1,
        vocab_size=80,
        num_actions=8,
        use_auxiliary_loss=False,
    )

    # Create test input with CLS token (52) and padding (-1)
    batch_size = 2
    seq_len = 10

    card_indices = torch.tensor(
        [
            [52, 0, 1, 2, -1, -1, -1, -1, -1, -1],  # CLS, cards, padding
            [52, 3, 4, 5, 6, -1, -1, -1, -1, -1],  # CLS, cards, padding
        ],
        device=device,
    )

    # Create dummy structured embeddings
    card_streets = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    card_visibility = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    card_order = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    action_actors = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    action_types = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    action_streets = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    action_size_bins = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    action_size_features = torch.zeros(
        batch_size, seq_len, 3, dtype=torch.float, device=device
    )
    context_pot_sizes = torch.zeros(
        batch_size, seq_len, 1, dtype=torch.float, device=device
    )
    context_stack_sizes = torch.zeros(
        batch_size, seq_len, 2, dtype=torch.float, device=device
    )
    context_positions = torch.zeros(
        batch_size, seq_len, dtype=torch.long, device=device
    )
    context_street_context = torch.zeros(
        batch_size, seq_len, 4, dtype=torch.float, device=device
    )

    # Forward pass should not crash
    test_data = StructuredEmbeddingData(
        token_ids=card_indices,
        card_ranks=torch.randint(0, 13, (batch_size, seq_len), device=device),
        card_suits=torch.randint(0, 4, (batch_size, seq_len), device=device),
        card_streets=card_streets,
        action_actors=action_actors,
        action_streets=action_streets,
        action_legal_masks=torch.ones(batch_size, seq_len, 8, device=device).bool(),
        context_features=torch.randint(0, 10, (batch_size, seq_len, 10), device=device),
    )

    with torch.no_grad():
        outputs = model(structured_data=test_data)

    print(f"✅ CLS/PAD handling test passed")
    print(
        f"   Output shapes: policy_logits={outputs['policy_logits'].shape}, value={outputs['value'].shape}"
    )

    return True


if __name__ == "__main__":
    try:
        test_transformer_pipeline()
        test_cls_pad_handling()
        print("\n🎉 All integration tests passed!")
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
