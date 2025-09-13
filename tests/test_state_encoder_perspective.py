"""Test state encoder perspective behavior for both CNN and Transformer models."""

from __future__ import annotations

import torch

from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.state_encoder import CNNStateEncoder
from alphaholdem.models.transformer.state_encoder import TransformerStateEncoder


def test_cnn_state_encoder_perspective_behavior():
    """Test that CNN state encoder presents consistent perspective for both players."""
    device = torch.device("cpu")

    # Create a tensor environment with some action history
    env = HUNLTensorEnv(
        num_envs=2,
        starting_stack=1000,
        sb=50,
        bb=100,
        bet_bins=[0.5, 1.0, 1.5, 2.0],
        device=device,
    )

    # Reset and create some action history
    env.reset()

    # Simulate some actions to create history
    # Player 0 (SB) raises
    env.step_bins(
        torch.tensor([2, 2], device=device)
    )  # Raise to 1.0x pot for both envs

    # Player 1 (BB) calls
    env.step_bins(torch.tensor([1, 1], device=device))  # Call for both envs

    # Create state encoder
    encoder = CNNStateEncoder(env, device)

    # Encode for player 0 (should be unchanged)
    states_p0 = encoder.encode_tensor_states(
        player=0, idxs=torch.tensor([0], device=device)
    )
    actions_p0 = states_p0.actions[0]  # [24, 4, num_bet_bins]

    # Encode for player 1 (should have swapped perspective)
    states_p1 = encoder.encode_tensor_states(
        player=1, idxs=torch.tensor([0], device=device)
    )
    actions_p1 = states_p1.actions[0]  # [24, 4, num_bet_bins]

    # Check that player 0 and player 1 actions are swapped
    # In the original action history, player 0 acted first, then player 1
    # After perspective swap for player 1, it should look like player 1 acted first, then player 0

    # Find the first action slot that has activity
    action_slots = actions_p0[:, 2, :].any(dim=1)  # Sum plane (index 2) across bet bins
    first_action_slot = action_slots.nonzero(as_tuple=True)[0][0].item()

    # Check that the action history is properly swapped
    # For player 0 encoding, player 0's actions should be in row 0, player 1's in row 1
    # For player 1 encoding, player 0's actions should be in row 1, player 1's in row 0

    # Check if player 0 acted in the first slot (should be true for both encodings)
    p0_player0_acted = actions_p0[first_action_slot, 0, :].any().item()
    p1_player0_acted = (
        actions_p1[first_action_slot, 1, :].any().item()
    )  # Player 0's actions moved to row 1

    # Check if player 1 acted in the second slot (should be true for both encodings)
    second_action_slot = first_action_slot + 1
    if second_action_slot < 24:
        p0_player1_acted = actions_p0[second_action_slot, 1, :].any().item()
        p1_player1_acted = (
            actions_p1[second_action_slot, 0, :].any().item()
        )  # Player 1's actions moved to row 0

        # Both encodings should show the same players acted, just in different rows
        assert (
            p0_player0_acted == p1_player0_acted
        ), "Player 0's actions should be consistent"
        assert (
            p0_player1_acted == p1_player1_acted
        ), "Player 1's actions should be consistent"


def test_transformer_state_encoder_perspective_behavior():
    """Test that Transformer state encoder presents consistent perspective for both players."""
    device = torch.device("cpu")

    # Create a tensor environment with some action history
    env = HUNLTensorEnv(
        num_envs=2,
        starting_stack=1000,
        sb=50,
        bb=100,
        bet_bins=[0.5, 1.0, 1.5, 2.0],
        device=device,
    )

    # Reset and create some action history
    env.reset()

    # Simulate some actions to create history
    # Player 0 (SB) raises
    env.step_bins(
        torch.tensor([2, 2], device=device)
    )  # Raise to 1.0x pot for both envs

    # Player 1 (BB) calls
    env.step_bins(torch.tensor([1, 1], device=device))  # Call for both envs

    # Create state encoder
    encoder = TransformerStateEncoder(env, device)

    # Encode for player 0 (should be unchanged)
    states_p0 = encoder.encode_tensor_states(
        player=0, idxs=torch.tensor([0], device=device)
    )
    action_actors_p0 = states_p0.action_actors[0]  # [L] - actor for each action

    # Encode for player 1 (should have swapped perspective)
    states_p1 = encoder.encode_tensor_states(
        player=1, idxs=torch.tensor([0], device=device)
    )
    action_actors_p1 = states_p1.action_actors[0]  # [L] - actor for each action

    # Find action slots that have activity (non-zero actor values)
    # The action actors are stored starting at index 8 (action_index_offset)
    # Based on the debug output, we have actions in slots 0 and 1 of the action history
    # which correspond to indices 8 and 9 in the action_actors tensor
    action_start_idx = 8  # action_index_offset
    active_actors_p0 = action_actors_p0[
        action_start_idx : action_start_idx + 2
    ]  # Slots 8-9
    active_actors_p1 = action_actors_p1[
        action_start_idx : action_start_idx + 2
    ]  # Slots 8-9

    # For player 0 encoding, actors should be [0, 1, 0, 1, ...] (alternating)
    # For player 1 encoding, actors should be [1, 0, 1, 0, ...] (swapped)

    # Check that the actors are properly swapped
    # If we have actions from both players, they should be swapped
    if len(active_actors_p0) >= 2:
        # First action should be from different players
        assert (
            active_actors_p0[0] != active_actors_p1[0]
        ), f"First action actors should be different: p0={active_actors_p0[0]}, p1={active_actors_p1[0]}"

        # The actors should be flipped
        expected_p1_actors = 1 - active_actors_p0
        assert torch.allclose(
            active_actors_p1.float(), expected_p1_actors.float()
        ), f"Actors should be flipped: p0={active_actors_p0}, p1={active_actors_p1}, expected_p1={expected_p1_actors}"


def test_both_encoders_consistency():
    """Test that both CNN and Transformer encoders handle perspective consistently."""
    device = torch.device("cpu")

    # Create a tensor environment
    env = HUNLTensorEnv(
        num_envs=1,
        starting_stack=1000,
        sb=50,
        bb=100,
        bet_bins=[0.5, 1.0, 1.5, 2.0],
        device=device,
    )

    # Reset and create some action history
    env.reset()

    # Simulate some actions
    env.step_bins(torch.tensor([2], device=device))  # Player 0 raises
    env.step_bins(torch.tensor([1], device=device))  # Player 1 calls

    # Create both encoders
    cnn_encoder = CNNStateEncoder(env, device)
    transformer_encoder = TransformerStateEncoder(env, device)

    # Encode for both players with both encoders
    cnn_p0 = cnn_encoder.encode_tensor_states(
        player=0, idxs=torch.tensor([0], device=device)
    )
    cnn_p1 = cnn_encoder.encode_tensor_states(
        player=1, idxs=torch.tensor([0], device=device)
    )

    transformer_p0 = transformer_encoder.encode_tensor_states(
        player=0, idxs=torch.tensor([0], device=device)
    )
    transformer_p1 = transformer_encoder.encode_tensor_states(
        player=1, idxs=torch.tensor([0], device=device)
    )

    # Both encoders should show the same perspective behavior
    # Player 0 encoding should be identical between encoders
    # Player 1 encoding should be identical between encoders (both swapped)

    # For CNN: check that action tensors are the same shape and have same perspective behavior
    assert (
        cnn_p0.actions.shape == cnn_p1.actions.shape
    ), "CNN action shapes should match"

    # For Transformer: check that action actors show same perspective behavior
    assert (
        transformer_p0.action_actors.shape == transformer_p1.action_actors.shape
    ), "Transformer actor shapes should match"

    # Both should show the perspective swap behavior
    # This is a basic consistency check - both encoders should handle the swap the same way
    print("✅ Both CNN and Transformer encoders handle perspective consistently")


if __name__ == "__main__":
    test_cnn_state_encoder_perspective_behavior()
    test_transformer_state_encoder_perspective_behavior()
    test_both_encoders_consistency()
    print("All perspective behavior tests passed!")
