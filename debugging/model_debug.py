#!/usr/bin/env python3
"""
Model Debugging Script

Investigates why the model shows no policy diversity.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from alphaholdem.rl.self_play import SelfPlayTrainer
from alphaholdem.env.hunl_env import HUNLEnv
from alphaholdem.encoding.cards_encoder import CardsPlanesV1
from alphaholdem.encoding.actions_encoder import ActionsHUEncoderV1


def debug_model_internals(trainer):
    """Debug the model's internal representations."""
    print("=== Model Internal Debugging ===")
    
    cards_encoder = CardsPlanesV1()
    actions_encoder = ActionsHUEncoderV1()
    
    # Create a simple test state
    env = HUNLEnv(starting_stack=400, sb=1, bb=2, seed=1)
    state = env.reset()
    
    # Encode state
    cards_tensor = cards_encoder.encode_cards(state, seat=state.to_act)
    actions_tensor = actions_encoder.encode_actions(state, seat=state.to_act, num_bet_bins=9)
    
    print(f"Input shapes:")
    print(f"  Cards tensor: {cards_tensor.shape}")
    print(f"  Actions tensor: {actions_tensor.shape}")
    print(f"  Cards norm: {torch.norm(cards_tensor).item():.6f}")
    print(f"  Actions norm: {torch.norm(actions_tensor).item():.6f}")
    
    # Check if inputs are different
    print(f"\nInput diversity:")
    print(f"  Cards tensor variance: {cards_tensor.var().item():.6f}")
    print(f"  Actions tensor variance: {actions_tensor.var().item():.6f}")
    
    # Model forward pass with intermediate outputs
    with torch.no_grad():
        # Get intermediate outputs
        x_cards = trainer.model.cards_trunk(cards_tensor.unsqueeze(0))
        x_actions = trainer.model.actions_trunk(actions_tensor.unsqueeze(0))
        x = torch.cat([x_cards, x_actions], dim=1)
        h = trainer.model.fusion(x)
        logits = trainer.model.policy_head(h)
        value = trainer.model.value_head(h).squeeze(-1)
        
        print(f"\nIntermediate representations:")
        print(f"  Cards trunk output: {x_cards.shape}, norm: {torch.norm(x_cards).item():.6f}")
        print(f"  Actions trunk output: {x_actions.shape}, norm: {torch.norm(x_actions).item():.6f}")
        print(f"  Concatenated: {x.shape}, norm: {torch.norm(x).item():.6f}")
        print(f"  Fusion output: {h.shape}, norm: {torch.norm(h).item():.6f}")
        print(f"  Policy logits: {logits.shape}, norm: {torch.norm(logits).item():.6f}")
        print(f"  Value: {value.shape}, value: {value.item():.6f}")
        
        # Check for dead neurons
        print(f"\nDead neuron analysis:")
        print(f"  Cards trunk - zero outputs: {(x_cards == 0).sum().item()}/{x_cards.numel()}")
        print(f"  Actions trunk - zero outputs: {(x_actions == 0).sum().item()}/{x_actions.numel()}")
        print(f"  Fusion - zero outputs: {(h == 0).sum().item()}/{h.numel()}")
        print(f"  Policy logits - zero outputs: {(logits == 0).sum().item()}/{logits.numel()}")


def debug_parameter_updates(trainer):
    """Check if parameters are actually being updated."""
    print("\n=== Parameter Update Analysis ===")
    
    # Store initial parameters
    initial_params = {}
    for name, param in trainer.model.named_parameters():
        initial_params[name] = param.data.clone()
    
    # Do a few training steps
    print("Performing 5 training steps...")
    for step in range(5):
        stats = trainer.train_step(num_trajectories=4)
        if step % 2 == 0:
            print(f"  Step {step}: Loss: {stats.get('total_loss', 'N/A')}")
    
    # Check parameter changes
    print(f"\nParameter changes:")
    total_change = 0
    for name, param in trainer.model.named_parameters():
        initial = initial_params[name]
        change = torch.norm(param.data - initial).item()
        total_change += change
        if change > 0.001:  # Only show significant changes
            print(f"  {name}: {change:.6f}")
    
    print(f"\nTotal parameter change: {total_change:.6f}")
    
    if total_change < 0.01:
        print("⚠️  WARNING: Very little parameter change! Model may not be learning.")
    elif total_change < 0.1:
        print("⚠️  WARNING: Low parameter change. Learning may be slow.")
    else:
        print("✅ Good parameter change detected.")


def debug_gradient_flow(trainer):
    """Check if gradients are flowing properly."""
    print("\n=== Gradient Flow Analysis ===")
    
    cards_encoder = CardsPlanesV1()
    actions_encoder = ActionsHUEncoderV1()
    
    env = HUNLEnv(starting_stack=400, sb=1, bb=2, seed=1)
    state = env.reset()
    
    cards_tensor = cards_encoder.encode_cards(state, seat=state.to_act)
    actions_tensor = actions_encoder.encode_actions(state, seat=state.to_act, num_bet_bins=9)
    
    # Get legal actions
    legal_actions = env.legal_actions()
    legal_mask = torch.zeros(9)
    for action in legal_actions:
        bin_idx = actions_encoder._action_to_bin(action, state, 9)
        if bin_idx is not None:
            legal_mask[bin_idx] = 1.0
    
    # Forward pass
    logits, value = trainer.model(cards_tensor.unsqueeze(0), actions_tensor.unsqueeze(0))
    logits = logits.squeeze(0)
    value = value.squeeze(0)
    
    # Apply legal mask
    masked_logits = logits.clone()
    masked_logits[legal_mask == 0] = -1e9
    probs = F.softmax(masked_logits, dim=-1)
    
    # Compute loss
    target_action = torch.tensor([0])  # fold
    loss = F.cross_entropy(logits.unsqueeze(0), target_action)
    
    # Backward pass
    trainer.optimizer.zero_grad()
    loss.backward()
    
    # Analyze gradients
    print(f"Loss: {loss.item():.6f}")
    print(f"Gradient analysis:")
    
    total_grad_norm = 0
    for name, param in trainer.model.named_parameters():
        if param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            total_grad_norm += grad_norm ** 2
            if grad_norm > 0.001:  # Only show significant gradients
                print(f"  {name}: {grad_norm:.6f}")
    
    total_grad_norm = total_grad_norm ** 0.5
    print(f"\nTotal gradient norm: {total_grad_norm:.6f}")
    
    if total_grad_norm < 0.01:
        print("⚠️  WARNING: Very small gradients! This could cause learning issues.")
    elif total_grad_norm < 0.1:
        print("⚠️  WARNING: Small gradients. Learning may be slow.")
    else:
        print("✅ Good gradient flow detected.")


def main():
    """Main function."""
    print("=== Model Debugging Script ===")
    print("This script investigates why the model shows no policy diversity.")
    print()
    
    # Initialize trainer
    trainer = SelfPlayTrainer(
        num_bet_bins=9,
        learning_rate=1e-4,
        batch_size=256,
        grad_clip=0.5,
    )
    
    # Debug model internals
    debug_model_internals(trainer)
    
    # Debug parameter updates
    debug_parameter_updates(trainer)
    
    # Debug gradient flow
    debug_gradient_flow(trainer)
    
    print("\n=== Debugging Complete ===")
    print("This analysis helps identify why the model is not learning properly.")


if __name__ == "__main__":
    main()
