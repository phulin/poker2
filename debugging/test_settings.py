#!/usr/bin/env python3
"""Test training with different hyperparameter settings."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from alphaholdem.rl.self_play import SelfPlayTrainer


def test_training_settings():
    print("=== Testing Different Training Settings ===\n")
    
    # Test different configurations
    configs = [
        {
            'name': 'Conservative',
            'lr': 1e-4,
            'batch_size': 32,
            'grad_clip': 0.5,
            'value_coef': 0.1,  # Lower value coefficient
        },
        {
            'name': 'Aggressive',
            'lr': 3e-4,
            'batch_size': 16,
            'grad_clip': 1.0,
            'value_coef': 1.0,  # Higher value coefficient
        },
        {
            'name': 'Balanced',
            'lr': 2e-4,
            'batch_size': 24,
            'grad_clip': 0.8,
            'value_coef': 0.5,  # Standard value coefficient
        },
    ]
    
    for config in configs:
        print(f"Testing {config['name']} configuration:")
        print(f"  LR: {config['lr']}, Batch: {config['batch_size']}, Grad Clip: {config['grad_clip']}, Value Coef: {config['value_coef']}")
        
        # Create trainer with these settings
        trainer = SelfPlayTrainer(
            num_bet_bins=9,
            learning_rate=config['lr'],
            batch_size=config['batch_size'],
            grad_clip=config['grad_clip'],
            value_coef=config['value_coef'],
        )
        
        # Track metrics
        losses = []
        rewards = []
        value_losses = []
        policy_losses = []
        
        # Run training for a few steps
        for step in range(5):
            stats = trainer.train_step(num_trajectories=4)
            
            if 'avg_loss' in stats:
                losses.append(stats['avg_loss'])
            
            rewards.append(stats['avg_reward'])
            
            # Get detailed loss components if available
            if step == 4:  # Last step
                # Run one more step to get detailed loss
                trainer.replay_buffer.clear()  # Clear buffer
                for _ in range(4):
                    trainer.collect_trajectory()
                
                if len(trainer.replay_buffer.trajectories) >= trainer.batch_size:
                    # Get detailed loss info
                    trajectories = trainer.replay_buffer.sample_trajectories(trainer.batch_size)
                    
                    # Compute values and advantages
                    for trajectory in trajectories:
                        rewards_traj = [t.reward for t in trajectory.transitions]
                        values = []
                        
                        for transition in trajectory.transitions:
                            obs = transition.observation
                            cards = obs[:(6 * 4 * 13)].reshape(1, 6, 4, 13)
                            actions_tensor = obs[(6 * 4 * 13):].reshape(1, 24, 4, trainer.num_bet_bins)
                            
                            with torch.no_grad():
                                _, value = trainer.model(cards, actions_tensor)
                                values.append(value.item())
                        
                        values.append(0.0)
                        
                        from alphaholdem.rl.replay import compute_gae_returns
                        advantages, returns = compute_gae_returns(
                            rewards_traj, values, 
                            gamma=trainer.gamma, 
                            lambda_=trainer.gae_lambda
                        )
                        
                        for i, transition in enumerate(trajectory.transitions):
                            transition.advantage = advantages[i]
                            transition.return_ = returns[i]
                    
                    # Prepare batch and compute loss
                    from alphaholdem.rl.replay import prepare_ppo_batch
                    batch = prepare_ppo_batch(trajectories)
                    
                    observations = batch['observations']
                    cards = observations[:, :(6 * 4 * 13)].reshape(-1, 6, 4, 13)
                    actions_tensor = observations[:, (6 * 4 * 13):].reshape(-1, 24, 4, trainer.num_bet_bins)
                    
                    with torch.no_grad():
                        logits, values = trainer.model(cards, actions_tensor)
                    
                    from alphaholdem.rl.losses import trinal_clip_ppo_loss
                    loss_dict = trinal_clip_ppo_loss(
                        logits=logits,
                        values=values,
                        actions=batch['actions'],
                        log_probs_old=batch['log_probs_old'],
                        advantages=batch['advantages'],
                        returns=batch['returns'],
                        legal_masks=batch['legal_masks'],
                        epsilon=trainer.epsilon,
                        delta1=trainer.delta1,
                        delta2=-100.0,
                        delta3=100.0,
                        value_coef=trainer.value_coef,
                        entropy_coef=trainer.entropy_coef,
                    )
                    
                    value_losses.append(loss_dict['value_loss'].item())
                    policy_losses.append(loss_dict['policy_loss'].item())
        
        # Analyze results
        final_loss = losses[-1] if losses else 'N/A'
        print(f"  Final loss: {final_loss}")
        print(f"  Loss trend: {'decreasing' if len(losses) > 1 and losses[-1] < losses[0] else 'stable/increasing'}")
        print(f"  Final reward: {rewards[-1]:.2f}")
        print(f"  Reward trend: {'improving' if len(rewards) > 1 and rewards[-1] > rewards[0] else 'stable/worsening'}")
        
        if value_losses:
            print(f"  Value loss: {value_losses[0]:.6f}")
            print(f"  Policy loss: {policy_losses[0]:.6f}")
        
        print()


if __name__ == "__main__":
    test_training_settings()
