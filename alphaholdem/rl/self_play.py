from __future__ import annotations

from typing import List, Optional
import torch
import torch.nn as nn
import torch.optim as optim

from ..env.hunl_env import HUNLEnv
from ..encoding.cards_encoder import CardsPlanesV1
from ..encoding.actions_encoder import ActionsHUEncoderV1
from ..encoding.action_mapping import bin_to_action, get_legal_mask
from ..models.siamese_convnet import SiameseConvNetV1
from ..models.heads import CategoricalPolicyV1
from ..rl.replay import ReplayBuffer, Transition, Trajectory, compute_gae_returns, prepare_ppo_batch
from ..rl.losses import trinal_clip_ppo_loss


class SelfPlayTrainer:
    def __init__(
        self,
        num_bet_bins: int = 9,
        learning_rate: float = 3e-4,
        batch_size: int = 64,
        num_epochs: int = 4,
        gamma: float = 0.999,
        gae_lambda: float = 0.95,
        epsilon: float = 0.2,
        delta1: float = 3.0,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        grad_clip: float = 1.0,
    ):
        self.num_bet_bins = num_bet_bins
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.delta1 = delta1
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.grad_clip = grad_clip
        
        # Initialize components
        self.env = HUNLEnv(starting_stack=1000, sb=50, bb=100)
        self.cards_encoder = CardsPlanesV1()
        self.actions_encoder = ActionsHUEncoderV1()
        self.model = SiameseConvNetV1(num_actions=num_bet_bins)
        self.policy = CategoricalPolicyV1()
        self.replay_buffer = ReplayBuffer(capacity=1000)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training stats
        self.episode_count = 0
        self.total_reward = 0.0

    def collect_trajectory(self) -> Trajectory:
        """Collect a single trajectory from self-play."""
        state = self.env.reset()
        transitions = []
        max_steps = 50  # Safety limit to prevent infinite loops
        
        step_count = 0
        while not state.terminal and step_count < max_steps:
            step_count += 1
            
            # Encode state
            cards = self.cards_encoder.encode_cards(state, seat=state.to_act)
            actions_tensor = self.actions_encoder.encode_actions(state, seat=state.to_act, num_bet_bins=self.num_bet_bins)
            
            # Get model prediction
            with torch.no_grad():
                logits, value = self.model(cards.unsqueeze(0), actions_tensor.unsqueeze(0))
                legal_mask = get_legal_mask(state, self.num_bet_bins)
                
                # Sample action
                action_idx, log_prob = self.policy.action(logits.squeeze(0), legal_mask)
                
                # Convert to concrete action
                action = bin_to_action(action_idx, state, self.num_bet_bins)
            
            # Take step
            next_state, reward, done, _ = self.env.step(action)
            
            # Record transition
            transition = Transition(
                observation=torch.cat([cards.flatten(), actions_tensor.flatten()]),  # Simplified obs
                action=action_idx,
                log_prob=log_prob,
                reward=reward,
                done=done,
                legal_mask=legal_mask,
                chips_placed=action.amount,
            )
            transitions.append(transition)
            
            state = next_state
        
        if step_count >= max_steps:
            print(f"Warning: Trajectory reached max steps ({max_steps}), forcing termination")
            # Force termination by setting final reward
            if transitions:
                transitions[-1].reward = 0.0
                transitions[-1].done = True
        
        # Compute final value for GAE
        final_value = 0.0  # Terminal state value is 0
        
        return Trajectory(transitions=transitions, final_value=final_value)

    def update_model(self) -> dict:
        """Perform PPO update on collected trajectories."""
        if len(self.replay_buffer.trajectories) < self.batch_size:
            return {}
        
        # Sample trajectories
        trajectories = self.replay_buffer.sample_trajectories(self.batch_size)
        
        # Compute values for GAE
        for trajectory in trajectories:
            rewards = [t.reward for t in trajectory.transitions]
            values = []
            
            # Compute values for each transition
            for transition in trajectory.transitions:
                # Split observation back to cards and actions
                obs = transition.observation
                cards = obs[:(6 * 4 * 13)].reshape(1, 6, 4, 13)
                actions_tensor = obs[(6 * 4 * 13):].reshape(1, 24, 4, self.num_bet_bins)
                
                with torch.no_grad():
                    _, value = self.model(cards, actions_tensor)
                    values.append(value.item())
            
            # Add final value (0 for terminal state)
            values.append(0.0)
            
            # Compute GAE advantages and returns
            advantages, returns = compute_gae_returns(
                rewards, values, 
                gamma=self.gamma, 
                lambda_=self.gae_lambda
            )
            
            # Update trajectory with computed advantages/returns
            for i, transition in enumerate(trajectory.transitions):
                transition.advantage = advantages[i]
                transition.return_ = returns[i]
        
        # Prepare batch
        batch = prepare_ppo_batch(trajectories)
        
        # Compute dynamic delta2 and delta3 bounds from chips placed
        all_chips_placed = []
        for trajectory in trajectories:
            for transition in trajectory.transitions:
                all_chips_placed.append(transition.chips_placed)
        
        if all_chips_placed:
            max_chips = max(all_chips_placed)
            # δ2: negative bound (opponent chips), δ3: positive bound (our chips)
            # According to paper: δ2 and δ3 represent total chips placed by players
            delta2 = -max_chips
            delta3 = max_chips
        else:
            delta2 = 0.0
            delta3 = 0.0
        
        # Multiple epochs of updates
        total_loss = 0.0
        for epoch in range(self.num_epochs):
            # Forward pass
            observations = batch['observations']
            # Split observations back to cards and actions
            cards = observations[:, :(6 * 4 * 13)].reshape(-1, 6, 4, 13)
            actions_tensor = observations[:, (6 * 4 * 13):].reshape(-1, 24, 4, self.num_bet_bins)
            
            logits, values = self.model(cards, actions_tensor)
            
            # Compute loss with dynamic delta bounds
            loss_dict = trinal_clip_ppo_loss(
                logits=logits,
                values=values,
                actions=batch['actions'],
                log_probs_old=batch['log_probs_old'],
                advantages=batch['advantages'],
                returns=batch['returns'],
                legal_masks=batch['legal_masks'],
                epsilon=self.epsilon,
                delta1=self.delta1,
                delta2=delta2,
                delta3=delta3,
                value_coef=self.value_coef,
                entropy_coef=self.entropy_coef,
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            
            total_loss += loss_dict['total_loss'].item()
        
        return {
            'avg_loss': total_loss / self.num_epochs,
            'num_trajectories': len(trajectories),
            'delta2': delta2,
            'delta3': delta3,
        }

    def train_step(self, num_trajectories: int = 4) -> dict:
        """Single training step: collect trajectories and update model."""
        # Collect trajectories
        for _ in range(num_trajectories):
            trajectory = self.collect_trajectory()
            self.replay_buffer.add_trajectory(trajectory)
            self.episode_count += 1
            self.total_reward += sum(t.reward for t in trajectory.transitions)
        
        # Update model
        update_stats = self.update_model()
        
        return {
            'episode_count': self.episode_count,
            'avg_reward': self.total_reward / max(1, self.episode_count),
            **update_stats,
        }
