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
        learning_rate: float = 1e-3,  # Increased from 3e-4 for better learning
        batch_size: int = 256,  # Larger batch size like AlphaHoldem (they used 16,384)
        num_epochs: int = 4,
        gamma: float = 0.999,
        gae_lambda: float = 0.95,
        epsilon: float = 0.2,
        delta1: float = 3.0,
        value_coef: float = 0.1,  # Reasonable value coefficient
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
        self._initialize_weights()  # Initialize with better weights
        self.policy = CategoricalPolicyV1()
        self.replay_buffer = ReplayBuffer(capacity=1000)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training stats
        self.episode_count = 0
        self.total_reward = 0.0

    def _initialize_weights(self):
        """Initialize model weights to prevent dead neurons."""
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

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
        # According to AlphaHoldem paper: δ2 and δ3 represent total chips placed by players
        # They should be dynamically calculated from the actual chips in the trajectory
        total_chips_placed = 0
        for trajectory in trajectories:
            for transition in trajectory.transitions:
                total_chips_placed += transition.chips_placed
        
        if total_chips_placed > 0:
            # δ2: negative bound (opponent chips), δ3: positive bound (our chips)
            # Use total chips placed as the bound, similar to paper's approach
            delta2 = -total_chips_placed
            delta3 = total_chips_placed
        else:
            # Fallback to reasonable bounds if no chips placed
            delta2 = -1000.0
            delta3 = 1000.0
        
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

    def save_checkpoint(self, path: str, step: int) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'step': step,
            'episode_count': self.episode_count,
            'total_reward': self.total_reward,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'num_bet_bins': self.num_bet_bins,
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'epsilon': self.epsilon,
                'delta1': self.delta1,
                'value_coef': self.value_coef,
                'entropy_coef': self.entropy_coef,
                'grad_clip': self.grad_clip,
            }
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> int:
        """Load model checkpoint. Returns the step number."""
        checkpoint = torch.load(path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_count = checkpoint['episode_count']
        self.total_reward = checkpoint['total_reward']
        
        print(f"Checkpoint loaded from {path}")
        return checkpoint['step']

    def get_preflop_range_grid(self, seat: int = 0) -> str:
        """Get preflop range as a grid showing action probabilities for button play."""
        # Create a 13x13 grid representing all possible hole card combinations
        # Rows/cols: A, K, Q, J, T, 9, 8, 7, 6, 5, 4, 3, 2
        ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
        
        # Initialize grid with action probabilities
        grid = []
        header = "    " + " ".join(f"{rank:>2}" for rank in ranks)
        grid.append(header)
        grid.append("   " + "-" * 39)  # Separator line
        
        for i, rank1 in enumerate(ranks):
            row = [f"{rank1:>2} |"]
            for j, rank2 in enumerate(ranks):
                if i == j:
                    # Same rank (pairs)
                    card1, card2 = f"{rank1}s", f"{rank1}h"  # Suited pair
                elif i < j:
                    # First rank higher (e.g., AK, AQ)
                    card1, card2 = f"{rank1}s", f"{rank2}h"  # Suited
                else:
                    # Second rank higher (e.g., KA, QA)
                    card1, card2 = f"{rank2}s", f"{rank1}h"  # Suited
                
                # Get action probability for this hand
                prob = self._get_preflop_action_probability(card1, card2, seat)
                row.append(f"{prob:>2}")
            
            grid.append(" ".join(row))
        
        return "\n".join(grid)
    
    def _get_preflop_action_probability(self, card1: str, card2: str, seat: int) -> str:
        """Get action probability for a specific hole card combination."""
        # Create a minimal game state for preflop button play
        from ..env.types import GameState, PlayerState
        
        # Create players
        p0 = PlayerState(stack=1000)
        p1 = PlayerState(stack=1000)
        
        # Set up preflop state (button acts last preflop)
        button = 1 - seat  # If seat=0 (button), then button=1
        p_sb = 0 if button == 0 else 1
        p_bb = 1 - p_sb
        
        # Post blinds
        p_states = [p0, p1]
        p_states[p_sb].stack -= 50  # small blind
        p_states[p_sb].committed += 50
        p_states[p_bb].stack -= 100  # big blind
        p_states[p_bb].committed += 100
        
        # Set hole cards for the seat we're analyzing
        p_states[seat].hole_cards = [self._card_str_to_int(card1), self._card_str_to_int(card2)]
        
        # Create game state
        state = GameState(
            button=button,
            street="preflop",
            deck=[],  # Not needed for this analysis
            board=[],
            pot=150,
            to_act=seat,
            small_blind=50,
            big_blind=100,
            min_raise=100,
            last_aggressive_amount=100,
            players=(p_states[0], p_states[1]),
            terminal=False,
        )
        
        # Encode state
        cards = self.cards_encoder.encode_cards(state, seat=seat)
        actions_tensor = self.actions_encoder.encode_actions(state, seat=seat, num_bet_bins=self.num_bet_bins)
        
        # Get model prediction
        with torch.no_grad():
            logits, _ = self.model(cards.unsqueeze(0), actions_tensor.unsqueeze(0))
            
            # For preflop button play, we know the legal actions:
            # fold, call, raise (pot-sized), all-in
            # Create a simple legal mask
            legal_mask = torch.zeros(self.num_bet_bins)
            legal_mask[0] = 1.0  # fold
            legal_mask[1] = 1.0  # call/check
            legal_mask[4] = 1.0  # pot-sized bet
            legal_mask[7] = 1.0  # all-in
            
            # Apply legal mask
            masked_logits = logits.clone()
            masked_logits[0, legal_mask == 0] = -1e9
            
            # Get probabilities
            probs = torch.softmax(masked_logits, dim=-1).squeeze(0)
            
            # Get probability of not folding (fold is action 0)
            not_fold_prob = 1.0 - probs[0].item()
            
            # Convert to percentage and format
            percentage = round(not_fold_prob * 100)
            return f"{percentage:2d}"
    
    def _card_str_to_int(self, card_str: str) -> int:
        """Convert card string (e.g., 'As', 'Kh') to integer representation."""
        rank_map = {'A': 12, 'K': 11, 'Q': 10, 'J': 9, 'T': 8, 
                   '9': 7, '8': 6, '7': 5, '6': 4, '5': 3, '4': 2, '3': 1, '2': 0}
        suit_map = {'s': 0, 'h': 1, 'd': 2, 'c': 3}
        
        rank = card_str[0]
        suit = card_str[1]
        
        return rank_map[rank] * 4 + suit_map[suit]

    def diagnose_model_health(self) -> dict:
        """Diagnose potential issues with the model and training setup."""
        diagnostics = {}
        
        # Check model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        diagnostics['total_params'] = total_params
        diagnostics['trainable_params'] = trainable_params
        
        # Check parameter norms
        param_norms = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                norm = torch.norm(param).item()
                param_norms.append((name, norm))
        diagnostics['param_norms'] = param_norms
        
        # Check if parameters are all zero or NaN
        has_nan = any(torch.isnan(param).any() for param in self.model.parameters())
        has_inf = any(torch.isinf(param).any() for param in self.model.parameters())
        diagnostics['has_nan'] = has_nan
        diagnostics['has_inf'] = has_inf
        
        # Test forward pass with dummy data
        dummy_cards = torch.randn(1, 6, 4, 13)
        dummy_actions = torch.randn(1, 24, 4, self.num_bet_bins)
        
        with torch.no_grad():
            logits, values = self.model(dummy_cards, dummy_actions)
            diagnostics['logits_shape'] = list(logits.shape)
            diagnostics['values_shape'] = list(values.shape)
            diagnostics['logits_norm'] = torch.norm(logits).item()
            diagnostics['values_norm'] = torch.norm(values).item()
            diagnostics['logits_has_nan'] = torch.isnan(logits).any().item()
            diagnostics['values_has_nan'] = torch.isnan(values).any().item()
        
        # Check optimizer state
        diagnostics['optimizer_lr'] = self.optimizer.param_groups[0]['lr']
        diagnostics['optimizer_momentum'] = self.optimizer.param_groups[0].get('betas', (0.9, 0.999))
        
        # Check replay buffer
        diagnostics['replay_buffer_size'] = len(self.replay_buffer.trajectories)
        
        return diagnostics
