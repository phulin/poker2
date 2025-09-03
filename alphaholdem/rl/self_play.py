from __future__ import annotations

from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.optim as optim

from alphaholdem.core.config_loader import get_config

from ..env.hunl_env import HUNLEnv
from ..encoding.cards_encoder import CardsPlanesV1
from ..encoding.actions_encoder import ActionsHUEncoderV1
from ..encoding.action_mapping import bin_to_action, get_legal_mask
from ..models.siamese_convnet import SiameseConvNetV1
from ..models.heads import CategoricalPolicyV1
from ..rl.replay import ReplayBuffer, Transition, Trajectory, compute_gae_returns, prepare_ppo_batch
from ..rl.losses import trinal_clip_ppo_loss
from ..rl.k_best_pool import KBestOpponentPool, AgentSnapshot
from ..core.config import RootConfig
from ..core.builders import build_components_from_config


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
        k_best_pool_size: int = 5,  # K-Best pool size
        min_elo_diff: float = 50.0,  # Minimum ELO difference for pool updates
        device: torch.device = None,
        config: Union[RootConfig, str, None] = None,
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
        
        # Set device
        self.device = device if device is not None else torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Initialize components
        self.env = HUNLEnv(starting_stack=1000, sb=50, bb=100)

        # Config-driven components (default to configs/default.yaml when config is None)
        cfg = get_config(config)
        ce, ae, model, policy, nb = build_components_from_config(cfg)
        self.cards_encoder = ce
        self.actions_encoder = ae
        self.model = model
        self.policy = policy
        self.num_bet_bins = nb

        self.model.to(self.device)  # Move model to device
        self._initialize_weights()  # Initialize with better weights
        self.replay_buffer = ReplayBuffer(capacity=1000)
        
        # K-Best opponent pool
        self.opponent_pool = KBestOpponentPool(k=k_best_pool_size, min_elo_diff=min_elo_diff)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training stats
        self.episode_count = 0
        self.total_reward = 0.0
        self.current_elo = 1200.0  # Starting ELO rating

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

    def collect_trajectory(self, opponent_snapshot: Optional[AgentSnapshot] = None) -> Trajectory:
        """
        Collect a single trajectory from self-play or against an opponent.
        
        Args:
            opponent_snapshot: Optional opponent snapshot to play against.
                             If None, plays against self (self-play).
        """
        state = self.env.reset()
        transitions = []
        max_steps = 50  # Safety limit to prevent infinite loops
        
        step_count = 0
        while not state.terminal and step_count < max_steps:
            step_count += 1
            
            # Determine which player's turn it is
            current_player = state.to_act
            
            # Encode state for current player
            cards = self.cards_encoder.encode_cards(state, seat=current_player)
            actions_tensor = self.actions_encoder.encode_actions(state, seat=current_player, num_bet_bins=self.num_bet_bins)
            
            # Move tensors to device
            cards = cards.to(self.device)
            actions_tensor = actions_tensor.to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                if opponent_snapshot is not None and current_player != 0:  # Opponent's turn
                    # Use opponent's model
                    logits, value = opponent_snapshot.model(cards.unsqueeze(0), actions_tensor.unsqueeze(0))
                else:
                    # Use current model (our turn or self-play)
                    logits, value = self.model(cards.unsqueeze(0), actions_tensor.unsqueeze(0))
                
                legal_mask = get_legal_mask(state, self.num_bet_bins)
                legal_mask = legal_mask.to(self.device)  # Move legal mask to device
                
                # Sample action
                action_idx, log_prob = self.policy.action(logits.squeeze(0), legal_mask)
                
                # Convert to concrete action
                action = bin_to_action(action_idx, state, self.num_bet_bins)
            
            # Take step
            next_state, reward, done, _ = self.env.step(action)
            
            # Record transition (only for our player's actions)
            if current_player == 0:  # Our player
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
                
                # Move tensors to device
                cards = cards.to(self.device)
                actions_tensor = actions_tensor.to(self.device)
                
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
        
        # Move batch tensors to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        
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
        """
        Single training step: collect trajectories against K-Best opponents and update model.
        
        Args:
            num_trajectories: Number of trajectories to collect
            
        Returns:
            Dictionary with training statistics
        """
        # Sample opponents from K-Best pool
        opponents = self.opponent_pool.sample(k=num_trajectories)
        
        # Collect trajectories
        for i in range(num_trajectories):
            opponent = opponents[i] if i < len(opponents) else None
            trajectory = self.collect_trajectory(opponent_snapshot=opponent)
            self.replay_buffer.add_trajectory(trajectory)
            self.episode_count += 1
            self.total_reward += sum(t.reward for t in trajectory.transitions)
            
            # Update ELO if we played against an opponent
            if opponent is not None:
                # Calculate game result based on final reward
                final_reward = sum(t.reward for t in trajectory.transitions)
                if final_reward > 0:
                    result = 'win'
                elif final_reward < 0:
                    result = 'loss'
                else:
                    result = 'draw'
                
                # Update ELO ratings
                self.opponent_pool.update_elo_after_game(opponent, result)
                self.current_elo = self.opponent_pool.current_elo
        
        # Update model
        update_stats = self.update_model()
        
        # Check if we should add current model to opponent pool
        if self.opponent_pool.should_add_snapshot(self.current_elo):
            self.opponent_pool.add_snapshot(self, self.current_elo)
        
        return {
            'episode_count': self.episode_count,
            'avg_reward': self.total_reward / max(1, self.episode_count),
            'current_elo': self.current_elo,
            'pool_stats': self.opponent_pool.get_pool_stats(),
            **update_stats,
        }

    def save_checkpoint(self, path: str, step: int) -> None:
        """Save model checkpoint and opponent pool."""
        checkpoint = {
            'step': step,
            'episode_count': self.episode_count,
            'total_reward': self.total_reward,
            'current_elo': self.current_elo,
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
        
        # Save opponent pool separately
        pool_path = path.replace('.pt', '_pool.pt')
        self.opponent_pool.save_pool(pool_path)
        print(f"Opponent pool saved to {pool_path}")

    def load_checkpoint(self, path: str) -> int:
        """Load model checkpoint and opponent pool. Returns the step number."""
        checkpoint = torch.load(path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_count = checkpoint['episode_count']
        self.total_reward = checkpoint['total_reward']
        self.current_elo = checkpoint.get('current_elo', 1200.0)
        
        print(f"Checkpoint loaded from {path}")
        
        # Load opponent pool if it exists
        pool_path = path.replace('.pt', '_pool.pt')
        try:
            from ..models.siamese_convnet import SiameseConvNetV1
            self.opponent_pool.load_pool(pool_path, SiameseConvNetV1)
            print(f"Opponent pool loaded from {pool_path}")
        except FileNotFoundError:
            print(f"No opponent pool found at {pool_path}, starting with empty pool")
        
        return checkpoint['step']

    def evaluate_against_pool(self, num_games: int = 100) -> dict:
        """
        Evaluate current model against all opponents in the pool.
        
        Args:
            num_games: Number of games to play against each opponent
            
        Returns:
            Dictionary with evaluation results
        """
        if not self.opponent_pool.snapshots:
            return {'error': 'No opponents in pool'}
        
        results = {}
        total_wins = 0
        total_games = 0
        
        for i, opponent in enumerate(self.opponent_pool.snapshots):
            wins = 0
            for _ in range(num_games):
                trajectory = self.collect_trajectory(opponent_snapshot=opponent)
                final_reward = sum(t.reward for t in trajectory.transitions)
                if final_reward > 0:
                    wins += 1
            
            win_rate = wins / num_games
            results[f'opponent_{i}_step_{opponent.step}'] = {
                'win_rate': win_rate,
                'opponent_elo': opponent.elo,
                'wins': wins,
                'total_games': num_games
            }
            total_wins += wins
            total_games += num_games
        
        overall_win_rate = total_wins / total_games if total_games > 0 else 0.0
        
        return {
            'overall_win_rate': overall_win_rate,
            'total_games': total_games,
            'opponent_results': results,
            'pool_stats': self.opponent_pool.get_pool_stats()
        }

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
                    # Same rank (pairs) - always suited
                    card1, card2 = f"{rank1}s", f"{rank1}h"  # Suited pair
                elif i < j:
                    # Top-right triangle: suited hands (e.g., AKs, AQs)
                    card1, card2 = f"{rank1}s", f"{rank2}h"  # Suited
                else:
                    # Bottom-left triangle: off-suit hands (e.g., KAs, QAs)
                    card1, card2 = f"{rank2}s", f"{rank1}d"  # Off-suit
                
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
        
        # Move tensors to device
        cards = cards.to(self.device)
        actions_tensor = actions_tensor.to(self.device)
        
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
            legal_mask = legal_mask.to(self.device)  # Move legal mask to device
            
            # Apply legal mask
            masked_logits = logits.clone()
            masked_logits[0, legal_mask == 0] = -1e9
            
            # Get probabilities
            probs = torch.softmax(masked_logits, dim=-1).squeeze(0)
            
            # Get probability of all-in (all-in is action 7)
            all_in_prob = probs[7].item()
            
            # Convert to percentage and format
            percentage = round(all_in_prob * 100)
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
