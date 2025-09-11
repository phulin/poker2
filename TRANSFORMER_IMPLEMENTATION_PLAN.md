# Transformer Model Implementation Plan

## Overview

This document outlines the detailed plan for implementing a transformer-based poker model to replace or complement the existing CNN-based approach in AlphaHoldem. The transformer will leverage the newly added 0-51 card index tracking in `HUNLTensorEnv` and follow the embedding patterns discussed.

## Architecture Design

### Core Components

#### 1. Token-Based State Representation
- **Card Tokens**: Individual cards as tokens (0-51 indices)
- **Action Tokens**: Historical actions as sequence tokens
- **Context Tokens**: Numeric game state (pot, stacks, etc.)
- **Special Tokens**: Street boundaries, position markers

#### 2. Embedding Structure
```
Card Token = rank_emb[rank] + suit_emb[suit] + stage_emb[stage] + visibility_emb[type] + order_emb[position]
Action Token = actor_emb[player] + action_type_emb[type] + street_emb[street] + size_emb[size]
Context Token = MLP(numeric_features)
```

#### 3. Model Architecture
- **Input Processing**: Token embedding + positional encoding
- **Transformer Encoder**: 2-4 layers, 4 heads, d_model=192-256
- **Output Heads**: Policy (categorical + continuous) + Value + Auxiliary (hand range)

## Implementation Steps

### Phase 1: Directory Structure and Base Classes

#### 1.1 Create Transformer Module Structure
```
alphaholdem/models/transformer/
├── __init__.py
├── poker_transformer.py      # Main transformer model
├── embeddings.py             # Card/action/context embeddings
├── heads.py                  # Policy/value/auxiliary heads
└── tokenizer.py             # State-to-tokens conversion
```

#### 1.2 Update Model Factory
- Add transformer support to `ModelFactory`
- Create `_create_transformer_model()` method
- Update `get_available_model_types()` to include "transformer"

#### 1.3 Create Base Classes
- `PokerTransformer`: Main model class implementing `Model` interface
- `CardEmbedding`: Handles card token embeddings
- `ActionEmbedding`: Handles action sequence embeddings
- `ContextEmbedding`: Handles numeric context embeddings

### Phase 2: Embedding Implementation

#### 2.1 Card Embeddings (`embeddings.py`)
```python
class CardEmbedding(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.rank_emb = nn.Embedding(13, d_model)      # A,2,3,...,K
        self.suit_emb = nn.Embedding(4, d_model)        # s,h,d,c
        self.stage_emb = nn.Embedding(4, d_model)       # hole,flop,turn,river
        self.visibility_emb = nn.Embedding(3, d_model) # self,opponent,public
        self.order_emb = nn.Embedding(5, d_model)      # position within stage
        
        # Relational attention biases
        self.rank_bias = nn.Parameter(torch.zeros(13, 13))
        self.suit_bias = nn.Parameter(torch.zeros(4, 4))
        
    def forward(self, card_indices, stages, visibility, order):
        # Compose embeddings additively
        # Add relational biases for attention
```

#### 2.2 Action Embeddings
```python
class ActionEmbedding(nn.Module):
    def __init__(self, d_model: int = 256, num_action_types: int = 6):
        super().__init__()
        self.actor_emb = nn.Embedding(2, d_model)           # P1, P2
        self.action_type_emb = nn.Embedding(num_action_types, d_model)
        self.street_emb = nn.Embedding(4, d_model)          # pre,flop,turn,river
        self.size_bin_emb = nn.Embedding(20, d_model)       # coarse size bins
        self.size_mlp = nn.Sequential(                       # fine size features
            nn.Linear(3, d_model),  # fraction_of_pot, fraction_of_stack, log_chips
            nn.LayerNorm(d_model)
        )
```

#### 2.3 Context Embeddings
```python
class ContextEmbedding(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.pot_emb = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model)
        )
        self.stack_emb = nn.Sequential(
            nn.Linear(2, d_model),  # effective stacks for both players
            nn.LayerNorm(d_model)
        )
        self.position_emb = nn.Embedding(2, d_model)  # SB, BB
```

### Phase 3: Tokenization System

#### 3.1 State Tokenizer (`tokenizer.py`)
```python
class PokerTokenizer:
    def __init__(self, max_sequence_length: int = 50):
        self.max_seq_len = max_sequence_length
        self.special_tokens = {
            'CLS': 0,
            'SEP': 1,
            'MASK': 2,
            'PAD': 3
        }
    
    def tokenize_state(self, tensor_env: HUNLTensorEnv, env_idx: int, player: int):
        """Convert game state to token sequence"""
        tokens = []
        
        # Add CLS token
        tokens.append(self.special_tokens['CLS'])
        
        # Add visible cards
        visible_cards = tensor_env.get_visible_card_indices(env_idx, player)
        for card_idx in visible_cards:
            tokens.append(card_idx.item() + 4)  # Offset by special tokens
        
        # Add action history
        action_history = tensor_env.get_action_history()[env_idx]
        # Process action history into tokens...
        
        # Add context tokens
        pot_token = self._encode_pot_context(tensor_env, env_idx)
        stack_token = self._encode_stack_context(tensor_env, env_idx)
        tokens.extend([pot_token, stack_token])
        
        # Pad to max length
        while len(tokens) < self.max_seq_len:
            tokens.append(self.special_tokens['PAD'])
            
        return torch.tensor(tokens[:self.max_seq_len])
```

### Phase 4: Main Transformer Model

#### 4.1 PokerTransformer (`poker_transformer.py`)
```python
@register_model("poker_transformer_v1")
class PokerTransformerV1(nn.Module, Model):
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        vocab_size: int = 60,  # 52 cards + 8 special tokens
        num_actions: int = 8,
        dropout: float = 0.1,
        use_auxiliary_loss: bool = True,
    ):
        super().__init__()
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(50, d_model)  # max sequence length
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Output heads
        self.policy_head = PolicyHead(d_model, num_actions)
        self.value_head = ValueHead(d_model)
        
        if use_auxiliary_loss:
            self.hand_range_head = HandRangeHead(d_model)
        
        # Attention biases for relational structure
        self.register_attention_biases()
    
    def forward(self, token_ids, attention_mask=None):
        # Token embeddings + positional encoding
        seq_len = token_ids.size(1)
        pos_ids = torch.arange(seq_len, device=token_ids.device)
        
        x = self.token_emb(token_ids) + self.pos_emb(pos_ids)
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        
        # Output heads
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        outputs = {
            'policy_logits': policy_logits,
            'value': value
        }
        
        if hasattr(self, 'hand_range_head'):
            hand_range_logits = self.hand_range_head(x)
            outputs['hand_range_logits'] = hand_range_logits
            
        return outputs
```

#### 4.2 Policy Head (Mixed Discrete-Continuous)
```python
class PolicyHead(nn.Module):
    def __init__(self, d_model: int, num_action_types: int = 4):
        super().__init__()
        self.action_type_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_action_types)  # fold,check/call,raise,allin
        )
        
        self.size_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2)  # Beta distribution parameters
        )
    
    def forward(self, x):
        # Use CLS token representation
        cls_repr = x[:, 0]  # [batch, d_model]
        
        action_logits = self.action_type_head(cls_repr)
        size_params = torch.sigmoid(self.size_head(cls_repr))  # [batch, 2]
        
        return {
            'action_logits': action_logits,
            'size_params': size_params
        }
```

#### 4.3 Auxiliary Hand Range Head
```python
class HandRangeHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1326)  # All possible two-card combinations
        )
    
    def forward(self, x):
        cls_repr = x[:, 0]
        return self.head(cls_repr)
```

### Phase 5: Training Integration

#### 5.1 Update State Encoder
```python
class TransformerStateEncoder(StateEncoder):
    def __init__(self, tokenizer: PokerTokenizer, device: torch.device):
        self.tokenizer = tokenizer
        self.device = device
    
    def encode_tensor_states(self, tensor_env: HUNLTensorEnv, num_envs: int, **kwargs):
        """Encode states for transformer model"""
        batch_tokens = []
        batch_masks = []
        
        for env_idx in range(num_envs):
            # Get current player
            current_player = tensor_env.to_act[env_idx].item()
            
            # Tokenize state
            tokens = self.tokenizer.tokenize_state(tensor_env, env_idx, current_player)
            attention_mask = (tokens != self.tokenizer.special_tokens['PAD']).float()
            
            batch_tokens.append(tokens)
            batch_masks.append(attention_mask)
        
        return {
            'token_ids': torch.stack(batch_tokens),
            'attention_masks': torch.stack(batch_masks)
        }
```

#### 5.2 Loss Functions
```python
def transformer_ppo_loss(
    policy_logits,
    value_pred,
    old_policy_logits,
    old_value,
    advantages,
    returns,
    hand_range_logits=None,
    hand_range_targets=None,
    legal_action_masks=None,
    **kwargs
):
    # Standard PPO policy loss with legal action masking
    policy_loss = compute_policy_loss(policy_logits, old_policy_logits, advantages, legal_action_masks)
    
    # Value loss
    value_loss = F.mse_loss(value_pred, returns)
    
    # Auxiliary hand range loss
    aux_loss = 0
    if hand_range_logits is not None and hand_range_targets is not None:
        aux_loss = F.cross_entropy(hand_range_logits, hand_range_targets)
    
    return {
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'aux_loss': aux_loss,
        'total_loss': policy_loss + value_loss + 0.1 * aux_loss
    }
```

### Phase 6: Configuration and Testing

#### 6.1 Configuration Updates
```yaml
# conf/config_transformer.yaml
model:
  name: "poker_transformer_v1"
  d_model: 256
  n_layers: 4
  n_heads: 4
  vocab_size: 60
  dropout: 0.1
  use_auxiliary_loss: true

training:
  transformer_lr: 0.0001
  warmup_steps: 1000
  aux_loss_weight: 0.1
```

#### 6.2 Testing Strategy
1. **Unit Tests**: Test individual components (embeddings, tokenizer, heads)
2. **Integration Tests**: Test full forward pass with mock data
3. **Ablation Studies**: Compare with CNN model performance
4. **Latency Tests**: Ensure <3ms inference time target

### Phase 7: Performance Optimization

#### 7.1 Memory Optimization
- Gradient checkpointing for transformer layers
- Mixed precision training
- Efficient attention implementation (FlashAttention if available)

#### 7.2 Inference Optimization
- Model quantization for deployment
- Batch processing optimizations
- Caching of frequent token sequences

## Implementation Timeline

### Week 1: Foundation
- [ ] Create transformer module structure
- [ ] Implement basic embeddings
- [ ] Create tokenizer system

### Week 2: Core Model
- [ ] Implement PokerTransformer class
- [ ] Add policy/value heads
- [ ] Integrate with ModelFactory

### Week 3: Training Integration
- [ ] Update StateEncoder for transformer
- [ ] Implement transformer-specific loss functions
- [ ] Add auxiliary hand range head

### Week 4: Testing and Optimization
- [ ] Comprehensive testing suite
- [ ] Performance benchmarking
- [ ] Configuration system updates

## Key Design Decisions

### 1. Token Composition
- **Additive embeddings** for cards: `rank + suit + stage + visibility + order`
- **Separate streams** for cards vs actions vs context
- **Relational biases** in attention for poker-specific patterns

### 2. Sequence Structure
```
[CLS] [hole_cards] [action_history] [context_tokens] [PAD...]
```

### 3. Output Design
- **Mixed discrete-continuous** policy head
- **Auxiliary hand range** prediction for opponent modeling
- **Legal action masking** at policy head level

### 4. Training Strategy
- **Auxiliary losses** for hand range prediction
- **Masked token modeling** for action history
- **Next street prediction** for board cards

## Expected Benefits

1. **Better Relational Reasoning**: Attention mechanism captures complex card interactions
2. **Variable-Length History**: No fixed tensor dimensions for action sequences
3. **End-to-End Learning**: No hand-crafted features, pure data-driven
4. **Scalable Architecture**: Easy to add new token types or modify sequence structure
5. **Interpretable Attention**: Can visualize what the model focuses on

## Risks and Mitigations

### 1. Sample Efficiency
- **Risk**: Transformers may need more data than CNNs
- **Mitigation**: Auxiliary losses, pre-training on synthetic data

### 2. Information Leakage
- **Risk**: Model might learn from hidden opponent cards
- **Mitigation**: Strict masking, unit tests, ablation studies

### 3. Inference Speed
- **Risk**: Transformer inference slower than CNN
- **Mitigation**: Model compression, efficient attention, hardware optimization

### 4. Overfitting
- **Risk**: Model might overfit to training opponents
- **Mitigation**: K-best self-play, regularization, diverse opponent pool

## Success Metrics

1. **Performance**: Beat CNN model in head-to-head evaluation
2. **Speed**: Maintain <3ms inference time
3. **Stability**: Stable training without divergence
4. **Interpretability**: Meaningful attention patterns
5. **Robustness**: Good performance against diverse opponents

This plan provides a comprehensive roadmap for implementing a transformer-based poker model that leverages modern deep learning techniques while maintaining the efficiency and performance requirements of the original AlphaHoldem system.
