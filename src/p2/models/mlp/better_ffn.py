from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn

from p2.core.structured_config import NonlinearityType
from p2.env.card_utils import NUM_HANDS, hand_combos_tensor
from p2.models.activation_utils import get_activation, SwiGLU
from p2.models.base_mlp_model import BaseMLPModel
from p2.models.mlp.better_feature_encoder import BetterFeatureEncoder
from p2.models.mlp.better_features import context_length
from p2.models.mlp.mlp_features import MLPFeatures
from p2.models.model_output import ModelOutput
from p2.utils.profiling import profile


HAND_STATIC_FEATURE_DIM = 8
HAND_DYNAMIC_FEATURE_DIM = 15


class ResidualBlock(nn.Module):
    def __init__(self, inner: nn.Module, alpha: float) -> None:
        super().__init__()
        self.inner = inner
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.inner(x) + x


def ffn_block(
    in_dim: int,
    hidden_dim: int,
    out_dim: int | None = None,
    nonlinearity: NonlinearityType = NonlinearityType.gelu,
) -> nn.Module:
    if out_dim is None:
        out_dim = in_dim
    if nonlinearity == NonlinearityType.swiglu:
        return nn.Sequential(
            OrderedDict(
                [
                    ("norm", nn.RMSNorm(in_dim, eps=1e-5)),
                    ("swiglu", SwiGLU(in_dim, hidden_dim, out_dim)),
                ]
            )
        )
    else:
        return nn.Sequential(
            OrderedDict(
                [
                    ("norm", nn.RMSNorm(in_dim, eps=1e-5)),
                    ("linear_in", nn.Linear(in_dim, hidden_dim, bias=False)),
                    ("activation", get_activation(nonlinearity)),
                    ("linear_out", nn.Linear(hidden_dim, out_dim)),
                ]
            )
        )


def output_projection(in_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        OrderedDict(
            [
                ("norm", nn.RMSNorm(in_dim, eps=1e-5)),
                ("linear_out", nn.Linear(in_dim, out_dim)),
            ]
        )
    )


class BetterFFN(BaseMLPModel):
    """Better PBS feed-forward poker model."""

    def __init__(
        self,
        num_actions: int,
        hidden_dim: int = 1024,
        range_hidden_dim: int = 256,
        ffn_dim: int = 1024,
        num_hidden_layers: int = 3,
        num_policy_layers: int = 3,
        num_value_layers: int = 3,
        num_players: int = 2,
        shared_trunk: bool = True,
        enforce_zero_sum: bool = True,
        board_interaction_dim: int = 0,
        policy_rank: int = 64,
        policy_hand_bias_rank: int = 32,
        policy_factor_scale: float = 0.5,
        nonlinearity: NonlinearityType = NonlinearityType.gelu,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_players = num_players
        self.shared_trunk = shared_trunk
        self.enforce_zero_sum = enforce_zero_sum
        self.board_interaction_dim = board_interaction_dim
        self.policy_rank = policy_rank
        self.policy_hand_bias_rank = policy_hand_bias_rank

        if range_hidden_dim < 0:
            raise ValueError("range_hidden_dim must be non-negative")
        if board_interaction_dim < 0:
            raise ValueError("board_interaction_dim must be non-negative")
        if policy_rank <= 0:
            raise ValueError("policy_rank must be positive")
        if policy_hand_bias_rank <= 0:
            raise ValueError("policy_hand_bias_rank must be positive")

        self.street_embedding = nn.Embedding(5, hidden_dim)
        self.rank_embedding = nn.Embedding(13 + 1, hidden_dim, padding_idx=13)
        self.suit_embedding = nn.Embedding(4 + 1, hidden_dim, padding_idx=4)
        self.card_embedding = nn.Embedding(52, hidden_dim)
        # Hand-aware belief encoder: project per-player belief vectors through a
        # hand embedding, then fuse across players. range_hidden_dim=0 keeps
        # this path and uses ffn_dim for the belief FFN width instead of
        # num_players * range_hidden_dim.
        combos = hand_combos_tensor()  # [NUM_HANDS, 2]
        self.register_buffer("hand_combos", combos, persistent=False)
        self.register_buffer("hand_card_a", combos[:, 0].long(), persistent=False)
        self.register_buffer("hand_card_b", combos[:, 1].long(), persistent=False)
        self.register_buffer("hand_ranks", combos % 13, persistent=False)
        self.register_buffer("hand_suits", combos // 13, persistent=False)
        hand_static_features = self._build_hand_static_features(
            self.hand_ranks, self.hand_suits
        )
        self.register_buffer(
            "hand_static_features", hand_static_features, persistent=False
        )
        hand_rank_pair_idx = self._unordered_pair_index(
            self.hand_ranks[:, 0], self.hand_ranks[:, 1], 13
        )
        self.register_buffer(
            "hand_rank_pair_one_hot",
            torch.nn.functional.one_hot(hand_rank_pair_idx, 91).to(torch.float32),
            persistent=False,
        )
        hand_suit_pair_idx = self._unordered_pair_index(
            self.hand_suits[:, 0], self.hand_suits[:, 1], 4
        )
        self.register_buffer(
            "hand_suit_pair_one_hot",
            torch.nn.functional.one_hot(hand_suit_pair_idx, 10).to(torch.float32),
            persistent=False,
        )
        if range_hidden_dim == 0 and ffn_dim % num_players != 0:
            raise ValueError(
                "ffn_dim must be divisible by num_players when range_hidden_dim is 0"
            )
        effective_range_hidden_dim = (
            ffn_dim // num_players if range_hidden_dim == 0 else range_hidden_dim
        )
        belief_in_dim = num_players * hidden_dim
        belief_hidden_dim = num_players * effective_range_hidden_dim
        self.hand_feature_proj = nn.Linear(
            HAND_STATIC_FEATURE_DIM, hidden_dim, bias=False
        )
        self.belief_proj = ffn_block(
            belief_in_dim, belief_hidden_dim, hidden_dim, nonlinearity
        )
        self.context_encoder = ffn_block(
            context_length(num_players), hidden_dim, hidden_dim, nonlinearity
        )
        if board_interaction_dim > 0:
            self.rank_pair_low_embedding = nn.Embedding(91, board_interaction_dim)
            self.board_rank_low = nn.Linear(13, board_interaction_dim, bias=False)
            self.rank_board_interaction_out = nn.Linear(
                num_players * board_interaction_dim, hidden_dim, bias=False
            )
            self.suit_pair_low_embedding = nn.Embedding(10, board_interaction_dim)
            self.board_suit_low = nn.Linear(4, board_interaction_dim, bias=False)
            self.suit_board_interaction_out = nn.Linear(
                num_players * board_interaction_dim, hidden_dim, bias=False
            )

        # Build trunk
        # Default alpha is always based on hidden + value layers
        alpha = 1 / math.sqrt(num_hidden_layers + num_value_layers)
        layers = [
            ResidualBlock(
                ffn_block(hidden_dim, ffn_dim, nonlinearity=nonlinearity), alpha
            )
            for _ in range(num_hidden_layers)
        ]
        self.trunk = nn.Sequential(*layers)

        # Heads
        # If shared_trunk is False, use separate alpha for policy_head based on num_policy_layers
        policy_alpha = alpha if shared_trunk else 1 / math.sqrt(num_policy_layers)

        layers = [
            ResidualBlock(
                ffn_block(hidden_dim, ffn_dim, nonlinearity=nonlinearity), policy_alpha
            )
            for _ in range(num_policy_layers)
        ]
        self.policy_tower = nn.Sequential(*layers)
        self.policy_hand_proj = output_projection(hidden_dim, self.policy_rank)
        self.policy_action_head = output_projection(
            hidden_dim, num_actions * self.policy_rank
        )
        self.policy_hand_gate = output_projection(hidden_dim, self.policy_rank)
        self.policy_dynamic_coeff = output_projection(
            hidden_dim, num_actions * HAND_DYNAMIC_FEATURE_DIM
        )
        self.policy_action_bias = output_projection(hidden_dim, num_actions)
        self.policy_hand_bias = output_projection(
            hidden_dim, self.policy_hand_bias_rank
        )
        self.policy_hand_bias_action = output_projection(
            hidden_dim, num_actions * self.policy_hand_bias_rank
        )
        self.policy_hand_norm = nn.RMSNorm(hidden_dim, eps=1e-5)
        self.policy_factor_scale = nn.Parameter(
            torch.tensor(float(policy_factor_scale))
        )

        layers = [
            ResidualBlock(
                ffn_block(hidden_dim, ffn_dim, nonlinearity=nonlinearity), alpha
            )
            for _ in range(num_value_layers)
        ]
        layers.append(output_projection(hidden_dim, num_players * NUM_HANDS))
        self.hand_value_head = nn.Sequential(*layers)

    @staticmethod
    def _unordered_pair_index(
        first: torch.Tensor, second: torch.Tensor, num_items: int
    ) -> torch.Tensor:
        lo = torch.minimum(first, second)
        hi = torch.maximum(first, second)
        return lo * num_items - (lo * (lo - 1)) // 2 + (hi - lo)

    @staticmethod
    def _index_counts(
        indices: torch.Tensor,
        valid: torch.Tensor,
        num_items: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        safe_indices = torch.where(valid, indices, torch.zeros_like(indices))
        counts = torch.zeros(
            indices.shape[0],
            num_items,
            device=indices.device,
            dtype=dtype,
        )
        return counts.scatter_add(1, safe_indices, valid.to(dtype))

    @staticmethod
    def _build_hand_static_features(
        hand_ranks: torch.Tensor, hand_suits: torch.Tensor
    ) -> torch.Tensor:
        rank_a = hand_ranks[:, 0].to(torch.float32)
        rank_b = hand_ranks[:, 1].to(torch.float32)
        suit_a = hand_suits[:, 0]
        suit_b = hand_suits[:, 1]
        hi = torch.maximum(rank_a, rank_b)
        lo = torch.minimum(rank_a, rank_b)
        gap = (hi - lo).clamp(min=0.0)
        return torch.stack(
            [
                (rank_a == rank_b).to(torch.float32),
                (suit_a == suit_b).to(torch.float32),
                gap / 12.0,
                hi / 12.0,
                lo / 12.0,
                (hi == 12).to(torch.float32),
                (lo >= 8).to(torch.float32),
                (gap <= 1).to(torch.float32),
            ],
            dim=-1,
        )

    def _hand_embedding(self) -> torch.Tensor:
        """Per-hand exact-card embedding — shape [NUM_HANDS, hidden_dim]."""
        card_emb = self.card_embedding(self.hand_combos)
        static = self.hand_static_features.to(dtype=card_emb.dtype)
        return card_emb.sum(dim=1) + self.hand_feature_proj(static)

    def _card_mass_and_unblocked_mass(
        self, belief: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        card_mass = self._card_mass(belief)
        return card_mass, self._unblocked_mass_from_card_mass(belief, card_mass)

    def _card_mass(self, belief: torch.Tensor) -> torch.Tensor:
        belief_batched = belief.view(-1, NUM_HANDS).float()
        card_mass = torch.zeros(
            belief_batched.shape[0],
            52,
            dtype=belief_batched.dtype,
            device=belief_batched.device,
        )
        card_a_idx = self.hand_card_a[None, :].expand(belief_batched.shape[0], -1)
        card_b_idx = self.hand_card_b[None, :].expand(belief_batched.shape[0], -1)
        card_mass.scatter_add_(1, card_a_idx, belief_batched)
        card_mass.scatter_add_(1, card_b_idx, belief_batched)
        return card_mass.view(*belief.shape[:-1], 52)

    def _unblocked_mass_from_card_mass(
        self, belief: torch.Tensor, card_mass: torch.Tensor
    ) -> torch.Tensor:
        belief = belief.float()
        card_mass = card_mass.float()
        total = belief.sum(dim=-1, keepdim=True)
        unblocked = (
            total
            - card_mass[..., self.hand_card_a]
            - card_mass[..., self.hand_card_b]
            + belief
        )
        return unblocked.clamp_min(0.0)

    def _calculate_unblocked_mass(self, target: torch.Tensor) -> torch.Tensor:
        _, unblocked = self._card_mass_and_unblocked_mass(target)
        return unblocked

    def _board_stats(
        self, board: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = board >= 0
        ranks = torch.where(valid, board % 13, torch.zeros_like(board))
        suits = torch.where(valid, board // 13, torch.zeros_like(board))
        rank_counts = self._index_counts(ranks, valid, 13, dtype)
        suit_counts = self._index_counts(suits, valid, 4, dtype)

        board_safe = torch.where(valid, board, torch.full_like(board, 52))
        board_onehot = torch.zeros(
            board.shape[0], 53, dtype=torch.bool, device=board.device
        )
        board_onehot.scatter_(1, board_safe, valid)
        board_onehot = board_onehot[:, :52]
        return rank_counts, suit_counts, board_onehot

    def _board_hand_feature_dot(
        self,
        board: torch.Tensor,
        coeff: torch.Tensor,
        board_stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        dtype = coeff.dtype
        if board_stats is None:
            rank_counts, suit_counts, board_onehot = self._board_stats(board, dtype)
        else:
            rank_counts, suit_counts, board_onehot = board_stats
            rank_counts = rank_counts.to(dtype=dtype)
            suit_counts = suit_counts.to(dtype=dtype)

        rank_a = rank_counts[:, self.hand_ranks[:, 0]]
        rank_b = rank_counts[:, self.hand_ranks[:, 1]]
        suit_a = suit_counts[:, self.hand_suits[:, 0]]
        suit_b = suit_counts[:, self.hand_suits[:, 1]]
        blocked = (
            board_onehot[:, self.hand_card_a] | board_onehot[:, self.hand_card_b]
        ).to(dtype)

        if coeff.dim() == 3:
            rank_a = rank_a[:, None]
            rank_b = rank_b[:, None]
            suit_a = suit_a[:, None]
            suit_b = suit_b[:, None]
            blocked = blocked[:, None]
            coeff = coeff[:, :, :, None]
        else:
            coeff = coeff[:, :, None]

        rank_sum = rank_a + rank_b
        suit_sum = suit_a + suit_b
        out = (rank_a / 4.0) * coeff[..., 0, :]
        out = out + (rank_b / 4.0) * coeff[..., 1, :]
        out = out + (rank_sum / 4.0) * coeff[..., 2, :]
        out = out + ((rank_a * rank_b) / 16.0) * coeff[..., 3, :]
        out = out + (suit_a / 5.0) * coeff[..., 4, :]
        out = out + (suit_b / 5.0) * coeff[..., 5, :]
        out = out + (suit_sum / 5.0) * coeff[..., 6, :]
        out = out + blocked * coeff[..., 7, :]
        return out

    def _board_hand_features_from_stats(
        self,
        board: torch.Tensor,
        dtype: torch.dtype,
        board_stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if board_stats is None:
            rank_counts, suit_counts, board_onehot = self._board_stats(board, dtype)
        else:
            rank_counts, suit_counts, board_onehot = board_stats
            rank_counts = rank_counts.to(dtype=dtype)
            suit_counts = suit_counts.to(dtype=dtype)

        rank_a = rank_counts[:, self.hand_ranks[:, 0]]
        rank_b = rank_counts[:, self.hand_ranks[:, 1]]
        suit_a = suit_counts[:, self.hand_suits[:, 0]]
        suit_b = suit_counts[:, self.hand_suits[:, 1]]
        blocked = (
            board_onehot[:, self.hand_card_a] | board_onehot[:, self.hand_card_b]
        ).to(dtype)
        rank_sum = rank_a + rank_b
        suit_sum = suit_a + suit_b
        return torch.stack(
            [
                rank_a / 4.0,
                rank_b / 4.0,
                rank_sum / 4.0,
                (rank_a * rank_b) / 16.0,
                suit_a / 5.0,
                suit_b / 5.0,
                suit_sum / 5.0,
                blocked,
            ],
            dim=-1,
        )

    def _dynamic_hand_features_from_stats(
        self,
        own_belief: torch.Tensor,
        own_card_mass: torch.Tensor,
        opp_card_mass: torch.Tensor,
        opp_unblocked: torch.Tensor,
        board: torch.Tensor,
        dtype: torch.dtype,
        board_stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        own_belief = own_belief.to(dtype=dtype)
        opp_unblocked = opp_unblocked.to(dtype=dtype)
        own_card_mass = own_card_mass.to(dtype=dtype)
        opp_card_mass = opp_card_mass.to(dtype=dtype)
        own_card_a = own_card_mass[..., self.hand_card_a]
        own_card_b = own_card_mass[..., self.hand_card_b]
        opp_card_a = opp_card_mass[..., self.hand_card_a]
        opp_card_b = opp_card_mass[..., self.hand_card_b]
        board_features = self._board_hand_features_from_stats(board, dtype, board_stats)
        if own_belief.dim() == 3:
            board_features = board_features[:, None].expand(
                -1, own_belief.shape[1], -1, -1
            )
        range_features = torch.stack(
            [
                own_belief,
                own_belief.clamp_min(1e-8).log(),
                opp_unblocked,
                own_card_a,
                own_card_b,
                opp_card_a,
                opp_card_b,
            ],
            dim=-1,
        )
        return torch.cat([range_features, board_features], dim=-1)

    def _dynamic_hand_feature_dot_from_stats(
        self,
        own_belief: torch.Tensor,
        own_card_mass: torch.Tensor,
        opp_card_mass: torch.Tensor,
        opp_unblocked: torch.Tensor,
        board: torch.Tensor,
        coeff: torch.Tensor,
        board_stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        dtype = coeff.dtype
        own_belief = own_belief.to(dtype=dtype)
        opp_unblocked = opp_unblocked.to(dtype=dtype)
        own_card_mass = own_card_mass.to(dtype=dtype)
        opp_card_mass = opp_card_mass.to(dtype=dtype)
        own_card_a = own_card_mass[..., self.hand_card_a]
        own_card_b = own_card_mass[..., self.hand_card_b]
        opp_card_a = opp_card_mass[..., self.hand_card_a]
        opp_card_b = opp_card_mass[..., self.hand_card_b]
        if coeff.dim() == own_belief.dim() + 1:
            own_belief = own_belief[:, None, :]
            opp_unblocked = opp_unblocked[:, None, :]
            own_card_a = own_card_a[:, None, :]
            own_card_b = own_card_b[:, None, :]
            opp_card_a = opp_card_a[:, None, :]
            opp_card_b = opp_card_b[:, None, :]

        out = own_belief * coeff[..., 0, None]
        out = out + own_belief.clamp_min(1e-8).log() * coeff[..., 1, None]
        out = out + opp_unblocked * coeff[..., 2, None]
        out = out + own_card_a * coeff[..., 3, None]
        out = out + own_card_b * coeff[..., 4, None]
        out = out + opp_card_a * coeff[..., 5, None]
        out = out + opp_card_b * coeff[..., 6, None]

        board_coeff = coeff[..., 7:HAND_DYNAMIC_FEATURE_DIM]
        return out + self._board_hand_feature_dot(board, board_coeff, board_stats)

    def _policy_dynamic_logits(
        self,
        player_beliefs: torch.Tensor,
        actor: torch.Tensor,
        board: torch.Tensor,
        coeff: torch.Tensor,
        board_stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.num_players != 2:
            raise ValueError("BetterFFN dynamic hand features require two players")
        actor_belief = player_beliefs.gather(
            1,
            actor[:, None, None].expand(-1, 1, NUM_HANDS),
        ).squeeze(1)
        opp = 1 - actor

        card_mass = self._card_mass(player_beliefs)
        actor_card_mass = card_mass.gather(
            1,
            actor[:, None, None].expand(-1, 1, 52),
        ).squeeze(1)
        opp_card_mass = card_mass.gather(
            1,
            opp[:, None, None].expand(-1, 1, 52),
        ).squeeze(1)
        opp_belief = player_beliefs.gather(
            1,
            opp[:, None, None].expand(-1, 1, NUM_HANDS),
        ).squeeze(1)
        opp_unblocked = self._unblocked_mass_from_card_mass(opp_belief, opp_card_mass)
        dynamic_logits = self._dynamic_hand_feature_dot_from_stats(
            actor_belief,
            actor_card_mass,
            opp_card_mass,
            opp_unblocked,
            board,
            coeff,
            board_stats,
        )
        return dynamic_logits.transpose(1, 2)

    def _policy_dynamic_features(
        self,
        player_beliefs: torch.Tensor,
        actor: torch.Tensor,
        board: torch.Tensor,
        dtype: torch.dtype,
        board_stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.num_players != 2:
            raise ValueError("BetterFFN dynamic hand features require two players")
        actor_belief = player_beliefs.gather(
            1,
            actor[:, None, None].expand(-1, 1, NUM_HANDS),
        ).squeeze(1)
        opp = 1 - actor

        card_mass = self._card_mass(player_beliefs)
        actor_card_mass = card_mass.gather(
            1,
            actor[:, None, None].expand(-1, 1, 52),
        ).squeeze(1)
        opp_card_mass = card_mass.gather(
            1,
            opp[:, None, None].expand(-1, 1, 52),
        ).squeeze(1)
        opp_belief = player_beliefs.gather(
            1,
            opp[:, None, None].expand(-1, 1, NUM_HANDS),
        ).squeeze(1)
        opp_unblocked = self._unblocked_mass_from_card_mass(opp_belief, opp_card_mass)
        return self._dynamic_hand_features_from_stats(
            actor_belief,
            actor_card_mass,
            opp_card_mass,
            opp_unblocked,
            board,
            dtype,
            board_stats,
        )

    def _hand_value_logits(
        self,
        value_input: torch.Tensor,
    ) -> torch.Tensor:
        return self.hand_value_head(value_input).view(-1, self.num_players, NUM_HANDS)

    def _policy_logits(
        self,
        policy_input: torch.Tensor,
        player_beliefs: torch.Tensor,
        actor: torch.Tensor,
        board: torch.Tensor,
        hand_emb: torch.Tensor,
        board_stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        policy_state = self.policy_tower(policy_input)
        action_emb = self.policy_action_head(policy_state).view(
            -1, self.num_actions, self.policy_rank
        )
        hand_gate = 1.0 + self.policy_hand_gate(policy_state).tanh()
        action_emb = action_emb * hand_gate[:, None, :]
        hand_vec = self.policy_hand_proj(hand_emb)
        logits = torch.einsum("hr,bar->bha", hand_vec, action_emb)
        logits = logits / math.sqrt(self.policy_rank)
        hand_bias = self.policy_hand_bias(hand_emb)
        hand_bias_action = self.policy_hand_bias_action(policy_state).view(
            -1, self.num_actions, self.policy_hand_bias_rank
        )
        logits = logits + torch.einsum("hk,bak->bha", hand_bias, hand_bias_action)

        dynamic_coeff = self.policy_dynamic_coeff(policy_state).view(
            -1, self.num_actions, HAND_DYNAMIC_FEATURE_DIM
        )
        if torch.is_grad_enabled():
            dynamic_features = self._policy_dynamic_features(
                player_beliefs, actor, board, policy_state.dtype, board_stats
            )
            logits = logits + torch.einsum(
                "bhf,baf->bha", dynamic_features, dynamic_coeff
            )
        else:
            logits = logits + self._policy_dynamic_logits(
                player_beliefs, actor, board, dynamic_coeff, board_stats
            )
        logits = logits + self.policy_action_bias(policy_state)[:, None, :]
        return logits * self.policy_factor_scale

    def _belief_board_interaction(
        self,
        player_beliefs: torch.Tensor,
        board_stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor | None:
        if self.board_interaction_dim <= 0:
            return None
        board_rank_counts, board_suit_counts, _ = board_stats

        rank_pair_mass = player_beliefs @ self.hand_rank_pair_one_hot.to(
            dtype=player_beliefs.dtype
        )
        rank_pair_low = rank_pair_mass @ self.rank_pair_low_embedding.weight
        board_rank_low = self.board_rank_low(board_rank_counts)
        rank_features = self.rank_board_interaction_out(
            (rank_pair_low * board_rank_low[:, None, :]).flatten(1)
        )

        suit_pair_mass = player_beliefs @ self.hand_suit_pair_one_hot.to(
            dtype=player_beliefs.dtype
        )
        suit_pair_low = suit_pair_mass @ self.suit_pair_low_embedding.weight
        board_suit_low = self.board_suit_low(board_suit_counts)
        suit_features = self.suit_board_interaction_out(
            (suit_pair_low * board_suit_low[:, None, :]).flatten(1)
        )

        return rank_features + suit_features

    def static_feature_base(self, features: MLPFeatures) -> torch.Tensor:
        """Feature contribution that is fixed for a CFR leaf row."""
        return self.static_feature_base_from_prefix(
            self.static_feature_prefix(features.context, features.street),
            features.board,
        )

    def static_feature_prefix(
        self, context: torch.Tensor, street: torch.Tensor
    ) -> torch.Tensor:
        """Feature contribution from context and street before board expansion."""
        return self.street_embedding(street) + self.context_encoder(context)

    def static_feature_base_from_prefix(
        self, prefix: torch.Tensor, board: torch.Tensor
    ) -> torch.Tensor:
        """Add board features to a precomputed context/street prefix."""
        ranks = torch.where(board >= 0, board % 13, torch.full_like(board, 13))
        suits = torch.where(board >= 0, board // 13, torch.full_like(board, 4))
        board_features = self.rank_embedding(ranks) + self.suit_embedding(suits)
        return board_features.sum(dim=1) + prefix

    def _forward_base(
        self,
        features: MLPFeatures,
        static_base_features: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        player_beliefs = features.beliefs.view(-1, self.num_players, NUM_HANDS)
        hand_emb = self._hand_embedding()  # [NUM_HANDS, hidden_dim]
        per_player_belief = player_beliefs @ hand_emb  # [B, P, H]
        belief_features = self.belief_proj(per_player_belief.flatten(1))

        if static_base_features is None:
            static_base_features = self.static_feature_base(features)
        flat_features = static_base_features + belief_features
        board_stats = self._board_stats(features.board, player_beliefs.dtype)
        interaction_features = self._belief_board_interaction(
            player_beliefs, board_stats
        )
        if interaction_features is not None:
            flat_features = flat_features + interaction_features
        # assert flat_features.isfinite().all()

        x = self.trunk(flat_features)
        # assert x.isfinite().all()
        return (
            player_beliefs,
            flat_features,
            x,
            self.policy_hand_norm(hand_emb),
            board_stats,
        )

    def forward_policy(
        self,
        features: MLPFeatures,
        latent=None,
        static_base_features: torch.Tensor | None = None,
    ) -> ModelOutput:
        player_beliefs, flat_features, x, hand_emb, board_stats = self._forward_base(
            features, static_base_features=static_base_features
        )
        policy_input = x if self.shared_trunk else flat_features.detach()
        policy_logits = self._policy_logits(
            policy_input,
            player_beliefs,
            features.to_act,
            features.board,
            hand_emb,
            board_stats,
        )
        return ModelOutput(policy_logits=policy_logits)

    def forward_value(
        self,
        features: MLPFeatures,
        latent=None,
        apply_zero_sum: bool = True,
        static_base_features: torch.Tensor | None = None,
    ) -> ModelOutput:
        """
        Value-only pass.

        apply_zero_sum controls where the zero-sum projection is applied, not
        whether it is required. If ``enforce_zero_sum`` is false this flag has no
        effect; if it is true and this flag is false, the caller must apply the
        projection after any value mixing.
        """
        player_beliefs, _, x, hand_emb, board_stats = self._forward_base(
            features, static_base_features=static_base_features
        )
        del hand_emb, board_stats
        hand_values_raw = self._hand_value_logits(x)
        if self.enforce_zero_sum and apply_zero_sum:
            hand_value_sums = (
                (hand_values_raw * player_beliefs)
                .sum(dim=2, keepdim=True)
                .mean(dim=1, keepdim=True)
            )
            hand_values = hand_values_raw - hand_value_sums
        else:
            hand_values = hand_values_raw
        value = hand_values.mean(dim=-1)
        return ModelOutput(value=value, hand_values=hand_values)

    def forward_both(
        self,
        features: MLPFeatures,
        latent=None,
        apply_zero_sum: bool = True,
        static_base_features: torch.Tensor | None = None,
    ) -> ModelOutput:
        player_beliefs, flat_features, x, hand_emb, board_stats = self._forward_base(
            features, static_base_features=static_base_features
        )
        policy_input = x if self.shared_trunk else flat_features.detach()
        policy_logits = self._policy_logits(
            policy_input,
            player_beliefs,
            features.to_act,
            features.board,
            hand_emb,
            board_stats,
        )
        hand_values_raw = self._hand_value_logits(x)
        if self.enforce_zero_sum and apply_zero_sum:
            hand_value_sums = (
                (hand_values_raw * player_beliefs)
                .sum(dim=2, keepdim=True)
                .mean(dim=1, keepdim=True)
            )
            hand_values = hand_values_raw - hand_value_sums
        else:
            hand_values = hand_values_raw
        value = hand_values.mean(dim=-1)
        return ModelOutput(
            policy_logits=policy_logits,
            value=value,
            hand_values=hand_values,
        )

    @profile
    def forward(
        self,
        features: MLPFeatures,
        include_policy: bool = True,
        include_value: bool = True,
        apply_zero_sum: bool = True,
        static_base_features: torch.Tensor | None = None,
        latent=None,
    ) -> ModelOutput:
        """Forward pass over flat feature vectors."""
        if include_policy and include_value:
            return self._call_forward_both(
                features,
                apply_zero_sum=apply_zero_sum,
                static_base_features=static_base_features,
            )
        if include_policy:
            return self._call_forward_policy(
                features, static_base_features=static_base_features
            )
        if include_value:
            return self._call_forward_value(
                features,
                apply_zero_sum=apply_zero_sum,
                static_base_features=static_base_features,
            )
        raise ValueError("At least one of include_policy/include_value must be true")

    def init_weights(self, rng: torch.Generator | None = None) -> None:
        """Initialize parameters following Xavier/RMSNorm defaults."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, generator=rng)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)

        expansion_gain = math.sqrt(self.ffn_dim / self.hidden_dim)
        for sequential in [self.trunk, self.policy_tower, self.hand_value_head]:
            for block in sequential.modules():
                if not isinstance(block, ResidualBlock):
                    continue
                inner = block.inner
                if "swiglu" in dict(inner.named_children()):
                    swiglu = inner.get_submodule("swiglu")
                    nn.init.orthogonal_(
                        swiglu.gate.weight, expansion_gain, generator=rng
                    )
                    nn.init.orthogonal_(swiglu.up.weight, expansion_gain, generator=rng)
                else:
                    # 1.532 is the gain for GELU nonlinearity.
                    nn.init.orthogonal_(
                        inner.get_submodule("linear_in").weight,
                        1.532 * expansion_gain,
                        generator=rng,
                    )

        # Guess hand values are around stddev 0.1.
        self.hand_value_head[-1].get_submodule("linear_out").weight.data.mul_(0.1)
        if self.board_interaction_dim > 0:
            self.rank_board_interaction_out.weight.data.mul_(0.1)
            self.suit_board_interaction_out.weight.data.mul_(0.1)

        # Start CFR warm-start policies close to uniform. The policy logits are
        # assembled from several additive branches; leaving all policy output
        # projections at orthogonal scale makes the random initial policy very
        # sharp, especially through dynamic log-belief features.
        self.policy_action_head.get_submodule("linear_out").weight.data.mul_(0.1)
        self.policy_hand_bias_action.get_submodule("linear_out").weight.data.mul_(
            0.01
        )
        self.policy_dynamic_coeff.get_submodule("linear_out").weight.data.mul_(0.01)
        self.policy_action_bias.get_submodule("linear_out").weight.data.mul_(0.01)

    def create_feature_encoder(
        self,
        env,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> BetterFeatureEncoder:
        return BetterFeatureEncoder(env=env, device=device, dtype=dtype)

    def repeat(
        self,
        features: MLPFeatures,
        count: int,
        include_policy: bool = False,
        include_value: bool = True,
    ) -> ModelOutput:
        return self(
            features, include_policy=include_policy, include_value=include_value
        )
