from __future__ import annotations

from types import SimpleNamespace

import torch

from p2.env.card_utils import PREFLOP_HANDS
from p2.env.pbs_env import PBSEnv
from p2.search.preflop_live_pair_distillation import uniform_live_pair_value_targets


class _IdentityBeliefEncoder:
    belief_dim = PREFLOP_HANDS

    def encode(
        self,
        beliefs: torch.Tensor,
        pre_chance_node: torch.Tensor | bool | None = None,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return beliefs


class _BeliefValueModel(torch.nn.Module):
    num_players = 2
    hand_dim = PREFLOP_HANDS

    def create_feature_encoder(
        self,
        *,
        env: PBSEnv,
        device: torch.device,
        dtype: torch.dtype,
    ) -> _IdentityBeliefEncoder:
        return _IdentityBeliefEncoder()

    def forward(
        self,
        features: torch.Tensor,
        *,
        include_policy: bool,
        apply_zero_sum: bool,
    ) -> SimpleNamespace:
        return SimpleNamespace(hand_values=features)


def _make_env(rows: int) -> PBSEnv:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)
    env = PBSEnv(
        num_envs=rows,
        num_players=6,
        mean_stack=10_000,
        sb=50,
        bb=100,
        default_bet_bins=[0.25, 0.5, 0.75, 1.0, 1.5],
        device=torch.device("cpu"),
        rng=generator,
        float_dtype=torch.float32,
        force_heads_up_preflop_flop=False,
    )
    env.reset()
    env.stacks.copy_(env.starting_stacks)
    env.has_folded.fill_(True)
    env.done.fill_(False)
    return env


def test_live_pair_targets_average_only_pairs_containing_player() -> None:
    env = _make_env(2)
    env.has_folded[0, :3] = False
    env.has_folded[1, 4:] = False

    beliefs = torch.zeros(2, 6, PREFLOP_HANDS, dtype=torch.float32)
    for player in range(6):
        beliefs[:, player, :] = 10.0 + float(player)

    targets = uniform_live_pair_value_targets(
        env,
        beliefs,
        target_model=_BeliefValueModel(),
        target_dtype=torch.float32,
    )

    for player in range(3):
        expected = torch.full((PREFLOP_HANDS,), 10.0 + float(player))
        torch.testing.assert_close(targets[0, player], expected)
    torch.testing.assert_close(targets[0, 3:], torch.zeros(3, PREFLOP_HANDS))

    torch.testing.assert_close(targets[1, :4], torch.zeros(4, PREFLOP_HANDS))
    torch.testing.assert_close(
        targets[1, 4],
        torch.full((PREFLOP_HANDS,), 14.0),
    )
    torch.testing.assert_close(
        targets[1, 5],
        torch.full((PREFLOP_HANDS,), 15.0),
    )
