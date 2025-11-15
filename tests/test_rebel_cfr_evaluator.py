from __future__ import annotations

from typing import Callable

import pytest
import torch
import torch.nn as nn

from alphaholdem.core.structured_config import CFRType
from alphaholdem.env.card_utils import (
    combo_blocking_tensor,
    combo_index,
    combo_to_onehot_tensor,
    hand_combos_tensor,
    mask_conflicting_combos,
)
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.env.rules import rank_hands
from alphaholdem.models.mlp.better_features import ScalarContext
from alphaholdem.models.mlp.better_ffn import BetterFFN
from alphaholdem.models.mlp.mlp_features import MLPFeatures
from alphaholdem.models.model_output import ModelOutput
from alphaholdem.search.cfr_evaluator import PublicBeliefState
from alphaholdem.search.rebel_cfr_evaluator import (
    NUM_HANDS,
    RebelCFREvaluator,
)
from alphaholdem.utils.model_utils import compute_masked_logits


def get_device() -> torch.device:
    return (
        torch.device("cuda")
        if torch.cuda.is_available()
        else (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    )


def make_env(num_envs: int = 4, device: torch.device = None) -> HUNLTensorEnv:
    env = HUNLTensorEnv(
        num_envs=num_envs,
        starting_stack=1000,
        sb=5,
        bb=10,
        device=device,
        float_dtype=torch.float32,
        flop_showdown=False,
    )
    env.reset()
    return env


class MockModel:
    """Flexible model stub that can return custom logits and hand values."""

    def __init__(
        self,
        logits: torch.Tensor | None = None,
        hand_values: torch.Tensor | None = None,
        num_actions: int | None = None,
        num_players: int = 2,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        custom_logits_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        custom_hand_values_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.num_actions = num_actions
        self.num_players = num_players
        self.device = device
        self.dtype = dtype
        self.custom_logits_fn = custom_logits_fn
        self.custom_hand_values_fn = custom_hand_values_fn

        # Set up default logits
        if logits is not None:
            if logits.dim() == 1:
                logits = logits.unsqueeze(0).unsqueeze(0).expand(1, NUM_HANDS, -1)
            elif logits.dim() == 2:
                logits = logits.unsqueeze(0)
            self.logits = logits
        else:
            self.logits = None

        # Set up default hand values
        if hand_values is not None:
            self.hand_values = hand_values
        else:
            self.hand_values = torch.zeros(1, num_players, NUM_HANDS)

    def __call__(self, features: MLPFeatures) -> ModelOutput:
        batch = len(features)
        device = features.context.device
        dtype = features.context.dtype

        # Handle logits
        if self.custom_logits_fn:
            logits = self.custom_logits_fn(features)
        elif self.logits is not None:
            logits = self.logits.to(device=device, dtype=dtype)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0).unsqueeze(0).expand(batch, NUM_HANDS, -1)
            elif logits.dim() == 2:
                logits = logits.unsqueeze(0).expand(batch, -1, -1)
            elif logits.shape[0] != batch:
                logits = logits.expand(batch, -1, -1)
        else:
            # Default zero logits
            logits = torch.zeros(
                batch, NUM_HANDS, self.num_actions or 8, device=device, dtype=dtype
            )

        # Handle hand values
        if self.custom_hand_values_fn:
            hand_values = self.custom_hand_values_fn(features)
        else:
            hand_values = self.hand_values.to(device=device, dtype=dtype)
            if hand_values.shape[0] > batch:
                hand_values = hand_values[:batch]
            elif hand_values.shape[0] < batch:
                hand_values = hand_values.expand(batch, -1, -1)

        return ModelOutput(
            policy_logits=logits,
            value=torch.zeros(batch, device=device, dtype=dtype),
            hand_values=hand_values,
        )

    def eval(self) -> None:
        pass

    def train(self) -> None:
        pass


def make_evaluator(
    *,
    batch_size: int = 2,
    max_depth: int = 2,
    bet_bins: list[float] | None = None,
    cfr_iterations: int = 4,
    device: torch.device = torch.device("cpu"),
) -> tuple[RebelCFREvaluator, HUNLTensorEnv]:
    env = make_env(batch_size, device=device)
    bet_bins = bet_bins or env.default_bet_bins
    model = MockModel(
        logits=torch.zeros(len(bet_bins) + 3, dtype=env.float_dtype),
        hand_values=torch.zeros(1, 2, NUM_HANDS, dtype=env.float_dtype),
        device=device,
    )
    evaluator = RebelCFREvaluator(
        search_batch_size=batch_size,
        env_proto=env,
        model=model,  # type: ignore[arg-type]
        bet_bins=bet_bins,
        max_depth=max_depth,
        cfr_iterations=max(32, cfr_iterations),
        device=env.device,
        float_dtype=env.float_dtype,
    )
    return evaluator, env


def _get_child_nodes(evaluator: RebelCFREvaluator, env: HUNLTensorEnv) -> torch.Tensor:
    """Get child nodes (non-root valid nodes) from the evaluator."""
    return torch.where(
        evaluator.valid_mask
        & (
            torch.arange(evaluator.total_nodes, device=env.device)
            >= evaluator.root_nodes
        )
    )[0]


def _copy_root_beliefs_to_children(evaluator: RebelCFREvaluator) -> None:
    """Utility to mirror the root public beliefs onto depth-1 nodes for tests."""
    num_actions = evaluator.num_actions
    if num_actions <= 0 or evaluator.total_nodes <= 1:
        return
    root_beliefs = evaluator.beliefs[0].clone()
    root_beliefs_avg = evaluator.beliefs_avg[0].clone()
    evaluator.beliefs[1 : 1 + num_actions] = root_beliefs
    evaluator.beliefs_avg[1 : 1 + num_actions] = root_beliefs_avg


RANKS = "23456789TJQKA"
SUITS = "shdc"


def card(rank: int, suit: int) -> int:
    """Return numeric card index given rank 0..12 (2..A) and suit 0..3."""
    return suit * 13 + rank


def make_board(cards: str, device: torch.device | None = None) -> torch.Tensor:
    """Return numeric board indices given a string of card names."""
    return torch.tensor(
        [card(RANKS.index(c[0]), SUITS.index(c[1])) for c in cards.split()],
        device=device,
    )


def parse_card(name: str) -> int:
    """Convert a human-readable card like 'As' to its numeric index."""
    return card(RANKS.index(name[0]), SUITS.index(name[1]))


def combo_from_strs(card_a: str, card_b: str) -> int:
    """Return combo index for two human-readable cards."""
    return combo_index(parse_card(card_a), parse_card(card_b))


def compute_reference_showdown_ev(
    board: torch.Tensor,
    opp_beliefs: torch.Tensor,
    potential: float,
) -> torch.Tensor:
    """
    Compute per-hand showdown EVs by enumerating opponent holdings.

    Args:
        board: [5] tensor of board card indices.
        opp_beliefs: [1326] prior over opponent combos.
        potential: Chips gained on hero win (and lost on hero loss).
        device: Target device.
        dtype: Floating dtype for computation.
    """

    if not isinstance(potential, torch.Tensor):
        potential = torch.tensor(potential)

    board_mask = mask_conflicting_combos(board)
    opp_masked = opp_beliefs * board_mask
    opp_probs = opp_masked / opp_masked.sum().clamp_min(1e-12)

    blocking = combo_blocking_tensor()
    hand_ranks, _ = rank_hands(board.unsqueeze(0))
    hand_ranks = hand_ranks.squeeze(0)

    ev_per_hand = torch.zeros(NUM_HANDS)
    valid_indices = torch.where(board_mask)[0]
    for combo_idx in valid_indices.tolist():
        compat_mask = (~blocking[combo_idx]) & board_mask
        opp_cond = opp_probs * compat_mask
        opp_total = opp_cond.sum()
        if opp_total <= 0:
            continue
        opp_cond = opp_cond / opp_total

        hero_rank = hand_ranks[combo_idx]
        opp_ranks = hand_ranks
        win_prob = (opp_cond * (opp_ranks < hero_rank)).sum()
        loss_prob = (opp_cond * (opp_ranks > hero_rank)).sum()

        ev_per_hand[combo_idx] = potential * (win_prob - loss_prob)

    return ev_per_hand


def test_initialize_subgame_sets_uniform_beliefs() -> None:
    evaluator, env = make_evaluator(batch_size=2, max_depth=1)
    roots = torch.arange(evaluator.root_nodes, device=env.device)
    evaluator.initialize_subgame(env, roots)

    assert torch.all(evaluator.valid_mask[roots])

    # Verify root nodes have self_reach initialized to 1.0
    root_self_reach = evaluator.self_reach[roots]
    torch.testing.assert_close(
        root_self_reach,
        torch.ones_like(root_self_reach),
        msg="Root nodes should have self_reach initialized to 1.0",
    )

    root_beliefs = evaluator.beliefs[roots]
    torch.testing.assert_close(
        root_beliefs.sum(dim=-1),
        torch.ones(
            (evaluator.root_nodes, evaluator.num_players),
            device=env.device,
            dtype=env.float_dtype,
        ),
    )
    uniform = torch.full((NUM_HANDS,), 1.0 / NUM_HANDS, dtype=env.float_dtype)
    torch.testing.assert_close(root_beliefs[0, 0].cpu(), uniform)
    torch.testing.assert_close(root_beliefs[0, 1].cpu(), uniform)


def test_initialize_subgame_marks_done_roots_as_leaves() -> None:
    evaluator, env = make_evaluator(batch_size=2, max_depth=1)
    env.done[0] = True
    roots = torch.arange(evaluator.root_nodes, device=env.device)
    evaluator.initialize_subgame(env, roots)
    assert evaluator.leaf_mask[0]
    assert not evaluator.leaf_mask[1]


def test_initialize_subgame_keeps_to_call_non_negative() -> None:
    evaluator, env = make_evaluator(batch_size=32, max_depth=2)
    roots = torch.arange(evaluator.root_nodes, device=env.device)
    evaluator.initialize_subgame(env, roots)

    env_state = evaluator.env
    indices = torch.arange(env_state.N, device=env_state.device)
    to_act = env_state.to_act
    opp = 1 - to_act
    to_call = env_state.committed[indices, opp] - env_state.committed[indices, to_act]
    mask = evaluator.valid_mask & ~env_state.done
    assert torch.all(to_call[mask] >= 0)


def test_initialize_subgame_keeps_stacks_non_negative() -> None:
    evaluator, env = make_evaluator(batch_size=32, max_depth=2)
    roots = torch.arange(evaluator.root_nodes, device=env.device)
    evaluator.initialize_subgame(env, roots)

    env_state = evaluator.env
    mask = evaluator.valid_mask & ~env_state.done
    assert torch.all(env_state.stacks[mask] >= 0)


def test_subgame_initialization_clones_states_and_marks_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator, env = make_evaluator(batch_size=1, max_depth=2)
    total_nodes = evaluator.total_nodes
    num_actions = evaluator.num_actions
    roots = torch.arange(evaluator.root_nodes, device=env.device)
    legal_mask = torch.zeros(
        (total_nodes, num_actions), dtype=torch.bool, device=env.device
    )
    # Make sure all valid nodes have at least one legal action
    legal_mask[0, 0:2] = True  # Root has actions 0 and 1
    # For child nodes, we need to ensure they have legal actions too
    for i in range(1, total_nodes):  # Cover first few child nodes
        legal_mask[i, 1] = True  # Give them at least action 1

    monkeypatch.setattr(
        evaluator.env,
        "legal_bins_mask",
        lambda bet_bins=None: legal_mask,
    )

    step_observer: dict[str, torch.Tensor] = {}

    def fake_step_bins(
        bin_indices: torch.Tensor,
        bin_amounts: torch.Tensor | None = None,
        legal_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        step_observer["bins"] = bin_indices.clone()
        rewards = torch.zeros_like(bin_indices, dtype=evaluator.float_dtype)
        new_streets = torch.full_like(bin_indices, -1, dtype=torch.long)
        dealt_cards = torch.full(
            (bin_indices.shape[0], 3),
            -1,
            dtype=torch.long,
            device=bin_indices.device,
        )
        return rewards, new_streets, dealt_cards

    monkeypatch.setattr(evaluator.env, "step_bins", fake_step_bins)

    evaluator.initialize_subgame(env, roots)

    # Check that child nodes were created
    child_nodes = _get_child_nodes(evaluator, env)
    assert child_nodes.numel() >= 2

    # Check that the first child node has the expected properties
    first_child = child_nodes[0]
    torch.testing.assert_close(
        evaluator.env.pot[first_child],
        evaluator.env.pot[roots[0]].expand(first_child.numel()).squeeze(),
    )
    # Check that at least some child nodes are marked as leaves
    assert torch.any(evaluator.leaf_mask[child_nodes])
    assert "bins" in step_observer


def test_initialize_subgame_depth4_street_changes_and_legal_masks() -> None:
    """Test that initialize_subgame at depth 4 correctly handles street changes and legal masks.

    Verifies:
    1. Only leaf nodes go to a new street from the root node
    2. Legal masks are True only for nodes that are NOT leaves AND NOT street changes
    """
    evaluator, env = make_evaluator(batch_size=8, max_depth=4)
    roots = torch.arange(evaluator.root_nodes, device=env.device)
    evaluator.initialize_subgame(env, roots)

    valid_nodes = torch.where(evaluator.valid_mask)[0]

    # Check 1: Only leaf nodes go to a new street from the root node
    # new_street_mask identifies nodes that transitioned to a new street (and are not root nodes)
    # All such nodes must be leaf nodes
    if evaluator.new_street_mask.any():
        new_street_nodes = torch.where(
            evaluator.new_street_mask & evaluator.valid_mask
        )[0]
        if new_street_nodes.numel() > 0:
            assert torch.all(
                evaluator.leaf_mask[new_street_nodes]
            ), "Only leaf nodes should go to a new street from the root node"

    # Check 2: Legal masks = True only for nodes that are NOT leaves AND NOT street changes
    # All street change nodes (identified by new_street_mask) should be marked as leaves
    if evaluator.new_street_mask.any():
        new_street_valid = evaluator.new_street_mask & evaluator.valid_mask
        if new_street_valid.any():
            assert torch.all(
                evaluator.leaf_mask[new_street_valid]
            ), "Street change nodes should be marked as leaves"

    # Legal mask should have at least one True for valid non-leaf nodes
    # (street changes are already leaves, so we just check non-leaf)
    non_leaf_nodes = valid_nodes[~evaluator.leaf_mask[valid_nodes]]
    if non_leaf_nodes.numel() > 0:
        legal_mask_has_action = evaluator.legal_mask[non_leaf_nodes].any(dim=-1)
        assert torch.all(
            legal_mask_has_action
        ), "All valid non-leaf nodes should have at least one legal action"

    # Verify that leaf nodes (including street changes) are excluded from legal mask checks.
    # This is enforced inside initialize_subgame when the tree expansion checks
    # the legal masks for valid_mask & ~leaf_mask nodes.


def test_initialize_policy_respects_legal_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator, env = make_evaluator(batch_size=1, max_depth=1)
    roots = torch.arange(evaluator.root_nodes, device=env.device)
    evaluator.initialize_subgame(env, roots)

    num_actions = evaluator.num_actions
    legal_mask = torch.zeros(
        (evaluator.total_nodes, num_actions),
        dtype=torch.bool,
        device=env.device,
    )
    legal_mask[0, 1:3] = True  # Only allow actions 1 and 2 at the root
    evaluator.valid_mask.zero_()
    evaluator.valid_mask[0] = True
    evaluator.valid_mask[2:4] = True

    monkeypatch.setattr(
        evaluator.env,
        "legal_bins_mask",
        lambda bet_bins=None: legal_mask,
    )

    # Initialize legal_masks before calling initialize_policy_and_beliefs
    evaluator.legal_mask = evaluator.env.legal_bins_mask()

    logits = torch.arange(float(num_actions), dtype=env.float_dtype)
    evaluator.model = MockModel(logits=logits)  # type: ignore[assignment]
    evaluator.initialize_policy_and_beliefs()

    expected = torch.softmax(
        compute_masked_logits(logits.unsqueeze(0), legal_mask[:1]), dim=-1
    )
    expected = expected * legal_mask[:1]
    expected = expected / expected.sum(dim=-1, keepdim=True)
    expected = expected.squeeze(0)

    root_policy = evaluator.policy_probs[1:9, 0]
    torch.testing.assert_close(root_policy, expected)


def test_initialize_beliefs_updates_child_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator, env = make_evaluator(batch_size=1, max_depth=2)
    total_nodes = evaluator.total_nodes
    num_actions = evaluator.num_actions
    legal_mask = torch.zeros(
        (total_nodes, num_actions), dtype=torch.bool, device=env.device
    )
    legal_mask[0, 0:2] = True
    legal_mask[1:, 1] = True

    monkeypatch.setattr(
        evaluator.env,
        "legal_bins_mask",
        lambda bet_bins=None: legal_mask,
    )

    roots = torch.arange(evaluator.root_nodes, device=env.device)
    evaluator.initialize_subgame(env, roots)

    root = 0
    child_nodes = _get_child_nodes(evaluator, env)
    assert child_nodes.numel() >= 2
    child_a, child_b = child_nodes[0], child_nodes[1]

    evaluator.env.to_act[root] = 0

    hand_ids = torch.arange(NUM_HANDS, device=env.device, dtype=env.float_dtype)
    root_actor = hand_ids + 1.0
    root_actor = root_actor / root_actor.sum()
    root_opp = torch.full_like(root_actor, 1.0 / NUM_HANDS)
    evaluator.beliefs[root, 0] = root_actor
    evaluator.beliefs[root, 1] = root_opp

    # Set up the model to return the expected policy probabilities
    action0_scores = hand_ids + 1.0
    action1_scores = torch.flip(action0_scores, dims=[0])
    score_sum = action0_scores + action1_scores
    policy_action0 = action0_scores / score_sum
    policy_action1 = action1_scores / score_sum

    # Create a model that returns the expected policy probabilities
    def custom_logits_fn(features):
        batch_size = len(features)
        logits = torch.zeros(
            batch_size,
            NUM_HANDS,
            evaluator.num_actions,
            device=env.device,
            dtype=env.float_dtype,
        )
        logits[:, :, 0] = action0_scores.log()
        logits[:, :, 1] = action1_scores.log()
        logits[:, :, 2:] = -float("inf")
        return logits

    evaluator.model = MockModel(
        custom_logits_fn=custom_logits_fn,
        num_actions=evaluator.num_actions,
        num_players=evaluator.num_players,
        device=env.device,
        dtype=env.float_dtype,
    )

    # Initialize legal_masks before calling initialize_policy_and_beliefs
    evaluator.legal_mask = evaluator.env.legal_bins_mask()

    evaluator.initialize_policy_and_beliefs()

    expected_child_a = root_actor * policy_action0
    expected_child_a = expected_child_a / expected_child_a.sum()
    expected_child_b = root_actor * policy_action1
    expected_child_b = expected_child_b / expected_child_b.sum()

    torch.testing.assert_close(evaluator.beliefs[child_a, 0], expected_child_a)
    torch.testing.assert_close(evaluator.beliefs[child_a, 1], root_opp)
    torch.testing.assert_close(evaluator.beliefs[child_b, 0], expected_child_b)
    torch.testing.assert_close(evaluator.beliefs[child_b, 1], root_opp)

    policy_probs_src = evaluator._pull_back(evaluator.policy_probs)
    valid_mask_src = evaluator.valid_mask[: policy_probs_src.shape[0]]
    prob_sum = policy_probs_src[valid_mask_src].sum(dim=1)
    # Only check nodes that have non-zero policy sums (nodes with policies set)
    has_policy = prob_sum > 1e-6
    torch.testing.assert_close(
        prob_sum[has_policy], torch.ones_like(prob_sum[has_policy])
    )
    belief_sum = evaluator.beliefs[evaluator.valid_mask].sum(dim=2)
    torch.testing.assert_close(belief_sum, torch.ones_like(belief_sum))


def test_fan_out_deep_repeats_root_beliefs() -> None:
    evaluator, _ = make_evaluator(batch_size=2, max_depth=2)
    root_data = torch.arange(
        evaluator.root_nodes * 3,
        device=evaluator.device,
        dtype=evaluator.float_dtype,
    ).view(evaluator.root_nodes, 3)

    broadcast = evaluator._fan_out_deep(root_data)
    for depth in range(evaluator.max_depth):
        offset = evaluator.depth_offsets[depth]
        offset_next = evaluator.depth_offsets[depth + 1]
        offset_next_next = evaluator.depth_offsets[depth + 2]
        expected = broadcast[offset:offset_next].repeat_interleave(
            evaluator.num_actions, dim=0
        )
        torch.testing.assert_close(broadcast[offset_next:offset_next_next], expected)


def test_compute_expected_values_matches_child_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator, env = make_evaluator(batch_size=1, max_depth=2)
    roots = torch.arange(evaluator.root_nodes)

    num_actions = evaluator.num_actions
    legal_mask = torch.ones((evaluator.total_nodes, num_actions), dtype=torch.bool)
    original_fn = evaluator.env.legal_bins_amounts_and_mask

    def fake_legal_bins_amounts_and_mask(bet_bins=None):
        bin_amounts, _ = original_fn(bet_bins)
        return bin_amounts.clone(), legal_mask.clone()

    monkeypatch.setattr(
        evaluator.env,
        "legal_bins_amounts_and_mask",
        fake_legal_bins_amounts_and_mask,
    )

    evaluator.initialize_subgame(env, roots)
    evaluator.initialize_policy_and_beliefs()
    evaluator.set_leaf_values(0)

    probs = torch.arange(1, num_actions + 1, dtype=env.float_dtype)
    probs = probs / probs.sum()
    probs_all = probs[None, :, None].expand(evaluator.total_nodes, -1, 1)
    bottom = evaluator.depth_offsets[1]
    # fixed policy probs for every node.
    evaluator.policy_probs[bottom:] = evaluator._push_down(probs_all)

    child_values = torch.arange(1, num_actions + 1, dtype=env.float_dtype)
    child_values_all = child_values[None, :, None].expand(evaluator.total_nodes, -1, 1)
    values_temp = evaluator._push_down(child_values_all)
    # fixed expected values for all the leaf nodes.
    values_bottom = torch.where(evaluator.leaf_mask[bottom:, None], values_temp, 0.0)

    # counterfactual value = EV * opponent reach
    # Root nodes should already be initialized to 1.0 by initialize_subgame
    reach_weights = evaluator.self_reach.clone()
    torch.testing.assert_close(
        reach_weights[: evaluator.root_nodes],
        torch.ones_like(reach_weights[: evaluator.root_nodes]),
        msg="Root nodes should start at 1.0 from initialize_subgame",
    )
    evaluator._calculate_reach_weights(reach_weights, evaluator.policy_probs)
    # Verify root nodes remain at 1.0 (never updated by _calculate_reach_weights)
    torch.testing.assert_close(
        reach_weights[: evaluator.root_nodes],
        torch.ones_like(reach_weights[: evaluator.root_nodes]),
        msg="Root nodes should remain at 1.0 after _calculate_reach_weights",
    )
    evaluator.latest_values[:] = 0.0
    evaluator.latest_values[bottom:, 0] = values_bottom * reach_weights[bottom:, 1]
    evaluator.latest_values[bottom:, 1] = -values_bottom * reach_weights[bottom:, 0]

    evaluator.compute_expected_values()

    # actor is player 1 at node 2.
    expected_value_actor = (probs[:, None] * evaluator.latest_values[17:25, 1]).sum(
        dim=0
    )
    expected_value_opp = evaluator.latest_values[17:25, 0].sum(dim=0)

    torch.testing.assert_close(evaluator.latest_values[2, 0], expected_value_opp)
    torch.testing.assert_close(evaluator.latest_values[2, 1], expected_value_actor)


def test_set_leaf_values_only_updates_marked_nodes() -> None:
    evaluator, env = make_evaluator(batch_size=1, max_depth=2)
    roots = torch.arange(evaluator.root_nodes)
    evaluator.initialize_subgame(env, roots)
    evaluator.initialize_policy_and_beliefs()

    leaf_indices = torch.tensor([2, 4])
    evaluator.valid_mask[leaf_indices] = True
    evaluator.leaf_mask.zero_()
    evaluator.leaf_mask[leaf_indices] = True

    evaluator.latest_values.zero_()

    hand_values = torch.zeros(
        evaluator.total_nodes,
        evaluator.num_players,
        NUM_HANDS,
        dtype=env.float_dtype,
    )
    hand_values[2] = 2.5
    hand_values[4] = -1.25

    # Mock the model to return the expected hand values
    def custom_hand_values_fn(features):
        batch_size = len(features)
        model_hand_values = torch.zeros(
            batch_size,
            evaluator.num_players,
            NUM_HANDS,
            dtype=env.float_dtype,
        )
        model_hand_values[:] = hand_values[leaf_indices]
        return model_hand_values

    evaluator.model = MockModel(
        custom_hand_values_fn=custom_hand_values_fn,
        num_actions=len(evaluator.bet_bins) + 3,
        num_players=evaluator.num_players,
        dtype=env.float_dtype,
    )

    evaluator.set_leaf_values(0)

    torch.testing.assert_close(evaluator.latest_values[2, 0], hand_values[2, 0])
    torch.testing.assert_close(evaluator.latest_values[2, 1], hand_values[2, 1])
    torch.testing.assert_close(evaluator.latest_values[4, 0], hand_values[4, 0])
    torch.testing.assert_close(evaluator.latest_values[4, 1], hand_values[4, 1])
    assert (
        torch.count_nonzero(
            evaluator.latest_values[~evaluator.env.done & ~evaluator.leaf_mask]
        )
        == 0
    )


def test_sample_leaf_copies_selected_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test sample_leaves returns a valid PBS with sampled nodes."""
    torch.manual_seed(0)
    evaluator, env = make_evaluator(batch_size=1, max_depth=2)
    roots = torch.arange(evaluator.root_nodes, device=env.device)
    evaluator.initialize_subgame(env, roots)
    evaluator.initialize_policy_and_beliefs()

    evaluator.generator = torch.Generator(device=env.device)
    evaluator.generator.manual_seed(42)
    evaluator.sample_epsilon = 0.0

    # Set up policy_probs_sample for sampling
    evaluator.policy_probs_sample[:] = evaluator.policy_probs

    # Initialize legal_mask in evaluator
    evaluator.legal_mask = evaluator.env.legal_bins_mask()

    pbs = evaluator.sample_leaves(training_mode=False)

    # sample_leaves returns a PBS with sampled nodes (non-root nodes)
    # If all sampled nodes are roots, it returns a PBS with 0 environments
    assert pbs is not None
    # The PBS should have the correct structure
    assert pbs.beliefs.shape[0] == pbs.env.N
    assert pbs.beliefs.shape == (pbs.env.N, evaluator.num_players, NUM_HANDS)
    # If we sampled a non-root node, check its properties
    if pbs.env.N > 0:
        # Check that beliefs are valid (sum to 1)
        belief_sums = pbs.beliefs.sum(dim=-1)
        torch.testing.assert_close(belief_sums, torch.ones_like(belief_sums))
        assert not hasattr(pbs, "pre_chance_beliefs")


def test_sample_leaf_handles_partial_masks() -> None:
    evaluator, env = make_evaluator(batch_size=3, max_depth=2)
    roots = torch.arange(evaluator.root_nodes, device=env.device)
    evaluator.initialize_subgame(env, roots)
    evaluator.initialize_policy_and_beliefs()

    # Set up policy_probs_sample for sampling
    evaluator.policy_probs_sample[:] = evaluator.policy_probs

    pbs = evaluator.sample_leaves(training_mode=True)

    # sample_leaves may return None if all sampled nodes are roots
    # If it returns a PBS, check its properties
    if pbs is not None:
        assert pbs.env.N >= 0
        assert pbs.beliefs.shape[0] == pbs.env.N
        assert pbs.beliefs.shape == (pbs.env.N, evaluator.num_players, NUM_HANDS)


def test_update_policy_uses_positive_regrets(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator, env = make_evaluator(batch_size=1, max_depth=2, device=get_device())
    roots = torch.arange(evaluator.root_nodes, device=env.device)
    evaluator.initialize_subgame(env, roots)
    evaluator.initialize_policy_and_beliefs()
    evaluator.set_leaf_values(0)

    num_actions = evaluator.num_actions
    legal_mask = torch.ones(
        (evaluator.total_nodes, num_actions),
        dtype=torch.bool,
        device=env.device,
    )
    monkeypatch.setattr(
        evaluator.env,
        "legal_bins_mask",
        lambda bet_bins=None: legal_mask,
    )

    root_index = 0
    child_indices = torch.arange(1, 1 + num_actions, device=env.device)

    evaluator.valid_mask[:] = False
    evaluator.valid_mask[root_index] = True
    evaluator.valid_mask[child_indices] = True
    evaluator.leaf_mask[:] = True
    evaluator.leaf_mask[root_index] = False
    evaluator.allowed_hands[root_index] = True
    evaluator.allowed_hands_prob[root_index] = 1.0 / NUM_HANDS
    evaluator.allowed_hands[child_indices] = True
    evaluator.allowed_hands_prob[child_indices] = 1.0 / NUM_HANDS

    evaluator.env.to_act[root_index] = 0
    evaluator.env.to_act[child_indices] = 1

    uniform = torch.full(
        (NUM_HANDS,), 1.0 / NUM_HANDS, dtype=env.float_dtype, device=env.device
    )
    evaluator.beliefs[:] = uniform[None, None, :]

    evaluator.policy_probs[:] = 1.0 / num_actions

    evaluator.latest_values[:] = 0.0
    evaluator.latest_values[1, 0] = 2.0  # Positive advantage for action 0
    evaluator.latest_values[2, 0] = -1.0  # Negative advantage for action 1
    evaluator.compute_expected_values()

    # Compute regrets first, then update policy
    regrets = evaluator.compute_instantaneous_regrets(evaluator.latest_values)
    evaluator.cumulative_regrets += regrets
    evaluator.update_policy(1)

    root_policy = evaluator.policy_probs[1 : num_actions + 1, 0]
    expected = torch.zeros(num_actions, dtype=env.float_dtype, device=env.device)
    expected[0] = 1.0
    torch.testing.assert_close(root_policy, expected)


def test_training_data_returns_root_batch() -> None:
    evaluator, env = make_evaluator(batch_size=2, max_depth=1)
    env.step_bins(torch.tensor([1, 1]))
    env.step_bins(torch.tensor([1, 1]))
    roots = torch.arange(evaluator.root_nodes, device=env.device)
    children = torch.arange(
        evaluator.depth_offsets[1], evaluator.depth_offsets[2], device=env.device
    )
    evaluator.initialize_subgame(env, roots)

    uniform_policy = torch.full(
        (NUM_HANDS, evaluator.num_actions),
        1.0 / evaluator.num_actions,
        device=env.device,
    )
    expected_policy = uniform_policy.unsqueeze(0).expand(evaluator.root_nodes, -1, -1)
    evaluator.policy_probs_avg[children] = expected_policy.permute(0, 2, 1).reshape(
        -1, NUM_HANDS
    )
    evaluator.latest_values[roots] = 0.5
    evaluator.values_avg[roots] = 0.5  # Sync values_avg from latest_values

    value_batch, pre_value_batch, policy_batch = evaluator.training_data()

    assert len(policy_batch.features) == evaluator.depth_offsets[-2]
    assert policy_batch.policy_targets.shape == (
        evaluator.depth_offsets[-2],
        NUM_HANDS,
        evaluator.num_actions,
    )
    assert value_batch.value_targets.shape == (
        evaluator.root_nodes,
        evaluator.num_players,
        NUM_HANDS,
    )
    assert pre_value_batch.value_targets.shape == (
        evaluator.root_nodes,
        evaluator.num_players,
        NUM_HANDS,
    )
    torch.testing.assert_close(policy_batch.policy_targets, expected_policy)


@pytest.mark.parametrize(
    "board",
    [
        [8, 39, 20, 28, -1],
        [8, 39, 20, -1, -1],
    ],
)
def test_turn_pre_batch_matches_enumerated_river_expectation(board: list[int]) -> None:
    bet_bins = [0.5, 1.0]
    env_proto = HUNLTensorEnv(
        num_envs=1,
        starting_stack=1000,
        sb=5,
        bb=10,
        default_bet_bins=bet_bins,
        device=torch.device("cpu"),
        float_dtype=torch.float32,
        flop_showdown=False,
    )
    env_proto.reset()

    class RandomModel(nn.Module):
        def forward(self, features: MLPFeatures):
            board = features.board.to(torch.long)
            seeds = (
                board.clamp(min=0)
                * torch.tensor([1, 31, 997, 17, 53], device=board.device)
            ).sum(dim=1)
            hand_values = torch.empty((features.context.shape[0], 2, NUM_HANDS))
            for i in range(features.context.shape[0]):
                torch.manual_seed(seeds[i])
                hand_values[i] = torch.rand((2, NUM_HANDS))

            class Output:
                def __init__(self, hv: torch.Tensor):
                    self.hand_values = hv

            return Output(hand_values)

    model = RandomModel()
    evaluator = RebelCFREvaluator(
        search_batch_size=1,
        env_proto=env_proto,
        model=model,  # type: ignore[arg-type]
        bet_bins=bet_bins,
        max_depth=1,
        cfr_iterations=1,
        float_dtype=torch.float32,
        device=torch.device("cpu"),
        warm_start_iterations=0,
        cfr_type=CFRType.linear,
        cfr_avg=True,
    )

    roots = torch.zeros(1, dtype=torch.long)
    env_proto.reset()

    # Force specific turn boards.
    pre_chance_board = torch.tensor(board, dtype=torch.long)
    pre_chance_street = (pre_chance_board >= 0).sum() - 2
    env_proto.board_indices[0] = pre_chance_board
    env_proto.last_board_indices[0] = pre_chance_board
    env_proto.street[0] = pre_chance_street + 1
    env_proto.actions_this_round[0] = 0

    # Pre-chance beliefs random.
    torch.manual_seed(1234)
    pre_beliefs = torch.rand(1, 2, NUM_HANDS)
    pre_beliefs /= pre_beliefs.sum(dim=-1, keepdim=True)

    evaluator.initialize_subgame(env_proto, roots, pre_beliefs)
    evaluator.latest_values[:] = torch.rand_like(evaluator.latest_values)
    evaluator.values_avg[:] = evaluator.latest_values
    evaluator.self_reach[:1] = 1.0
    evaluator.self_reach_avg[:1] = 1.0
    if evaluator.legal_mask is None:
        evaluator.legal_mask = torch.ones(
            evaluator.total_nodes,
            evaluator.num_actions,
            dtype=torch.bool,
        )

    start_features = evaluator.feature_encoder.encode(
        evaluator.beliefs_avg, pre_chance_node=False
    )
    _, pre_value_batch, _ = evaluator.training_data(exclude_start=False)

    combo_onehot = combo_to_onehot_tensor()
    manual_expected = torch.zeros_like(pre_value_batch.value_targets)

    board_turn = env_proto.board_indices[0].clone()
    used_cards = board_turn[board_turn >= 0]
    available = [c for c in range(52) if c not in used_cards.tolist()]

    sum_values = torch.zeros(2, NUM_HANDS)
    pre_belief = pre_beliefs[0]
    context = start_features.context[:1]
    street_tensor = start_features.street[:1]
    to_act_tensor = start_features.to_act[:1]

    torch.manual_seed(1234)
    for card in available:
        board_river = board_turn.clone()
        empty_pos = (board_river == -1).nonzero(as_tuple=False)[0]
        board_river[empty_pos] = card

        allowed = ~combo_onehot[:, board_river].any(dim=1)
        post_belief = pre_belief.clone()
        post_belief[..., ~allowed] = 0.0
        post_belief /= post_belief.sum(dim=-1, keepdim=True)

        env_card = HUNLTensorEnv.from_proto(env_proto, num_envs=1)
        env_card.reset()
        env_card.copy_state_from(
            env_proto,
            torch.tensor([0], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
        )
        env_card.board_indices[0] = board_river
        env_card.street[0] = pre_chance_street + 1
        env_card.actions_this_round[0] = 0
        env_card.last_board_indices[0] = board_turn

        belief_feature = (post_belief.reshape(1, -1) * 2) - 1
        features = MLPFeatures(
            context=context,
            street=street_tensor,
            to_act=to_act_tensor,
            board=board_river.unsqueeze(0),
            beliefs=belief_feature,
        )
        hand_values = model(features).hand_values.squeeze(0)
        sum_values += hand_values

    manual_expected[0] = sum_values / len(available)

    torch.testing.assert_close(pre_value_batch.value_targets[:1], manual_expected[:1])


def setup_showdown_evaluator(
    board: str | list[int],
    beliefs: torch.Tensor | None = None,
) -> tuple[RebelCFREvaluator, HUNLTensorEnv, torch.Tensor, torch.Tensor]:
    """Common initialization for showdown_value tests."""
    evaluator, env = make_evaluator(batch_size=1, max_depth=0)
    if isinstance(board, str):
        board = make_board(board)
    else:
        board = torch.tensor(board)
    all_cards = torch.arange(52)
    used_mask = torch.isin(all_cards, board)
    unused_cards = torch.where(~used_mask)[0]
    deck = torch.concat([unused_cards[:4], board])
    env.reset(force_deck=deck[None, :])
    for _ in range(8):
        env.step_bins(torch.tensor([1]))
    env.chips_placed[:] = 100
    env.pot[:] = 200
    env.stacks[:] = 900

    roots = torch.arange(evaluator.root_nodes)
    if beliefs is None:
        beliefs = torch.full((1, 2, NUM_HANDS), 1.0 / NUM_HANDS)
    evaluator.initialize_subgame(env, roots, beliefs)
    evaluator.initialize_policy_and_beliefs()
    idx = torch.tensor([0])
    pot = evaluator.env.pot[idx].unsqueeze(-1)
    potentials = (
        evaluator.env.stacks[idx] + pot - evaluator.env.starting_stack
    ) / evaluator.env.scale

    return evaluator, env, idx, potentials.squeeze(0)


def test_showdown_value_uniform_beliefs_matches_reference() -> None:
    evaluator, _, idx, potentials = setup_showdown_evaluator(
        "Ac Qh 9d 7s 4h",
    )

    opp_beliefs_p0 = evaluator.beliefs[idx, 1].squeeze(0)
    expected_p0 = compute_reference_showdown_ev(
        evaluator.env.board_indices[0], opp_beliefs_p0, float(potentials[0])
    )
    actual_p0 = evaluator._showdown_value(0, idx).squeeze(0)
    torch.testing.assert_close(actual_p0, expected_p0)

    opp_beliefs_p1 = evaluator.beliefs[idx, 0].squeeze(0)
    expected_p1 = compute_reference_showdown_ev(
        evaluator.env.board_indices[0], opp_beliefs_p1, float(potentials[1])
    )
    actual_p1 = evaluator._showdown_value(1, idx).squeeze(0)
    torch.testing.assert_close(actual_p1, expected_p1)


def test_showdown_value_top_delta_vs_uniform_wins() -> None:
    # Give us a guaranteed win
    hero_combo = combo_index(card(8, 2), card(7, 2))  # Ten and Nine hearts
    beliefs = torch.zeros(1, 2, NUM_HANDS)
    beliefs[0, 0, hero_combo] = 1.0
    beliefs[0, 1] = 1.0 / (NUM_HANDS - 1)
    beliefs[0, 1, hero_combo] = 0.0
    evaluator, _, idx, potentials = setup_showdown_evaluator("Ad Kd Qd Jd 2s", beliefs)

    opp_beliefs = evaluator.beliefs[idx, 1].squeeze(0)
    expected = compute_reference_showdown_ev(
        evaluator.env.board_indices[0], opp_beliefs, float(potentials[0])
    )
    actual = evaluator._showdown_value(0, idx).squeeze(0)
    torch.testing.assert_close(actual, expected)
    assert actual[hero_combo] == pytest.approx(float(potentials[0]))

    opp_beliefs_p1 = evaluator.beliefs[idx, 0].squeeze(0)
    expected_p1 = compute_reference_showdown_ev(
        evaluator.env.board_indices[0], opp_beliefs_p1, float(potentials[1])
    )
    actual_p1 = evaluator._showdown_value(1, idx).squeeze(0)
    torch.testing.assert_close(actual_p1, expected_p1)


def test_showdown_value_uniform_vs_top_delta_loses() -> None:
    # Give opponent a guaranteed win
    hero_hand = torch.tensor([card(8, 2), card(7, 2)])
    hero_combo = combo_index(hero_hand[0], hero_hand[1])  # Td / 9d
    beliefs = torch.zeros(1, 2, NUM_HANDS)
    beliefs[0, 1, hero_combo] = 1.0
    beliefs[0, 0] = 1.0 / (NUM_HANDS - 1)
    beliefs[0, 0, hero_combo] = 0.0
    evaluator, _, idx, potentials = setup_showdown_evaluator("Ad Kd Qd Jd 2s", beliefs)

    actual = evaluator._showdown_value(0, idx).squeeze(0)
    valid_hands_board_mask = mask_conflicting_combos(evaluator.env.board_indices[0])
    valid_hands_hand_mask = mask_conflicting_combos(hero_hand)

    # Valid hands: lost.
    valid = actual[valid_hands_board_mask & valid_hands_hand_mask]
    torch.testing.assert_close(valid, torch.full_like(valid, -float(potentials[0])))

    # Invalid hands: no reward/penalty.
    invalid = actual[~valid_hands_board_mask | ~valid_hands_hand_mask]
    torch.testing.assert_close(invalid, torch.zeros_like(invalid))

    opp_beliefs_p1 = evaluator.beliefs[idx, 0].squeeze(0)
    expected_p1 = compute_reference_showdown_ev(
        evaluator.env.board_indices[0], opp_beliefs_p1, float(potentials[1])
    )
    actual_p1 = evaluator._showdown_value(1, idx).squeeze(0)
    torch.testing.assert_close(actual_p1, expected_p1)


def test_showdown_value_all_ties_returns_zero() -> None:
    evaluator, _, idx, _ = setup_showdown_evaluator("Ah Kh Qh Jh Th")
    actual_p0 = evaluator._showdown_value(0, idx).squeeze(0)
    actual_p1 = evaluator._showdown_value(1, idx).squeeze(0)
    torch.testing.assert_close(actual_p0, torch.zeros_like(actual_p0))
    torch.testing.assert_close(actual_p1, torch.zeros_like(actual_p1))


def test_showdown_value_fail_when_opponent_range_invalid() -> None:
    board_tensor = make_board("Jh 8s 6c 4h 2d")[None, :]
    beliefs = torch.zeros(1, 2, NUM_HANDS)
    board_mask = mask_conflicting_combos(board_tensor[0])
    conflicting_indices = torch.where(~board_mask)[0]
    assert conflicting_indices.numel() > 0

    beliefs[0].zero_()
    beliefs[0, :, conflicting_indices[:10]] = 0.1

    evaluator, _, idx, _ = setup_showdown_evaluator("Jh 8s 6c 4h 2d", beliefs)

    # Should get uniform beliefs since all beliefs were blocked
    allowed = evaluator.allowed_hands[idx].float() / evaluator.allowed_hands[idx].sum(
        dim=-1, keepdim=True
    )

    torch.testing.assert_close(
        evaluator.beliefs[idx],
        allowed[:, None, :].expand(1, 2, -1),
    )


@pytest.mark.parametrize(
    ("board_str", "opp_hand"),
    [
        ("As Jh 7c 5s 2d", ("Ah", "Kh")),
        ("Tc Th Ts 7d 7h", ("Ah", "Kc")),
        ("7d 6d 5d 4d 3d", ("Ah", "Kc")),
    ],
)
def test_showdown_value_matches_reference_on_diverse_boards(
    board_str: str, opp_hand: tuple[str, str]
) -> None:
    beliefs = torch.zeros(1, 2, NUM_HANDS)
    opp_combo = combo_from_strs(*opp_hand)
    beliefs[0, 1, opp_combo] = 1.0

    evaluator, _, idx, potentials = setup_showdown_evaluator(board_str, beliefs)
    actual = evaluator._showdown_value(0, idx).squeeze(0)

    board_tensor = evaluator.env.board_indices[idx].int()
    ranks, _ = rank_hands(board_tensor)
    ranks = ranks.squeeze(0)
    opp_cards = hand_combos_tensor()[opp_combo]
    valid_mask = evaluator.allowed_hands[idx].squeeze(0) & mask_conflicting_combos(
        opp_cards
    )
    opp_rank = ranks[opp_combo]

    better = torch.where((ranks > opp_rank) & valid_mask)[0]
    worse = torch.where((ranks < opp_rank) & valid_mask)[0]

    assert better.numel() > 0

    better_vals = actual[better[: min(3, better.numel())]]
    torch.testing.assert_close(
        better_vals, torch.full_like(better_vals, float(potentials[0]))
    )

    if worse.numel() > 0:
        worse_vals = actual[worse[: min(3, worse.numel())]]
        torch.testing.assert_close(
            worse_vals, torch.full_like(worse_vals, -float(potentials[0]))
        )

    actual_p1 = evaluator._showdown_value(1, idx).squeeze(0)
    opp_beliefs_p1 = evaluator.beliefs[idx, 0].squeeze(0)
    expected_p1 = compute_reference_showdown_ev(
        evaluator.env.board_indices[0], opp_beliefs_p1, float(potentials[1])
    )
    torch.testing.assert_close(actual_p1, expected_p1)


@pytest.mark.parametrize(
    "hero_hand",
    [
        combo_index(card(12, 0), card(11, 1)),  # AK suited
        combo_index(card(10, 2), card(9, 3)),  # QJ offsuit
        combo_index(card(8, 0), card(7, 0)),  # 98 suited
        combo_index(card(6, 1), card(5, 1)),  # 76 suited
    ],
)
def test_showdown_value_delta_beliefs_each_hand(hero_hand) -> None:
    """Test showdown value with delta beliefs (all mass on single hand)."""
    # Create a simple board
    board = "Ac Kd Qh Js 9c"

    # Create delta belief for opponent (setup_showdown_evaluator expects opponent beliefs)
    beliefs = torch.zeros(1, 2, NUM_HANDS)
    beliefs[0, 1, hero_hand] = 1.0

    evaluator, _, idx, potentials = setup_showdown_evaluator(board, beliefs)

    actual = evaluator._showdown_value(0, idx).squeeze(0)
    board_tensor = evaluator.env.board_indices[idx].int()
    ranks, _ = rank_hands(board_tensor)
    ranks = ranks.squeeze(0)
    hero_rank = ranks[hero_hand]

    allowed = evaluator.allowed_hands[idx].squeeze(0)
    hero_cards = hand_combos_tensor()[hero_hand]
    hero_block = mask_conflicting_combos(hero_cards)
    valid_mask = allowed & hero_block

    better = torch.where((ranks > hero_rank) & valid_mask)[0]
    worse = torch.where((ranks < hero_rank) & valid_mask)[0]
    ties = torch.where((ranks == hero_rank) & valid_mask)[0]

    assert better.numel() > 0 or worse.numel() > 0

    if better.numel() > 0:
        better_vals = actual[better[: min(3, better.numel())]]
        torch.testing.assert_close(
            better_vals, torch.full_like(better_vals, float(potentials[0]))
        )

    if worse.numel() > 0:
        worse_vals = actual[worse[: min(3, worse.numel())]]
        torch.testing.assert_close(
            worse_vals, torch.full_like(worse_vals, -float(potentials[0]))
        )

    if ties.numel() > 0:
        tie_vals = actual[ties[: min(3, ties.numel())]]
        torch.testing.assert_close(tie_vals, torch.zeros_like(tie_vals))

    torch.testing.assert_close(actual[hero_hand], torch.zeros_like(actual[hero_hand]))

    actual_p1 = evaluator._showdown_value(1, idx).squeeze(0)
    opp_beliefs_p1 = evaluator.beliefs[idx, 0].squeeze(0)
    expected_p1 = compute_reference_showdown_ev(
        evaluator.env.board_indices[0], opp_beliefs_p1, float(potentials[1])
    )
    torch.testing.assert_close(actual_p1, expected_p1)


def test_showdown_value_double_half_delta_beliefs() -> None:
    """Test showdown value with 0.5-delta beliefs (half mass on each of two hands)."""
    board = "Ac Kd Qh Js 9c"

    # Two test hands
    hand1 = combo_index(card(12, 0), card(11, 1))  # AK suited
    hand2 = combo_index(card(10, 2), card(9, 3))  # QJ offsuit

    beliefs = torch.zeros(1, 2, NUM_HANDS)
    beliefs[0, 1, hand1] = 0.5
    beliefs[0, 1, hand2] = 0.5

    evaluator, _, idx, potentials = setup_showdown_evaluator(board, beliefs)
    actual = evaluator._showdown_value(0, idx).squeeze(0)

    winner = combo_from_strs("Ts", "2h")  # Makes Broadway straight, beats both hands.
    splitter = combo_from_strs("Ks", "Qc")  # Beats QJ, loses to AK.
    loser = combo_from_strs("4s", "2d")  # High-card hand loses to both.

    torch.testing.assert_close(
        actual[winner], torch.full_like(actual[winner], float(potentials[0]))
    )
    torch.testing.assert_close(actual[splitter], torch.zeros_like(actual[splitter]))
    torch.testing.assert_close(
        actual[loser], torch.full_like(actual[loser], -float(potentials[0]))
    )

    actual_p1 = evaluator._showdown_value(1, idx).squeeze(0)
    opp_beliefs_p1 = evaluator.beliefs[idx, 0].squeeze(0)
    expected_p1 = compute_reference_showdown_ev(
        evaluator.env.board_indices[0], opp_beliefs_p1, float(potentials[1])
    )
    torch.testing.assert_close(actual_p1, expected_p1)


def test_showdown_value_belief_normalization() -> None:
    """Test that showdown values are invariant to belief normalization."""
    board = "Ac Kd Qh Js 9c"

    # Create unnormalized beliefs (sum to 2.0 instead of 1.0)
    beliefs_unnorm = torch.full((1, 2, NUM_HANDS), 2.0 / NUM_HANDS)

    # Create normalized beliefs
    beliefs_norm = torch.full((1, 2, NUM_HANDS), 1.0 / NUM_HANDS)

    # Test with unnormalized beliefs
    evaluator_unnorm, _, idx_unnorm, potentials = setup_showdown_evaluator(
        board, beliefs_unnorm
    )
    values_unnorm = evaluator_unnorm._showdown_value(0, idx_unnorm).squeeze(0)

    # Test with normalized beliefs
    evaluator_norm, _, idx_norm, potentials_norm = setup_showdown_evaluator(
        board, beliefs_norm
    )
    values_norm = evaluator_norm._showdown_value(0, idx_norm).squeeze(0)

    # Values should be identical (beliefs get normalized internally)
    board_mask = mask_conflicting_combos(evaluator_unnorm.env.board_indices[0])
    torch.testing.assert_close(values_unnorm[board_mask], values_norm[board_mask])

    values_unnorm_p1 = evaluator_unnorm._showdown_value(1, idx_unnorm).squeeze(0)
    values_norm_p1 = evaluator_norm._showdown_value(1, idx_norm).squeeze(0)
    torch.testing.assert_close(values_unnorm_p1[board_mask], values_norm_p1[board_mask])


def test_showdown_value_edge_case_zero_opponent_beliefs() -> None:
    """Test showdown value when opponent has zero beliefs (should fallback to uniform)."""
    board = "Ac Kd Qh Js 9c"

    # Create beliefs where opponent has zero mass on all hands
    beliefs = torch.zeros(1, 2, NUM_HANDS)

    evaluator, _, idx, _ = setup_showdown_evaluator(board, beliefs)

    # The evaluator should have normalized opponent beliefs to uniform
    opp_beliefs = evaluator.beliefs[idx, 1].squeeze(0)
    board_mask = mask_conflicting_combos(evaluator.env.board_indices[0])
    uniform_prob = 1.0 / board_mask.sum().float()

    # Opponent beliefs should be uniform over valid hands
    torch.testing.assert_close(
        opp_beliefs[board_mask], torch.full_like(opp_beliefs[board_mask], uniform_prob)
    )

    # Should not crash and return valid values
    actual_p0 = evaluator._showdown_value(0, idx).squeeze(0)
    actual_p1 = evaluator._showdown_value(1, idx).squeeze(0)
    assert torch.all(torch.isfinite(actual_p0[board_mask]))
    assert torch.all(torch.isfinite(actual_p1[board_mask]))


def test_self_play_iteration_returns_public_belief_state() -> None:
    evaluator, env = make_evaluator(batch_size=2, max_depth=1)
    roots = torch.arange(evaluator.root_nodes, device=env.device)
    evaluator.initialize_subgame(env, roots)
    evaluator.warm_start_iterations = 0
    evaluator.cfr_iterations = 2
    evaluator.generator = torch.Generator(device=env.device)
    evaluator.generator.manual_seed(1)

    next_pbs = evaluator.evaluate_cfr(training_mode=False)

    assert next_pbs is not None
    assert next_pbs.env.N == 2
    assert next_pbs.beliefs.shape == (2, evaluator.num_players, NUM_HANDS)
    torch.testing.assert_close(
        next_pbs.beliefs.sum(dim=-1),
        torch.ones((2, evaluator.num_players), device=env.device),
    )


def test_linear_cfr_policy_averaging() -> None:
    """Test that discounted CFR policy averaging weights by max(0, t - delay)."""

    device = torch.device("cpu")
    float_dtype = torch.float32
    bet_bins = [0.5, 1.5]

    # Create a random model
    model = MockModel(num_actions=5)

    # Create environment
    env = HUNLTensorEnv(
        num_envs=1,
        starting_stack=1000,
        sb=5,
        bb=10,
        device=device,
        default_bet_bins=bet_bins,
        float_dtype=float_dtype,
    )
    env.reset()

    # Create evaluator with depth=0 for simplicity
    evaluator = RebelCFREvaluator(
        search_batch_size=1,
        env_proto=env,
        model=model,
        bet_bins=bet_bins,
        max_depth=1,
        cfr_iterations=20,
        warm_start_iterations=0,
        device=device,
        float_dtype=float_dtype,
        cfr_type=CFRType.linear,
        cfr_avg=True,
    )

    # Initialize search
    roots = torch.tensor([0], device=device)
    evaluator.initialize_subgame(env, roots)
    evaluator.initialize_policy_and_beliefs()

    # Generate random policy sequences (seeded)
    torch.manual_seed(42)
    all_policies = torch.rand(20, NUM_HANDS, device=device)

    # Set reach weights to make the averaging work properly
    evaluator.self_reach.fill_(1.0)
    evaluator.self_reach_avg.fill_(1.0)

    # Run iterations and track average policy
    for t in range(20):
        # Set the policy for this iteration
        evaluator.policy_probs[evaluator.valid_mask] = all_policies[t]
        evaluator.policy_probs[0] = 0.0
        evaluator.update_average_policy(t)

        if t == 0:
            assert torch.allclose(
                evaluator.policy_probs_avg[evaluator.valid_mask][1:], all_policies[t]
            )
        else:
            weights = torch.arange(t + 1, dtype=torch.float32)
            weights = weights.clamp(min=0)
            weights /= weights.sum()
            expected = (all_policies[: t + 1] * weights[:, None]).sum(dim=0)
            assert torch.allclose(
                evaluator.policy_probs_avg[evaluator.valid_mask][1:], expected
            )


def test_discounted_cfr_policy_averaging() -> None:
    """Test that discounted CFR policy averaging weights by max(0, t - delay)."""

    device = torch.device("cpu")
    float_dtype = torch.float32
    bet_bins = [0.5, 1.5]

    # Create a random model
    model = MockModel(num_actions=5)

    # Create environment
    env = HUNLTensorEnv(
        num_envs=1,
        starting_stack=1000,
        sb=5,
        bb=10,
        device=device,
        default_bet_bins=bet_bins,
        float_dtype=float_dtype,
    )
    env.reset()

    # Create evaluator with depth=0 for simplicity
    # Note: discounted_plus uses delay-based weights, not discounted
    dcfr_delay = 10
    evaluator = RebelCFREvaluator(
        search_batch_size=1,
        env_proto=env,
        model=model,
        bet_bins=bet_bins,
        max_depth=1,
        cfr_iterations=20,
        warm_start_iterations=0,
        dcfr_delay=dcfr_delay,
        device=device,
        float_dtype=float_dtype,
        cfr_type=CFRType.discounted_plus,
        cfr_avg=True,
    )

    # Initialize search
    roots = torch.tensor([0], device=device)
    evaluator.initialize_subgame(env, roots)
    evaluator.initialize_policy_and_beliefs()

    # Generate random policy sequences (seeded)
    torch.manual_seed(42)
    all_policies = torch.rand(20, NUM_HANDS, device=device)

    # Set reach weights to make the averaging work properly
    evaluator.self_reach.fill_(1.0)
    evaluator.self_reach_avg.fill_(1.0)

    # Run iterations and track average policy
    for t in range(20):
        # Set the policy for this iteration
        evaluator.policy_probs[evaluator.valid_mask] = all_policies[t]
        evaluator.policy_probs[0] = 0.0
        evaluator.update_average_policy(t)

        if t <= dcfr_delay:
            assert torch.allclose(
                evaluator.policy_probs_avg[evaluator.valid_mask][1:], all_policies[t]
            )
        else:
            weights = torch.arange(t + 1, dtype=torch.float32) - dcfr_delay
            weights = weights.clamp(min=0)
            weights /= weights.sum()
            expected = (all_policies[: t + 1] * weights[:, None]).sum(dim=0)
            assert torch.allclose(
                evaluator.policy_probs_avg[evaluator.valid_mask][1:], expected
            )


def test_flop_blocking_over_iterations() -> None:
    """Preflop nodes remain unblocked; flop nodes block board cards while pre-chance stays intact."""

    device = torch.device("cpu")
    uniform = torch.full((1, 2, NUM_HANDS), 1.0 / NUM_HANDS, device=device)

    bet_bins = [1.0]
    env = HUNLTensorEnv(
        num_envs=1,
        starting_stack=1000,
        sb=5,
        bb=10,
        default_bet_bins=bet_bins,
        device=device,
        float_dtype=torch.float32,
        flop_showdown=False,
    )
    env.reset()
    model = MockModel(
        logits=torch.zeros(len(bet_bins) + 3, dtype=env.float_dtype),
        hand_values=torch.zeros(1, 2, NUM_HANDS, dtype=env.float_dtype),
        num_actions=len(bet_bins) + 3,
        device=device,
        dtype=env.float_dtype,
    )
    evaluator = RebelCFREvaluator(
        search_batch_size=1,
        env_proto=env,
        model=model,  # type: ignore[arg-type]
        bet_bins=bet_bins,
        max_depth=2,
        cfr_iterations=4,
        device=device,
        float_dtype=env.float_dtype,
        warm_start_iterations=0,
    )
    roots = torch.arange(evaluator.root_nodes, device=device)
    evaluator.initialize_subgame(env, roots, uniform)
    evaluator.initialize_policy_and_beliefs()

    preflop_valid = torch.where(evaluator.valid_mask)[0]
    assert evaluator.allowed_hands[preflop_valid].all()

    num_actions = evaluator.num_actions
    first_call = evaluator.depth_offsets[1] + 1
    second_call = evaluator.depth_offsets[2] + 1 * num_actions + 1
    assert evaluator.valid_mask[first_call]
    assert evaluator.valid_mask[second_call]
    assert evaluator.env.street[second_call] == 1

    flop_pbs = PublicBeliefState.from_proto(
        env_proto=evaluator.env,
        beliefs=torch.zeros_like(uniform),
        num_envs=1,
    )
    flop_pbs.env.copy_state_from(
        evaluator.env,
        torch.tensor([second_call], device=device),
        torch.tensor([0], device=device),
    )
    flop_pbs.beliefs[0] = evaluator.beliefs[second_call]

    evaluator.initialize_subgame(
        flop_pbs.env,
        torch.tensor([0], device=device),
        flop_pbs.beliefs,
    )
    evaluator.root_pre_chance_beliefs[:1] = evaluator.root_pre_chance_beliefs[:1]
    evaluator.initialize_policy_and_beliefs()

    flop_valid = torch.where(evaluator.valid_mask)[0]
    board = evaluator.env.board_indices[flop_valid[0]]
    expected_mask = mask_conflicting_combos(board, device=device)

    for idx in flop_valid:
        assert torch.equal(evaluator.allowed_hands[idx], expected_mask)
        for player in range(evaluator.num_players):
            assert torch.all(evaluator.beliefs[idx, player, expected_mask] > 0)
            assert torch.all(evaluator.beliefs[idx, player, ~expected_mask] == 0)

    torch.testing.assert_close(
        evaluator.root_pre_chance_beliefs[0],
        uniform[0],
    )


def test_local_exploitability_depth_limited() -> None:
    evaluator, _ = make_evaluator(batch_size=1, max_depth=1, cfr_iterations=2)
    device = evaluator.device
    dtype = evaluator.float_dtype
    num_hands = NUM_HANDS
    num_actions = evaluator.num_actions
    total_nodes = evaluator.total_nodes

    evaluator.valid_mask.zero_()
    evaluator.valid_mask[0] = True
    evaluator.valid_mask[1 : 1 + num_actions] = True

    evaluator.leaf_mask.zero_()
    evaluator.leaf_mask[1 : 1 + num_actions] = True

    evaluator.env.to_act.zero_()
    evaluator.env.to_act[0] = 0
    evaluator.env.to_act[1 : 1 + num_actions] = 1

    evaluator.legal_mask = torch.zeros(
        total_nodes, num_actions, dtype=torch.bool, device=device
    )
    good_action = 1
    bad_action = 2
    evaluator.legal_mask[0, good_action] = True
    evaluator.legal_mask[0, bad_action] = True

    policy = torch.zeros(total_nodes, num_hands, device=device, dtype=dtype)
    policy[1 + good_action].fill_(0.2)
    policy[1 + bad_action].fill_(0.8)
    evaluator.policy_probs[:] = policy
    evaluator.policy_probs_avg[:] = policy

    values = torch.zeros(total_nodes, 2, num_hands, device=device, dtype=dtype)
    values[1 + good_action, 0].fill_(1.0)
    values[1 + good_action, 1].fill_(-1.0)
    values[1 + bad_action, 0].fill_(0.2)
    values[1 + bad_action, 1].fill_(-0.2)
    base_root_value = 0.2 * 1.0 + 0.8 * 0.2
    values[0, 0].fill_(base_root_value)
    values[0, 1].fill_(-base_root_value)
    evaluator.latest_values[:] = values
    evaluator.values_avg[:] = values

    evaluator.self_reach.zero_()
    evaluator.self_reach_avg.zero_()
    evaluator.self_reach[0].fill_(1.0)
    evaluator.self_reach[1 : 1 + num_actions].fill_(1.0)
    evaluator.self_reach_avg[:] = evaluator.self_reach

    beliefs = torch.zeros(total_nodes, 2, num_hands, device=device, dtype=dtype)
    uniform = torch.full((num_hands,), 1.0 / num_hands, device=device, dtype=dtype)
    beliefs[0, 0] = uniform
    beliefs[0, 1] = uniform
    evaluator.beliefs[:] = beliefs
    evaluator.beliefs_avg[:] = beliefs
    _copy_root_beliefs_to_children(evaluator)

    evaluator.allowed_hands = torch.ones(
        total_nodes, num_hands, dtype=torch.bool, device=device
    )
    evaluator.allowed_hands_prob = torch.full(
        (total_nodes, num_hands),
        1.0 / num_hands,
        device=device,
        dtype=dtype,
    )

    evaluator.stats.clear()
    evaluator._compute_exploitability()

    value_batch, pre_value_batch, policy_batch = evaluator.training_data(
        exclude_start=False
    )
    assert "local_exploitability" in value_batch.statistics
    assert "local_exploitability" not in policy_batch.statistics

    torch.testing.assert_close(
        value_batch.statistics["local_exploitability"][0],
        torch.tensor(0.64, dtype=dtype),
    )
    torch.testing.assert_close(
        value_batch.statistics["local_best_response_values"][0],
        torch.tensor([1.0, -base_root_value], dtype=dtype),
    )


def test_local_exploitability_not_scaled_by_opponent_reach() -> None:
    evaluator, _ = make_evaluator(batch_size=1, max_depth=1, cfr_iterations=2)
    device = evaluator.device
    dtype = evaluator.float_dtype
    num_hands = NUM_HANDS
    num_actions = evaluator.num_actions
    total_nodes = evaluator.total_nodes

    evaluator.valid_mask.zero_()
    evaluator.valid_mask[0] = True
    evaluator.valid_mask[1 : 1 + num_actions] = True

    evaluator.leaf_mask.zero_()
    evaluator.leaf_mask[1 : 1 + num_actions] = True

    evaluator.env.to_act.zero_()
    evaluator.env.to_act[0] = 0
    evaluator.env.to_act[1 : 1 + num_actions] = 1

    evaluator.legal_mask = torch.zeros(
        total_nodes, num_actions, dtype=torch.bool, device=device
    )
    good_action = 1
    bad_action = 2
    evaluator.legal_mask[0, good_action] = True
    evaluator.legal_mask[0, bad_action] = True

    policy = torch.zeros(total_nodes, num_hands, device=device, dtype=dtype)
    policy[1 + good_action].fill_(0.2)
    policy[1 + bad_action].fill_(0.8)
    evaluator.policy_probs[:] = policy
    evaluator.policy_probs_avg[:] = policy

    values = torch.zeros(total_nodes, 2, num_hands, device=device, dtype=dtype)
    values[1 + good_action, 0].fill_(1.0)
    values[1 + good_action, 1].fill_(-1.0)
    values[1 + bad_action, 0].fill_(0.2)
    values[1 + bad_action, 1].fill_(-0.2)
    base_root_value = 0.2 * 1.0 + 0.8 * 0.2
    values[0, 0].fill_(base_root_value)
    values[0, 1].fill_(-base_root_value)
    evaluator.latest_values[:] = values
    evaluator.values_avg[:] = values

    evaluator.self_reach.zero_()
    evaluator.self_reach_avg.zero_()
    evaluator.self_reach[0, 0].fill_(0.5)
    evaluator.self_reach[0, 1].fill_(0.1)
    evaluator.self_reach[1 : 1 + num_actions, 0].fill_(0.5)
    evaluator.self_reach[1 : 1 + num_actions, 1].fill_(0.1)
    evaluator.self_reach_avg[:] = evaluator.self_reach

    beliefs = torch.zeros(total_nodes, 2, num_hands, device=device, dtype=dtype)
    uniform = torch.full((num_hands,), 1.0 / num_hands, device=device, dtype=dtype)
    beliefs[0, 0] = uniform
    beliefs[0, 1] = uniform
    evaluator.beliefs[:] = beliefs
    evaluator.beliefs_avg[:] = beliefs
    _copy_root_beliefs_to_children(evaluator)

    evaluator.allowed_hands = torch.ones(
        total_nodes, num_hands, dtype=torch.bool, device=device
    )
    evaluator.allowed_hands_prob = torch.full(
        (total_nodes, num_hands),
        1.0 / num_hands,
        device=device,
        dtype=dtype,
    )

    stats = evaluator._compute_exploitability()

    torch.testing.assert_close(
        stats.local_exploitability[0], torch.tensor(0.64, device=device, dtype=dtype)
    )


def test_local_exploitability_uses_correct_player_beliefs() -> None:
    evaluator, _ = make_evaluator(batch_size=1, max_depth=1, cfr_iterations=2)
    device = evaluator.device
    dtype = evaluator.float_dtype
    num_hands = NUM_HANDS
    num_actions = evaluator.num_actions
    total_nodes = evaluator.total_nodes

    evaluator.valid_mask.zero_()
    evaluator.valid_mask[0] = True
    evaluator.valid_mask[1 : 1 + num_actions] = True

    evaluator.leaf_mask.zero_()
    evaluator.leaf_mask[1 : 1 + num_actions] = True

    evaluator.env.to_act.zero_()
    evaluator.env.to_act[0] = 1
    evaluator.env.to_act[1 : 1 + num_actions] = 0

    evaluator.legal_mask = torch.zeros(
        total_nodes, num_actions, dtype=torch.bool, device=device
    )
    good_action = 1
    bad_action = 2
    evaluator.legal_mask[0, good_action] = True
    evaluator.legal_mask[0, bad_action] = True

    policy = torch.zeros(total_nodes, num_hands, device=device, dtype=dtype)
    policy[1 + good_action].fill_(0.2)
    policy[1 + bad_action].fill_(0.8)
    evaluator.policy_probs[:] = policy
    evaluator.policy_probs_avg[:] = policy

    values = torch.zeros(total_nodes, 2, num_hands, device=device, dtype=dtype)
    good_values = torch.zeros(num_hands, device=device, dtype=dtype)
    bad_values = torch.zeros_like(good_values)
    combo_a, combo_b = 0, 1
    good_values[combo_a] = 1.5
    good_values[combo_b] = 0.1
    bad_values[combo_a] = -0.5
    bad_values[combo_b] = 0.4

    values[1 + good_action, 1] = good_values
    values[1 + good_action, 0] = -good_values
    values[1 + bad_action, 1] = bad_values
    values[1 + bad_action, 0] = -bad_values

    root_values_p1 = 0.2 * good_values + 0.8 * bad_values
    values[0, 1] = root_values_p1
    values[0, 0] = -root_values_p1
    evaluator.latest_values[:] = values
    evaluator.values_avg[:] = values

    evaluator.self_reach.zero_()
    evaluator.self_reach_avg.zero_()
    evaluator.self_reach[0].fill_(1.0)
    evaluator.self_reach[1 : 1 + num_actions].fill_(1.0)
    evaluator.self_reach_avg[:] = evaluator.self_reach

    beliefs = torch.zeros(total_nodes, 2, num_hands, device=device, dtype=dtype)
    beliefs[0, 0, combo_b] = 1.0
    beliefs[0, 1, combo_a] = 1.0
    evaluator.beliefs[:] = beliefs
    evaluator.beliefs_avg[:] = beliefs
    _copy_root_beliefs_to_children(evaluator)

    evaluator.allowed_hands = torch.ones(
        total_nodes, num_hands, dtype=torch.bool, device=device
    )
    evaluator.allowed_hands_prob = torch.full(
        (total_nodes, num_hands),
        1.0 / num_hands,
        device=device,
        dtype=dtype,
    )

    stats = evaluator._compute_exploitability()

    expected_improvement = torch.tensor(1.6, device=device, dtype=dtype)
    torch.testing.assert_close(
        stats.local_exploitability[0],
        expected_improvement,
    )


def test_local_exploitability_uses_policy_evaluation_for_baseline() -> None:
    evaluator, _ = make_evaluator(batch_size=1, max_depth=1, cfr_iterations=2)
    device = evaluator.device
    dtype = evaluator.float_dtype
    num_hands = NUM_HANDS
    num_actions = evaluator.num_actions

    evaluator.valid_mask.zero_()
    evaluator.valid_mask[0] = True
    evaluator.valid_mask[1 : 1 + num_actions] = True
    evaluator.leaf_mask.zero_()
    evaluator.leaf_mask[1 : 1 + num_actions] = True

    evaluator.env.to_act.zero_()
    evaluator.env.to_act[0] = 0
    evaluator.env.to_act[1 : 1 + num_actions] = 1

    evaluator.legal_mask = torch.ones(
        evaluator.total_nodes, num_actions, dtype=torch.bool, device=device
    )

    policy = torch.zeros(evaluator.total_nodes, num_hands, device=device, dtype=dtype)
    policy[1].fill_(0.75)
    policy[2].fill_(0.25)
    evaluator.policy_probs[:] = policy
    evaluator.policy_probs_avg[:] = policy

    values = torch.zeros(
        evaluator.total_nodes, 2, num_hands, device=device, dtype=dtype
    )
    good = torch.full((num_hands,), 2.0, device=device, dtype=dtype)
    bad = torch.full((num_hands,), -1.0, device=device, dtype=dtype)
    values[1, 0] = good
    values[1, 1] = -good
    values[2, 0] = bad
    values[2, 1] = -bad

    # Intentionally set inconsistent cached root values; these should be ignored
    # when re-evaluating the policy.
    values[0, 0].fill_(10.0)
    values[0, 1].fill_(-10.0)

    evaluator.latest_values[:] = values
    evaluator.values_avg[:] = values

    evaluator.self_reach.fill_(1.0)
    evaluator.self_reach_avg.fill_(1.0)

    uniform = torch.full((num_hands,), 1.0 / num_hands, dtype=dtype, device=device)
    evaluator.beliefs[:] = uniform
    evaluator.beliefs_avg[:] = evaluator.beliefs
    evaluator.allowed_hands[:] = True
    evaluator.allowed_hands_prob.fill_(1.0 / num_hands)

    stats = evaluator._compute_exploitability()

    torch.testing.assert_close(
        stats.local_exploitability[0],
        torch.tensor(0.75, device=device, dtype=dtype),
    )


def test_set_leaf_values_cfr_avg_branches() -> None:
    """Test all branches of the CFR-AVG modified formula in set_leaf_values."""
    device = torch.device("cpu")
    float_dtype = torch.float32
    bet_bins = [0.5, 1.5]

    def make_model(value: float) -> MockModel:
        return MockModel(
            hand_values=torch.full(
                (1, 2, NUM_HANDS), value, device=device, dtype=float_dtype
            ),
            device=device,
            dtype=float_dtype,
        )

    def setup_evaluator(
        model: MockModel, cfr_type: CFRType, cfr_avg: bool
    ) -> RebelCFREvaluator:
        env = HUNLTensorEnv(num_envs=1, starting_stack=1000, sb=5, bb=10, device=device)
        env.reset()
        evaluator = RebelCFREvaluator(
            search_batch_size=1,
            env_proto=env,
            model=model,
            bet_bins=bet_bins,
            max_depth=0,
            cfr_iterations=10,
            warm_start_iterations=0,
            device=device,
            float_dtype=float_dtype,
            cfr_type=cfr_type,
            cfr_avg=cfr_avg,
        )
        roots = torch.tensor([0], device=device)
        evaluator.initialize_subgame(env, roots)
        evaluator.initialize_policy_and_beliefs()
        leaf_idx = 0
        evaluator.leaf_mask[leaf_idx] = True
        evaluator.valid_mask[leaf_idx] = True
        evaluator.env.done[leaf_idx] = False
        evaluator.self_reach[leaf_idx] = 1.0
        evaluator.folded_mask[leaf_idx] = False
        return evaluator

    leaf_idx = 0

    # Test Case 1: t <= 1 with cfr_avg=True (direct assignment)
    for t in [0, 1]:
        evaluator = setup_evaluator(make_model(1.0), CFRType.standard, cfr_avg=True)
        evaluator.set_leaf_values(t)
        expected = torch.full((2, NUM_HANDS), 1.0, device=device, dtype=float_dtype)
        torch.testing.assert_close(
            evaluator.latest_values[leaf_idx], expected, atol=1e-6, rtol=1e-6
        )

    # Test Case 2: cfr_avg=False with t > 1 (direct assignment)
    evaluator = setup_evaluator(make_model(2.0), CFRType.standard, cfr_avg=False)
    evaluator.last_model_values = torch.full(
        (1, 2, NUM_HANDS), 100.0, device=device, dtype=float_dtype
    )
    evaluator.set_leaf_values(5)
    expected = torch.full((2, NUM_HANDS), 2.0, device=device, dtype=float_dtype)
    torch.testing.assert_close(
        evaluator.latest_values[leaf_idx], expected, atol=1e-6, rtol=1e-6
    )

    # Test Case 3: last_model_values is None with t > 1 and cfr_avg=True (direct assignment)
    evaluator = setup_evaluator(make_model(3.0), CFRType.standard, cfr_avg=True)
    assert evaluator.last_model_values is None
    evaluator.set_leaf_values(3)
    expected = torch.full((2, NUM_HANDS), 3.0, device=device, dtype=float_dtype)
    torch.testing.assert_close(
        evaluator.latest_values[leaf_idx], expected, atol=1e-6, rtol=1e-6
    )

    # Test Cases 4-6: Formula branches with different CFR types
    test_cases = [
        (
            CFRType.standard,
            5.0,
            4.0,
            6,
            lambda t, new, old: t * new - (t - 1) * old,
        ),  # 6*5 - 5*4 = 10
        (
            CFRType.linear,
            7.0,
            6.0,
            4,
            lambda t, new, old: ((t + 1) * new - (t - 1) * old) / 2,
        ),  # (5*7 - 3*6) / 2 = 8.5
        (
            CFRType.discounted,
            9.0,
            8.0,
            5,
            lambda t, new_val, old_val: (
                ((t - 1) ** 2 + t**2) * new_val - (t - 1) ** 2 * old_val
            )
            / (t**2),
        ),
    ]

    for cfr_type, new_val, old_val, t_iter, formula in test_cases:
        evaluator = setup_evaluator(make_model(new_val), cfr_type, cfr_avg=True)
        evaluator.set_leaf_values(1)  # Initialize last_model_values
        evaluator.last_model_values = torch.full(
            (1, 2, NUM_HANDS), old_val, device=device, dtype=float_dtype
        )
        evaluator.set_leaf_values(t_iter)
        expected = torch.full(
            (2, NUM_HANDS),
            formula(t_iter, new_val, old_val),
            device=device,
            dtype=float_dtype,
        )
        torch.testing.assert_close(
            evaluator.latest_values[leaf_idx], expected, atol=1e-6, rtol=1e-6
        )


def test_pre_chance_features_share_root_context() -> None:
    """Pre-chance nodes descended from the same root encode identical context."""

    device = torch.device("cpu")
    bet_bins = [0.5, 1.0]
    env = HUNLTensorEnv(
        num_envs=1,
        starting_stack=1000,
        sb=5,
        bb=10,
        default_bet_bins=bet_bins,
        device=device,
        float_dtype=torch.float32,
    )
    _advance_env_to_flop(env, button=1)

    model = BetterFFN(
        num_actions=len(bet_bins) + 3,
        hidden_dim=32,
        range_hidden_dim=32,
        ffn_dim=64,
        num_hidden_layers=1,
        num_policy_layers=1,
        num_value_layers=1,
    )
    model.eval()

    evaluator = RebelCFREvaluator(
        search_batch_size=1,
        env_proto=env,
        model=model,
        bet_bins=bet_bins,
        max_depth=2,
        cfr_iterations=2,
        warm_start_iterations=0,
        device=device,
        float_dtype=torch.float32,
    )

    roots = torch.tensor([0], device=device)
    evaluator.initialize_subgame(env, roots)

    pre_chance_mask = evaluator.new_street_mask & evaluator.valid_mask
    assert pre_chance_mask.any(), "expected at least one start-of-street node"

    features = evaluator.feature_encoder.encode(
        evaluator.beliefs, pre_chance_node=pre_chance_mask
    )
    root_ids = _compute_root_ids(evaluator)
    masked_indices = torch.where(pre_chance_mask)[0]
    actions_idx = ScalarContext.ACTIONS_ROUND.value

    for root_id in torch.unique(root_ids[masked_indices]):
        node_indices = masked_indices[root_ids[masked_indices] == root_id]
        reference = node_indices[0]

        expected_street = features.street[reference]
        expected_board = features.board[reference]
        expected_actions = features.context[reference, actions_idx]

        assert torch.all(features.street[node_indices] == expected_street)
        assert torch.all(features.board[node_indices] == expected_board)
        torch.testing.assert_close(
            features.context[node_indices, actions_idx],
            torch.full_like(
                features.context[node_indices, actions_idx], expected_actions
            ),
        )


def _advance_env_to_flop(env: HUNLTensorEnv, button: int = 1) -> None:
    """Advance a tensor env from reset to a flop state with no flop actions taken."""

    env.reset(force_button=torch.tensor([button], device=env.device))
    bet_bin_index = min(3, len(env.default_bet_bins) + 2)
    bet_action = torch.full(
        (env.N,), bet_bin_index, dtype=torch.long, device=env.device
    )
    call_action = torch.ones(env.N, dtype=torch.long, device=env.device)

    # SB open-raises and BB calls to close the preflop round.
    env.step_bins(bet_action)
    env.step_bins(call_action)


def _compute_root_ids(evaluator: RebelCFREvaluator) -> torch.Tensor:
    """Map every node in the constructed subgame back to its root index."""

    root_ids = torch.full(
        (evaluator.total_nodes,),
        -1,
        dtype=torch.long,
        device=evaluator.device,
    )

    for depth in range(evaluator.max_depth + 1):
        offset = evaluator.depth_offsets[depth]
        offset_next = evaluator.depth_offsets[depth + 1]
        per_root = evaluator.num_actions**depth
        span = torch.arange(
            offset_next - offset,
            device=evaluator.device,
            dtype=torch.long,
        )
        root_ids[offset:offset_next] = span // per_root

    return root_ids


def test_calculate_reach_weights() -> None:
    """Test _calculate_reach_weights computes reach weights correctly."""
    evaluator, env = make_evaluator(batch_size=1, max_depth=2)
    roots = torch.arange(evaluator.root_nodes, device=env.device)
    evaluator.initialize_subgame(env, roots)
    evaluator.initialize_policy_and_beliefs()

    # Set up a simple policy
    num_actions = evaluator.num_actions
    evaluator.policy_probs.zero_()
    evaluator.policy_probs[1 : 1 + num_actions, :] = 1.0 / num_actions

    # Calculate reach weights (root nodes should already be 1.0 from initialize_subgame)
    reach_weights = evaluator.self_reach.clone()
    # Verify root nodes start at 1.0
    torch.testing.assert_close(
        reach_weights[: evaluator.root_nodes],
        torch.ones_like(reach_weights[: evaluator.root_nodes]),
    )

    evaluator._calculate_reach_weights(reach_weights, evaluator.policy_probs)

    # Root should still have reach 1.0 for both players (never updated by _calculate_reach_weights)
    torch.testing.assert_close(
        reach_weights[: evaluator.root_nodes],
        torch.ones_like(reach_weights[: evaluator.root_nodes]),
    )

    # Child nodes should have reach based on policy
    child_nodes = torch.arange(
        evaluator.root_nodes, evaluator.root_nodes + num_actions, device=env.device
    )
    child_reach = reach_weights[child_nodes]
    # Reach should be non-zero for valid nodes
    assert torch.all(child_reach[evaluator.valid_mask[child_nodes]] >= 0)


def test_propagate_all_beliefs() -> None:
    """Test _propagate_all_beliefs propagates beliefs correctly."""
    evaluator, env = make_evaluator(batch_size=1, max_depth=1)
    roots = torch.arange(evaluator.root_nodes, device=env.device)
    evaluator.initialize_subgame(env, roots)

    # Set up initial beliefs
    uniform = torch.full(
        (NUM_HANDS,), 1.0 / NUM_HANDS, device=env.device, dtype=env.float_dtype
    )
    evaluator.beliefs[0, 0] = uniform
    evaluator.beliefs[0, 1] = uniform

    # Set up policy and reach
    evaluator.policy_probs.zero_()
    evaluator.policy_probs[1 : 1 + evaluator.num_actions, :] = (
        1.0 / evaluator.num_actions
    )
    evaluator.self_reach.zero_()
    evaluator.self_reach[0] = 1.0

    # Propagate beliefs
    evaluator._propagate_all_beliefs()

    # Beliefs should be normalized
    belief_sums = evaluator.beliefs[evaluator.valid_mask].sum(dim=-1)
    torch.testing.assert_close(belief_sums, torch.ones_like(belief_sums))


def test_propagate_level_beliefs() -> None:
    """Test _propagate_level_beliefs propagates beliefs from one level to the next."""
    evaluator, env = make_evaluator(batch_size=1, max_depth=1)
    roots = torch.arange(evaluator.root_nodes, device=env.device)
    evaluator.initialize_subgame(env, roots)

    # Set up root beliefs
    uniform = torch.full(
        (NUM_HANDS,), 1.0 / NUM_HANDS, device=env.device, dtype=env.float_dtype
    )
    evaluator.beliefs[0, 0] = uniform
    evaluator.beliefs[0, 1] = uniform

    # Set up policy for depth 0
    evaluator.policy_probs[1 : 1 + evaluator.num_actions, :] = (
        1.0 / evaluator.num_actions
    )

    # Propagate from depth 0 to depth 1
    evaluator._propagate_level_beliefs(0)

    # Child beliefs should be updated (but not normalized yet)
    child_nodes = torch.arange(1, 1 + evaluator.num_actions, device=env.device)
    child_beliefs = evaluator.beliefs[child_nodes]
    # Check that beliefs were propagated (non-zero for valid nodes)
    valid_children = child_nodes[evaluator.valid_mask[child_nodes]]
    if valid_children.numel() > 0:
        # Beliefs should be non-zero (they get normalized later)
        assert torch.any(child_beliefs[evaluator.valid_mask[child_nodes]] > 0)


def test_block_beliefs() -> None:
    """Test _block_beliefs blocks beliefs based on board cards."""
    evaluator, env = make_evaluator(batch_size=1, max_depth=0)
    roots = torch.arange(evaluator.root_nodes, device=env.device)
    evaluator.initialize_subgame(env, roots)

    # Set up beliefs with non-zero values for all hands
    uniform = torch.full(
        (NUM_HANDS,), 1.0 / NUM_HANDS, device=env.device, dtype=env.float_dtype
    )
    evaluator.beliefs[0, 0] = uniform
    evaluator.beliefs[0, 1] = uniform

    # Block beliefs
    evaluator._block_beliefs()

    # Beliefs should be zero for blocked hands
    blocked_hands = ~evaluator.allowed_hands[0]
    if blocked_hands.any():
        torch.testing.assert_close(
            evaluator.beliefs[0, :, blocked_hands],
            torch.zeros_like(evaluator.beliefs[0, :, blocked_hands]),
        )


def test_normalize_beliefs() -> None:
    """Test _normalize_beliefs normalizes beliefs correctly."""
    evaluator, env = make_evaluator(batch_size=1, max_depth=0)
    roots = torch.arange(evaluator.root_nodes, device=env.device)
    evaluator.initialize_subgame(env, roots)

    # Set up unnormalized beliefs
    unnormalized = torch.rand(1, 2, NUM_HANDS, device=env.device, dtype=env.float_dtype)
    evaluator.beliefs[0] = unnormalized[0]

    # Normalize beliefs
    evaluator._normalize_beliefs()

    # Beliefs should sum to 1
    belief_sums = evaluator.beliefs[0].sum(dim=-1)
    torch.testing.assert_close(belief_sums, torch.ones_like(belief_sums))

    # Test with zero beliefs (should fallback to uniform)
    evaluator.beliefs[0].zero_()
    evaluator._normalize_beliefs()
    # Should use allowed_hands_prob as fallback
    belief_sums = evaluator.beliefs[0].sum(dim=-1)
    torch.testing.assert_close(belief_sums, torch.ones_like(belief_sums))


def test_get_mixing_weights() -> None:
    """Test _get_mixing_weights returns correct weights for different CFR types."""
    evaluator, env = make_evaluator(batch_size=1, max_depth=0)

    # Test standard CFR
    evaluator.cfr_type = CFRType.standard
    old, new = evaluator._get_mixing_weights(5)
    assert old == 4
    assert new == 1

    # Test linear CFR
    evaluator.cfr_type = CFRType.linear
    old, new = evaluator._get_mixing_weights(5)
    assert old == 4
    assert new == 2

    # Test discounted CFR
    evaluator.cfr_type = CFRType.discounted
    evaluator.dcfr_gamma = 2.0
    old, new = evaluator._get_mixing_weights(5)
    assert old == 16  # (5-1)^2
    assert new == 25  # 5^2

    # Test discounted_plus CFR with delay
    evaluator.cfr_type = CFRType.discounted_plus
    evaluator.dcfr_delay = 3
    old, new = evaluator._get_mixing_weights(5)
    assert old == 1  # (5-3-1) = 1
    assert new == 2

    # Test discounted_plus CFR before delay
    old, new = evaluator._get_mixing_weights(2)
    assert old == 0
    assert new == 1


def test_get_sampling_schedule() -> None:
    """Test _get_sampling_schedule generates correct sampling schedule."""
    evaluator, env = make_evaluator(batch_size=4, max_depth=1)
    evaluator.generator = torch.Generator(device=env.device)
    evaluator.generator.manual_seed(42)

    # Test with standard CFR
    evaluator.cfr_type = CFRType.standard
    evaluator.warm_start_iterations = 5
    evaluator.cfr_iterations = 20
    schedule = evaluator._get_sampling_schedule()
    assert schedule.shape[0] == evaluator.total_nodes
    # Should be in range [warm_start_iterations + 1, cfr_iterations]
    assert schedule.min() >= evaluator.warm_start_iterations + 1
    assert schedule.max() < evaluator.cfr_iterations

    # Test with discounted_plus CFR
    evaluator.cfr_type = CFRType.discounted_plus
    evaluator.dcfr_delay = 10
    schedule = evaluator._get_sampling_schedule()
    assert schedule.min() >= max(
        evaluator.warm_start_iterations + 1, evaluator.dcfr_delay + 1
    )


def test_record_stats() -> None:
    """Test _record_stats records policy update statistics."""
    evaluator, env = make_evaluator(batch_size=1, max_depth=1)
    roots = torch.arange(evaluator.root_nodes, device=env.device)
    evaluator.initialize_subgame(env, roots)
    evaluator.initialize_policy_and_beliefs()

    # Set up old policy
    old_policy = evaluator.policy_probs.clone()

    # Update policy
    evaluator.policy_probs[:] = torch.rand_like(evaluator.policy_probs)
    evaluator.policy_probs[evaluator.valid_mask] /= evaluator.policy_probs[
        evaluator.valid_mask
    ].sum(dim=-1, keepdim=True)

    # Record stats at a tracked iteration
    evaluator.warm_start_iterations = 0
    evaluator.cfr_iterations = 100
    t = 25  # Should be tracked
    evaluator._record_stats(t, old_policy)

    # Should have recorded a stat
    assert any("cfr_delta" in key for key in evaluator.stats.keys())


def test_record_cfr_entropy() -> None:
    """Test _record_cfr_entropy records policy entropy."""
    evaluator, env = make_evaluator(batch_size=1, max_depth=1)
    roots = torch.arange(evaluator.root_nodes, device=env.device)
    evaluator.initialize_subgame(env, roots)
    evaluator.initialize_policy_and_beliefs()

    # Set up a policy
    evaluator.policy_probs_avg[:] = evaluator.policy_probs

    # Record entropy
    evaluator._record_cfr_entropy()

    # Should have recorded entropy
    assert "cfr_entropy" in evaluator.stats
    assert evaluator.stats["cfr_entropy"] >= 0

    # Test with max_depth=0 (should return early)
    evaluator.max_depth = 0
    evaluator.stats.clear()
    evaluator._record_cfr_entropy()
    assert "cfr_entropy" not in evaluator.stats


def test_record_cumulative_regret() -> None:
    """Test _record_cumulative_regret records regret statistics."""
    evaluator, env = make_evaluator(batch_size=1, max_depth=1)
    roots = torch.arange(evaluator.root_nodes, device=env.device)
    evaluator.initialize_subgame(env, roots)
    evaluator.initialize_policy_and_beliefs()

    # Set up some regrets
    evaluator.cumulative_regrets[:] = (
        torch.rand_like(evaluator.cumulative_regrets) - 0.5
    )
    evaluator.regret_weight_sums[:] = torch.ones_like(evaluator.regret_weight_sums)

    # Record regret stats
    evaluator._record_cumulative_regret()

    # Should have recorded stats
    assert "mean_positive_regret" in evaluator.stats
    assert "mean_regret_bound" in evaluator.stats
    assert evaluator.stats["mean_positive_regret"] >= 0


def test_best_response_values() -> None:
    """Test _best_response_values computes best response values correctly."""
    evaluator, env = make_evaluator(batch_size=1, max_depth=1)
    roots = torch.arange(evaluator.root_nodes, device=env.device)
    evaluator.initialize_subgame(env, roots)
    evaluator.initialize_policy_and_beliefs()

    # Set up base values
    base_values = torch.zeros_like(evaluator.latest_values)
    base_values[evaluator.leaf_mask] = torch.rand_like(base_values[evaluator.leaf_mask])

    # Set up policy
    policy = evaluator.policy_probs_avg

    # Compute best response (deviating_player defaults to None, using env.to_act)
    br_values_p0 = evaluator._best_response_values(policy, base_values)

    # Best response values should be at least as good as base values
    # (for the deviating player)
    br_root_p0 = br_values_p0[0, 0]
    base_root_p0 = base_values[0, 0]
    # Best response should be >= base value (in expectation)
    assert torch.all(br_root_p0 >= base_root_p0 - 1e-5) or torch.allclose(
        br_root_p0, base_root_p0
    )


def test_record_action_mix() -> None:
    """Test _record_action_mix records action distribution statistics."""
    evaluator, env = make_evaluator(batch_size=1, max_depth=1)
    roots = torch.arange(evaluator.root_nodes, device=env.device)
    evaluator.initialize_subgame(env, roots)
    evaluator.initialize_policy_and_beliefs()

    # Set up policy
    evaluator.policy_probs_avg[:] = evaluator.policy_probs

    # Record action mix
    evaluator._record_action_mix()

    # Should have recorded action mix stats
    assert "action_mix" in evaluator.stats
    action_mix = evaluator.stats["action_mix"]
    assert "fold" in action_mix
    assert "call" in action_mix
    assert "bet" in action_mix
    assert "allin" in action_mix
    # All should be between 0 and 1
    assert 0 <= action_mix["fold"] <= 1
    assert 0 <= action_mix["call"] <= 1
    assert 0 <= action_mix["bet"] <= 1
    assert 0 <= action_mix["allin"] <= 1


def test_valid_nodes() -> None:
    """Test _valid_nodes yields correct node indices."""
    evaluator, env = make_evaluator(batch_size=2, max_depth=1)
    roots = torch.arange(evaluator.root_nodes, device=env.device)
    evaluator.initialize_subgame(env, roots)

    # Collect all valid nodes
    valid_nodes_list = []
    for depth, indices in evaluator._valid_nodes():
        valid_nodes_list.extend(indices.tolist())
        # All indices should be valid
        assert torch.all(evaluator.valid_mask[indices])

    # Test bottom_up=True
    valid_nodes_bottom_up = []
    for depth, indices in evaluator._valid_nodes(bottom_up=True):
        valid_nodes_bottom_up.extend(indices.tolist())

    # Should get same nodes (order may differ)
    assert set(valid_nodes_list) == set(valid_nodes_bottom_up)


def test_pull_back() -> None:
    """Test _pull_back reshapes data correctly."""
    evaluator, env = make_evaluator(batch_size=1, max_depth=1)
    roots = torch.arange(evaluator.root_nodes, device=env.device)
    evaluator.initialize_subgame(env, roots)

    # Create test data at child nodes
    num_actions = evaluator.num_actions
    bottom = evaluator.depth_offsets[1]
    top = evaluator.depth_offsets[2]
    child_data = torch.randn(
        top - bottom, NUM_HANDS, device=env.device, dtype=env.float_dtype
    )

    # Pull back with level=0 (method now expects full tensor and slices internally)
    full_data = torch.zeros(
        evaluator.total_nodes, NUM_HANDS, device=env.device, dtype=env.float_dtype
    )
    full_data[bottom:top] = child_data
    pulled = evaluator._pull_back(full_data, level=0)

    # Should have shape [num_parents, num_actions, NUM_HANDS]
    num_parents = evaluator.root_nodes
    assert pulled.shape == (num_parents, num_actions, NUM_HANDS)

    # Test without level (data includes root nodes, use all levels)
    pulled_full = evaluator._pull_back(full_data, level=None)
    assert pulled_full.shape == (num_parents, num_actions, NUM_HANDS)


def test_push_down() -> None:
    """Test _push_down reshapes data correctly."""
    evaluator, env = make_evaluator(batch_size=1, max_depth=1)
    roots = torch.arange(evaluator.root_nodes, device=env.device)
    evaluator.initialize_subgame(env, roots)

    # Create test data at parent nodes
    num_actions = evaluator.num_actions
    top = evaluator.depth_offsets[-2]
    parent_data = torch.randn(
        top, num_actions, NUM_HANDS, device=env.device, dtype=env.float_dtype
    )

    # Push down
    pushed = evaluator._push_down(parent_data)

    # Should have shape [num_children, NUM_HANDS]
    num_children = evaluator.total_nodes - evaluator.root_nodes
    assert pushed.shape == (num_children, NUM_HANDS)


def test_fan_out() -> None:
    """Test _fan_out broadcasts data to children correctly."""
    evaluator, env = make_evaluator(batch_size=1, max_depth=1)
    roots = torch.arange(evaluator.root_nodes, device=env.device)
    evaluator.initialize_subgame(env, roots)

    # Create test data at parent nodes
    num_actions = evaluator.num_actions
    top = evaluator.depth_offsets[-2]
    parent_data = torch.randn(top, 3, device=env.device, dtype=env.float_dtype)

    # Fan out
    fanned = evaluator._fan_out(parent_data, level=None)

    # Should have shape [num_children, 3]
    num_children = evaluator.total_nodes - evaluator.root_nodes
    assert fanned.shape == (num_children, 3)

    # Each parent's data should be repeated num_actions times
    for i in range(top):
        parent_val = parent_data[i]
        child_start = i * num_actions
        child_end = child_start + num_actions
        for j in range(child_start, child_end):
            torch.testing.assert_close(fanned[j], parent_val)

    # Test with level=0 (method now expects full tensor and slices internally)
    full_data_for_level = torch.zeros(
        evaluator.total_nodes, 3, device=env.device, dtype=env.float_dtype
    )
    full_data_for_level[:top] = parent_data
    fanned_sliced = evaluator._fan_out(full_data_for_level, level=0)
    assert fanned_sliced.shape == (num_actions, 3)
