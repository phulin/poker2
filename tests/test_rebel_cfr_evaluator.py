from __future__ import annotations

import torch
import pytest
from typing import Callable

from alphaholdem.env.card_utils import (
    combo_blocking_tensor,
    combo_index,
    mask_conflicting_combos,
)
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.env.rules import rank_hands
from alphaholdem.models.model_output import ModelOutput
from alphaholdem.search.rebel_cfr_evaluator import (
    RebelCFREvaluator,
    NUM_HANDS,
    PublicBeliefState,
)
from alphaholdem.utils.model_utils import compute_masked_logits


def make_env(num_envs: int = 4) -> HUNLTensorEnv:
    env = HUNLTensorEnv(
        num_envs=num_envs,
        starting_stack=1000,
        sb=5,
        bb=10,
        device=torch.device("cpu"),
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

    def __call__(self, features: torch.Tensor) -> ModelOutput:
        batch = features.shape[0]
        device = features.device
        dtype = features.dtype

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


def make_evaluator(
    *,
    batch_size: int = 2,
    max_depth: int = 2,
    bet_bins: list[float] | None = None,
    cfr_iterations: int = 4,
) -> tuple[RebelCFREvaluator, HUNLTensorEnv]:
    env = make_env(batch_size)
    bet_bins = bet_bins or env.default_bet_bins
    model = MockModel(
        logits=torch.zeros(len(bet_bins) + 3, dtype=env.float_dtype),
        hand_values=torch.zeros(1, 2, NUM_HANDS, dtype=env.float_dtype),
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
            >= evaluator.search_batch_size
        )
    )[0]


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
        tie_prob = (opp_cond * (opp_ranks == hero_rank)).sum()

        ev_per_hand[combo_idx] = potential * (win_prob + 0.5 * tie_prob)

    return ev_per_hand


def test_initialize_search_sets_uniform_beliefs() -> None:
    evaluator, env = make_evaluator(batch_size=2, max_depth=1)
    roots = torch.arange(evaluator.search_batch_size, device=env.device)
    evaluator.initialize_search(env, roots)

    assert torch.all(evaluator.valid_mask[roots])
    assert torch.count_nonzero(evaluator.valid_mask) == evaluator.search_batch_size
    assert not torch.any(evaluator.leaf_mask)
    assert torch.count_nonzero(evaluator.policy_probs) == 0
    assert torch.count_nonzero(evaluator.policy_probs_avg) == 0
    assert torch.count_nonzero(evaluator.values) == 0

    root_beliefs = evaluator.beliefs[roots]
    torch.testing.assert_close(
        root_beliefs.sum(dim=-1),
        torch.ones(
            (evaluator.search_batch_size, evaluator.num_players),
            device=env.device,
            dtype=env.float_dtype,
        ),
    )
    uniform = torch.full((NUM_HANDS,), 1.0 / NUM_HANDS, dtype=env.float_dtype)
    torch.testing.assert_close(root_beliefs[0, 0].cpu(), uniform)
    torch.testing.assert_close(root_beliefs[0, 1].cpu(), uniform)


def test_initialize_search_marks_done_roots_as_leaves() -> None:
    evaluator, env = make_evaluator(batch_size=2, max_depth=1)
    env.done[0] = True
    roots = torch.arange(evaluator.search_batch_size, device=env.device)
    evaluator.initialize_search(env, roots)
    assert evaluator.leaf_mask[0]
    assert not evaluator.leaf_mask[1]


def test_construct_subgame_clones_states_and_marks_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator, env = make_evaluator(batch_size=1, max_depth=2)
    roots = torch.arange(evaluator.search_batch_size, device=env.device)
    evaluator.initialize_search(env, roots)

    total_nodes = evaluator.total_nodes
    num_actions = evaluator.num_actions
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

    evaluator.construct_subgame()

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


def test_initialize_policy_respects_legal_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator, env = make_evaluator(batch_size=1, max_depth=1)
    roots = torch.arange(evaluator.search_batch_size, device=env.device)
    evaluator.initialize_search(env, roots)
    evaluator.construct_subgame()

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
    roots = torch.arange(evaluator.search_batch_size, device=env.device)
    evaluator.initialize_search(env, roots)

    total_nodes = evaluator.total_nodes
    num_actions = evaluator.num_actions
    legal_mask = torch.zeros(
        (total_nodes, num_actions), dtype=torch.bool, device=env.device
    )
    legal_mask[0, 0:2] = True
    # Ensure all nodes have at least one legal action
    for i in range(total_nodes):
        if not legal_mask[i].any():
            legal_mask[i, 0] = True

    monkeypatch.setattr(
        evaluator.env,
        "legal_bins_mask",
        lambda bet_bins=None: legal_mask,
    )

    evaluator.construct_subgame()

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
        batch_size = features.shape[0]
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


def test_compute_expected_values_matches_child_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator, env = make_evaluator(batch_size=1, max_depth=2)
    roots = torch.arange(evaluator.search_batch_size)
    evaluator.initialize_search(env, roots)

    num_actions = evaluator.num_actions
    legal_mask = torch.ones((evaluator.total_nodes, num_actions), dtype=torch.bool)
    bin_amounts, _ = evaluator.env.legal_bins_amounts_and_mask()
    monkeypatch.setattr(
        evaluator.env,
        "legal_bins_amounts_and_mask",
        lambda bet_bins=None: (bin_amounts.clone(), legal_mask.clone()),
    )

    evaluator.construct_subgame()
    evaluator.initialize_policy_and_beliefs()
    evaluator.set_leaf_values()

    probs = torch.arange(1, num_actions + 1, dtype=env.float_dtype)
    probs = probs / probs.sum()
    probs_all = probs[None, :, None].expand(evaluator.total_nodes, -1, 1)
    bottom = evaluator.depth_offsets[1]
    evaluator.policy_probs[bottom:] = evaluator._push_down(probs_all)

    child_values = torch.arange(1, num_actions + 1, dtype=env.float_dtype)
    child_values_all = child_values[None, :, None, None].expand(
        evaluator.total_nodes, -1, 1, 1
    )
    values_temp = evaluator._push_down(child_values_all)
    values_bottom = torch.zeros_like(values_temp)
    values_bottom[evaluator.leaf_mask[bottom:]] = values_temp[
        evaluator.leaf_mask[bottom:]
    ]
    evaluator.values[:] = 0.0
    evaluator.values[bottom:, 0] = values_bottom[:, 0]
    evaluator.values[bottom:, 1] = -values_bottom[:, 0]

    new_values = evaluator.compute_expected_values()
    expected_value = (probs * child_values).sum()

    torch.testing.assert_close(
        new_values[2, 0],
        torch.full((NUM_HANDS,), expected_value, dtype=env.float_dtype),
    )
    torch.testing.assert_close(
        new_values[2, 1],
        torch.full((NUM_HANDS,), -expected_value, dtype=env.float_dtype),
    )


def test_set_leaf_values_only_updates_marked_nodes() -> None:
    evaluator, env = make_evaluator(batch_size=1, max_depth=2)
    roots = torch.arange(evaluator.search_batch_size, device=env.device)
    evaluator.initialize_search(env, roots)

    leaf_indices = torch.tensor([1, 2], device=env.device)
    evaluator.valid_mask[leaf_indices] = True
    evaluator.leaf_mask.zero_()
    evaluator.leaf_mask[leaf_indices] = True

    evaluator.values.zero_()

    hand_values = torch.zeros(
        evaluator.total_nodes,
        evaluator.num_players,
        NUM_HANDS,
        device=env.device,
        dtype=env.float_dtype,
    )
    hand_values[leaf_indices[0]] = torch.full(
        (evaluator.num_players, NUM_HANDS),
        2.5,
        device=env.device,
        dtype=env.float_dtype,
    )
    hand_values[leaf_indices[1]] = torch.full(
        (evaluator.num_players, NUM_HANDS),
        -1.25,
        device=env.device,
        dtype=env.float_dtype,
    )

    # Mock the model to return the expected hand values
    def custom_hand_values_fn(features):
        batch_size = features.shape[0]
        model_hand_values = torch.zeros(
            batch_size,
            evaluator.num_players,
            NUM_HANDS,
            device=env.device,
            dtype=env.float_dtype,
        )
        for i in range(batch_size):
            model_hand_values[i] = hand_values[leaf_indices[i].item()]
        return model_hand_values

    evaluator.model = MockModel(
        custom_hand_values_fn=custom_hand_values_fn,
        num_actions=len(evaluator.bet_bins) + 3,
        num_players=evaluator.num_players,
        device=env.device,
        dtype=env.float_dtype,
    )

    evaluator.set_leaf_values()

    torch.testing.assert_close(
        evaluator.values[leaf_indices[0]],
        hand_values[leaf_indices[0]],
    )
    torch.testing.assert_close(
        evaluator.values[leaf_indices[1]],
        hand_values[leaf_indices[1]],
    )
    assert torch.count_nonzero(evaluator.values[roots]) == 0


def test_sample_leaf_copies_selected_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    evaluator, env = make_evaluator(batch_size=1, max_depth=2)
    roots = torch.arange(evaluator.search_batch_size, device=env.device)
    evaluator.initialize_search(env, roots)

    evaluator.generator = torch.Generator(device=env.device)
    evaluator.generator.manual_seed(42)
    evaluator.sample_epsilon = 0.0

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

    root = 0
    child_call = 2  # action index 1 child
    child_fold = 1
    evaluator.valid_mask.zero_()
    evaluator.valid_mask[root] = True
    evaluator.valid_mask[child_call] = True
    evaluator.valid_mask[child_fold] = True
    evaluator.leaf_mask.zero_()
    evaluator.leaf_mask[child_call] = True
    evaluator.leaf_mask[child_fold] = True
    evaluator.env.done.zero_()

    actor_belief = torch.full(
        (NUM_HANDS,), 1.0 / NUM_HANDS, device=env.device, dtype=env.float_dtype
    )
    evaluator.beliefs.zero_()
    evaluator.beliefs[root, 0] = actor_belief
    evaluator.beliefs[root, 1] = actor_belief
    evaluator.beliefs[child_call, 0] = actor_belief
    evaluator.beliefs[child_call, 1] = actor_belief
    evaluator.beliefs[child_fold, 0] = actor_belief
    evaluator.beliefs[child_fold, 1] = actor_belief

    evaluator.policy_probs.zero_()
    evaluator.policy_probs[child_call, :] = 1.0
    evaluator.policy_probs_avg[:] = evaluator.policy_probs

    evaluator.env.pot[child_call] = evaluator.env.pot[root] + 10
    evaluator.env.to_act[root] = 0
    evaluator.env.to_act[child_call] = 1

    pbs = PublicBeliefState.from_proto(
        env_proto=evaluator.env,
        beliefs=torch.zeros(1, evaluator.num_players, NUM_HANDS, device=env.device),
        num_envs=1,
    )

    evaluator.sample_leaf(
        torch.tensor([root], device=env.device),
        pbs,
        0,
        training_mode=False,
    )

    expected_index = child_call
    torch.testing.assert_close(pbs.beliefs[0], evaluator.beliefs[expected_index])
    assert pbs.env.to_act[0] == evaluator.env.to_act[expected_index]
    assert pbs.env.pot[0] == evaluator.env.pot[expected_index]


def test_sample_leaf_handles_partial_masks() -> None:
    evaluator, env = make_evaluator(batch_size=3, max_depth=2)
    roots = torch.arange(evaluator.search_batch_size, device=env.device)
    evaluator.initialize_search(env, roots)

    evaluator.construct_subgame()
    evaluator.initialize_policy_and_beliefs()

    root_indices = torch.tensor([0, 2], device=env.device)
    pbs = PublicBeliefState.from_proto(
        env_proto=evaluator.env,
        beliefs=torch.zeros(2, evaluator.num_players, NUM_HANDS, device=env.device),
        num_envs=2,
    )

    evaluator.sample_leaf(root_indices, pbs, 0, training_mode=True)

    assert pbs.env.N == 2
    assert pbs.beliefs.shape == (2, evaluator.num_players, NUM_HANDS)


def test_update_policy_uses_positive_regrets(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator, env = make_evaluator(batch_size=1, max_depth=2)
    roots = torch.arange(evaluator.search_batch_size, device=env.device)
    evaluator.initialize_search(env, roots)
    evaluator.construct_subgame()

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

    evaluator.env.to_act[root_index] = 0
    evaluator.env.to_act[child_indices] = 1

    uniform = torch.full((NUM_HANDS,), 1.0 / NUM_HANDS, dtype=env.float_dtype)
    evaluator.beliefs[:] = uniform[None, None, :]

    evaluator.policy_probs[:] = 1.0 / num_actions

    evaluator.values[:] = 0.0
    evaluator.values[1, 0] = 2.0  # Positive advantage for action 0
    evaluator.values[2, 0] = -1.0  # Negative advantage for action 1

    # Compute regrets first, then update policy
    regrets = evaluator.compute_instantaneous_regrets(evaluator.values)
    evaluator.cumulative_regrets += regrets
    evaluator.update_policy()

    root_policy = evaluator.policy_probs[1 : num_actions + 1, 0]
    expected = torch.zeros(num_actions, dtype=env.float_dtype)
    expected[0] = 1.0
    torch.testing.assert_close(root_policy, expected)


def test_sample_data_returns_root_batch() -> None:
    evaluator, env = make_evaluator(batch_size=2, max_depth=1)
    roots = torch.arange(evaluator.search_batch_size, device=env.device)
    children = torch.arange(
        evaluator.depth_offsets[1], evaluator.depth_offsets[2], device=env.device
    )
    evaluator.initialize_search(env, roots)

    uniform_policy = torch.full(
        (NUM_HANDS, evaluator.num_actions),
        1.0 / evaluator.num_actions,
        device=env.device,
    )
    expected_policy = uniform_policy.unsqueeze(0).expand(
        evaluator.search_batch_size, -1, -1
    )
    evaluator.policy_probs_avg[children] = expected_policy.permute(0, 2, 1).reshape(
        -1, NUM_HANDS
    )
    evaluator.values[roots] = 0.5

    batch = evaluator.sample_data()

    assert batch.features.shape == (
        evaluator.search_batch_size,
        evaluator.feature_encoder.feature_dim,
    )
    assert batch.policy_targets.shape == (
        evaluator.search_batch_size,
        NUM_HANDS,
        evaluator.num_actions,
    )
    assert batch.value_targets.shape == (
        evaluator.search_batch_size,
        evaluator.num_players,
        NUM_HANDS,
    )
    torch.testing.assert_close(batch.policy_targets, expected_policy)


def setup_showdown_evaluator(
    board: str | list[int],
    beliefs: torch.Tensor | None = None,
) -> tuple[RebelCFREvaluator, HUNLTensorEnv, torch.Tensor]:
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

    roots = torch.arange(evaluator.search_batch_size)
    if beliefs is None:
        beliefs = torch.full((1, NUM_HANDS), 1.0 / NUM_HANDS)
    evaluator.initialize_search(env, roots, beliefs)
    evaluator.construct_subgame()
    evaluator.initialize_policy_and_beliefs()
    idx = torch.tensor([0])
    potential = (
        evaluator.env.stacks[idx, 0]
        + evaluator.env.pot[idx]
        - evaluator.env.starting_stack
    )

    return evaluator, env, idx, potential


def test_showdown_value_uniform_beliefs_matches_reference() -> None:
    evaluator, _, idx, potential = setup_showdown_evaluator(
        "Ac Qh 9d 7s 4h",
    )

    opp_beliefs = evaluator.beliefs[idx, 1].squeeze(0)
    expected = compute_reference_showdown_ev(
        evaluator.env.board_indices[0], opp_beliefs, potential
    )
    expected /= evaluator.env.scale

    actual = evaluator._showdown_value(idx).squeeze(0)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_showdown_value_single_hand_belief_returns_potential() -> None:
    hero_combo = combo_index(card(8, 2), card(7, 2))  # Ten and Nine hearts
    beliefs = torch.zeros(1, NUM_HANDS)
    beliefs[0, hero_combo] = 1.0
    evaluator, _, idx, potential = setup_showdown_evaluator("Ad Kd Qd Jd 2s", beliefs)

    opp_beliefs = evaluator.beliefs[idx, 1].squeeze(0)
    expected = compute_reference_showdown_ev(
        evaluator.env.board_indices[0], opp_beliefs, potential
    )
    expected = expected / evaluator.env.scale

    actual = evaluator._showdown_value(idx).squeeze(0)
    torch.testing.assert_close(actual, expected)


def test_showdown_value_all_ties_returns_half_potential() -> None:
    evaluator, _, idx, potential = setup_showdown_evaluator("Ah Kh Qh Jh Th")

    opp_beliefs = evaluator.beliefs[idx, 1].squeeze(0)
    expected = compute_reference_showdown_ev(
        evaluator.env.board_indices[0], opp_beliefs, potential
    )
    expected = expected / evaluator.env.scale
    actual = evaluator._showdown_value(idx).squeeze(0)
    board_mask = mask_conflicting_combos(evaluator.env.board_indices[0])
    torch.testing.assert_close(
        actual[~board_mask], torch.zeros_like(actual[~board_mask])
    )
    torch.testing.assert_close(actual, expected)


def test_showdown_value_fail_when_opponent_range_invalid() -> None:
    board_tensor = make_board("Jh 8s 6c 4h 2d")[None, :]
    beliefs = torch.zeros(1, NUM_HANDS)
    board_mask = mask_conflicting_combos(board_tensor[0])
    conflicting_indices = torch.where(~board_mask)[0]
    assert conflicting_indices.numel() > 0

    beliefs[0].zero_()
    beliefs[0, conflicting_indices[:10]] = 0.1

    evaluator, _, idx, _ = setup_showdown_evaluator("Jh 8s 6c 4h 2d", beliefs)

    # Should get uniform beliefs since all beliefs were blocked
    torch.testing.assert_close(
        evaluator.beliefs[idx], torch.full((1, 2, NUM_HANDS), 1.0 / NUM_HANDS)
    )

    with pytest.raises(AssertionError):
        evaluator._showdown_value(idx).squeeze(0)


@pytest.mark.parametrize(
    "board_cards",
    [
        [card(12, 0), card(9, 1), card(5, 3), card(3, 0), card(0, 2)],
        [card(8, 3), card(8, 1), card(8, 0), card(5, 2), card(5, 1)],
        [card(7, 2), card(6, 2), card(5, 2), card(4, 2), card(3, 2)],
    ],
)
def test_showdown_value_matches_reference_on_diverse_boards(
    board_cards: list[int],
) -> None:
    evaluator, _, idx, potential = setup_showdown_evaluator(board_cards)

    opp_beliefs = evaluator.beliefs[idx, 1].squeeze(0)
    expected = compute_reference_showdown_ev(
        evaluator.env.board_indices[0], opp_beliefs, potential
    )
    expected = expected / evaluator.env.scale
    actual = evaluator._showdown_value(idx).squeeze(0)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_self_play_iteration_returns_public_belief_state() -> None:
    evaluator, env = make_evaluator(batch_size=2, max_depth=1)
    roots = torch.arange(evaluator.search_batch_size, device=env.device)
    evaluator.initialize_search(env, roots)
    # evaluator.warm_start_iterations = 0
    evaluator.cfr_iterations = 2
    evaluator.generator = torch.Generator(device=env.device)
    evaluator.generator.manual_seed(1)

    next_pbs = evaluator.self_play_iteration(training_mode=False)

    assert next_pbs is not None
    assert next_pbs.env.N == 1
    assert next_pbs.beliefs.shape == (1, evaluator.num_players, NUM_HANDS)
    torch.testing.assert_close(
        next_pbs.beliefs.sum(dim=-1),
        torch.ones((1, evaluator.num_players), device=env.device),
    )
