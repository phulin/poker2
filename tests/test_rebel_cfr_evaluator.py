import torch
import pytest

from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
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


class ConstantModel:
    """Simple model stub that returns fixed logits and hand values."""

    def __init__(
        self,
        logits: torch.Tensor,
        hand_values: torch.Tensor | None = None,
    ) -> None:
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        self.logits = logits
        if hand_values is None:
            hand_values = torch.zeros(1, 2, NUM_HANDS)
        self.hand_values = hand_values

    def __call__(self, features: torch.Tensor) -> ModelOutput:
        batch = features.shape[0]
        device = features.device
        dtype = features.dtype
        logits = self.logits.to(device=device, dtype=dtype)
        if logits.shape[0] != batch:
            logits = logits.expand(batch, -1)
        hand_values = self.hand_values.to(device=device, dtype=dtype)
        if hand_values.shape[0] != batch:
            hand_values = hand_values.expand(batch, -1, -1)
        value = torch.zeros(batch, device=device, dtype=dtype)
        return ModelOutput(
            policy_logits=logits,
            value=value,
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
    model = ConstantModel(
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


def test_initialize_policy_respects_legal_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator, env = make_evaluator(batch_size=1, max_depth=1)
    roots = torch.arange(evaluator.search_batch_size, device=env.device)
    evaluator.initialize_search(env, roots)

    num_actions = evaluator.num_actions
    legal_mask = torch.zeros(
        (evaluator.total_nodes, num_actions),
        dtype=torch.bool,
        device=env.device,
    )
    legal_mask[0, 1:3] = True  # Only allow actions 1 and 2 at the root

    monkeypatch.setattr(
        evaluator.env,
        "legal_bins_mask",
        lambda bet_bins=None: legal_mask,
    )

    logits = torch.arange(float(num_actions), dtype=env.float_dtype)
    evaluator.model = ConstantModel(logits=logits)  # type: ignore[assignment]
    model_output = evaluator.evaluate_model_on_all()
    evaluator.initialize_policy(model_output)

    expected = torch.softmax(
        compute_masked_logits(logits.unsqueeze(0), legal_mask[:1]), dim=-1
    )
    expected = expected * legal_mask[:1]
    expected = expected / expected.sum(dim=-1, keepdim=True)
    expected = expected.squeeze(0)

    root_policy = evaluator.policy_probs[0, 0]
    torch.testing.assert_close(root_policy, expected)


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
    legal_mask[0, 0:2] = True

    monkeypatch.setattr(
        evaluator.env,
        "legal_bins_mask",
        lambda bet_bins=None: legal_mask,
    )

    step_observer: dict[str, torch.Tensor] = {}

    def fake_step_bins(
        bin_indices: torch.Tensor,
        legal_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        step_observer["bins"] = bin_indices.clone()
        rewards = torch.zeros_like(bin_indices, dtype=evaluator.float_dtype)
        dones = evaluator.env.done.clone()
        to_act = evaluator.env.to_act.clone()
        new_streets = torch.full_like(bin_indices, -1, dtype=torch.long)
        dealt_cards = torch.full(
            (bin_indices.shape[0], 3),
            -1,
            dtype=torch.long,
            device=bin_indices.device,
        )
        return rewards, dones, to_act, new_streets, dealt_cards

    monkeypatch.setattr(evaluator.env, "step_bins", fake_step_bins)

    evaluator.construct_subgame()

    child_indices = torch.tensor([1, 2], device=env.device)
    assert torch.all(evaluator.valid_mask[child_indices])
    assert not torch.any(evaluator.valid_mask[3:])
    torch.testing.assert_close(
        evaluator.env.pot[child_indices],
        evaluator.env.pot[roots].expand(child_indices.shape[0]),
    )
    assert torch.all(evaluator.leaf_mask[child_indices])
    assert "bins" in step_observer
    torch.testing.assert_close(
        step_observer["bins"][child_indices],
        torch.tensor([0, 1], device=env.device),
    )


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

    monkeypatch.setattr(
        evaluator.env,
        "legal_bins_mask",
        lambda bet_bins=None: legal_mask,
    )

    root = 0
    child_a = 1
    child_b = 2
    evaluator.valid_mask.zero_()
    evaluator.leaf_mask.zero_()
    evaluator.valid_mask[root] = True
    evaluator.valid_mask[child_a] = True
    evaluator.valid_mask[child_b] = True
    evaluator.leaf_mask[child_a] = True
    evaluator.leaf_mask[child_b] = True

    evaluator.env.to_act[root] = 0

    hand_ids = torch.arange(NUM_HANDS, device=env.device, dtype=env.float_dtype)
    root_actor = hand_ids + 1.0
    root_actor = root_actor / root_actor.sum()
    root_opp = torch.full_like(root_actor, 1.0 / NUM_HANDS)
    evaluator.beliefs[root, 0] = root_actor
    evaluator.beliefs[root, 1] = root_opp

    action0_scores = hand_ids + 1.0
    action1_scores = torch.flip(action0_scores, dims=[0])
    score_sum = action0_scores + action1_scores
    policy_action0 = action0_scores / score_sum
    policy_action1 = action1_scores / score_sum

    evaluator.policy_probs[root, :, 0] = policy_action0
    evaluator.policy_probs[root, :, 1] = policy_action1

    dummy_output = ModelOutput(
        policy_logits=torch.zeros(
            evaluator.total_nodes,
            evaluator.num_actions,
            device=env.device,
            dtype=env.float_dtype,
        ),
        value=torch.zeros(
            evaluator.total_nodes, device=env.device, dtype=env.float_dtype
        ),
        hand_values=torch.zeros(
            evaluator.total_nodes,
            evaluator.num_players,
            NUM_HANDS,
            device=env.device,
            dtype=env.float_dtype,
        ),
    )

    evaluator.initialize_beliefs(dummy_output)

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
    roots = torch.arange(evaluator.search_batch_size, device=env.device)
    evaluator.initialize_search(env, roots)

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

    root_index = torch.tensor([0], device=env.device)
    child_indices = torch.arange(1, evaluator.total_nodes, device=env.device)

    evaluator.valid_mask[root_index] = True
    evaluator.valid_mask[child_indices] = True
    evaluator.leaf_mask[root_index] = False
    evaluator.leaf_mask[child_indices] = True

    root_probs = torch.arange(1, num_actions + 1, dtype=env.float_dtype)
    root_probs = root_probs / root_probs.sum()
    evaluator.policy_probs[0] = root_probs.unsqueeze(0).expand(NUM_HANDS, -1)

    child_values = torch.arange(1, num_actions + 1, dtype=env.float_dtype)
    for action, idx in enumerate(child_indices):
        evaluator.values[idx, 0] = child_values[action % child_values.numel()]
        evaluator.values[idx, 1] = -child_values[action % child_values.numel()]

    evaluator.values[0] = 0.0

    new_values = evaluator.compute_expected_values()
    expected_value = (root_probs * child_values).sum()

    torch.testing.assert_close(
        new_values[0, 0],
        torch.full((NUM_HANDS,), expected_value, dtype=env.float_dtype),
    )
    torch.testing.assert_close(
        new_values[0, 1],
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

    dummy_output = ModelOutput(
        policy_logits=torch.zeros(
            evaluator.total_nodes,
            evaluator.num_actions,
            device=env.device,
            dtype=env.float_dtype,
        ),
        value=torch.zeros(
            evaluator.total_nodes, device=env.device, dtype=env.float_dtype
        ),
        hand_values=hand_values,
    )

    evaluator.set_leaf_values(dummy_output)

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
    evaluator.policy_probs[root, :, 1] = 1.0
    evaluator.policy_probs_avg.copy_(evaluator.policy_probs)

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


def test_update_policy_uses_positive_regrets(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator, env = make_evaluator(batch_size=1, max_depth=2)
    roots = torch.arange(evaluator.search_batch_size, device=env.device)
    evaluator.initialize_search(env, roots)

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
    child_indices = torch.arange(1, evaluator.total_nodes, device=env.device)

    evaluator.valid_mask[:] = False
    evaluator.valid_mask[root_index] = True
    evaluator.valid_mask[child_indices] = True
    evaluator.leaf_mask[:] = True
    evaluator.leaf_mask[root_index] = False

    evaluator.env.to_act[root_index] = 0
    evaluator.combo_compat = torch.eye(
        NUM_HANDS, device=env.device, dtype=env.float_dtype
    )

    uniform = torch.full((NUM_HANDS,), 1.0 / NUM_HANDS, dtype=env.float_dtype)
    evaluator.beliefs[root_index, 0] = uniform
    evaluator.beliefs[root_index, 1] = uniform
    for idx in child_indices:
        evaluator.beliefs[idx, 0] = uniform
        evaluator.beliefs[idx, 1] = uniform

    evaluator.values[:] = 0.0
    evaluator.values[1, 0] = 2.0  # Positive advantage for action 0
    evaluator.values[2, 0] = -1.0  # Negative advantage for action 1

    evaluator.update_policy()

    root_policy = evaluator.policy_probs[root_index, 0]
    expected = torch.zeros(num_actions, dtype=env.float_dtype)
    expected[0] = 1.0
    torch.testing.assert_close(root_policy, expected)


def test_sample_data_returns_root_batch() -> None:
    evaluator, env = make_evaluator(batch_size=2, max_depth=1)
    roots = torch.arange(evaluator.search_batch_size, device=env.device)
    evaluator.initialize_search(env, roots)

    uniform_policy = torch.full(
        (NUM_HANDS, evaluator.num_actions),
        1.0 / evaluator.num_actions,
        device=env.device,
        dtype=env.float_dtype,
    )
    expected_policy = uniform_policy.unsqueeze(0).expand(
        evaluator.search_batch_size, -1, -1
    )
    evaluator.policy_probs_avg[roots] = expected_policy
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
    torch.testing.assert_close(
        batch.legal_masks,
        evaluator.env.legal_bins_mask()[roots],
    )
    torch.testing.assert_close(batch.policy_targets, expected_policy)


def test_self_play_iteration_returns_public_belief_state() -> None:
    evaluator, env = make_evaluator(batch_size=1, max_depth=1)
    roots = torch.arange(evaluator.search_batch_size, device=env.device)
    evaluator.initialize_search(env, roots)
    evaluator.warm_start_iterations = 0
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
