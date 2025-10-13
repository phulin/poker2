import torch
import pytest

from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.model_output import ModelOutput
from alphaholdem.search.rebel_cfr_evaluator import RebelCFREvaluator, NUM_HANDS
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
        sample_count=1,
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
    model_output = evaluator.evaluate_model_on_valid()
    evaluator.initialize_policy(model_output)

    expected = torch.softmax(
        compute_masked_logits(logits.unsqueeze(0), legal_mask[:1]), dim=-1
    )
    expected = expected * legal_mask[:1]
    expected = expected / expected.sum(dim=-1, keepdim=True)
    expected = expected.squeeze(0)

    root_policy = evaluator.policy_probs[0, 0]
    torch.testing.assert_close(root_policy, expected)


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
