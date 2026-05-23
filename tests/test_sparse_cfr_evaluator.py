from __future__ import annotations

import torch
from torch.testing import assert_close

from p2.core.structured_config import (
    CFRType,
    Config,
    EnvConfig,
    ModelConfig,
    ModelType,
    SearchConfig,
    TrainingConfig,
)
from p2.env.card_utils import NUM_HANDS
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.models.mlp.mlp_features import MLPFeatures
from p2.models.model_output import ModelOutput
from p2.search.sparse_cfr_evaluator import SparseCFREvaluator


STREET_CUTOFF_SCHEDULE = [
    [0.25, 0.5, 0.75, 1.0, 1.5],
    [0.5, 1.0],
    [1.0],
    [1.0],
    [1.0],
    [],
]


def get_device() -> torch.device:
    return torch.device("cpu")


def make_env(num_envs: int = 1, device: torch.device | None = None) -> HUNLTensorEnv:
    """Create a test environment."""
    if device is None:
        device = get_device()
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
    """Simple mock model for testing."""

    def __init__(
        self,
        num_actions: int = 5,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_actions = num_actions
        self.device = device
        self.dtype = dtype
        self.enforce_zero_sum = True
        self.hidden_dim = 1

    def create_feature_encoder(self, env, device=None, dtype=None):
        from p2.models.mlp.rebel_feature_encoder import RebelFeatureEncoder

        return RebelFeatureEncoder(env=env, device=device, dtype=dtype)

    def __call__(
        self, features: MLPFeatures, include_policy: bool = True
    ) -> ModelOutput:
        batch = len(features)
        device = features.context.device
        dtype = features.context.dtype

        # Default uniform logits
        logits = torch.zeros(
            batch, NUM_HANDS, self.num_actions, device=device, dtype=dtype
        )
        # Default zero hand values
        hand_values = torch.zeros(batch, 2, NUM_HANDS, device=device, dtype=dtype)

        return ModelOutput(
            policy_logits=logits,
            value=torch.zeros(batch, device=device, dtype=dtype),
            hand_values=hand_values,
        )

    def eval(self) -> None:
        pass


class PreOnlyMockModel(MockModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.forward_pre_calls = 0

    def forward_pre(self, features: MLPFeatures) -> torch.Tensor:
        self.forward_pre_calls += 1
        batch = len(features)
        return torch.zeros(
            batch,
            2,
            NUM_HANDS,
            device=features.context.device,
            dtype=features.context.dtype,
        )


class DeterministicModel:
    """Deterministic policy/value head used for sparse vs rebel parity tests."""

    def __init__(
        self,
        num_actions: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.num_actions = num_actions
        self.device = device
        self.dtype = dtype
        logits = torch.linspace(
            -0.25, 0.75, steps=num_actions, device=device, dtype=dtype
        )
        self.base_logits = logits.view(1, 1, -1)
        hand_values = torch.linspace(
            -0.4, 0.4, steps=NUM_HANDS, device=device, dtype=dtype
        )
        self.base_values = torch.stack([hand_values, -hand_values], dim=0)
        self.hidden_dim = 1

    def eval(self) -> None:
        pass

    def train(self) -> None:
        pass

    def create_feature_encoder(self, env, device=None, dtype=None):
        from p2.models.mlp.rebel_feature_encoder import RebelFeatureEncoder

        return RebelFeatureEncoder(env=env, device=device, dtype=dtype)

    def __call__(
        self, features: MLPFeatures, include_policy: bool = True
    ) -> ModelOutput:
        batch = len(features)
        logits = self.base_logits.expand(batch, NUM_HANDS, self.num_actions).clone()
        values = self.base_values.expand(batch, 2, NUM_HANDS).clone()
        return ModelOutput(
            policy_logits=logits,
            value=torch.zeros(batch, device=self.device, dtype=self.dtype),
            hand_values=values,
        )


def make_config(bet_bins: list[float] | None = None) -> Config:
    """Create a test configuration."""
    cfg = Config()
    cfg.device = "cpu"
    cfg.seed = 42

    cfg.train = TrainingConfig()

    cfg.model = ModelConfig()
    cfg.model.name = ModelType.rebel_ffn
    cfg.model.input_dim = 2661
    cfg.model.hidden_dim = 128
    cfg.model.num_hidden_layers = 2
    cfg.model.num_actions = -1  # Will be set below
    cfg.model.value_head_type = "scalar"
    cfg.model.detach_value_head = True

    cfg.env = EnvConfig()
    cfg.env.stack = 1000
    cfg.env.sb = 5
    cfg.env.bb = 10
    cfg.env.bet_bins = bet_bins or [0.5, 1.5]
    cfg.env.flop_showdown = False

    cfg.search = SearchConfig()
    cfg.search.iterations = 1
    cfg.search.warm_start_iterations = 0
    cfg.search.depth = 1
    cfg.search.branching = 4
    cfg.search.dcfr_alpha = 1.5
    cfg.search.dcfr_beta = 0.0
    cfg.search.dcfr_gamma = 2.0
    cfg.search.cfr_type = CFRType.linear
    cfg.search.cfr_avg = True

    cfg.model.num_actions = len(cfg.env.bet_bins) + 3

    return cfg


def make_street_cutoff_config() -> Config:
    cfg = make_config([0.25, 0.5, 0.75, 1.0, 1.5])
    cfg.search.bet_bins_by_depth = STREET_CUTOFF_SCHEDULE
    cfg.search.allin_by_depth = [True, True, True, True, True, False]
    cfg.search.depth = len(STREET_CUTOFF_SCHEDULE)
    cfg.model.num_actions = len(cfg.env.bet_bins) + 3
    return cfg


def make_sparse_evaluator(
    model: MockModel | None = None,
    env: HUNLTensorEnv | None = None,
    cfg: Config | None = None,
    device: torch.device | None = None,
) -> tuple[SparseCFREvaluator, HUNLTensorEnv, Config]:
    """Create a sparse CFR evaluator for testing."""
    if device is None:
        device = get_device()
    if env is None:
        env = make_env(1, device=device)
    if cfg is None:
        cfg = make_config(env.default_bet_bins)
    if model is None:
        model = MockModel(
            num_actions=len(cfg.env.bet_bins) + 3, device=device, dtype=env.float_dtype
        )

    evaluator = SparseCFREvaluator(
        model=model,  # type: ignore[arg-type]
        device=device,
        cfg=cfg,
    )

    return evaluator, env, cfg


def test_sparse_evaluator_initialization() -> None:
    """Test that sparse evaluator can be initialized."""
    device = get_device()
    evaluator, env, cfg = make_sparse_evaluator(device=device)

    assert evaluator.model is not None
    assert evaluator.device == device
    assert evaluator.num_players == 2
    assert evaluator.num_actions == len(cfg.env.bet_bins) + 3
    assert evaluator.total_nodes == 0  # Not initialized yet
    assert len(evaluator.depth_offsets) == 1
    assert evaluator.depth_offsets[0] == 0


def test_initialize_subgame() -> None:
    """Test that subgame can be initialized."""
    device = get_device()
    evaluator, env, cfg = make_sparse_evaluator(device=device)

    root_indices = torch.tensor([0], dtype=torch.long, device=device)
    evaluator.initialize_subgame(env, root_indices)

    # After initialization, should have at least root nodes
    assert evaluator.total_nodes > 0
    assert evaluator.root_nodes == 1
    assert len(evaluator.depth_offsets) >= 2
    assert evaluator.depth_offsets[0] == 0
    assert evaluator.depth_offsets[1] == evaluator.root_nodes

    # Verify root nodes have self_reach initialized to 1.0
    root_self_reach = evaluator.self_reach[: evaluator.root_nodes]
    torch.testing.assert_close(
        root_self_reach,
        torch.ones_like(root_self_reach),
        msg="Root nodes should have self_reach initialized to 1.0",
    )

    # Check tensor shapes
    assert evaluator.beliefs.shape == (evaluator.total_nodes, 2, NUM_HANDS)
    assert evaluator.policy_probs.shape == (
        evaluator.total_nodes,
        NUM_HANDS,
    )
    assert evaluator.latest_values.shape == (evaluator.total_nodes, 2, NUM_HANDS)
    assert evaluator.leaf_mask.shape == (evaluator.total_nodes,)
    assert evaluator.child_mask.shape == (evaluator.total_nodes, evaluator.num_actions)


def test_initialize_policy_and_beliefs() -> None:
    """Test that policy and beliefs can be initialized."""
    device = get_device()
    evaluator, env, cfg = make_sparse_evaluator(device=device)

    root_indices = torch.tensor([0], dtype=torch.long, device=device)
    evaluator.initialize_subgame(env, root_indices)

    # Set initial beliefs at root
    uniform_beliefs = (
        torch.ones(1, 2, NUM_HANDS, device=device, dtype=torch.float32) / NUM_HANDS
    )
    evaluator.beliefs[: evaluator.root_nodes] = uniform_beliefs

    # Initialize policy and beliefs
    evaluator.initialize_policy_and_beliefs()

    # Check that policy probabilities are valid
    assert evaluator.policy_probs.shape == (
        evaluator.total_nodes,
        NUM_HANDS,
    )
    # Policy should sum to 1 over actions (for legal actions)
    # But we can't easily check this without masking, so just check shape and non-negative
    assert (evaluator.policy_probs >= 0).all()
    assert (evaluator.policy_probs <= 1).all()

    # Check that beliefs are propagated (non-root nodes should have non-zero beliefs if they have children)
    assert evaluator.beliefs.shape == (evaluator.total_nodes, 2, NUM_HANDS)
    assert (evaluator.beliefs >= 0).all()


def test_update_policy() -> None:
    """Test that policy can be updated."""
    device = get_device()
    evaluator, env, cfg = make_sparse_evaluator(device=device)

    root_indices = torch.tensor([0], dtype=torch.long, device=device)
    evaluator.initialize_subgame(env, root_indices)

    # Set up some cumulative regrets
    evaluator.cumulative_regrets = torch.ones(
        evaluator.total_nodes,
        NUM_HANDS,
        device=device,
        dtype=torch.float32,
    )
    # Update policy
    evaluator.update_policy(t=0)

    # Check that policy is updated
    assert evaluator.policy_probs.shape == (
        evaluator.total_nodes,
        NUM_HANDS,
    )
    assert (evaluator.policy_probs >= 0).all()


def test_compute_expected_values() -> None:
    """Test that expected values can be computed."""
    device = get_device()
    evaluator, env, cfg = make_sparse_evaluator(device=device)

    root_indices = torch.tensor([0], dtype=torch.long, device=device)
    evaluator.initialize_subgame(env, root_indices)

    # Set some leaf values
    evaluator.latest_values = torch.randn(
        evaluator.total_nodes, 2, NUM_HANDS, device=device, dtype=torch.float32
    )

    # Initialize policy for value backup
    evaluator.initialize_policy_and_beliefs()

    # Compute expected values
    evaluator.compute_expected_values()

    # Check that values are finite
    assert evaluator.latest_values.isfinite().all()


def test_cfr_iteration() -> None:
    """Test that a single CFR iteration runs."""
    device = get_device()
    evaluator, env, cfg = make_sparse_evaluator(device=device)

    root_indices = torch.tensor([0], dtype=torch.long, device=device)
    evaluator.initialize_subgame(env, root_indices)

    # Initialize policy and beliefs
    evaluator.initialize_policy_and_beliefs()

    evaluator.latest_values = torch.randn_like(evaluator.latest_values)
    evaluator.set_leaf_values = lambda t: None  # type: ignore[assignment]

    # Run one CFR iteration
    evaluator.t_sample = evaluator._get_sampling_schedule()
    evaluator.cfr_iteration(t=0)

    # Check that cumulative regrets are updated
    assert evaluator.cumulative_regrets.shape == (
        evaluator.total_nodes,
        NUM_HANDS,
    )
    assert (evaluator.cumulative_regrets >= 0).all()  # CFR+ clamps to non-negative

    # Check that values are updated
    assert evaluator.latest_values.isfinite().all()
    assert evaluator.values_avg.isfinite().all()


def test_evaluate_cfr_basic() -> None:
    """Test that evaluate_cfr runs without errors."""
    device = get_device()
    evaluator, env, cfg = make_sparse_evaluator(device=device)

    root_indices = torch.tensor([0], dtype=torch.long, device=device)
    evaluator.initialize_subgame(env, root_indices)

    evaluator.cfr_iterations = 1
    evaluator.warm_start_iterations = 0

    def _set_zero_leaf_values(t: int, beliefs: torch.Tensor | None = None) -> None:
        evaluator.latest_values.zero_()

    def _noop_sample_leaves(training_mode: bool) -> None:
        return None

    evaluator.set_leaf_values = _set_zero_leaf_values  # type: ignore[assignment]
    evaluator.sample_leaves = _noop_sample_leaves  # type: ignore[assignment]
    evaluator._record_action_mix = lambda: None  # type: ignore[method-assign]
    evaluator._record_cfr_entropy = lambda: None  # type: ignore[method-assign]
    evaluator._record_cumulative_regret = lambda: None  # type: ignore[method-assign]
    evaluator.evaluate_cfr()

    # Check that averages are updated
    assert evaluator.policy_probs_avg.shape == (
        evaluator.total_nodes,
        NUM_HANDS,
    )
    assert evaluator.values_avg.shape == (evaluator.total_nodes, 2, NUM_HANDS)
    assert evaluator.values_avg.isfinite().all()


def test_variable_stack_root_training_data_is_finite() -> None:
    device = get_device()
    env = make_env(2, device=device)
    env.starting_stacks[:] = torch.tensor([[200, 200], [1500, 1500]], device=device)
    env.scale[:] = env.starting_stacks[:, 0].to(env.float_dtype)
    env.stacks[:] = env.starting_stacks
    env.stacks[torch.arange(env.N, device=device), env.button] -= env.sb
    env.stacks[torch.arange(env.N, device=device), 1 - env.button] -= env.bb

    evaluator, _, _ = make_sparse_evaluator(env=env, device=device)
    roots = torch.arange(env.N, device=device)
    evaluator.initialize_subgame(env, roots)
    evaluator.evaluate_cfr(training_mode=False)

    value_batch, pre_value_batch, policy_batch = evaluator.training_data(
        exclude_start=False
    )
    assert torch.isfinite(value_batch.value_targets).all()
    assert torch.isfinite(policy_batch.policy_targets).all()
    if len(pre_value_batch) > 0:
        assert torch.isfinite(pre_value_batch.value_targets).all()


def test_leaf_mask() -> None:
    """Test that leaf mask is correctly set."""
    device = get_device()
    evaluator, env, cfg = make_sparse_evaluator(device=device)

    root_indices = torch.tensor([0], dtype=torch.long, device=device)
    evaluator.initialize_subgame(env, root_indices)

    # Root should not be a leaf (unless there are no legal actions)
    if evaluator.total_nodes > 1:
        assert not evaluator.leaf_mask[0] or evaluator.child_count[0] == 0

    # Check that leaf mask matches done states or new street transitions
    # (This is a sanity check - exact logic depends on tree structure)
    assert evaluator.leaf_mask.shape == (evaluator.total_nodes,)


def test_sparse_sample_leaf_uses_acting_players_sampled_hand() -> None:
    device = torch.device("cpu")
    env = make_env(1, device)
    cfg = make_config(env.default_bet_bins)
    cfg.search.depth = 2
    model = MockModel(
        num_actions=len(cfg.env.bet_bins) + 3,
        device=device,
        dtype=torch.float32,
    )
    evaluator = SparseCFREvaluator(
        model=model,  # type: ignore[arg-type]
        device=device,
        cfg=cfg,
    )
    evaluator.initialize_subgame(env, torch.arange(1, device=device))
    evaluator.initialize_policy_and_beliefs()

    p0_hand = 3
    p1_hand = 17
    evaluator.beliefs.zero_()
    evaluator.beliefs[:, 0, p0_hand] = 1.0
    evaluator.beliefs[:, 1, p1_hand] = 1.0

    root_action = 1
    expected_action = 1
    wrong_action = 2
    root_child = evaluator.child_offsets[0] + 1
    expected_node = evaluator.child_offsets[root_child]
    wrong_node = evaluator.child_offsets[root_child] + 1
    assert evaluator.action_from_parent[root_child] == root_action
    assert evaluator.action_from_parent[expected_node] == expected_action
    assert evaluator.action_from_parent[wrong_node] == wrong_action
    assert evaluator.env.to_act[0] == 0
    assert evaluator.env.to_act[root_child] == 1

    evaluator.policy_probs_sample.zero_()
    evaluator.policy_probs_sample[root_child, p0_hand] = 1.0
    evaluator.policy_probs_sample[expected_node, p1_hand] = 1.0
    evaluator.policy_probs_sample[wrong_node, p0_hand] = 1.0
    expected_beliefs = torch.zeros_like(evaluator.beliefs[expected_node])
    expected_beliefs[0, 23] = 1.0
    expected_beliefs[1, 29] = 1.0
    evaluator.beliefs_sample.zero_()
    evaluator.beliefs_sample[expected_node] = expected_beliefs

    pbs = evaluator.sample_leaves(training_mode=False)

    assert pbs.env.N == 1
    assert_close(pbs.beliefs[0], expected_beliefs)
    assert_close(pbs.env.pot, evaluator.env.pot[expected_node : expected_node + 1])
    assert_close(
        pbs.env.committed,
        evaluator.env.committed[expected_node : expected_node + 1],
    )
    assert not torch.equal(
        pbs.env.committed,
        evaluator.env.committed[wrong_node : wrong_node + 1],
    )


def test_sparse_allin_call_leaf_does_not_create_descendants() -> None:
    device = get_device()
    env = make_env(1, device=device)
    cfg = make_config(env.default_bet_bins)
    cfg.search.depth = 3

    env.street[0] = 1
    env.board_indices[0] = torch.tensor([0, 1, 2, -1, -1], device=device)
    env.board_onehot[0].zero_()
    env.board_onehot[0, 0] = env.card_onehot_cache[0]
    env.board_onehot[0, 1] = env.card_onehot_cache[1]
    env.board_onehot[0, 2] = env.card_onehot_cache[2]
    env.to_act[0] = 0
    env.actions_this_round[0] = 1
    env.stacks[0] = torch.tensor([100, 0], device=device)
    env.starting_stacks[0] = torch.tensor([100, 100], device=device)
    env.scale[0] = 100
    env.committed[0] = torch.tensor([0, 100], device=device)
    env.pot[0] = 100
    env.is_allin[0] = torch.tensor([False, True], device=device)

    evaluator, _, _ = make_sparse_evaluator(env=env, cfg=cfg, device=device)
    evaluator.initialize_subgame(env, torch.tensor([0], device=device))

    assert evaluator.total_nodes == 3
    assert evaluator.action_from_parent.tolist() == [-1, 0, 1]
    assert evaluator.allin_call_indices.tolist() == [2]
    assert evaluator.leaf_mask[2]
    assert evaluator.child_count[2] == 0


def test_parent_child_indices() -> None:
    """Test that parent and child indices are correctly set."""
    device = get_device()
    evaluator, env, cfg = make_sparse_evaluator(device=device)

    root_indices = torch.tensor([0], dtype=torch.long, device=device)
    evaluator.initialize_subgame(env, root_indices)

    # Root nodes should have parent_index = -1
    root_mask = (
        torch.arange(evaluator.total_nodes, device=device) < evaluator.root_nodes
    )
    assert (evaluator.parent_index[root_mask] == -1).all()

    # Non-root nodes should have valid parent indices
    if evaluator.total_nodes > evaluator.root_nodes:
        non_root_mask = ~root_mask
        parent_indices = evaluator.parent_index[non_root_mask]
        assert (parent_indices >= 0).all()
        assert (parent_indices < evaluator.total_nodes).all()
        # Parent indices should be less than child indices
        child_indices = torch.where(non_root_mask)[0]
        assert (parent_indices < child_indices).all()

    # Check action_from_parent
    if evaluator.total_nodes > evaluator.root_nodes:
        actions = evaluator.action_from_parent[~root_mask]
        assert (actions >= 0).all()
        assert (actions < evaluator.num_actions).all()


def test_child_mask() -> None:
    """Test that child mask correctly identifies legal actions."""
    device = get_device()
    evaluator, env, cfg = make_sparse_evaluator(device=device)

    root_indices = torch.tensor([0], dtype=torch.long, device=device)
    evaluator.initialize_subgame(env, root_indices)

    # Child mask should match legal mask for non-leaf nodes
    for i in range(min(evaluator.total_nodes, 100)):  # Check first 100 nodes
        if not evaluator.leaf_mask[i]:
            # Child mask should be subset of legal mask
            assert (evaluator.child_mask[i] <= evaluator.legal_mask[i]).all()


def test_tree_structure_consistency() -> None:
    """Test that tree structure is internally consistent."""
    device = get_device()
    evaluator, env, cfg = make_sparse_evaluator(device=device)

    root_indices = torch.tensor([0], dtype=torch.long, device=device)
    evaluator.initialize_subgame(env, root_indices)

    # Check depth offsets are monotonic
    for i in range(len(evaluator.depth_offsets) - 1):
        assert evaluator.depth_offsets[i] <= evaluator.depth_offsets[i + 1]

    # Check that total_nodes matches final offset
    assert evaluator.total_nodes == evaluator.depth_offsets[-1]

    # Check that child_count matches actual number of children
    non_leaf = ~evaluator.leaf_mask
    for i in range(min(evaluator.total_nodes, 100)):
        child_total = evaluator.child_count[i].item()
        if not non_leaf[i] or child_total == 0:
            continue
        expected_children = evaluator.child_mask[i].sum().item()
        assert child_total == expected_children


def test_street_cutoff_action_masks_by_depth() -> None:
    device = get_device()
    cfg = make_street_cutoff_config()
    evaluator, _, _ = make_sparse_evaluator(cfg=cfg, device=device)

    counts = evaluator.action_masks_by_depth.sum(dim=1).tolist()
    assert counts == [8, 5, 4, 4, 4, 2]
    assert evaluator.action_masks_by_depth[0].tolist() == [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    assert evaluator.action_masks_by_depth[1].tolist() == [
        True,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
    ]
    assert evaluator.action_masks_by_depth[5].tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]


def test_street_cutoff_depth_five_expands_fold_call_only() -> None:
    device = get_device()
    cfg = make_street_cutoff_config()
    env = make_env(1, device=device)
    env.default_bet_bins = cfg.env.bet_bins
    env.num_bet_bins = cfg.model.num_actions
    model = MockModel(num_actions=cfg.model.num_actions, device=device)
    evaluator, _, _ = make_sparse_evaluator(model=model, env=env, cfg=cfg, device=device)

    evaluator.initialize_subgame(env, torch.tensor([0], dtype=torch.long, device=device))

    assert len(evaluator.depth_offsets) >= 8
    depth5_start = evaluator.depth_offsets[5]
    depth5_end = evaluator.depth_offsets[6]
    depth6_start = evaluator.depth_offsets[6]
    depth6_end = evaluator.depth_offsets[7]
    depth6_children = torch.arange(depth6_start, depth6_end, device=device)

    depth5_child_mask = (
        (evaluator.parent_index[depth6_children] >= depth5_start)
        & (evaluator.parent_index[depth6_children] < depth5_end)
    )
    depth5_children = depth6_children[depth5_child_mask]
    assert depth5_children.numel() > 0
    assert (evaluator.action_from_parent[depth5_children] <= 1).all()

    call_children = depth5_children[evaluator.action_from_parent[depth5_children] == 1]
    assert call_children.numel() > 0
    assert evaluator.new_street_mask[call_children[~evaluator.env.done[call_children]]].all()

    model_leaf_mask = evaluator.leaf_mask & ~evaluator.env.done
    model_leaf_mask &= ~evaluator.allin_call_mask
    assert evaluator.new_street_mask[model_leaf_mask].all()


def test_street_cutoff_model_leaves_use_forward_pre() -> None:
    device = get_device()
    cfg = make_street_cutoff_config()
    env = make_env(1, device=device)
    env.default_bet_bins = cfg.env.bet_bins
    env.num_bet_bins = cfg.model.num_actions
    model = PreOnlyMockModel(num_actions=cfg.model.num_actions, device=device)
    evaluator, _, _ = make_sparse_evaluator(model=model, env=env, cfg=cfg, device=device)

    evaluator.initialize_subgame(env, torch.tensor([0], dtype=torch.long, device=device))
    evaluator.set_leaf_values(0)

    assert evaluator.model_indices.numel() > 0
    assert model.forward_pre_calls == 1
