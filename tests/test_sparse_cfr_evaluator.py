from __future__ import annotations

import torch
from torch.testing import assert_close

from alphaholdem.core.structured_config import (
    CFRType,
    Config,
    EnvConfig,
    ModelConfig,
    ModelType,
    SearchConfig,
    TrainingConfig,
)
from alphaholdem.env.card_utils import NUM_HANDS
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.mlp.mlp_features import MLPFeatures
from alphaholdem.models.model_output import ModelOutput
from alphaholdem.search.rebel_cfr_evaluator import RebelCFREvaluator
from alphaholdem.search.sparse_cfr_evaluator import SparseCFREvaluator


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
        from alphaholdem.models.mlp.rebel_feature_encoder import RebelFeatureEncoder

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
        from alphaholdem.models.mlp.rebel_feature_encoder import RebelFeatureEncoder

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
    cfg.search.iterations = 10
    cfg.search.warm_start_iterations = 5
    cfg.search.depth = 3
    cfg.search.branching = 4
    cfg.search.dcfr_alpha = 1.5
    cfg.search.dcfr_beta = 0.0
    cfg.search.dcfr_gamma = 2.0
    cfg.search.cfr_type = CFRType.linear
    cfg.search.cfr_avg = True

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


def make_rebel_evaluator(
    env: HUNLTensorEnv,
    cfg: Config,
    model: DeterministicModel,
    device: torch.device,
) -> RebelCFREvaluator:
    return RebelCFREvaluator(
        search_batch_size=env.N,
        env_proto=env,
        model=model,  # type: ignore[arg-type]
        bet_bins=cfg.env.bet_bins,
        max_depth=cfg.search.depth,
        cfr_iterations=cfg.search.iterations,
        device=device,
        float_dtype=env.float_dtype,
        warm_start_iterations=cfg.search.warm_start_iterations,
        cfr_type=cfg.search.cfr_type,
        cfr_avg=cfg.search.cfr_avg,
        sample_epsilon=cfg.search.sample_epsilon,
        dcfr_alpha=cfg.search.dcfr_alpha,
        dcfr_beta=cfg.search.dcfr_beta,
        dcfr_gamma=cfg.search.dcfr_gamma,
    )


def setup_sparse_and_rebel(
    *,
    num_envs: int = 2,
    depth: int = 2,
    seed: int = 123,
) -> tuple[SparseCFREvaluator, RebelCFREvaluator]:
    device = torch.device("cpu")
    env = make_env(num_envs=num_envs, device=device)
    env.rng.manual_seed(seed)
    env.reset()

    cfg = make_config(env.default_bet_bins)
    cfg.search.depth = depth
    cfg.search.iterations = 2
    cfg.search.warm_start_iterations = 0

    model = DeterministicModel(
        num_actions=len(cfg.env.bet_bins) + 3,
        device=device,
        dtype=env.float_dtype,
    )
    root_indices = torch.arange(env.N, device=device)
    initial_beliefs = torch.full(
        (env.N, 2, NUM_HANDS),
        1.0 / NUM_HANDS,
        dtype=env.float_dtype,
        device=device,
    )

    sparse = SparseCFREvaluator(model=model, device=device, cfg=cfg)
    sparse.initialize_subgame(env, root_indices, initial_beliefs=initial_beliefs)

    rebel = make_rebel_evaluator(env, cfg, model, device)
    rebel.initialize_subgame(env, root_indices, initial_beliefs=initial_beliefs)
    return sparse, rebel


def get_rebel_parent_index(rebel: RebelCFREvaluator) -> torch.Tensor:
    parent = torch.full(
        (rebel.total_nodes,),
        -1,
        dtype=torch.long,
        device=rebel.device,
    )
    for depth in range(rebel.max_depth):
        child_start = rebel.depth_offsets[depth + 1]
        child_end = rebel.depth_offsets[depth + 2]
        if child_end <= child_start:
            continue
        parent_start = rebel.depth_offsets[depth]
        span = child_end - child_start
        local = torch.arange(span, device=rebel.device)
        parent_indices = parent_start + (local // rebel.num_actions)
        parent[child_start:child_end] = parent_indices
    return parent


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
    evaluator.regret_weight_sums = torch.ones_like(evaluator.cumulative_regrets)

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

    # Set leaf values
    evaluator.set_leaf_values(0)

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

    # Run a few CFR iterations
    evaluator.cfr_iterations = 3
    evaluator.warm_start_iterations = min(
        evaluator.warm_start_iterations, max(1, evaluator.cfr_iterations - 1)
    )

    def _noop_sample_leaves(training_mode: bool) -> None:
        return None

    evaluator.sample_leaves = _noop_sample_leaves  # type: ignore[assignment]
    evaluator.evaluate_cfr()

    # Check that averages are updated
    assert evaluator.policy_probs_avg.shape == (
        evaluator.total_nodes,
        NUM_HANDS,
    )
    assert evaluator.values_avg.shape == (evaluator.total_nodes, 2, NUM_HANDS)
    assert evaluator.values_avg.isfinite().all()


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


def test_sparse_rebel_tree_state_alignment() -> None:
    """Sparse evaluator tree should mirror Rebel's valid nodes."""
    sparse, rebel = setup_sparse_and_rebel(num_envs=2, depth=2)
    valid_indices = torch.where(rebel.valid_mask)[0]

    assert sparse.total_nodes == valid_indices.numel()

    for attr in ("street", "to_act", "actions_this_round", "pot"):
        sparse_attr = getattr(sparse.env, attr)
        rebel_attr = getattr(rebel.env, attr)[valid_indices]
        assert_close(sparse_attr, rebel_attr)

    assert_close(sparse.env.board_indices, rebel.env.board_indices[valid_indices])
    assert_close(
        sparse.env.last_board_indices, rebel.env.last_board_indices[valid_indices]
    )
    assert_close(sparse.new_street_mask, rebel.new_street_mask[valid_indices])

    parent_dense = get_rebel_parent_index(rebel)
    dense_to_sparse = torch.full(
        (rebel.total_nodes,),
        -1,
        dtype=torch.long,
        device=rebel.device,
    )
    dense_to_sparse[valid_indices] = torch.arange(
        valid_indices.numel(), device=rebel.device
    )
    parent_sparse = torch.where(
        parent_dense >= 0,
        dense_to_sparse[parent_dense],
        torch.full_like(parent_dense, -1),
    )
    rebel_parent_index = parent_sparse[valid_indices].to(sparse.parent_index.device)
    assert_close(sparse.parent_index, rebel_parent_index)

    rebel_child_mask = (rebel.child_mask & (~rebel.leaf_mask)[:, None])[valid_indices]
    non_leaf_sparse = ~sparse.leaf_mask
    non_leaf_rebel = ~rebel.leaf_mask[valid_indices]
    compare_mask = non_leaf_sparse & non_leaf_rebel
    assert_close(
        sparse.child_mask[compare_mask],
        rebel_child_mask[compare_mask],
    )


def test_sparse_rebel_initial_policy_and_beliefs_alignment() -> None:
    """Sparse and Rebel initial policies/beliefs should match exactly."""
    sparse, rebel = setup_sparse_and_rebel(num_envs=2, depth=2)
    sparse.initialize_policy_and_beliefs()
    rebel.initialize_policy_and_beliefs()

    valid_indices = torch.where(rebel.valid_mask)[0]
    assert_close(sparse.beliefs, rebel.beliefs[valid_indices])
    assert_close(sparse.beliefs_avg, rebel.beliefs_avg[valid_indices])
    assert_close(sparse.allowed_hands, rebel.allowed_hands[valid_indices])
    assert_close(sparse.allowed_hands_prob, rebel.allowed_hands_prob[valid_indices])
    assert_close(sparse.self_reach, rebel.self_reach[valid_indices])
    assert_close(sparse.self_reach_avg, rebel.self_reach_avg[valid_indices])

    policy_sparse = sparse._pull_back(sparse.policy_probs)
    policy_rebel = rebel._pull_back(rebel.policy_probs)
    top_sparse = sparse.depth_offsets[-2]
    top_rebel = rebel.depth_offsets[-2]
    mask_sparse = ~sparse.leaf_mask[:top_sparse]
    mask_rebel = rebel.valid_mask[:top_rebel] & ~rebel.leaf_mask[:top_rebel]
    assert_close(
        policy_sparse[:top_sparse][mask_sparse].permute(0, 2, 1),
        policy_rebel[:top_rebel][mask_rebel].permute(0, 2, 1),
    )
