from __future__ import annotations

import torch

from alphaholdem.core.structured_config import (
    Config,
    SearchConfig,
    EnvConfig,
    ModelConfig,
    TrainingConfig,
    CFRType,
    ModelType,
)
from alphaholdem.env.card_utils import NUM_HANDS
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.mlp.mlp_features import MLPFeatures
from alphaholdem.models.model_output import ModelOutput
from alphaholdem.search.sparse_cfr_evaluator import SparseCFREvaluator


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

    def __call__(self, features: MLPFeatures) -> ModelOutput:
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

    def train(self) -> None:
        pass


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

    # Check tensor shapes
    assert evaluator.beliefs.shape == (evaluator.total_nodes, 2, NUM_HANDS)
    assert evaluator.policy_probs.shape == (
        evaluator.total_nodes,
        NUM_HANDS,
        evaluator.num_actions,
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
        evaluator.num_actions,
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
        evaluator.num_actions,
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
        evaluator.num_actions,
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
    evaluator.cfr_iteration(t=0, training_mode=False)

    # Check that cumulative regrets are updated
    assert evaluator.cumulative_regrets.shape == (
        evaluator.total_nodes,
        NUM_HANDS,
        evaluator.num_actions,
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
    evaluator.evaluate_cfr(num_iterations=3)

    # Check that averages are updated
    assert evaluator.policy_probs_avg.shape == (
        evaluator.total_nodes,
        NUM_HANDS,
        evaluator.num_actions,
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
            # Legal mask should be available
            if evaluator.legal_mask is not None:
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
    for i in range(min(evaluator.total_nodes, 100)):  # Check first 100 nodes
        expected_children = evaluator.child_mask[i].sum().item()
        assert evaluator.child_count[i] == expected_children
