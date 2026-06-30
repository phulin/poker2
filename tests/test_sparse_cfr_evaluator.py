from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch.testing import assert_close

from p2.core.structured_config import (
    CFRType,
    Config,
    EnvConfig,
    ModelConfig,
    ModelScope,
    ModelType,
    SearchConfig,
    TrainingConfig,
)
from p2.env.card_utils import (
    NUM_HANDS,
    PREFLOP_HANDS,
    preflop_class_multiplicity_tensor,
)
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.env.pbs_env import PBSEnv
from p2.models.mlp.mlp_features import MLPFeatures
from p2.models.model_output import ModelOutput
from p2.rl.target_provenance import (
    TARGET_SOURCE_CFR_BACKUP,
    TARGET_SOURCE_CHANCE_EXPECTATION,
    TARGET_SOURCE_CLOSING_NET,
    TARGET_SOURCE_EXACT_TERMINAL,
)
from p2.search.preflop_sparse_cfr_evaluator import PreflopSparseCFREvaluator
from p2.search.sparse_cfr_evaluator import SparseCFREvaluator


STREET_CUTOFF_SCHEDULE = [
    [0.25, 0.5, 0.75, 1.0, 1.5],
    [0.5, 1.0],
    [1.0],
    [1.0],
    [1.0],
    [],
]


def _scope_cfg(scope: ModelScope = ModelScope.mixed_street) -> Config:
    cfg = Config(device="cpu")
    cfg.search.model_scope = scope
    return cfg


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

    hand_dim = NUM_HANDS

    def __init__(
        self,
        num_actions: int = 5,
        num_players: int = 2,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_actions = num_actions
        self.num_players = num_players
        self.device = device
        self.dtype = dtype
        self.enforce_zero_sum = True
        self.hidden_dim = 1

    def create_feature_encoder(self, env, device=None, dtype=None):
        from p2.models.mlp.rebel_feature_encoder import RebelFeatureEncoder

        return RebelFeatureEncoder(env=env, device=device, dtype=dtype)

    def __call__(
        self,
        features: MLPFeatures,
        include_policy: bool = True,
        include_value: bool = True,
        **kwargs,
    ) -> ModelOutput:
        del include_value, kwargs
        batch = len(features)
        device = features.context.device
        dtype = features.context.dtype

        # Default uniform logits
        logits = torch.zeros(
            batch, NUM_HANDS, self.num_actions, device=device, dtype=dtype
        )
        # Default zero hand values
        hand_values = torch.zeros(
            batch,
            self.num_players,
            NUM_HANDS,
            device=device,
            dtype=dtype,
        )

        return ModelOutput(
            policy_logits=logits,
            value=torch.zeros(batch, device=device, dtype=dtype),
            hand_values=hand_values,
        )

    def eval(self) -> None:
        pass


class CompactPreflopMockModel:
    """Compact 169-hand policy/value mock for preflop evaluator tests."""

    hand_dim = PREFLOP_HANDS

    def __init__(
        self,
        num_actions: int,
        num_players: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_actions = num_actions
        self.num_players = num_players
        self.device = device
        self.dtype = dtype
        self.enforce_zero_sum = False
        self.hidden_dim = 1
        hand_values = torch.linspace(
            -0.2, 0.2, steps=PREFLOP_HANDS, device=device, dtype=dtype
        )
        self.base_values = torch.stack(
            [hand_values + 0.01 * player for player in range(num_players)], dim=0
        )

    def create_feature_encoder(self, env, device=None, dtype=None):
        from p2.models.mlp.better_feature_encoder import (
            BetterPreflopPolicyFeatureEncoder,
        )

        return BetterPreflopPolicyFeatureEncoder(env=env, device=device, dtype=dtype)

    def __call__(
        self,
        features: MLPFeatures,
        include_policy: bool = True,
        include_value: bool = True,
        **kwargs,
    ) -> ModelOutput:
        del kwargs
        batch = len(features)
        device = features.context.device
        dtype = features.context.dtype
        logits = None
        if include_policy:
            action_logits = torch.linspace(
                -0.25, 0.25, steps=self.num_actions, device=device, dtype=dtype
            )
            logits = action_logits.view(1, 1, -1).expand(
                batch, PREFLOP_HANDS, self.num_actions
            )
        values = None
        if include_value:
            values = self.base_values.to(device=device, dtype=dtype).expand(
                batch, self.num_players, PREFLOP_HANDS
            )
        return ModelOutput(
            policy_logits=logits,
            value=None if values is None else values.mean(dim=-1),
            hand_values=values,
        )

    def eval(self) -> None:
        pass


class CompactRecordingValueModel:
    hand_dim = PREFLOP_HANDS

    def __init__(self, value: float, *, num_players: int, device: torch.device) -> None:
        self.value = float(value)
        self.num_players = num_players
        self.device = device
        self.enforce_zero_sum = False
        self.call_contexts: list[torch.Tensor] = []
        self.call_beliefs: list[torch.Tensor] = []

    def __call__(
        self,
        features: MLPFeatures,
        include_policy: bool = False,
        **kwargs,
    ) -> ModelOutput:
        del include_policy, kwargs
        self.call_contexts.append(features.context.detach().clone())
        self.call_beliefs.append(
            features.beliefs.view(len(features), self.num_players, PREFLOP_HANDS)
            .detach()
            .clone()
        )
        hand_values = torch.full(
            (len(features), self.num_players, PREFLOP_HANDS),
            self.value,
            device=features.context.device,
            dtype=features.context.dtype,
        )
        return ModelOutput(
            policy_logits=None,
            value=hand_values.mean(dim=-1),
            hand_values=hand_values,
        )


class PreOnlyMockModel(MockModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.forward_pre_calls = 0

    def forward_pre(self, features: MLPFeatures, **kwargs) -> torch.Tensor:
        del kwargs
        self.forward_pre_calls += 1
        batch = len(features)
        return torch.zeros(
            batch,
            2,
            NUM_HANDS,
            device=features.context.device,
            dtype=features.context.dtype,
        )


class ConstantValueModel:
    hand_dim = NUM_HANDS

    def __init__(
        self,
        value: float,
        *,
        num_players: int = 2,
        output_dtype: torch.dtype | None = None,
    ) -> None:
        self.value = float(value)
        self.num_players = num_players
        self.output_dtype = output_dtype
        self.enforce_zero_sum = False
        self.call_contexts: list[torch.Tensor] = []
        self.forward_pre_contexts: list[torch.Tensor] = []

    def __call__(
        self,
        features: MLPFeatures,
        include_policy: bool = False,
        **kwargs,
    ) -> ModelOutput:
        del include_policy, kwargs
        self.call_contexts.append(features.context.detach().clone())
        batch = len(features)
        hand_values = torch.full(
            (batch, self.num_players, NUM_HANDS),
            self.value,
            device=features.context.device,
            dtype=self.output_dtype or features.context.dtype,
        )
        return ModelOutput(
            policy_logits=None,
            value=torch.zeros(batch, device=features.context.device),
            hand_values=hand_values,
        )

    def forward_pre(self, features: MLPFeatures, **kwargs) -> torch.Tensor:
        del kwargs
        self.forward_pre_contexts.append(features.context.detach().clone())
        return torch.full(
            (len(features), self.num_players, NUM_HANDS),
            self.value,
            device=features.context.device,
            dtype=self.output_dtype or features.context.dtype,
        )


class OffsetFeatureEncoder:
    def __init__(self, offset: float) -> None:
        self.offset = float(offset)

    def encode(
        self,
        beliefs: torch.Tensor,
        pre_chance_node: torch.Tensor | bool | None = None,
        indices: torch.Tensor | None = None,
    ) -> MLPFeatures:
        assert indices is not None
        selected = beliefs[indices]
        batch = indices.numel()
        return MLPFeatures(
            context=indices.to(torch.float32).view(batch, 1) + self.offset,
            street=torch.zeros(batch, dtype=torch.long, device=indices.device),
            to_act=torch.zeros(batch, dtype=torch.long, device=indices.device),
            board=torch.full((batch, 5), -1, dtype=torch.long, device=indices.device),
            beliefs=selected.reshape(batch, -1),
            hand_dim=selected.shape[-1],
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
        self,
        features: MLPFeatures,
        include_policy: bool = True,
        **kwargs,
    ) -> ModelOutput:
        del include_policy, kwargs
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


def test_model_leaf_values_use_closing_model_when_model_leaves_close_street() -> None:
    evaluator = object.__new__(SparseCFREvaluator)
    current_model = ConstantValueModel(1.0)
    closing_model = ConstantValueModel(2.0)
    evaluator.model = current_model
    evaluator.value_model = current_model
    evaluator.closing_leaf_value_model = closing_model
    evaluator.closing_leaf_value_encoder = None
    evaluator.cfg = _scope_cfg(ModelScope.end_of_street)
    evaluator.float_dtype = torch.float32
    evaluator.num_players = 2
    evaluator.cfr_avg = False
    evaluator.latest_values = torch.zeros(3, 2, NUM_HANDS)
    evaluator.model_indices = torch.tensor([0, 1, 2], dtype=torch.long)
    evaluator.new_street_mask = torch.tensor([True, True, True])
    evaluator.action_schedule = None
    evaluator.last_model_values = None

    features = MLPFeatures(
        context=torch.arange(3, dtype=torch.float32).view(3, 1),
        street=torch.zeros(3, dtype=torch.long),
        to_act=torch.zeros(3, dtype=torch.long),
        board=torch.full((3, 5), -1, dtype=torch.long),
        beliefs=torch.zeros(3, 2 * NUM_HANDS),
    )
    beliefs = torch.zeros(3, 2, NUM_HANDS)

    new_values, last_model_values = evaluator._set_model_values_impl(
        0, beliefs, features
    )

    torch.testing.assert_close(new_values[0], torch.full((2, NUM_HANDS), 2.0))
    torch.testing.assert_close(new_values[1], torch.full((2, NUM_HANDS), 2.0))
    torch.testing.assert_close(new_values[2], torch.full((2, NUM_HANDS), 2.0))
    torch.testing.assert_close(
        last_model_values[:, 0, 0], torch.tensor([2.0, 2.0, 2.0])
    )
    assert len(current_model.call_contexts) == 0
    assert len(current_model.forward_pre_contexts) == 0
    torch.testing.assert_close(
        closing_model.call_contexts[0].flatten(), torch.tensor([0.0, 1.0, 2.0])
    )
    assert len(closing_model.forward_pre_contexts) == 0


def test_model_leaf_values_mixed_scope_splits_cutoff_and_closing_leaves() -> None:
    evaluator = object.__new__(SparseCFREvaluator)
    current_model = ConstantValueModel(1.0)
    closing_model = ConstantValueModel(2.0)
    evaluator.model = current_model
    evaluator.value_model = current_model
    evaluator.closing_leaf_value_model = closing_model
    evaluator.closing_leaf_value_encoder = OffsetFeatureEncoder(100.0)
    evaluator.cfg = _scope_cfg()
    evaluator.float_dtype = torch.float32
    evaluator.num_players = 2
    evaluator.cfr_avg = False
    evaluator.latest_values = torch.zeros(3, 2, NUM_HANDS)
    evaluator.beliefs = torch.zeros(3, 2, NUM_HANDS)
    evaluator.model_indices = torch.tensor([0, 1, 2], dtype=torch.long)
    evaluator.new_street_mask = torch.tensor([False, True, False])
    evaluator.action_schedule = None
    evaluator.last_model_values = None

    features = MLPFeatures(
        context=torch.arange(3, dtype=torch.float32).view(3, 1),
        street=torch.zeros(3, dtype=torch.long),
        to_act=torch.zeros(3, dtype=torch.long),
        board=torch.full((3, 5), -1, dtype=torch.long),
        beliefs=torch.zeros(3, 2 * NUM_HANDS),
    )
    beliefs = torch.zeros(3, 2, NUM_HANDS)

    new_values, last_model_values = evaluator._set_model_values_impl(
        0, beliefs, features
    )

    torch.testing.assert_close(new_values[0], torch.ones(2, NUM_HANDS))
    torch.testing.assert_close(new_values[1], torch.full((2, NUM_HANDS), 2.0))
    torch.testing.assert_close(new_values[2], torch.ones(2, NUM_HANDS))
    torch.testing.assert_close(
        last_model_values[:, 0, 0], torch.tensor([1.0, 2.0, 1.0])
    )
    torch.testing.assert_close(
        current_model.call_contexts[0].flatten(), torch.tensor([0.0, 2.0])
    )
    torch.testing.assert_close(
        closing_model.call_contexts[0].flatten(), torch.tensor([101.0])
    )
    assert len(closing_model.forward_pre_contexts) == 0


def test_model_leaf_values_project_multiway_heads_up_closing_leaf() -> None:
    evaluator = object.__new__(SparseCFREvaluator)
    current_model = ConstantValueModel(1.0, num_players=4)
    closing_model = ConstantValueModel(2.0, num_players=2)
    evaluator.model = current_model
    evaluator.value_model = current_model
    evaluator.closing_leaf_value_model = closing_model
    evaluator.closing_leaf_value_encoder = None
    evaluator.cfg = _scope_cfg()
    evaluator.float_dtype = torch.float32
    evaluator.num_players = 4
    evaluator.hand_dim = PREFLOP_HANDS
    evaluator.cfr_avg = False
    evaluator.latest_values = torch.zeros(1, 4, PREFLOP_HANDS)
    evaluator.model_indices = torch.tensor([0], dtype=torch.long)
    evaluator.new_street_mask = torch.tensor([True])
    evaluator.action_schedule = None
    evaluator.last_model_values = None
    evaluator.env = PBSEnv(
        num_envs=1,
        num_players=4,
        mean_stack=1000,
        sb=5,
        bb=10,
        device="cpu",
    )
    evaluator.env.reset(force_button=torch.zeros(1, dtype=torch.long))
    evaluator.env.has_folded[0] = torch.tensor([True, False, True, False])
    evaluator.env.stacks[0] = torch.tensor([900, 1000, 800, 1000])
    evaluator.env.starting_stacks[0] = torch.tensor([1000, 1000, 1000, 1000])
    evaluator.env.scale[0] = 1000

    features = MLPFeatures(
        context=torch.zeros(1, 1),
        street=torch.zeros(1, dtype=torch.long),
        to_act=torch.zeros(1, dtype=torch.long),
        board=torch.full((1, 5), -1, dtype=torch.long),
        beliefs=torch.full((1, 4 * PREFLOP_HANDS), 1.0 / PREFLOP_HANDS),
        hand_dim=PREFLOP_HANDS,
    )
    beliefs = features.beliefs.view(1, 4, PREFLOP_HANDS)

    new_values, _ = evaluator._set_model_values_impl(0, beliefs, features)

    torch.testing.assert_close(new_values[0, 0], torch.full((PREFLOP_HANDS,), -0.1))
    torch.testing.assert_close(new_values[0, 1], torch.full((PREFLOP_HANDS,), 0.15))
    torch.testing.assert_close(new_values[0, 2], torch.full((PREFLOP_HANDS,), -0.2))
    torch.testing.assert_close(new_values[0, 3], torch.full((PREFLOP_HANDS,), 0.15))
    assert closing_model.call_contexts


def test_fused_model_leaf_values_mixed_scope_splits_cutoff_and_closing_leaves() -> None:
    from p2.search.fused_sparse_cfr_evaluator import FusedSparseCFREvaluator

    evaluator = object.__new__(FusedSparseCFREvaluator)
    current_model = ConstantValueModel(1.0, output_dtype=torch.bfloat16)
    closing_model = ConstantValueModel(2.0, output_dtype=torch.bfloat16)
    evaluator.model = current_model
    evaluator.value_model = current_model
    evaluator.closing_leaf_value_model = closing_model
    evaluator.closing_leaf_value_encoder = OffsetFeatureEncoder(100.0)
    evaluator.cfg = _scope_cfg()
    evaluator.float_dtype = torch.float32
    evaluator.num_players = 2
    evaluator.latest_values = torch.zeros(3, 2, NUM_HANDS)
    evaluator.beliefs = torch.zeros(3, 2, NUM_HANDS)
    evaluator.model_indices = torch.tensor([0, 1, 2], dtype=torch.long)
    evaluator.new_street_mask = torch.tensor([False, True, False])
    evaluator.action_schedule = None

    features = MLPFeatures(
        context=torch.arange(3, dtype=torch.float32).view(3, 1),
        street=torch.zeros(3, dtype=torch.long),
        to_act=torch.zeros(3, dtype=torch.long),
        board=torch.full((3, 5), -1, dtype=torch.long),
        beliefs=torch.zeros(3, 2 * NUM_HANDS),
    )

    hand_values, model_applied_zero_sum = (
        evaluator._model_leaf_values_for_fused_writeback(features)
    )

    torch.testing.assert_close(hand_values[0], torch.ones(2, NUM_HANDS))
    torch.testing.assert_close(hand_values[1], torch.full((2, NUM_HANDS), 2.0))
    torch.testing.assert_close(hand_values[2], torch.ones(2, NUM_HANDS))
    assert hand_values.dtype == evaluator.latest_values.dtype
    assert not model_applied_zero_sum
    torch.testing.assert_close(
        current_model.call_contexts[0],
        torch.tensor([[0.0], [2.0]]),
    )
    torch.testing.assert_close(
        closing_model.call_contexts[0],
        torch.tensor([[101.0]]),
    )


def test_fused_model_leaf_values_project_multiway_heads_up_closing_leaf() -> None:
    from p2.search.fused_sparse_cfr_evaluator import FusedSparseCFREvaluator

    evaluator = object.__new__(FusedSparseCFREvaluator)
    current_model = ConstantValueModel(1.0, num_players=4)
    closing_model = ConstantValueModel(2.0, num_players=2)
    evaluator.model = current_model
    evaluator.value_model = current_model
    evaluator.closing_leaf_value_model = closing_model
    evaluator.closing_leaf_value_encoder = None
    evaluator.cfg = _scope_cfg()
    evaluator.float_dtype = torch.float32
    evaluator.num_players = 4
    evaluator.hand_dim = PREFLOP_HANDS
    evaluator.latest_values = torch.zeros(1, 4, PREFLOP_HANDS)
    evaluator.model_indices = torch.tensor([0], dtype=torch.long)
    evaluator.new_street_mask = torch.tensor([True])
    evaluator.action_schedule = None
    evaluator.env = PBSEnv(
        num_envs=1,
        num_players=4,
        mean_stack=1000,
        sb=5,
        bb=10,
        device="cpu",
    )
    evaluator.env.reset(force_button=torch.zeros(1, dtype=torch.long))
    evaluator.env.has_folded[0] = torch.tensor([True, False, True, False])
    evaluator.env.stacks[0] = torch.tensor([900, 1000, 800, 1000])
    evaluator.env.starting_stacks[0] = torch.tensor([1000, 1000, 1000, 1000])
    evaluator.env.scale[0] = 1000

    features = MLPFeatures(
        context=torch.zeros(1, 1),
        street=torch.zeros(1, dtype=torch.long),
        to_act=torch.zeros(1, dtype=torch.long),
        board=torch.full((1, 5), -1, dtype=torch.long),
        beliefs=torch.full((1, 4 * PREFLOP_HANDS), 1.0 / PREFLOP_HANDS),
        hand_dim=PREFLOP_HANDS,
    )

    hand_values, _ = evaluator._model_leaf_values_for_fused_writeback(features)

    torch.testing.assert_close(hand_values[0, 0], torch.full((PREFLOP_HANDS,), -0.1))
    torch.testing.assert_close(hand_values[0, 1], torch.full((PREFLOP_HANDS,), 2.0))
    torch.testing.assert_close(hand_values[0, 2], torch.full((PREFLOP_HANDS,), -0.2))
    torch.testing.assert_close(hand_values[0, 3], torch.full((PREFLOP_HANDS,), 2.0))
    assert closing_model.call_contexts


def test_fused_model_leaf_values_use_closing_model_when_model_leaves_close_street() -> (
    None
):
    from p2.search.fused_sparse_cfr_evaluator import FusedSparseCFREvaluator

    evaluator = object.__new__(FusedSparseCFREvaluator)
    current_model = ConstantValueModel(1.0)
    closing_model = ConstantValueModel(2.0)
    evaluator.model = current_model
    evaluator.value_model = current_model
    evaluator.closing_leaf_value_model = closing_model
    evaluator.closing_leaf_value_encoder = None
    evaluator.cfg = _scope_cfg(ModelScope.end_of_street)
    evaluator.float_dtype = torch.float32
    evaluator.num_players = 2
    evaluator.latest_values = torch.zeros(3, 2, NUM_HANDS)
    evaluator.model_indices = torch.tensor([0, 1, 2], dtype=torch.long)
    evaluator.new_street_mask = torch.tensor([True, True, True])
    evaluator.action_schedule = None

    features = MLPFeatures(
        context=torch.arange(3, dtype=torch.float32).view(3, 1),
        street=torch.zeros(3, dtype=torch.long),
        to_act=torch.zeros(3, dtype=torch.long),
        board=torch.full((3, 5), -1, dtype=torch.long),
        beliefs=torch.zeros(3, 2 * NUM_HANDS),
    )

    hand_values, model_applied_zero_sum = (
        evaluator._model_leaf_values_for_fused_writeback(features)
    )

    torch.testing.assert_close(hand_values[0], torch.full((2, NUM_HANDS), 2.0))
    torch.testing.assert_close(hand_values[1], torch.full((2, NUM_HANDS), 2.0))
    torch.testing.assert_close(hand_values[2], torch.full((2, NUM_HANDS), 2.0))
    assert not model_applied_zero_sum
    assert len(current_model.call_contexts) == 0
    assert len(current_model.forward_pre_contexts) == 0
    assert len(closing_model.call_contexts) == 1
    assert len(closing_model.forward_pre_contexts) == 0
    torch.testing.assert_close(
        closing_model.call_contexts[0],
        torch.tensor([[0.0], [1.0], [2.0]]),
    )


def test_mid_street_value_roots_expected_only_in_mixed_continuation_mode() -> None:
    evaluator = object.__new__(SparseCFREvaluator)
    evaluator.cfg = _scope_cfg()
    evaluator.cfg.search.continuation_value_target_sampling = True
    assert evaluator._mid_street_value_roots_are_expected()

    evaluator.cfg.search.model_scope = ModelScope.single_street
    assert not evaluator._mid_street_value_roots_are_expected()

    evaluator.cfg.search.model_scope = ModelScope.mixed_street
    evaluator.cfg.search.continuation_value_target_sampling = False
    assert not evaluator._mid_street_value_roots_are_expected()


def test_root_leaf_target_source_counts_split_terminal_and_closing_leaves() -> None:
    evaluator = object.__new__(SparseCFREvaluator)
    evaluator.device = torch.device("cpu")
    evaluator.total_nodes = 5
    evaluator.depth_offsets = [0, 2, 5]
    evaluator.parent_index = torch.tensor([-1, -1, 0, 0, 1], dtype=torch.long)
    evaluator.leaf_mask = torch.tensor([False, False, True, True, True])
    evaluator.valid_mask = torch.ones(5, dtype=torch.bool)
    evaluator.new_street_mask = torch.tensor([False, False, True, False, False])
    evaluator.env = SimpleNamespace(
        done=torch.tensor([False, False, False, True, False])
    )
    evaluator.allin_call_indices = torch.tensor([4], dtype=torch.long)

    counts = evaluator._root_leaf_target_source_counts(num_roots=2)

    torch.testing.assert_close(
        counts["leaf_total_count"], torch.tensor([2, 1], dtype=torch.long)
    )
    torch.testing.assert_close(
        counts[f"leaf_target_source_{TARGET_SOURCE_CLOSING_NET}_count"],
        torch.tensor([1, 0], dtype=torch.long),
    )
    torch.testing.assert_close(
        counts[f"leaf_target_source_{TARGET_SOURCE_EXACT_TERMINAL}_count"],
        torch.tensor([1, 1], dtype=torch.long),
    )
    torch.testing.assert_close(
        counts[f"leaf_target_source_{TARGET_SOURCE_CFR_BACKUP}_count"],
        torch.tensor([0, 0], dtype=torch.long),
    )


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


def test_initialize_pbs_multiway_subgame() -> None:
    device = get_device()
    num_players = 3
    cfg = make_config([0.5])
    cfg.env.num_players = num_players
    cfg.search.allin_call_terminal_abstraction = False
    env = PBSEnv(
        num_envs=2,
        num_players=num_players,
        starting_stack=1000,
        sb=5,
        bb=10,
        default_bet_bins=cfg.env.bet_bins,
        device=device,
    )
    env.reset(
        force_button=torch.zeros(2, dtype=torch.long, device=device),
        force_deck=torch.tensor(
            [[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
            dtype=torch.long,
            device=device,
        ),
    )
    model = MockModel(
        num_actions=len(cfg.env.bet_bins) + 3,
        num_players=num_players,
        device=device,
    )
    evaluator = SparseCFREvaluator(
        model=model,  # type: ignore[arg-type]
        device=device,
        cfg=cfg,
    )
    beliefs = torch.full((2, num_players, NUM_HANDS), 1.0 / NUM_HANDS)

    evaluator.initialize_subgame(env, torch.arange(2, device=device), beliefs)

    assert evaluator.num_players == num_players
    assert isinstance(evaluator.env, PBSEnv)
    assert evaluator.beliefs.shape == (evaluator.total_nodes, num_players, NUM_HANDS)
    assert evaluator.latest_values.shape == evaluator.beliefs.shape


def test_evaluate_pbs_multiway_cfr_smoke() -> None:
    device = get_device()
    num_players = 3
    cfg = make_config([0.5])
    cfg.env.num_players = num_players
    cfg.search.allin_call_terminal_abstraction = False
    cfg.search.iterations = 1
    env = PBSEnv(
        num_envs=2,
        num_players=num_players,
        starting_stack=1000,
        sb=5,
        bb=10,
        default_bet_bins=cfg.env.bet_bins,
        device=device,
    )
    env.reset(
        force_button=torch.zeros(2, dtype=torch.long, device=device),
        force_deck=torch.tensor(
            [[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
            dtype=torch.long,
            device=device,
        ),
    )
    model = MockModel(
        num_actions=len(cfg.env.bet_bins) + 3,
        num_players=num_players,
        device=device,
    )
    evaluator = SparseCFREvaluator(
        model=model,  # type: ignore[arg-type]
        device=device,
        cfg=cfg,
    )
    beliefs = torch.full((1, num_players, NUM_HANDS), 1.0 / NUM_HANDS)
    evaluator.initialize_subgame(env, torch.arange(1, device=device), beliefs)

    next_pbs = evaluator.evaluate_cfr()
    value_batch, augmented_value_batch, policy_batch = evaluator.training_data()

    assert isinstance(next_pbs.env, PBSEnv)
    assert next_pbs.beliefs.shape[1:] == (num_players, NUM_HANDS)
    assert value_batch.features.beliefs.shape[1] == num_players * NUM_HANDS
    assert augmented_value_batch.features.beliefs.shape[1] == num_players * NUM_HANDS
    assert policy_batch.features.beliefs.shape[1] == num_players * NUM_HANDS


def test_preflop_sparse_evaluator_enforces_pbs_preflop_roots() -> None:
    device = get_device()
    num_players = 3
    cfg = make_config([0.5])
    cfg.env.num_players = num_players
    cfg.search.allin_call_terminal_abstraction = False
    env = PBSEnv(
        num_envs=2,
        num_players=num_players,
        mean_stack=1000,
        sb=5,
        bb=10,
        default_bet_bins=cfg.env.bet_bins,
        device=device,
    )
    env.reset(
        force_button=torch.zeros(2, dtype=torch.long, device=device),
        force_deck=torch.tensor(
            [[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
            dtype=torch.long,
            device=device,
        ),
    )
    model = CompactPreflopMockModel(
        num_actions=len(cfg.env.bet_bins) + 3,
        num_players=num_players,
        device=device,
    )
    evaluator = PreflopSparseCFREvaluator(
        model=model,  # type: ignore[arg-type]
        device=device,
        cfg=cfg,
    )
    beliefs = preflop_class_multiplicity_tensor(device=device).to(torch.float32)
    beliefs = (beliefs / beliefs.sum()).expand(1, num_players, PREFLOP_HANDS).clone()

    evaluator.initialize_subgame(env, torch.arange(1, device=device), beliefs)

    assert evaluator._continuation_value_target_sampling_enabled()
    assert evaluator._continuation_value_target_streets() == (0,)

    env.street[0] = 1
    with pytest.raises(ValueError, match="street-0"):
        evaluator.initialize_subgame(env, torch.arange(1, device=device), beliefs)

    hunl_env = make_env(1, device=device)
    with pytest.raises(TypeError, match="PBSEnv"):
        evaluator.initialize_subgame(hunl_env, torch.arange(1, device=device))


def test_preflop_sparse_evaluator_uses_compact_169_tensors() -> None:
    device = get_device()
    num_players = 3
    cfg = make_config([0.5])
    cfg.env.num_players = num_players
    cfg.search.allin_call_terminal_abstraction = False
    env = PBSEnv(
        num_envs=1,
        num_players=num_players,
        mean_stack=1000,
        sb=5,
        bb=10,
        default_bet_bins=cfg.env.bet_bins,
        device=device,
    )
    env.reset(
        force_button=torch.zeros(1, dtype=torch.long, device=device),
        force_deck=torch.tensor(
            [[10, 11, 12, 13, 14]], dtype=torch.long, device=device
        ),
    )
    model = CompactPreflopMockModel(
        num_actions=len(cfg.env.bet_bins) + 3,
        num_players=num_players,
        device=device,
    )
    evaluator = PreflopSparseCFREvaluator(
        model=model,  # type: ignore[arg-type]
        device=device,
        cfg=cfg,
    )

    evaluator.initialize_subgame(env, torch.arange(1, device=device))

    prior = preflop_class_multiplicity_tensor(device=device).to(torch.float32)
    prior = prior / prior.sum()
    torch.testing.assert_close(evaluator.beliefs[0], prior.expand(num_players, -1))
    for tensor in (
        evaluator.beliefs,
        evaluator.beliefs_avg,
        evaluator.self_reach,
        evaluator.latest_values,
        evaluator.values_avg,
        evaluator.policy_probs,
        evaluator.policy_probs_avg,
        evaluator.cumulative_regrets,
        evaluator.average_policy_numerator,
        evaluator.average_policy_denominator,
    ):
        assert tensor.shape[-1] == PREFLOP_HANDS


def test_preflop_sparse_new_street_leaves_are_heads_up() -> None:
    device = get_device()
    num_players = 4
    cfg = make_config([0.5])
    cfg.env.num_players = num_players
    cfg.search.depth = num_players
    cfg.search.allin_call_terminal_abstraction = False
    env = PBSEnv(
        num_envs=1,
        num_players=num_players,
        mean_stack=1000,
        sb=5,
        bb=10,
        default_bet_bins=cfg.env.bet_bins,
        device=device,
    )
    env.reset(
        force_button=torch.zeros(1, dtype=torch.long, device=device),
        force_deck=torch.tensor(
            [[10, 11, 12, 13, 14]], dtype=torch.long, device=device
        ),
    )
    model = CompactPreflopMockModel(
        num_actions=len(cfg.env.bet_bins) + 3,
        num_players=num_players,
        device=device,
    )
    evaluator = PreflopSparseCFREvaluator(
        model=model,  # type: ignore[arg-type]
        device=device,
        cfg=cfg,
    )

    evaluator.initialize_subgame(env, torch.arange(1, device=device))

    assert env.force_heads_up_preflop_flop is True
    assert evaluator.new_street_mask.any()
    live_counts = (~evaluator.env.has_folded[evaluator.new_street_mask]).sum(dim=1)
    assert int(live_counts.max()) <= 2


def test_preflop_sparse_evaluator_runs_compact_iteration_and_policy_batch() -> None:
    device = get_device()
    num_players = 3
    cfg = make_config([0.5])
    cfg.env.num_players = num_players
    cfg.search.allin_call_terminal_abstraction = False
    cfg.search.iterations = 1
    env = PBSEnv(
        num_envs=1,
        num_players=num_players,
        mean_stack=1000,
        sb=5,
        bb=10,
        default_bet_bins=cfg.env.bet_bins,
        device=device,
    )
    env.reset(
        force_button=torch.zeros(1, dtype=torch.long, device=device),
        force_deck=torch.tensor(
            [[10, 11, 12, 13, 14]], dtype=torch.long, device=device
        ),
    )
    model = CompactPreflopMockModel(
        num_actions=len(cfg.env.bet_bins) + 3,
        num_players=num_players,
        device=device,
    )
    evaluator = PreflopSparseCFREvaluator(
        model=model,  # type: ignore[arg-type]
        device=device,
        cfg=cfg,
    )
    beliefs = preflop_class_multiplicity_tensor(device=device).to(torch.float32)
    beliefs = (beliefs / beliefs.sum()).expand(1, num_players, PREFLOP_HANDS).clone()

    evaluator.initialize_subgame(env, torch.arange(1, device=device), beliefs)
    next_pbs = evaluator.evaluate_cfr()
    value_batch, pre_value_batch, policy_batch = evaluator.training_data(
        include_pre_chance_value_batch=False
    )

    assert next_pbs.beliefs.shape[1:] == (num_players, PREFLOP_HANDS)
    assert value_batch.features.beliefs.shape[1] == num_players * PREFLOP_HANDS
    assert pre_value_batch is None
    assert policy_batch is not None
    assert policy_batch.policy_targets.shape[1:] == (
        PREFLOP_HANDS,
        evaluator.num_actions,
    )
    assert policy_batch.features.beliefs.shape[1] == num_players * PREFLOP_HANDS
    assert "has_folded" in value_batch.statistics
    assert "has_folded" in policy_batch.statistics


def _assert_rebel_batches_close(actual, expected, *, atol: float, rtol: float) -> None:
    if actual is None or expected is None:
        assert actual is expected
        return
    assert len(actual) == len(expected)
    torch.testing.assert_close(actual.features.context, expected.features.context)
    torch.testing.assert_close(actual.features.street, expected.features.street)
    torch.testing.assert_close(actual.features.to_act, expected.features.to_act)
    torch.testing.assert_close(actual.features.board, expected.features.board)
    torch.testing.assert_close(
        actual.features.beliefs,
        expected.features.beliefs,
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(actual.legal_masks, expected.legal_masks)
    if actual.value_targets is None or expected.value_targets is None:
        assert actual.value_targets is expected.value_targets
    else:
        torch.testing.assert_close(
            actual.value_targets,
            expected.value_targets,
            atol=atol,
            rtol=rtol,
        )
    if actual.policy_targets is None or expected.policy_targets is None:
        assert actual.policy_targets is expected.policy_targets
    else:
        torch.testing.assert_close(
            actual.policy_targets,
            expected.policy_targets,
            atol=atol,
            rtol=rtol,
        )
    assert actual.statistics.keys() == expected.statistics.keys()
    for key, actual_value in actual.statistics.items():
        expected_value = expected.statistics[key]
        if actual_value.dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            torch.testing.assert_close(
                actual_value,
                expected_value,
                atol=atol,
                rtol=rtol,
            )
        else:
            torch.testing.assert_close(actual_value, expected_value)


def test_fused_preflop_sparse_evaluator_matches_non_fused_training_targets() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA to exercise the fused preflop evaluator")
    pytest.importorskip("triton")
    from p2.search.fused_preflop_sparse_cfr_evaluator import (
        FusedPreflopSparseCFREvaluator,
    )

    device = torch.device("cuda")
    num_players = 3
    bet_bins = [0.5]
    force_deck = torch.tensor(
        [[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
        dtype=torch.long,
        device=device,
    )
    prior = preflop_class_multiplicity_tensor(device=device).to(torch.float32)
    prior = prior / prior.sum()
    initial_beliefs = prior.expand(2, num_players, PREFLOP_HANDS).clone()

    def make_preflop_cfg(*, sparse_fused: bool) -> Config:
        cfg = make_config(bet_bins)
        cfg.device = "cuda"
        cfg.env.num_players = num_players
        cfg.search.depth = 2
        cfg.search.iterations = 3
        cfg.search.cfr_type = CFRType.discounted
        cfg.search.cfr_avg = True
        cfg.search.sparse_fused = sparse_fused
        cfg.search.allin_call_terminal_abstraction = False
        cfg.model.compile = "off"
        return cfg

    def make_preflop_env(cfg: Config) -> PBSEnv:
        env = PBSEnv(
            num_envs=2,
            num_players=num_players,
            mean_stack=1000,
            sb=5,
            bb=10,
            default_bet_bins=cfg.env.bet_bins,
            device=device,
        )
        env.reset(
            force_button=torch.zeros(2, dtype=torch.long, device=device),
            force_deck=force_deck,
        )
        return env

    def make_preflop_model(cfg: Config) -> CompactPreflopMockModel:
        return CompactPreflopMockModel(
            num_actions=len(cfg.env.bet_bins) + 3,
            num_players=num_players,
            device=device,
        )

    non_fused_cfg = make_preflop_cfg(sparse_fused=False)
    fused_cfg = make_preflop_cfg(sparse_fused=True)
    non_fused = PreflopSparseCFREvaluator(
        model=make_preflop_model(non_fused_cfg),  # type: ignore[arg-type]
        device=device,
        cfg=non_fused_cfg,
        generator=torch.Generator(device=device).manual_seed(17),
    )
    fused = FusedPreflopSparseCFREvaluator(
        model=make_preflop_model(fused_cfg),  # type: ignore[arg-type]
        device=device,
        cfg=fused_cfg,
        generator=torch.Generator(device=device).manual_seed(17),
        compile_model=False,
    )

    root_indices = torch.arange(2, device=device)
    non_fused.initialize_subgame(
        make_preflop_env(non_fused_cfg),
        root_indices,
        initial_beliefs,
    )
    fused.initialize_subgame(
        make_preflop_env(fused_cfg),
        root_indices,
        initial_beliefs,
    )

    non_fused.evaluate_cfr(training_mode=False, sample_continuation=False)
    fused.evaluate_cfr(training_mode=False, sample_continuation=False)
    non_fused_batches = non_fused.training_data(
        exclude_start=False,
        include_pre_chance_value_batch=False,
    )
    fused_batches = fused.training_data(
        exclude_start=False,
        include_pre_chance_value_batch=False,
    )

    torch.testing.assert_close(
        fused.policy_probs_avg,
        non_fused.policy_probs_avg,
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        fused.beliefs_avg,
        non_fused.beliefs_avg,
        atol=2e-5,
        rtol=2e-5,
    )
    for fused_batch, non_fused_batch in zip(
        fused_batches,
        non_fused_batches,
        strict=True,
    ):
        _assert_rebel_batches_close(
            fused_batch,
            non_fused_batch,
            atol=2e-5,
            rtol=2e-5,
        )


def test_fused_preflop_prepared_partitions_match_legacy_mixed_street() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA to exercise the fused preflop evaluator")
    pytest.importorskip("triton")
    from p2.search.fused_preflop_sparse_cfr_evaluator import (
        FusedPreflopSparseCFREvaluator,
    )

    class LegacyPartitionFusedPreflopSparseCFREvaluator(
        FusedPreflopSparseCFREvaluator
    ):
        def _try_set_model_values_prepared_partitions(
            self,
            t: int,
            beliefs: torch.Tensor,
        ) -> bool:
            del t, beliefs
            return False

    device = torch.device("cuda")
    num_players = 6
    bet_bins = [0.5]
    force_deck = torch.tensor(
        [[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
        dtype=torch.long,
        device=device,
    )
    prior = preflop_class_multiplicity_tensor(device=device).to(torch.float32)
    prior = prior / prior.sum()
    initial_beliefs = prior.expand(2, num_players, PREFLOP_HANDS).clone()

    def make_cfg() -> Config:
        cfg = make_config(bet_bins)
        cfg.device = "cuda"
        cfg.env.num_players = num_players
        cfg.search.depth = 4
        cfg.search.iterations = 7
        cfg.search.cfr_type = CFRType.discounted
        cfg.search.cfr_avg = False
        cfg.search.sparse_fused = True
        cfg.search.model_scope = ModelScope.mixed_street
        cfg.search.allin_call_terminal_abstraction = True
        cfg.model.compile = "off"
        return cfg

    def make_env(cfg: Config) -> PBSEnv:
        env = PBSEnv(
            num_envs=2,
            num_players=num_players,
            mean_stack=1000,
            sb=5,
            bb=10,
            default_bet_bins=cfg.env.bet_bins,
            device=device,
        )
        env.reset(
            force_button=torch.zeros(2, dtype=torch.long, device=device),
            force_deck=force_deck,
        )
        return env

    cfg = make_cfg()
    model = CompactPreflopMockModel(
        num_actions=len(cfg.env.bet_bins) + 3,
        num_players=num_players,
        device=device,
    )
    closing_model = CompactPreflopMockModel(
        num_actions=len(cfg.env.bet_bins) + 3,
        num_players=num_players,
        device=device,
    )
    prepared = FusedPreflopSparseCFREvaluator(
        model=model,  # type: ignore[arg-type]
        device=device,
        cfg=cfg,
        generator=torch.Generator(device=device).manual_seed(23),
        closing_leaf_model=closing_model,  # type: ignore[arg-type]
        compile_model=False,
    )
    legacy = LegacyPartitionFusedPreflopSparseCFREvaluator(
        model=CompactPreflopMockModel(
            num_actions=len(cfg.env.bet_bins) + 3,
            num_players=num_players,
            device=device,
        ),  # type: ignore[arg-type]
        device=device,
        cfg=cfg,
        generator=torch.Generator(device=device).manual_seed(23),
        closing_leaf_model=CompactPreflopMockModel(
            num_actions=len(cfg.env.bet_bins) + 3,
            num_players=num_players,
            device=device,
        ),  # type: ignore[arg-type]
        compile_model=False,
    )

    root_indices = torch.arange(2, device=device)
    prepared.initialize_subgame(make_env(cfg), root_indices, initial_beliefs)
    legacy.initialize_subgame(make_env(cfg), root_indices, initial_beliefs)

    prepared.evaluate_cfr(training_mode=False, sample_continuation=False)
    legacy.evaluate_cfr(training_mode=False, sample_continuation=False)

    torch.testing.assert_close(
        prepared.latest_values,
        legacy.latest_values,
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        prepared.policy_probs,
        legacy.policy_probs,
        atol=2e-5,
        rtol=2e-5,
    )
    prepared_batches = prepared.training_data(
        exclude_start=False,
        include_pre_chance_value_batch=False,
    )
    legacy_batches = legacy.training_data(
        exclude_start=False,
        include_pre_chance_value_batch=False,
    )
    for prepared_batch, legacy_batch in zip(
        prepared_batches,
        legacy_batches,
        strict=True,
    ):
        _assert_rebel_batches_close(
            prepared_batch,
            legacy_batch,
            atol=2e-5,
            rtol=2e-5,
        )


def test_preflop_model_batch_segments_use_static_tail_buckets() -> None:
    from p2.search.fused_preflop_sparse_cfr_evaluator import (
        _preflop_model_batch_segments,
    )

    assert _preflop_model_batch_segments(0, 4096) == ()
    assert _preflop_model_batch_segments(4096, 4096) == ((0, 4096, 4096),)
    assert _preflop_model_batch_segments(4097, 4096) == (
        (0, 4096, 4096),
        (4096, 1, 512),
    )
    assert _preflop_model_batch_segments(65024, 4096)[-3:] == (
        (61440, 2048, 2048),
        (63488, 1024, 1024),
        (64512, 512, 512),
    )
    assert len(_preflop_model_batch_segments(65536, 4096)) == 16
    assert _preflop_model_batch_segments(44300, 4096)[-3:] == (
        (40960, 2048, 2048),
        (43008, 1024, 1024),
        (44032, 268, 512),
    )
    assert _preflop_model_batch_segments(21236, 4096)[-1:] == (
        (20480, 756, 1024),
    )
    assert _preflop_model_batch_segments(7936, 4096)[-2:] == (
        (4096, 2048, 2048),
        (6144, 1792, 2048),
    )


def test_static_compile_leaves_policy_uncompiled_by_default(monkeypatch) -> None:
    from p2.rl.cfr_trainer import _compile_kwargs as trainer_compile_kwargs
    from p2.search.fused_sparse_cfr_evaluator import _compile_kwargs_from_env

    cfg = Config()
    cfg.model.compile = "static"
    monkeypatch.delenv("P2_POLICY_COMPILE", raising=False)
    monkeypatch.delenv("P2_POLICY_COMPILE_DYNAMIC", raising=False)

    kwargs = _compile_kwargs_from_env(cfg)

    assert kwargs["dynamic"] is False
    assert kwargs["policy_dynamic"] is True
    assert kwargs["policy_compile"] is False
    assert trainer_compile_kwargs(cfg) == kwargs

    monkeypatch.setenv("P2_POLICY_COMPILE", "1")

    kwargs = _compile_kwargs_from_env(cfg)

    assert kwargs["policy_compile"] is True
    assert trainer_compile_kwargs(cfg)["policy_compile"] is True


def test_fused_preflop_rejects_heads_up_closing_projection_mode() -> None:
    from p2.search.fused_preflop_sparse_cfr_evaluator import (
        FusedPreflopSparseCFREvaluator,
    )

    evaluator = object.__new__(FusedPreflopSparseCFREvaluator)
    evaluator.cfg = _scope_cfg()
    evaluator.num_players = 6
    evaluator.hand_dim = PREFLOP_HANDS

    evaluator.closing_leaf_value_model = CompactRecordingValueModel(
        1.0,
        num_players=6,
        device=torch.device("cpu"),
    )
    assert evaluator._preflop_mixed_street_closing_model_enabled()
    assert not evaluator._can_project_heads_up_closing_model()

    evaluator.closing_leaf_value_model = CompactRecordingValueModel(
        1.0,
        num_players=2,
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="same compact shape"):
        evaluator._preflop_mixed_street_closing_model_enabled()

    evaluator.closing_leaf_value_model = ConstantValueModel(1.0, num_players=6)
    with pytest.raises(ValueError, match="same compact shape"):
        evaluator._preflop_mixed_street_closing_model_enabled()


def test_fused_preflop_prepared_partitions_gather_direct_beliefs() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA to exercise the fused preflop evaluator")
    pytest.importorskip("triton")
    from p2.search.fused_preflop_sparse_cfr_evaluator import (
        FusedPreflopSparseCFREvaluator,
    )

    device = torch.device("cuda")
    num_players = 3
    evaluator = object.__new__(FusedPreflopSparseCFREvaluator)
    evaluator.cfg = _scope_cfg()
    evaluator.device = device
    evaluator.float_dtype = torch.float32
    evaluator.num_players = num_players
    evaluator.hand_dim = PREFLOP_HANDS
    evaluator.cfr_type = CFRType.discounted
    evaluator.cfr_avg = False
    evaluator.value_model = CompactRecordingValueModel(
        1.0, num_players=num_players, device=device
    )
    evaluator.model = evaluator.value_model
    evaluator.closing_leaf_value_model = CompactRecordingValueModel(
        2.0, num_players=num_players, device=device
    )
    evaluator.value_feature_encoder = OffsetFeatureEncoder(0.0)
    evaluator.closing_leaf_value_encoder = OffsetFeatureEncoder(100.0)
    evaluator.model_indices = torch.tensor([1, 2, 3], dtype=torch.long, device=device)
    evaluator.new_street_mask = torch.tensor(
        [False, False, True, False, False], device=device
    )
    evaluator.beliefs = torch.zeros(
        5, num_players, PREFLOP_HANDS, device=device, dtype=torch.float32
    )
    evaluator._model_leaf_scatter_enabled = False
    evaluator._preflop_model_beliefs_buf = None
    evaluator._preflop_model_beliefs_key = None
    evaluator._preflop_model_features = None
    evaluator._static_model_feature_fields = None
    evaluator._static_model_feature_key = None
    evaluator._preflop_partition_node_cache = {}
    evaluator._preflop_partition_beliefs_cache = {}
    evaluator._preflop_partition_feature_cache = {}
    evaluator._preflop_encoded_partition_encoder = None
    evaluator._preflop_encoded_partition_positions = None
    evaluator._preflop_encoded_partition_features = None
    evaluator._preflop_cutoff_belief_indices = None
    evaluator._preflop_cutoff_features = None
    evaluator._preflop_new_street_belief_indices = None
    evaluator._preflop_new_street_features = None
    evaluator._preflop_static_model_base_fn_cache = {}
    evaluator._preflop_static_hand_values_fn_cache = {}
    evaluator._preflop_static_model_base_features_cache = {}
    evaluator._preflop_partition_position_slices = {}
    evaluator._preflop_partition_last_values_valid = False
    evaluator._preflop_partition_last_values_marker = None
    evaluator._preflop_partition_eval_stream = None
    evaluator._subgame_generation = 1

    writes: list[tuple[torch.Tensor, torch.Tensor, str]] = []

    def record_writeback(self, **kwargs) -> None:
        del self
        position_beliefs = kwargs["position_beliefs"].view(
            int(kwargs["positions"].numel()), num_players, PREFLOP_HANDS
        )
        writes.append(
            (
                kwargs["positions"].detach().clone(),
                position_beliefs.detach().clone(),
                kwargs["last_attr"],
            )
        )

    evaluator._writeback_model_values_partition = MethodType(
        record_writeback, evaluator
    )
    evaluator._prepare_preflop_model_feature_tensors()

    dynamic_beliefs = torch.arange(
        5 * num_players * PREFLOP_HANDS,
        device=device,
        dtype=torch.float32,
    ).view(5, num_players, PREFLOP_HANDS)

    assert evaluator._try_set_model_values_prepared_partitions(0, dynamic_beliefs)

    torch.testing.assert_close(
        evaluator.value_model.call_beliefs[0],
        dynamic_beliefs[torch.tensor([1, 3], device=device)],
    )
    torch.testing.assert_close(
        evaluator.closing_leaf_value_model.call_beliefs[0],
        dynamic_beliefs[torch.tensor([2], device=device)],
    )
    torch.testing.assert_close(
        evaluator.value_model.call_contexts[0].flatten(),
        torch.tensor([1.0, 3.0], device=device),
    )
    torch.testing.assert_close(
        evaluator.closing_leaf_value_model.call_contexts[0].flatten(),
        torch.tensor([102.0], device=device),
    )
    assert [write[2] for write in writes] == [
        "_preflop_cutoff_last_values_buf",
        "_preflop_new_street_last_values_buf",
    ]


def test_preflop_sparse_ev_uses_marginal_action_policy_for_folded_players() -> None:
    device = get_device()
    num_players = 3
    cfg = make_config([0.5])
    cfg.env.num_players = num_players
    cfg.search.depth = 1
    cfg.search.allin_call_terminal_abstraction = False
    env = PBSEnv(
        num_envs=1,
        num_players=num_players,
        mean_stack=1000,
        sb=5,
        bb=10,
        default_bet_bins=cfg.env.bet_bins,
        device=device,
    )
    env.reset(
        force_button=torch.zeros(1, dtype=torch.long, device=device),
        force_deck=torch.tensor(
            [[10, 11, 12, 13, 14]], dtype=torch.long, device=device
        ),
    )
    env.to_act[0] = 0
    env.has_folded[0, 2] = True
    model = CompactPreflopMockModel(
        num_actions=len(cfg.env.bet_bins) + 3,
        num_players=num_players,
        device=device,
    )
    evaluator = PreflopSparseCFREvaluator(
        model=model,  # type: ignore[arg-type]
        device=device,
        cfg=cfg,
    )
    prior = preflop_class_multiplicity_tensor(device=device).to(torch.float32)
    prior = prior / prior.sum()
    beliefs = prior.expand(1, num_players, PREFLOP_HANDS).clone()
    evaluator.initialize_subgame(env, torch.arange(1, device=device), beliefs)
    evaluator.env.has_folded[:, 2] = True

    children = torch.arange(1, evaluator.total_nodes, device=device)
    policy = torch.zeros_like(evaluator.policy_probs)
    hand_axis = torch.linspace(0.0, 1.0, PREFLOP_HANDS, device=device)
    policy[children[0]] = 0.10 + 0.10 * hand_axis
    policy[children[1]] = 0.35 - 0.05 * hand_axis
    policy[children[2]] = 1.0 - policy[children[0]] - policy[children[1]]
    torch.testing.assert_close(policy[children].sum(dim=0), torch.ones(PREFLOP_HANDS))

    leaf_values = torch.zeros_like(evaluator.latest_values)
    action_values = torch.tensor([1.0, -0.5, 3.0], device=device)
    for child, value in zip(children, action_values, strict=True):
        leaf_values[child, 2] = value
    values = torch.empty_like(evaluator.latest_values)
    evaluator.compute_expected_values(
        policy=policy, leaf_values=leaf_values, values=values
    )

    actor_belief = evaluator.beliefs[0, 0]
    marginal_policy = (actor_belief[None, :] * policy[children]).sum(dim=-1)
    expected = (marginal_policy * action_values).sum()
    torch.testing.assert_close(
        values[0, 2],
        expected.expand(PREFLOP_HANDS),
        rtol=1e-6,
        atol=1e-6,
    )


def test_preflop_sparse_continuation_sampling_returns_abort_root_not_value_target() -> (
    None
):
    device = get_device()
    num_players = 3
    cfg = make_config([0.5])
    cfg.env.num_players = num_players
    cfg.search.depth = 2
    cfg.search.allin_call_terminal_abstraction = False
    cfg.search.continuation_value_target_min_depth = 1
    cfg.search.continuation_value_target_max_depth = 1
    env = PBSEnv(
        num_envs=1,
        num_players=num_players,
        mean_stack=1000,
        sb=5,
        bb=10,
        default_bet_bins=cfg.env.bet_bins,
        device=device,
    )
    env.reset(
        force_button=torch.zeros(1, dtype=torch.long, device=device),
        force_deck=torch.tensor(
            [[10, 11, 12, 13, 14]], dtype=torch.long, device=device
        ),
    )
    model = CompactPreflopMockModel(
        num_actions=len(cfg.env.bet_bins) + 3,
        num_players=num_players,
        device=device,
    )
    evaluator = PreflopSparseCFREvaluator(
        model=model,  # type: ignore[arg-type]
        device=device,
        cfg=cfg,
    )
    evaluator.initialize_subgame(env, torch.arange(1, device=device))
    evaluator.initialize_policy_and_beliefs()

    evaluator.policy_probs_sample.zero_()
    root_children = torch.arange(
        evaluator.child_offsets[0],
        evaluator.child_offsets[0] + evaluator.child_count[0],
        device=device,
    )
    continue_children = root_children[evaluator.action_from_parent[root_children] != 0]
    assert continue_children.numel() > 0
    evaluator.policy_probs_sample[continue_children[0]] = 1.0
    evaluator.beliefs_sample[:] = evaluator.beliefs

    next_pbs = evaluator.sample_leaves(training_mode=True)
    value_batch, _, _ = evaluator.training_data()

    assert evaluator.continuation_value_target_indices.numel() == 0
    assert next_pbs.env.N == 1
    assert int(next_pbs.env.street[0].item()) == 0
    assert int(next_pbs.env.actions_this_round[0].item()) > 0
    assert len(value_batch) == 0


def test_preflop_sparse_allin_call_mask_requires_no_other_betting_player() -> None:
    device = get_device()
    num_players = 4
    cfg = make_config([0.5])
    cfg.env.num_players = num_players
    cfg.search.allin_call_terminal_abstraction = True
    env = PBSEnv(
        num_envs=2,
        num_players=num_players,
        mean_stack=1000,
        sb=5,
        bb=10,
        default_bet_bins=cfg.env.bet_bins,
        device=device,
    )
    env.reset(
        force_button=torch.zeros(2, dtype=torch.long, device=device),
        force_deck=torch.tensor(
            [[10, 11, 12, 13, 14], [10, 11, 12, 13, 14]],
            dtype=torch.long,
            device=device,
        ),
    )
    env.street[:] = 0
    env.to_act[:] = 0
    env.committed[:] = torch.tensor(
        [[0, 100, 100, 0], [0, 100, 100, 100]],
        dtype=torch.long,
        device=device,
    )
    env.stacks[:] = torch.tensor(
        [[100, 0, 0, 1000], [200, 0, 0, 900]],
        dtype=torch.long,
        device=device,
    )
    env.is_allin[:] = torch.tensor(
        [[False, True, True, False], [False, True, True, False]],
        dtype=torch.bool,
        device=device,
    )
    env.has_folded[:] = torch.tensor(
        [[False, False, False, True], [False, False, False, False]],
        dtype=torch.bool,
        device=device,
    )
    model = CompactPreflopMockModel(
        num_actions=len(cfg.env.bet_bins) + 3,
        num_players=num_players,
        device=device,
    )
    evaluator = PreflopSparseCFREvaluator(
        model=model,  # type: ignore[arg-type]
        device=device,
        cfg=cfg,
    )

    mask = evaluator._allin_call_child_mask(
        env,
        torch.arange(2, device=device),
        torch.ones(2, dtype=torch.long, device=device),
    )

    assert mask.tolist() == [True, False]


def test_preflop_sparse_allin_call_leaves_match_terminal_runouts() -> None:
    device = get_device()
    num_players = 4
    cfg = make_config([0.5])
    cfg.env.num_players = num_players
    cfg.search.depth = 1
    cfg.search.allin_call_terminal_abstraction = True
    env = PBSEnv(
        num_envs=1,
        num_players=num_players,
        mean_stack=1000,
        sb=5,
        bb=10,
        default_bet_bins=cfg.env.bet_bins,
        device=device,
    )
    env.reset(
        force_button=torch.zeros(1, dtype=torch.long, device=device),
        force_deck=torch.tensor(
            [[10, 11, 12, 13, 14]], dtype=torch.long, device=device
        ),
    )
    env.street[0] = 0
    env.to_act[0] = 0
    env.actions_this_round[0] = 1
    env.acted_this_round[0] = torch.tensor(
        [False, True, True, True],
        dtype=torch.bool,
        device=device,
    )
    env.committed[0] = torch.tensor([0, 100, 100, 0], dtype=torch.long, device=device)
    env.chips_placed[0] = env.committed[0]
    env.pot[0] = 200
    env.stacks[0] = torch.tensor([200, 0, 0, 1000], dtype=torch.long, device=device)
    env.starting_stacks[0] = torch.tensor(
        [200, 100, 100, 1000],
        dtype=torch.long,
        device=device,
    )
    env.is_allin[0] = torch.tensor(
        [False, True, True, False],
        dtype=torch.bool,
        device=device,
    )
    env.has_folded[0] = torch.tensor(
        [False, False, False, True],
        dtype=torch.bool,
        device=device,
    )
    model = CompactPreflopMockModel(
        num_actions=len(cfg.env.bet_bins) + 3,
        num_players=num_players,
        device=device,
    )
    evaluator = PreflopSparseCFREvaluator(
        model=model,  # type: ignore[arg-type]
        device=device,
        cfg=cfg,
    )

    evaluator.initialize_subgame(env, torch.arange(1, device=device))

    child_indices = torch.arange(
        evaluator.root_nodes, evaluator.total_nodes, device=device
    )
    parent, _ = evaluator._parent_action_for_nodes(child_indices)
    live = ~evaluator.env.has_folded[child_indices]
    eligible = (
        live
        & ~evaluator.env.is_allin[child_indices]
        & (evaluator.env.stacks[child_indices] > 0)
    )
    terminal_runouts = child_indices[
        evaluator.env.done[child_indices]
        & (evaluator.env.street[child_indices] == 4)
        & (evaluator.env.street[parent] == 0)
        & (live.sum(dim=1) > 1)
        & ~eligible.any(dim=1)
    ]

    assert terminal_runouts.numel() > 0
    assert evaluator.allin_call_indices.tolist() == terminal_runouts.tolist()
    assert evaluator.preflop_allin_indices_by_live_count[2].tolist() == [1]
    assert evaluator.leaf_mask[terminal_runouts].all()
    assert not evaluator.new_street_mask[terminal_runouts].any()


def test_preflop_sparse_allin_values_route_by_precomputed_live_count() -> None:
    device = get_device()
    num_players = 6
    cfg = make_config([0.5])
    cfg.env.num_players = num_players
    cfg.search.allin_call_terminal_abstraction = False
    env = PBSEnv(
        num_envs=2,
        num_players=num_players,
        mean_stack=1000,
        sb=5,
        bb=10,
        default_bet_bins=cfg.env.bet_bins,
        device=device,
    )
    env.reset(
        force_button=torch.zeros(2, dtype=torch.long, device=device),
        force_deck=torch.tensor(
            [[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
            dtype=torch.long,
            device=device,
        ),
    )
    model = CompactPreflopMockModel(
        num_actions=len(cfg.env.bet_bins) + 3,
        num_players=num_players,
        device=device,
    )
    evaluator = PreflopSparseCFREvaluator(
        model=model,  # type: ignore[arg-type]
        device=device,
        cfg=cfg,
    )
    evaluator.initialize_subgame(env, torch.arange(2, device=device))
    node_idx = torch.arange(5, device=device)
    evaluator.allin_call_indices = node_idx
    evaluator.allin_call_parent_indices = torch.zeros_like(node_idx)
    evaluator.allin_call_mask = torch.zeros(
        evaluator.total_nodes, dtype=torch.bool, device=device
    )
    evaluator.allin_call_mask[node_idx] = True
    evaluator.env.has_folded[node_idx] = torch.tensor(
        [
            [False, False, True, True, True, True],
            [False, False, False, True, True, True],
            [False, False, False, False, True, True],
            [False, False, False, False, False, True],
            [False, False, False, False, False, False],
        ],
        dtype=torch.bool,
        device=device,
    )
    evaluator.env.is_allin[node_idx] = ~evaluator.env.has_folded[node_idx]
    evaluator.env.chips_placed[node_idx] = 100
    evaluator.env.stacks[node_idx] = 0
    evaluator.env.starting_stacks[node_idx] = 100
    evaluator.env.scale[node_idx] = 100
    evaluator._cache_allin_call_street_partitions(
        evaluator.env.street[evaluator.allin_call_parent_indices]
    )

    class FakeOracle:
        def __init__(self) -> None:
            self.live_counts: list[int] = []
            self.batch_sizes: list[int] = []
            self.live2_entries: list[tuple[list[int], list[int], list[int]]] = []
            self.live3_entries: list[
                tuple[list[int], list[int], list[int], list[int]]
            ] = []

        def values(self, *, beliefs, live_players, **kwargs):
            del kwargs
            self.live_counts.append(live_players)
            self.batch_sizes.append(int(beliefs.shape[0]))
            return torch.full_like(beliefs, float(live_players))

        def values_for_live2_entries(
            self,
            *,
            beliefs,
            live_entry_rows,
            hero_players,
            opp_players,
            **kwargs,
        ):
            del kwargs
            self.live2_entries.append(
                (
                    live_entry_rows.tolist(),
                    hero_players.tolist(),
                    opp_players.tolist(),
                )
            )
            return torch.full_like(beliefs, 2.0)

        def values_for_live3_entries(
            self,
            *,
            beliefs,
            live_entry_rows,
            hero_players,
            opp0_players,
            opp1_players,
            **kwargs,
        ):
            del kwargs
            self.live3_entries.append(
                (
                    live_entry_rows.tolist(),
                    hero_players.tolist(),
                    opp0_players.tolist(),
                    opp1_players.tolist(),
                )
            )
            return torch.full_like(beliefs, 3.0)

    fake = FakeOracle()
    evaluator.preflop_allin_169_oracle = fake
    evaluator._set_allin_call_values(evaluator.beliefs)

    assert [x.tolist() for x in evaluator.preflop_allin_indices_by_live_count[2:]] == [
        [0],
        [1],
        [2],
        [3],
        [4],
    ]
    assert evaluator.preflop_allin_exact_indices.tolist() == [0, 1]
    assert evaluator.preflop_allin_model_indices.tolist() == [2, 3, 4]
    assert fake.live_counts == [6]
    assert fake.batch_sizes == [3]
    assert fake.live2_entries == [([0, 0], [0, 1], [1, 0])]
    assert fake.live3_entries == [([0, 0, 0], [0, 1, 2], [1, 0, 0], [2, 2, 1])]
    assert_close(
        evaluator.latest_values[0], torch.full((num_players, PREFLOP_HANDS), 2.0)
    )
    assert_close(
        evaluator.latest_values[1], torch.full((num_players, PREFLOP_HANDS), 3.0)
    )
    assert_close(
        evaluator.latest_values[2], torch.full((num_players, PREFLOP_HANDS), 6.0)
    )
    assert_close(
        evaluator.latest_values[3], torch.full((num_players, PREFLOP_HANDS), 6.0)
    )
    assert_close(
        evaluator.latest_values[4], torch.full((num_players, PREFLOP_HANDS), 6.0)
    )


def test_preflop_sparse_evaluator_rejects_fused_config() -> None:
    device = get_device()
    cfg = make_config([0.5])
    cfg.search.sparse_fused = True
    model = MockModel(
        num_actions=len(cfg.env.bet_bins) + 3,
        device=device,
    )

    with pytest.raises(ValueError, match="non-fused sparse CFR"):
        PreflopSparseCFREvaluator(
            model=model,  # type: ignore[arg-type]
            device=device,
            cfg=cfg,
        )


def test_fused_preflop_sparse_evaluator_is_compact_only(monkeypatch) -> None:
    from p2.search.fused_preflop_sparse_cfr_evaluator import (
        FusedPreflopSparseCFREvaluator,
    )
    from p2.search.fused_sparse_cfr_evaluator import FusedSparseCFREvaluator

    monkeypatch.setattr(
        FusedSparseCFREvaluator,
        "__init__",
        SparseCFREvaluator.__init__,
    )
    device = get_device()
    num_players = 3
    cfg = make_config([0.5])
    cfg.env.num_players = num_players
    cfg.search.allin_call_terminal_abstraction = False
    env = PBSEnv(
        num_envs=1,
        num_players=num_players,
        mean_stack=1000,
        sb=5,
        bb=10,
        default_bet_bins=cfg.env.bet_bins,
        device=device,
    )
    env.reset(
        force_button=torch.zeros(1, dtype=torch.long, device=device),
        force_deck=torch.tensor(
            [[10, 11, 12, 13, 14]], dtype=torch.long, device=device
        ),
    )
    model = CompactPreflopMockModel(
        num_actions=len(cfg.env.bet_bins) + 3,
        num_players=num_players,
        device=device,
    )
    evaluator = FusedPreflopSparseCFREvaluator(
        model=model,  # type: ignore[arg-type]
        device=device,
        cfg=cfg,
    )

    evaluator.initialize_subgame(env, torch.arange(1, device=device))
    beliefs_at_model = evaluator._model_beliefs_for_values(evaluator.beliefs)
    torch.testing.assert_close(
        beliefs_at_model,
        evaluator.beliefs[evaluator.model_indices],
    )
    assert beliefs_at_model.is_contiguous()
    assert evaluator._model_beliefs_for_values(evaluator.beliefs).data_ptr() == (
        beliefs_at_model.data_ptr()
    )
    features = evaluator._model_features_for_beliefs(
        beliefs_at_model
    )

    assert evaluator.hand_dim == PREFLOP_HANDS
    assert evaluator.beliefs.shape[-1] == PREFLOP_HANDS
    assert evaluator.policy_probs.shape[-1] == PREFLOP_HANDS
    assert evaluator.latest_values.shape[-1] == PREFLOP_HANDS
    assert features.hand_dim == PREFLOP_HANDS
    assert features.beliefs.shape[1] == num_players * PREFLOP_HANDS
    partition_positions = (
        torch.cat(
            (evaluator.new_street_model_positions, evaluator.cutoff_model_positions)
        )
        .sort()
        .values
    )
    torch.testing.assert_close(
        partition_positions,
        torch.arange(evaluator.model_indices.numel(), device=device),
    )
    assert evaluator.new_street_indices.numel() + evaluator.cutoff_indices.numel() == (
        evaluator.model_indices.numel()
    )

    bad_model = MockModel(
        num_actions=len(cfg.env.bet_bins) + 3,
        num_players=num_players,
        device=device,
    )
    with pytest.raises(ValueError, match="compact-only"):
        FusedPreflopSparseCFREvaluator(
            model=bad_model,  # type: ignore[arg-type]
            device=device,
            cfg=cfg,
        )


def test_fused_preflop_sparse_new_street_leaves_are_heads_up(monkeypatch) -> None:
    from p2.search.fused_preflop_sparse_cfr_evaluator import (
        FusedPreflopSparseCFREvaluator,
    )
    from p2.search.fused_sparse_cfr_evaluator import FusedSparseCFREvaluator

    monkeypatch.setattr(
        FusedSparseCFREvaluator,
        "__init__",
        SparseCFREvaluator.__init__,
    )
    device = get_device()
    num_players = 4
    cfg = make_config([0.5])
    cfg.env.num_players = num_players
    cfg.search.depth = num_players
    cfg.search.allin_call_terminal_abstraction = False
    env = PBSEnv(
        num_envs=1,
        num_players=num_players,
        mean_stack=1000,
        sb=5,
        bb=10,
        default_bet_bins=cfg.env.bet_bins,
        device=device,
    )
    env.reset(
        force_button=torch.zeros(1, dtype=torch.long, device=device),
        force_deck=torch.tensor(
            [[10, 11, 12, 13, 14]], dtype=torch.long, device=device
        ),
    )
    model = CompactPreflopMockModel(
        num_actions=len(cfg.env.bet_bins) + 3,
        num_players=num_players,
        device=device,
    )
    evaluator = FusedPreflopSparseCFREvaluator(
        model=model,  # type: ignore[arg-type]
        device=device,
        cfg=cfg,
    )

    evaluator.initialize_subgame(env, torch.arange(1, device=device))

    assert env.force_heads_up_preflop_flop is True
    assert evaluator.new_street_mask.any()
    live_counts = (~evaluator.env.has_folded[evaluator.new_street_mask]).sum(dim=1)
    assert int(live_counts.max()) <= 2


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


def test_action_mix_records_root_mix_and_total_bet_mass() -> None:
    device = get_device()
    evaluator = SparseCFREvaluator.__new__(SparseCFREvaluator)
    evaluator.device = device
    evaluator.root_nodes = 2
    evaluator.num_actions = 6
    evaluator.total_nodes = 20
    evaluator.depth_offsets = [0, 2, 14, 20]
    evaluator.stats = {}

    evaluator.parent_index = torch.tensor(
        [0, 0] + [0] * 6 + [1] * 6 + [2] * 6,
        dtype=torch.long,
        device=device,
    )
    evaluator.action_from_parent = torch.tensor(
        [0, 0]
        + list(range(evaluator.num_actions))
        + list(range(evaluator.num_actions))
        + list(range(evaluator.num_actions)),
        dtype=torch.long,
        device=device,
    )
    evaluator.leaf_mask = torch.ones(
        evaluator.total_nodes, dtype=torch.bool, device=device
    )
    evaluator.leaf_mask[:2] = False
    evaluator.leaf_mask[2] = False

    hands = 3
    evaluator.allowed_hands = torch.ones(
        evaluator.total_nodes, hands, dtype=torch.bool, device=device
    )
    evaluator.policy_probs_avg = torch.zeros(
        evaluator.total_nodes, hands, dtype=torch.float32, device=device
    )

    parent_probs = torch.tensor(
        [
            [0.10, 0.20, 0.15, 0.25, 0.05, 0.25],
            [0.20, 0.10, 0.05, 0.05, 0.20, 0.40],
            [0.00, 0.50, 0.10, 0.10, 0.10, 0.20],
        ],
        dtype=torch.float32,
        device=device,
    )
    parent_rows = torch.tensor([0, 1, 2], dtype=torch.long, device=device)
    for row, probs in zip(parent_rows, parent_probs):
        child_rows = torch.where(evaluator.parent_index == row)[0]
        child_actions = evaluator.action_from_parent[child_rows]
        evaluator.policy_probs_avg[child_rows] = probs[child_actions, None]

    evaluator._record_action_mix()

    root_mix = evaluator.stats["root_action_mix"]
    assert_close(torch.tensor(root_mix["fold"]), torch.tensor(0.15))
    assert_close(torch.tensor(root_mix["call"]), torch.tensor(0.15))
    assert_close(torch.tensor(root_mix["bet"]), torch.tensor(0.375))
    assert_close(torch.tensor(root_mix["allin"]), torch.tensor(0.325))
    assert_close(torch.tensor(sum(root_mix.values())), torch.tensor(1.0))

    tree_mix = evaluator.stats["action_mix"]
    assert_close(torch.tensor(tree_mix["bet"]), torch.tensor(0.35))


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


def test_record_stats_aggregates_child_delta_like_pull_back() -> None:
    device = get_device()
    evaluator, env, _cfg = make_sparse_evaluator(device=device)

    root_indices = torch.tensor([0], dtype=torch.long, device=device)
    evaluator.initialize_subgame(env, root_indices)

    torch.manual_seed(17)
    old_policy_probs = torch.rand_like(evaluator.policy_probs)
    evaluator.policy_probs = torch.rand_like(evaluator.policy_probs)
    evaluator.self_reach.zero_()

    parent_count = evaluator.depth_offsets[-2]
    reachable = torch.rand(parent_count, NUM_HANDS, device=device) > 0.2
    reachable[0, 0] = True
    evaluator.self_reach[:parent_count, 0] = reachable.to(evaluator.float_dtype)

    diff = evaluator._pull_back(evaluator.policy_probs) - evaluator._pull_back(
        old_policy_probs
    )
    diff.masked_fill_(~reachable[:, None, :], 0.0)
    reachable_hand_count = reachable.sum(dim=-1)
    reachable_nodes = reachable_hand_count > 0
    diff_sum_nodes = diff.abs().sum(dim=1).sum(dim=1)
    expected = (
        torch.where(reachable_nodes, diff_sum_nodes / reachable_hand_count, 0.0).sum()
        / reachable_nodes.sum()
    )

    evaluator.warm_start_iterations = 0
    evaluator.cfr_iterations = 5
    evaluator._record_stats(2, old_policy_probs)

    assert_close(
        torch.tensor(evaluator.stats["cfr_delta.50"], device=device),
        expected,
    )


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
    assert torch.equal(
        value_batch.statistics["target_source"],
        torch.full_like(
            value_batch.statistics["target_source"],
            TARGET_SOURCE_CFR_BACKUP,
        ),
    )
    leaf_total = value_batch.statistics["leaf_total_count"]
    leaf_cfr = value_batch.statistics[
        f"leaf_target_source_{TARGET_SOURCE_CFR_BACKUP}_count"
    ]
    leaf_exact = value_batch.statistics[
        f"leaf_target_source_{TARGET_SOURCE_EXACT_TERMINAL}_count"
    ]
    leaf_closing = value_batch.statistics[
        f"leaf_target_source_{TARGET_SOURCE_CLOSING_NET}_count"
    ]
    assert torch.equal(leaf_total, leaf_cfr + leaf_exact + leaf_closing)
    assert (leaf_total > 0).all()
    assert torch.isfinite(policy_batch.policy_targets).all()
    if len(pre_value_batch) > 0:
        assert torch.isfinite(pre_value_batch.value_targets).all()
        assert torch.equal(
            pre_value_batch.statistics["target_source"],
            torch.full_like(
                pre_value_batch.statistics["target_source"],
                TARGET_SOURCE_CHANCE_EXPECTATION,
            ),
        )


def test_training_data_excludes_start_of_hand_but_keeps_river_roots() -> None:
    device = get_device()
    env = make_env(4, device=device)
    cfg = make_config(env.default_bet_bins)
    cfg.search.depth = 2
    evaluator, _, _ = make_sparse_evaluator(env=env, cfg=cfg, device=device)
    roots = torch.arange(env.N, device=device)
    evaluator.initialize_subgame(env, roots)
    evaluator.evaluate_cfr(training_mode=False)

    evaluator.env.street[: env.N] = torch.tensor([0, 3, 3, 1], device=device)
    evaluator.env.actions_this_round[: env.N] = torch.tensor(
        [0, 0, evaluator.max_depth - 1, 0],
        device=device,
    )
    evaluator.env.done[: env.N] = False

    value_batch, pre_value_batch, policy_batch = evaluator.training_data(
        include_pre_chance_value_batch=False,
        include_policy_batch=False,
    )

    assert pre_value_batch is None
    assert policy_batch is None
    assert len(value_batch) == 3
    assert value_batch.statistics["street"].tolist() == [3, 3, 1]
    assert value_batch.statistics["actions_this_round"].tolist() == [
        0,
        evaluator.max_depth - 1,
        0,
    ]
    assert value_batch.statistics["continuation_value_target"].tolist() == [
        False,
        False,
        False,
    ]
    assert value_batch.statistics["local_exploitability"].shape[0] == 3

    raw_value_batch, _, _ = evaluator.training_data(
        exclude_start=False,
        include_pre_chance_value_batch=False,
        include_policy_batch=False,
    )
    assert len(raw_value_batch) == env.N


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


def test_sparse_continuation_sampling_returns_abort_root_without_value_target() -> None:
    device = get_device()
    env = make_env(1, device=device)
    cfg = make_config(env.default_bet_bins)
    cfg.search.depth = 2
    cfg.search.continuation_value_target_sampling = True
    cfg.search.continuation_value_target_min_depth = 1
    cfg.search.continuation_value_target_max_depth = 1

    evaluator, _, _ = make_sparse_evaluator(env=env, cfg=cfg, device=device)
    evaluator.initialize_subgame(env, torch.tensor([0], device=device))
    evaluator.initialize_policy_and_beliefs()

    evaluator.policy_probs_sample.zero_()
    root_children = torch.arange(
        evaluator.child_offsets[0],
        evaluator.child_offsets[0] + evaluator.child_count[0],
        device=device,
    )
    call_children = root_children[evaluator.action_from_parent[root_children] == 1]
    assert call_children.numel() == 1
    evaluator.policy_probs_sample[call_children[0]] = 1.0
    evaluator.beliefs_sample[:] = evaluator.beliefs

    next_pbs = evaluator.sample_leaves(training_mode=True)
    value_batch, _, _ = evaluator.training_data(exclude_start=False)

    assert evaluator.continuation_value_target_indices.numel() == 0
    assert next_pbs.env.N == 1
    assert_close(next_pbs.beliefs[0], evaluator.beliefs_sample[call_children[0]])
    assert len(value_batch) == 1
    assert value_batch.statistics["continuation_value_target"].tolist() == [False]
    assert value_batch.statistics["node_depth"].tolist() == [0]


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


def test_sparse_allin_call_abstraction_skips_preflop_even_with_table() -> None:
    device = get_device()
    env = make_env(1, device=device)
    cfg = make_config(env.default_bet_bins)
    cfg.search.depth = 2
    cfg.search.allin_call_terminal_abstraction = True
    cfg.search.preflop_allin_table_path = "outputs/preflop_allin_table.pt.zst"

    env.street[0] = 0
    env.board_indices[0] = -1
    env.board_onehot[0].zero_()
    env.to_act[0] = 0
    env.actions_this_round[0] = 1
    env.stacks[0] = torch.tensor([100, 0], device=device)
    env.starting_stacks[0] = torch.tensor([100, 100], device=device)
    env.scale[0] = 100
    env.committed[0] = torch.tensor([0, 100], device=device)
    env.pot[0] = 100
    env.is_allin[0] = torch.tensor([False, True], device=device)

    evaluator, _, _ = make_sparse_evaluator(env=env, cfg=cfg, device=device)
    call_mask = evaluator._allin_call_child_mask(
        env,
        torch.tensor([0], device=device),
        torch.tensor([1], device=device),
    )

    evaluator.initialize_subgame(env, torch.tensor([0], device=device))

    assert not call_mask.item()
    assert evaluator.allin_call_indices.numel() == 0
    assert not evaluator.allin_call_mask.any()


def test_sparse_allin_call_abstraction_applies_postflop_with_table() -> None:
    device = get_device()
    env = make_env(1, device=device)
    cfg = make_config(env.default_bet_bins)
    cfg.search.allin_call_terminal_abstraction = True
    cfg.search.preflop_allin_table_path = "outputs/preflop_allin_table.pt.zst"

    env.street[0] = 1
    env.to_act[0] = 0
    env.actions_this_round[0] = 1
    env.stacks[0] = torch.tensor([100, 0], device=device)
    env.committed[0] = torch.tensor([0, 100], device=device)
    env.is_allin[0] = torch.tensor([False, True], device=device)

    evaluator, _, _ = make_sparse_evaluator(env=env, cfg=cfg, device=device)
    call_mask = evaluator._allin_call_child_mask(
        env,
        torch.tensor([0], device=device),
        torch.tensor([1], device=device),
    )

    assert call_mask.item()


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
    evaluator, _, _ = make_sparse_evaluator(
        model=model, env=env, cfg=cfg, device=device
    )

    evaluator.initialize_subgame(
        env, torch.tensor([0], dtype=torch.long, device=device)
    )

    assert len(evaluator.depth_offsets) >= 8
    depth5_start = evaluator.depth_offsets[5]
    depth5_end = evaluator.depth_offsets[6]
    depth6_start = evaluator.depth_offsets[6]
    depth6_end = evaluator.depth_offsets[7]
    depth6_children = torch.arange(depth6_start, depth6_end, device=device)

    depth5_child_mask = (evaluator.parent_index[depth6_children] >= depth5_start) & (
        evaluator.parent_index[depth6_children] < depth5_end
    )
    depth5_children = depth6_children[depth5_child_mask]
    assert depth5_children.numel() > 0
    assert (evaluator.action_from_parent[depth5_children] <= 1).all()

    call_children = depth5_children[evaluator.action_from_parent[depth5_children] == 1]
    assert call_children.numel() > 0
    assert evaluator.new_street_mask[
        call_children[~evaluator.env.done[call_children]]
    ].all()

    model_leaf_mask = evaluator.leaf_mask & ~evaluator.env.done
    model_leaf_mask &= ~evaluator.allin_call_mask
    assert evaluator.new_street_mask[model_leaf_mask].all()


def test_street_cutoff_model_leaves_use_configured_end_model() -> None:
    device = get_device()
    cfg = make_street_cutoff_config()
    cfg.search.model_scope = ModelScope.end_of_street
    env = make_env(1, device=device)
    env.default_bet_bins = cfg.env.bet_bins
    env.num_bet_bins = cfg.model.num_actions
    model = PreOnlyMockModel(num_actions=cfg.model.num_actions, device=device)
    evaluator, _, _ = make_sparse_evaluator(
        model=model, env=env, cfg=cfg, device=device
    )

    evaluator.initialize_subgame(
        env, torch.tensor([0], dtype=torch.long, device=device)
    )
    evaluator.set_leaf_values(0)

    assert evaluator.model_indices.numel() > 0
    assert model.forward_pre_calls == 0
