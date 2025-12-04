import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from alphaholdem.core.structured_config import Config, ModelType, NonlinearityType
from alphaholdem.core.structured_config import CFRType
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.mlp.better_feature_encoder import BetterFeatureEncoder
from alphaholdem.models.mlp.better_ffn import BetterFFN
from alphaholdem.models.mlp.better_trm import BetterTRM
from alphaholdem.models.model_output import ModelOutput
from alphaholdem.rl.cfr_trainer import RebelCFRTrainer
from debugging.debug_cfr_iterations import (
    UniformPolicyWrapper,
    load_model_from_checkpoint,
)
from alphaholdem.search.rebel_cfr_evaluator import RebelCFREvaluator
from alphaholdem.search.sparse_cfr_evaluator import SparseCFREvaluator


def _make_cfg(bet_bins: list[float] | None = None) -> Config:
    cfg = Config()
    cfg.device = "cpu"
    cfg.num_envs = 1
    cfg.model.compile = False
    cfg.train.batch_size = 2
    cfg.train.replay_buffer_batches = 1
    cfg.train.value_reuse_goal = 1
    cfg.train.policy_capacity_factor = 1
    if bet_bins is not None:
        cfg.env.bet_bins = bet_bins
    else:
        cfg.env.bet_bins = [0.5]
    cfg.model.num_actions = len(cfg.env.bet_bins) + 3
    return cfg


def test_load_model_uses_checkpoint_config_for_better_ffn(tmp_path) -> None:
    cfg_save = _make_cfg()
    cfg_save.model.name = ModelType.better_ffn
    cfg_save.model.hidden_dim = 32
    cfg_save.model.range_hidden_dim = 8
    cfg_save.model.ffn_dim = 64
    cfg_save.model.num_hidden_layers = 2
    cfg_save.model.num_policy_layers = 3
    cfg_save.model.num_value_layers = 1
    cfg_save.model.shared_trunk = False
    cfg_save.model.enforce_zero_sum = False

    trainer = RebelCFRTrainer(cfg_save, torch.device(cfg_save.device))
    ckpt_path = tmp_path / "better_ffn.pt"
    trainer.save_checkpoint(str(ckpt_path), step=7, save_optimizer=False)

    cfg_load = _make_cfg(bet_bins=[0.75])
    model = load_model_from_checkpoint(
        str(ckpt_path), cfg_load, torch.device(cfg_load.device)
    )

    assert isinstance(model, BetterFFN)
    assert cfg_load.model.name == ModelType.better_ffn
    assert cfg_load.model.hidden_dim == cfg_save.model.hidden_dim
    assert cfg_load.model.range_hidden_dim == cfg_save.model.range_hidden_dim
    assert cfg_load.model.ffn_dim == cfg_save.model.ffn_dim
    assert cfg_load.model.num_hidden_layers == cfg_save.model.num_hidden_layers
    assert cfg_load.model.num_policy_layers == cfg_save.model.num_policy_layers
    assert cfg_load.model.num_value_layers == cfg_save.model.num_value_layers
    assert cfg_load.env.bet_bins == cfg_save.env.bet_bins


def test_load_model_resets_nonlinearity_from_state(tmp_path) -> None:
    cfg_save = _make_cfg()
    cfg_save.model.name = ModelType.better_trm
    cfg_save.model.nonlinearity = NonlinearityType.gelu

    trainer = RebelCFRTrainer(cfg_save, torch.device(cfg_save.device))
    ckpt_path = tmp_path / "better_trm_gelu.pt"
    trainer.save_checkpoint(str(ckpt_path), step=1, save_optimizer=False)

    cfg_load = _make_cfg()
    cfg_load.model.name = ModelType.better_trm
    cfg_load.model.nonlinearity = NonlinearityType.swiglu  # user override

    model = load_model_from_checkpoint(
        str(ckpt_path), cfg_load, torch.device(cfg_load.device)
    )

    assert isinstance(model, BetterTRM)
    assert cfg_load.model.nonlinearity == NonlinearityType.gelu


def test_load_model_uses_checkpoint_config_for_better_trm(tmp_path) -> None:
    cfg_save = _make_cfg()
    cfg_save.model.name = ModelType.better_trm
    cfg_save.model.hidden_dim = 40
    cfg_save.model.range_hidden_dim = 10
    cfg_save.model.ffn_dim = 80
    cfg_save.model.num_hidden_layers = 2
    cfg_save.model.num_policy_layers = 2
    cfg_save.model.num_value_layers = 3
    cfg_save.model.num_recursions = 4
    cfg_save.model.num_iterations = 5
    cfg_save.model.shared_trunk = True
    cfg_save.model.enforce_zero_sum = False
    cfg_save.model.nonlinearity = NonlinearityType.swiglu

    trainer = RebelCFRTrainer(cfg_save, torch.device(cfg_save.device))
    ckpt_path = tmp_path / "better_trm.pt"
    trainer.save_checkpoint(str(ckpt_path), step=3, save_optimizer=False)

    cfg_load = _make_cfg(bet_bins=[1.0])
    model = load_model_from_checkpoint(
        str(ckpt_path), cfg_load, torch.device(cfg_load.device)
    )

    assert isinstance(model, BetterTRM)
    assert cfg_load.model.name == ModelType.better_trm
    assert cfg_load.model.hidden_dim == cfg_save.model.hidden_dim
    assert cfg_load.model.range_hidden_dim == cfg_save.model.range_hidden_dim
    assert cfg_load.model.ffn_dim == cfg_save.model.ffn_dim
    assert cfg_load.model.num_policy_layers == cfg_save.model.num_policy_layers
    assert cfg_load.model.num_value_layers == cfg_save.model.num_value_layers
    assert cfg_load.model.num_recursions == cfg_save.model.num_recursions
    assert cfg_load.model.num_iterations == cfg_save.model.num_iterations
    assert cfg_load.model.nonlinearity == cfg_save.model.nonlinearity
    assert cfg_load.model.range_hidden_dim == cfg_save.model.range_hidden_dim
    assert cfg_load.model.num_hidden_layers == cfg_save.model.num_hidden_layers
    assert cfg_load.env.bet_bins == cfg_save.env.bet_bins


def test_load_model_overrides_hidden_layers_from_state(tmp_path) -> None:
    cfg_save = _make_cfg()
    cfg_save.model.name = ModelType.better_trm
    cfg_save.model.num_hidden_layers = 2
    trainer = RebelCFRTrainer(cfg_save, torch.device(cfg_save.device))
    ckpt_path = tmp_path / "better_trm_hidden.pt"
    trainer.save_checkpoint(str(ckpt_path), step=1, save_optimizer=False)

    cfg_load = _make_cfg()
    cfg_load.model.name = ModelType.better_trm
    cfg_load.model.num_hidden_layers = 5

    load_model_from_checkpoint(str(ckpt_path), cfg_load, torch.device(cfg_load.device))

    assert cfg_load.model.num_hidden_layers == cfg_save.model.num_hidden_layers


class _DummyPolicyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._logits = torch.tensor([[[0.2, -0.3, 0.5]]], dtype=torch.float32)
        self._values = torch.tensor([0.5], dtype=torch.float32)

    def forward(self, *args, include_policy: bool = True, **kwargs) -> ModelOutput:
        policy_logits = self._logits if include_policy else None
        return ModelOutput(policy_logits=policy_logits, value=self._values)


def test_uniform_policy_wrapper_overrides_policy_logits() -> None:
    base_model = _DummyPolicyModel()
    model = UniformPolicyWrapper(base_model)

    output_with_policy = model(None, include_policy=True)
    torch.testing.assert_close(
        output_with_policy.policy_logits, torch.ones_like(base_model._logits)
    )
    torch.testing.assert_close(output_with_policy.value, base_model._values)

    output_without_policy = model(None, include_policy=False)
    assert output_without_policy.policy_logits is None
    torch.testing.assert_close(output_without_policy.value, base_model._values)


def _make_better_ffn(num_actions: int) -> BetterFFN:
    return BetterFFN(
        num_actions=num_actions,
        hidden_dim=8,
        range_hidden_dim=4,
        ffn_dim=8,
        num_hidden_layers=1,
        num_policy_layers=1,
        num_value_layers=1,
        shared_trunk=True,
        enforce_zero_sum=False,
    )


def _make_env(bet_bins: list[float], device: torch.device) -> HUNLTensorEnv:
    env = HUNLTensorEnv(
        num_envs=1,
        starting_stack=1000,
        sb=5,
        bb=10,
        default_bet_bins=bet_bins,
        device=device,
        float_dtype=torch.float32,
    )
    env.reset()
    return env


def test_rebel_cfr_uses_better_encoder_for_uniform_wrapper() -> None:
    device = torch.device("cpu")
    bet_bins = [0.5]
    model = UniformPolicyWrapper(_make_better_ffn(num_actions=len(bet_bins) + 3))
    env = _make_env(bet_bins, device)

    evaluator = RebelCFREvaluator(
        search_batch_size=1,
        env_proto=env,
        model=model,
        bet_bins=bet_bins,
        max_depth=1,
        cfr_iterations=2,
        device=device,
        float_dtype=torch.float32,
        generator=torch.Generator(device=device),
        warm_start_iterations=1,
        num_supervisions=1,
        cfr_type=CFRType.linear,
        cfr_avg=True,
        dcfr_alpha=1.5,
        dcfr_beta=0.0,
        dcfr_gamma=2.0,
        dcfr_delay=0,
        sample_epsilon=0.25,
        value_targets_from_final_policy=False,
    )

    assert isinstance(evaluator.feature_encoder, BetterFeatureEncoder)


def test_sparse_cfr_uses_better_encoder_for_uniform_wrapper() -> None:
    device = torch.device("cpu")
    bet_bins = [0.5]
    cfg = _make_cfg(bet_bins=bet_bins)
    cfg.search.depth = 1
    cfg.search.iterations = 2
    cfg.search.warm_start_iterations = 1
    cfg.search.cfr_type = CFRType.linear
    cfg.search.cfr_avg = True
    cfg.num_envs = 1
    cfg.model.num_actions = len(bet_bins) + 3

    env = _make_env(bet_bins, device)
    model = UniformPolicyWrapper(_make_better_ffn(num_actions=cfg.model.num_actions))

    evaluator = SparseCFREvaluator(
        model=model,
        device=device,
        cfg=cfg,
        generator=torch.Generator(device=device),
    )
    evaluator.initialize_subgame(env, torch.tensor([0], device=device))

    assert isinstance(evaluator.feature_encoder, BetterFeatureEncoder)
