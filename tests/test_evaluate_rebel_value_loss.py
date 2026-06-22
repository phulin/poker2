from __future__ import annotations

import importlib.util
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from p2.core.structured_config import Config
from p2.models.mlp.better_ffn import BetterSplitFFN


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "evaluate_rebel_value_loss.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_rebel_value_loss_script", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeSplitLeaf(torch.nn.Linear):
    def __init__(self) -> None:
        super().__init__(1, 1)
        self.hidden_dim = 1
        self.num_players = 2
        self.num_actions = 2
        self.enforce_zero_sum = False


def _fake_model() -> BetterSplitFFN:
    return BetterSplitFFN(
        policy_model=_FakeSplitLeaf(),
        value_model=_FakeSplitLeaf(),
    )


class _FakeTrainer:
    def __init__(
        self, cfg: Config, device: torch.device, pregeneration_only: bool = False
    ) -> None:
        assert pregeneration_only is True
        self.cfg = cfg
        self.device = device
        self.float_dtype = torch.float32
        self.model = _fake_model()
        self.synced = False

    def _sync_inference_model(self) -> None:
        self.synced = True


class _FakeValidationEvaluator:
    def __init__(
        self,
        *,
        trainer,
        cfg,
        dataset_path: str,
        batch_size: int,
        max_examples: int | None = None,
    ) -> None:
        self.trainer = trainer
        self.cfg = cfg
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.max_examples = max_examples

    def evaluate(self) -> dict[str, float | int]:
        assert self.dataset_path == "/tmp/solved"
        assert self.batch_size == 321
        assert self.max_examples == 1000
        assert self.cfg.use_wandb is False
        assert self.cfg.model.compile == "off"
        return {
            "validation_examples": 1000,
            "validation_value_loss": 0.25,
            "validation_element_mean_weighted_square_error": 0.125,
            "validation_batch_loss_mean": 0.5,
            "validation_num_batches": 4,
        }


def test_config_rebel_evaluate_value_loss_resolves() -> None:
    with hydra.initialize_config_dir(
        config_dir=str(REPO_ROOT / "conf"),
        version_base=None,
    ):
        dict_config: DictConfig = hydra.compose(
            config_name="config_rebel_evaluate_value_loss",
            overrides=[
                "device=cpu",
                "resume_from=/tmp/model.pt",
                "validation_set.dataset=/tmp/solved",
            ],
        )

    cfg = Config.from_dict_config(dict_config)

    assert cfg.device == "cpu"
    assert cfg.use_wandb is False
    assert cfg.resume_from == "/tmp/model.pt"
    assert cfg.validation_set.dataset == "/tmp/solved"
    assert cfg.validation_set.batch_size == 1024
    assert cfg.model.compile == "off"


def test_evaluate_value_loss_ignores_checkpoint_embedded_config(
    monkeypatch, tmp_path
) -> None:
    module = _load_script_module()
    model = _fake_model()
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "step": 7,
            "config": {"model": {"policy_factor_scale": 2.0}},
        },
        checkpoint,
    )

    monkeypatch.setattr(module, "RebelCFRTrainer", _FakeTrainer)
    monkeypatch.setattr(module, "RebelValueValidationSetEvaluator", _FakeValidationEvaluator)
    monkeypatch.setattr(module, "setup_torch_runtime", lambda cfg, device: None)

    cfg = Config(
        device="cpu",
        resume_from=str(checkpoint),
        use_wandb=True,
    )
    cfg.validation_set.dataset = "/tmp/solved"
    cfg.validation_set.batch_size = 321
    cfg.validation_set.max_examples = 1000

    result = module.evaluate_value_loss(cfg)

    assert result == {
        "checkpoint": str(checkpoint),
        "dataset": "/tmp/solved",
        "device": "cpu",
        "examples": 1000,
        "batch_size": 321,
        "value_loss": 0.25,
        "element_mean_weighted_square_error": 0.125,
        "batch_loss_mean": 0.5,
        "num_batches": 4,
    }
