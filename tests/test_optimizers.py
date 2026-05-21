import pytest
import torch
import torch.nn as nn

from p2.core.structured_config import LrSchedule, TrainingConfig
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.rl.optimizers import SplitOptimizer, build_optimizer


def test_muon_optimizer_splits_matrix_params_from_other_params():
    if not hasattr(torch.optim, "Muon"):
        pytest.skip("torch.optim.Muon is not available")

    model = nn.Sequential(
        nn.Embedding(8, 4),
        nn.Linear(4, 3),
        nn.LayerNorm(3),
    )
    cfg = TrainingConfig(optimizer="muon", learning_rate=1e-3, weight_decay=0.01)

    optimizer = build_optimizer(model, cfg, torch.device("cpu"))

    assert isinstance(optimizer, SplitOptimizer)
    assert [name for name, _ in optimizer.optimizers] == ["muon", "adamw"]
    assert isinstance(optimizer.optimizers[0][1], torch.optim.Muon)
    assert isinstance(optimizer.optimizers[1][1], torch.optim.AdamW)
    assert optimizer.optimizers[0][1].param_groups[0]["params"][0] is model[1].weight
    assert set(optimizer.optimizers[1][1].param_groups[0]["params"]) == {
        model[0].weight,
        model[1].bias,
        model[2].weight,
        model[2].bias,
    }


def test_muon_optimizer_uses_separate_policy_head_lr():
    if not hasattr(torch.optim, "Muon"):
        pytest.skip("torch.optim.Muon is not available")

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = nn.Linear(4, 4)
            self.policy_tower = nn.Linear(4, 4)
            self.policy_head = nn.Sequential(nn.Linear(4, 3))
            self.value_head = nn.Linear(4, 2)

    model = Model()
    cfg = TrainingConfig(
        optimizer="muon",
        learning_rate=1e-3,
        policy_head_muon_learning_rate=0.05,
        weight_decay=0.01,
    )

    optimizer = build_optimizer(model, cfg, torch.device("cpu"))

    assert isinstance(optimizer, SplitOptimizer)
    assert [name for name, _ in optimizer.optimizers] == [
        "muon",
        "policy_head_muon",
        "adamw",
    ]
    assert set(optimizer.optimizers[0][1].param_groups[0]["params"]) == {
        model.trunk.weight,
        model.value_head.weight,
    }
    assert set(optimizer.optimizers[1][1].param_groups[0]["params"]) == {
        model.policy_tower.weight,
        model.policy_head[0].weight,
    }
    assert optimizer.optimizers[1][1].param_groups[0]["lr"] == 0.05
    assert optimizer.optimizers[1][1].param_groups[0]["lr_role"] == "policy_head_muon"


def test_cfr_schedule_preserves_policy_head_muon_lr():
    trainer = RebelCFRTrainer.__new__(RebelCFRTrainer)
    trainer.cfg = type("_Cfg", (), {})()
    trainer.cfg.num_steps = 100
    trainer.cfg.train = type("_Train", (), {})()
    trainer.cfg.train.learning_rate = 1e-3
    trainer.cfg.train.learning_rate_final = 1e-4
    trainer.cfg.train.lr_schedule = LrSchedule.linear
    trainer.cfg.train.warmup_steps = 0
    trainer.cfg.train.policy_head_muon_learning_rate = 0.05
    trainer.cfg.search = type("_Search", (), {})()
    trainer.cfg.search.iterations = 100
    trainer.cfg.search.iterations_final = None
    trainer.optimizer = type("_Opt", (), {})()
    trainer.optimizer.param_groups = [
        {"lr": 1e-3},
        {"lr": 0.05, "lr_role": "policy_head_muon"},
    ]
    trainer.cfr_evaluator = type("_Evaluator", (), {})()

    trainer._apply_schedules(50)

    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(5.5e-4)
    assert trainer.optimizer.param_groups[1]["lr"] == 0.05


def test_muon_split_optimizer_steps_matrix_and_non_matrix_params():
    if not hasattr(torch.optim, "Muon"):
        pytest.skip("torch.optim.Muon is not available")

    model = nn.Sequential(nn.Linear(4, 3), nn.LayerNorm(3))
    cfg = TrainingConfig(optimizer="muon", learning_rate=1e-3, weight_decay=0.0)
    optimizer = build_optimizer(model, cfg, torch.device("cpu"))
    before = {name: param.detach().clone() for name, param in model.named_parameters()}

    loss = model(torch.randn(8, 4)).square().mean()
    loss.backward()
    optimizer.step()

    for name, param in model.named_parameters():
        assert not torch.equal(before[name], param.detach())


def test_muon_split_optimizer_state_dict_round_trips():
    if not hasattr(torch.optim, "Muon"):
        pytest.skip("torch.optim.Muon is not available")

    cfg = TrainingConfig(optimizer="muon", learning_rate=1e-3, weight_decay=0.0)
    model = nn.Sequential(nn.Linear(4, 3), nn.LayerNorm(3))
    optimizer = build_optimizer(model, cfg, torch.device("cpu"))

    loss = model(torch.randn(8, 4)).square().mean()
    loss.backward()
    optimizer.step()

    new_model = nn.Sequential(nn.Linear(4, 3), nn.LayerNorm(3))
    new_optimizer = build_optimizer(new_model, cfg, torch.device("cpu"))
    new_optimizer.load_state_dict(optimizer.state_dict())

    assert optimizer.state_dict()["optimizer_order"] == ["muon", "adamw"]


def test_default_optimizer_is_adamw():
    model = nn.Linear(4, 3)
    cfg = TrainingConfig()

    optimizer = build_optimizer(model, cfg, torch.device("cpu"))

    assert isinstance(optimizer, torch.optim.AdamW)


def test_invalid_optimizer_name_raises():
    model = nn.Linear(4, 3)
    cfg = TrainingConfig(optimizer="rmsprop")

    with pytest.raises(ValueError, match="train.optimizer"):
        build_optimizer(model, cfg, torch.device("cpu"))
