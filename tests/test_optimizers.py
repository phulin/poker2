import pytest
import torch
import torch.nn as nn

from p2.core.structured_config import LrSchedule, TrainingConfig
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.rl.optimizers import SplitOptimizer, build_optimizer


def _adamw_param_groups(optimizer: torch.optim.AdamW):
    decay_groups = [
        group for group in optimizer.param_groups if group.get("weight_decay", 0.0) > 0
    ]
    no_decay_groups = [
        group
        for group in optimizer.param_groups
        if group.get("weight_decay", 0.0) == 0.0
    ]
    return decay_groups, no_decay_groups


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
    adamw = optimizer.optimizers[1][1]
    decay_groups, no_decay_groups = _adamw_param_groups(adamw)
    assert set(decay_groups[0]["params"]) == {
        model[0].weight,
        model[1].bias,
    }
    assert set(no_decay_groups[0]["params"]) == {
        model[2].weight,
        model[2].bias,
    }
    assert all(group["lr_role"] == "adamw" for group in adamw.param_groups)


def test_adamw_optimizer_uses_separate_adamw_lr():
    model = nn.Sequential(nn.Linear(4, 3), nn.LayerNorm(3))
    cfg = TrainingConfig(
        optimizer="adamw",
        learning_rate=1e-3,
        adamw_learning_rate=3e-4,
        weight_decay=0.01,
    )

    optimizer = build_optimizer(model, cfg, torch.device("cpu"))

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 3e-4
    assert optimizer.param_groups[0]["lr_role"] == "adamw"
    _, no_decay_groups = _adamw_param_groups(optimizer)
    assert set(no_decay_groups[0]["params"]) == {
        model[1].weight,
        model[1].bias,
    }


def test_muon_optimizer_uses_separate_adamw_lr_for_all_adamw_params():
    if not hasattr(torch.optim, "Muon"):
        pytest.skip("torch.optim.Muon is not available")

    model = nn.Sequential(
        nn.Embedding(8, 4),
        nn.Linear(4, 3),
        nn.LayerNorm(3),
    )
    cfg = TrainingConfig(
        optimizer="muon",
        learning_rate=1e-3,
        adamw_learning_rate=3e-4,
        weight_decay=0.01,
    )

    optimizer = build_optimizer(model, cfg, torch.device("cpu"))

    assert isinstance(optimizer, SplitOptimizer)
    assert [name for name, _ in optimizer.optimizers] == ["muon", "adamw"]
    adamw = optimizer.optimizers[1][1]
    decay_groups, no_decay_groups = _adamw_param_groups(adamw)
    assert all(group["lr"] == 3e-4 for group in adamw.param_groups)
    assert all(group["lr_role"] == "adamw" for group in adamw.param_groups)
    assert set(decay_groups[0]["params"]) == {
        model[0].weight,
        model[1].bias,
    }
    assert set(no_decay_groups[0]["params"]) == {
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


def test_adamw_excludes_norm_params_from_weight_decay():
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4)
            self.norm = nn.RMSNorm(4)

    model = Model()
    cfg = TrainingConfig(optimizer="adamw", learning_rate=1e-3, weight_decay=0.01)

    optimizer = build_optimizer(model, cfg, torch.device("cpu"))

    assert isinstance(optimizer, torch.optim.AdamW)
    decay_groups, no_decay_groups = _adamw_param_groups(optimizer)
    assert set(decay_groups[0]["params"]) == {
        model.linear.weight,
        model.linear.bias,
    }
    assert set(no_decay_groups[0]["params"]) == {
        model.norm.weight,
    }


def test_cfr_schedule_scales_policy_head_muon_lr():
    trainer = RebelCFRTrainer.__new__(RebelCFRTrainer)
    trainer.cfg = type("_Cfg", (), {})()
    trainer.cfg.num_steps = 100
    trainer.cfg.train = type("_Train", (), {})()
    trainer.cfg.train.learning_rate = 1e-3
    trainer.cfg.train.learning_rate_final = 1e-4
    trainer.cfg.train.lr_schedule = LrSchedule.linear
    trainer.cfg.train.warmup_steps = 0
    trainer.cfg.train.policy_head_muon_learning_rate = 0.05
    trainer.cfg.train.adamw_learning_rate = 2e-4
    trainer.cfg.search = type("_Search", (), {})()
    trainer.cfg.search.iterations = 100
    trainer.cfg.search.iterations_final = None
    trainer.optimizer = type("_Opt", (), {})()
    trainer.optimizer.param_groups = [
        {"lr": 1e-3},
        {"lr": 0.05, "lr_role": "policy_head_muon"},
        {"lr": 2e-4, "lr_role": "adamw"},
    ]
    trainer.cfr_evaluator = type("_Evaluator", (), {})()

    trainer._apply_schedules(50)

    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(5.5e-4)
    assert trainer.optimizer.param_groups[1]["lr"] == pytest.approx(0.0275)
    assert trainer.optimizer.param_groups[2]["lr"] == pytest.approx(1.1e-4)


def test_cfr_train_step_logs_schedule_and_cfr_iterations():
    trainer = RebelCFRTrainer.__new__(RebelCFRTrainer)
    trainer._apply_schedules = lambda step: None
    trainer._update_model = lambda step: {"loss": 0.0}
    trainer.model = nn.Module()
    trainer.optimizer = type("_Opt", (), {})()
    trainer.optimizer.param_groups = [{"lr": 1e-3}]
    trainer.cfr_evaluator = type("_Evaluator", (), {"cfr_iterations": 400})()

    metrics = trainer.train_step(7)

    assert metrics["step"] == 8
    assert "policy_factor_scale" not in metrics
    assert metrics["cfr_iterations"] == 400


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
