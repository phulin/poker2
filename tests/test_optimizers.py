import pytest
import torch
import torch.nn as nn

from p2.core.structured_config import Config, LrSchedule, TrainingConfig
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.rl.optimizers import (
    NorMuon,
    SplitOptimizer,
    _normuon_matrix_update,
    build_optimizer,
)


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


def test_normuon_optimizer_splits_matrix_params_from_other_params():
    model = nn.Sequential(
        nn.Embedding(8, 4),
        nn.Linear(4, 3),
        nn.LayerNorm(3),
    )
    cfg = TrainingConfig(
        optimizer="normuon",
        learning_rate=1e-3,
        adamw_learning_rate=3e-4,
        weight_decay=0.01,
        normuon_beta1=0.9,
        normuon_beta2=0.99,
        normuon_eps=1e-6,
    )

    optimizer = build_optimizer(model, cfg, torch.device("cpu"))

    assert isinstance(optimizer, SplitOptimizer)
    assert [name for name, _ in optimizer.optimizers] == ["normuon", "adamw"]
    assert isinstance(optimizer.optimizers[0][1], NorMuon)
    normuon_group = optimizer.optimizers[0][1].param_groups[0]
    assert normuon_group["params"][0] is model[1].weight
    assert normuon_group["beta1"] == 0.9
    assert normuon_group["beta2"] == 0.99
    assert normuon_group["eps"] == 1e-6
    adamw = optimizer.optimizers[1][1]
    decay_groups, no_decay_groups = _adamw_param_groups(adamw)
    assert all(group["lr"] == 3e-4 for group in adamw.param_groups)
    assert set(decay_groups[0]["params"]) == {
        model[0].weight,
        model[1].bias,
    }
    assert set(no_decay_groups[0]["params"]) == {
        model[2].weight,
        model[2].bias,
    }


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
    trainer.cfg = Config()
    trainer.cfg.num_steps = 100
    trainer.cfg.train.learning_rate = 1e-3
    trainer.cfg.train.learning_rate_final = 1e-4
    trainer.cfg.train.lr_schedule = LrSchedule.linear
    trainer.cfg.train.warmup_steps = 0
    trainer.cfg.train.policy_head_muon_learning_rate = 0.05
    trainer.cfg.train.adamw_learning_rate = 2e-4
    trainer.cfg.search.iterations = 100
    trainer.cfg.search.iterations_final = None
    trainer.num_players = 2
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
    trainer.cfr_target_model = None
    trainer.target_update_block_batches = 0
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


def test_normuon_split_optimizer_steps_matrix_and_non_matrix_params():
    model = nn.Sequential(nn.Linear(4, 3), nn.LayerNorm(3))
    cfg = TrainingConfig(optimizer="normuon", learning_rate=1e-3, weight_decay=0.0)
    optimizer = build_optimizer(model, cfg, torch.device("cpu"))
    before = {name: param.detach().clone() for name, param in model.named_parameters()}

    loss = model(torch.randn(8, 4)).square().mean()
    loss.backward()
    optimizer.step()

    for name, param in model.named_parameters():
        assert not torch.equal(before[name], param.detach())


def test_normuon_split_optimizer_state_dict_round_trips():
    cfg = TrainingConfig(optimizer="normuon", learning_rate=1e-3, weight_decay=0.0)
    model = nn.Sequential(nn.Linear(4, 3), nn.LayerNorm(3))
    optimizer = build_optimizer(model, cfg, torch.device("cpu"))

    loss = model(torch.randn(8, 4)).square().mean()
    loss.backward()
    optimizer.step()

    new_model = nn.Sequential(nn.Linear(4, 3), nn.LayerNorm(3))
    new_optimizer = build_optimizer(new_model, cfg, torch.device("cpu"))
    new_optimizer.load_state_dict(optimizer.state_dict())

    assert optimizer.state_dict()["optimizer_order"] == ["normuon", "adamw"]


def test_normuon_matrix_update_is_torch_compileable():
    compiled_update = torch.compile(_normuon_matrix_update, fullgraph=True)
    param = torch.randn(3, 4)
    grad = torch.randn_like(param)
    momentum = torch.zeros_like(param)
    row_second_moment = torch.zeros(param.shape[0])
    scalar = torch.tensor(1.0e-3)

    next_param, next_momentum, next_row_second_moment = compiled_update(
        param,
        grad,
        momentum,
        row_second_moment,
        scalar,
        torch.tensor(0.01),
        torch.tensor(0.95),
        torch.tensor(0.99),
        torch.tensor(1.0e-8),
        torch.tensor(1.0e-7),
        5,
    )

    assert next_param.shape == param.shape
    assert next_momentum.shape == momentum.shape
    assert next_row_second_moment.shape == row_second_moment.shape
    assert torch.isfinite(next_param).all()


def test_normuon_update_scale_matches_pytorch_muon_original_mode():
    for shape in [(256, 8), (512, 512), (512, 1024), (1, 512)]:
        torch.manual_seed(123)
        param = torch.randn(shape)
        grad = torch.randn_like(param)
        momentum = torch.zeros_like(param)
        row_second_moment = torch.zeros(shape[0])

        next_param, _, _ = _normuon_matrix_update(
            param,
            grad,
            momentum,
            row_second_moment,
            torch.tensor(1.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(1.0e-8),
            torch.tensor(1.0e-7),
            5,
        )
        normuon_update_rms = (next_param - param).square().mean().sqrt()

        muon_param = torch.nn.Parameter(param.detach().clone())
        muon_param.grad = grad.detach().clone()
        muon = torch.optim.Muon(
            [muon_param],
            lr=1.0,
            weight_decay=0.0,
            momentum=0.0,
            nesterov=False,
            eps=1.0e-7,
            ns_steps=5,
            adjust_lr_fn=None,
        )
        muon.step()
        muon_update_rms = (muon_param.detach() - param).square().mean().sqrt()

        torch.testing.assert_close(
            normuon_update_rms,
            muon_update_rms,
            rtol=0.25,
            atol=1.0e-5,
        )


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
