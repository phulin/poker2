import math

import torch

from alphaholdem.rl.self_play import SelfPlayTrainer


class _Cfg:
    def __init__(self, num_steps: int):
        self.num_steps = num_steps


class _Train:
    def __init__(
        self,
        lr: float,
        lr_final: float,
        lr_schedule: str,
        ent_start: float,
        ent_final: float,
        ent_portion: float,
    ):
        self.learning_rate = lr
        self.learning_rate_final = lr_final
        self.lr_schedule = lr_schedule
        self.entropy_coef = ent_start
        self.entropy_coef_final = ent_final
        self.entropy_decay_portion = ent_portion


def _make_dummy_trainer(
    num_steps: int = 100,
    lr: float = 3e-4,
    lr_final: float = 1e-5,
    ent_start: float = 0.01,
    ent_final: float = 0.002,
    ent_portion: float = 0.6,
):
    trainer = SelfPlayTrainer.__new__(SelfPlayTrainer)

    # Minimal cfg with only fields used by _apply_schedules
    trainer.cfg = _Cfg(num_steps)
    trainer.cfg.train = _Train(
        lr, lr_final, "cosine", ent_start, ent_final, ent_portion
    )

    # Store originals as on trainer in __init__ normally
    trainer.learning_rate = lr
    trainer.learning_rate_final = lr_final
    trainer.lr_schedule = "cosine"
    trainer.entropy_coef_start = ent_start
    trainer.entropy_coef_final = ent_final
    trainer.entropy_decay_portion = ent_portion
    trainer.entropy_coef = ent_start

    # Optimizer stub with param groups and scales
    trainer._optimizer_group_scales = [0.1, 1.0]
    trainer.optimizer = type("_Opt", (), {})()
    trainer.optimizer.param_groups = [
        {"lr": lr * 0.1},
        {"lr": lr},
    ]

    return trainer


def test_cosine_lr_decay_monotonic_and_endpoints():
    num_steps = 100
    trainer = _make_dummy_trainer(num_steps=num_steps)

    lrs = []
    for step in range(0, num_steps + 1, 10):
        trainer._apply_schedules(step)
        lrs.append(trainer.optimizer.param_groups[-1]["lr"])  # main group lr

    # Start and end match configured values
    assert math.isclose(lrs[0], 3e-4, rel_tol=1e-6, abs_tol=1e-12)
    assert math.isclose(lrs[-1], 1e-5, rel_tol=1e-6, abs_tol=1e-12)

    # Cosine schedule is non-increasing over [0, 1]
    for a, b in zip(lrs, lrs[1:]):
        assert b <= a + 1e-12


def test_entropy_linear_decay_with_floor():
    num_steps = 100
    ent_start = 0.01
    ent_final = 0.002
    portion = 0.6
    trainer = _make_dummy_trainer(
        num_steps=num_steps,
        ent_start=ent_start,
        ent_final=ent_final,
        ent_portion=portion,
    )

    # Step 0: should be start
    trainer._apply_schedules(0)
    assert math.isclose(trainer.entropy_coef, ent_start, rel_tol=1e-9)

    # Midway through decay portion
    step_mid = int(num_steps * portion * 0.5)
    trainer._apply_schedules(step_mid)
    expected_mid = ent_start + (ent_final - ent_start) * 0.5
    assert math.isclose(trainer.entropy_coef, expected_mid, rel_tol=1e-6, abs_tol=1e-12)

    # At end of decay portion
    step_end_decay = int(num_steps * portion)
    trainer._apply_schedules(step_end_decay)
    assert math.isclose(trainer.entropy_coef, ent_final, rel_tol=1e-6, abs_tol=1e-12)

    # After decay portion: held at floor
    trainer._apply_schedules(num_steps)
    assert math.isclose(trainer.entropy_coef, ent_final, rel_tol=1e-6, abs_tol=1e-12)
