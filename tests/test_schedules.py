import math

from p2.core.structured_config import LrSchedule
from p2.rl.exponential_controller import ExponentialController
from p2.rl.self_play import SelfPlayTrainer
from p2.utils.ema import EMA


class _Cfg:
    def __init__(self, num_steps: int):
        self.num_steps = num_steps


class _Train:
    def __init__(
        self,
        lr: float,
        lr_final: float,
        lr_schedule: LrSchedule,
        ent_start: float,
        ent_final: float,
        ent_portion: float,
        value_head_lr: float = None,
        value_head_lr_final: float = None,
        policy_trunk_lr: float = None,
        policy_trunk_lr_final: float = None,
    ):
        self.learning_rate = lr
        self.learning_rate_final = lr_final
        self.lr_schedule = lr_schedule
        self.entropy_coef = ent_start
        self.entropy_coef_final = ent_final
        self.entropy_decay_portion = ent_portion
        # Use provided values or fall back to main learning rate
        self.value_head_learning_rate = (
            value_head_lr if value_head_lr is not None else lr
        )
        self.value_head_learning_rate_final = (
            value_head_lr_final if value_head_lr_final is not None else lr_final
        )
        self.policy_trunk_learning_rate = (
            policy_trunk_lr if policy_trunk_lr is not None else lr
        )
        self.policy_trunk_learning_rate_final = (
            policy_trunk_lr_final if policy_trunk_lr_final is not None else lr_final
        )


def _make_dummy_trainer(
    num_steps: int = 100,
    lr: float = 3e-4,
    lr_final: float = 1e-5,
    ent_start: float = 0.01,
    ent_final: float = 0.002,
    ent_portion: float = 0.6,
    value_head_lr: float = None,
    value_head_lr_final: float = None,
    policy_trunk_lr: float = None,
    policy_trunk_lr_final: float = None,
):
    trainer = SelfPlayTrainer.__new__(SelfPlayTrainer)

    # Minimal cfg with only fields used by _apply_schedules
    trainer.cfg = _Cfg(num_steps)
    trainer.cfg.train = _Train(
        lr,
        lr_final,
        "cosine",
        ent_start,
        ent_final,
        ent_portion,
        value_head_lr,
        value_head_lr_final,
        policy_trunk_lr,
        policy_trunk_lr_final,
    )

    # Store originals as on trainer in __init__ normally
    trainer.learning_rate = lr
    trainer.learning_rate_final = lr_final
    trainer.lr_schedule = "cosine"
    trainer.entropy_coef_start = ent_start
    trainer.entropy_coef_final = ent_final
    trainer.entropy_decay_portion = ent_portion
    trainer.entropy_coef = ent_start

    # Store separate learning rates
    trainer.value_head_learning_rate = (
        value_head_lr if value_head_lr is not None else lr
    )
    trainer.value_head_learning_rate_final = (
        value_head_lr_final if value_head_lr_final is not None else lr_final
    )
    trainer.policy_trunk_learning_rate = (
        policy_trunk_lr if policy_trunk_lr is not None else lr
    )
    trainer.policy_trunk_learning_rate_final = (
        policy_trunk_lr_final if policy_trunk_lr_final is not None else lr_final
    )

    # Optimizer stubs with separate optimizers
    trainer.policy_trunk_optimizer = type("_PolicyOpt", (), {})()
    trainer.policy_trunk_optimizer.param_groups = [
        {"lr": trainer.policy_trunk_learning_rate},  # policy/trunk
    ]

    trainer.value_head_optimizer = type("_ValueOpt", (), {})()
    trainer.value_head_optimizer.param_groups = [
        {"lr": trainer.value_head_learning_rate},  # value_head
    ]

    trainer.kl_ema = EMA()
    trainer.lr_scaling_controller = ExponentialController(1.0, 1.0, 1.0, 1.0)

    return trainer


def test_cosine_lr_decay_monotonic_and_endpoints():
    num_steps = 100
    trainer = _make_dummy_trainer(num_steps=num_steps)
    # Initialize minimal KL EMA fields for schedule
    trainer.kl_ema.initialized = False

    policy_lrs = []
    value_lrs = []
    for step in range(0, num_steps + 1, 10):
        trainer._apply_schedules(step)
        policy_lrs.append(
            trainer.policy_trunk_optimizer.param_groups[0]["lr"]
        )  # policy/trunk group lr
        value_lrs.append(
            trainer.value_head_optimizer.param_groups[0]["lr"]
        )  # value head group lr

    # Start and end match configured values for both groups
    assert math.isclose(policy_lrs[0], 3e-4, rel_tol=1e-6, abs_tol=1e-12)
    assert math.isclose(policy_lrs[-1], 1e-5, rel_tol=1e-6, abs_tol=1e-12)
    assert math.isclose(value_lrs[0], 3e-4, rel_tol=1e-6, abs_tol=1e-12)
    assert math.isclose(value_lrs[-1], 1e-5, rel_tol=1e-6, abs_tol=1e-12)

    # Cosine schedule is non-increasing over [0, 1] for both groups
    for a, b in zip(policy_lrs, policy_lrs[1:]):
        assert b <= a + 1e-12
    for a, b in zip(value_lrs, value_lrs[1:]):
        assert b <= a + 1e-12


def test_separate_learning_rates():
    """Test that value head and policy/trunk can have different learning rates."""
    num_steps = 100
    trainer = _make_dummy_trainer(
        num_steps=num_steps,
        lr=3e-4,  # main LR (fallback)
        lr_final=1e-5,  # main LR final (fallback)
        value_head_lr=1e-4,  # separate value head LR
        value_head_lr_final=5e-6,  # separate value head LR final
        policy_trunk_lr=2e-4,  # separate policy/trunk LR
        policy_trunk_lr_final=8e-6,  # separate policy/trunk LR final
    )
    trainer.kl_ema.initialized = False

    # Test initial values
    trainer._apply_schedules(0)
    assert math.isclose(
        trainer.policy_trunk_optimizer.param_groups[0]["lr"],
        2e-4,
        rel_tol=1e-6,
        abs_tol=1e-12,
    )
    assert math.isclose(
        trainer.value_head_optimizer.param_groups[0]["lr"],
        1e-4,
        rel_tol=1e-6,
        abs_tol=1e-12,
    )

    # Test final values
    trainer._apply_schedules(num_steps)
    assert math.isclose(
        trainer.policy_trunk_optimizer.param_groups[0]["lr"],
        8e-6,
        rel_tol=1e-6,
        abs_tol=1e-12,
    )
    assert math.isclose(
        trainer.value_head_optimizer.param_groups[0]["lr"],
        5e-6,
        rel_tol=1e-6,
        abs_tol=1e-12,
    )


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
    trainer.kl_ema.initialized = False

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
