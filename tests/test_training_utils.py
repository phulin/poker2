from alphaholdem.core.structured_config import Config
from alphaholdem.utils.training_utils import _resolve_search_iterations


class _TrainerStub:
    def __init__(
        self,
        cfg: Config,
        cfr_iterations=None,
        initial_iterations=None,
    ):
        self.cfg = cfg
        if initial_iterations is not None:
            self.initial_iterations = initial_iterations
        if cfr_iterations is not None:
            self.cfr_evaluator = type(
                "EvalStub", (), {"cfr_iterations": cfr_iterations}
            )


def test_resolve_iterations_prefers_override():
    cfg = Config()
    trainer = _TrainerStub(cfg, cfr_iterations=99)
    assert _resolve_search_iterations(trainer, iterations_override=5, step=10) == 5


def test_resolve_iterations_reads_trainer_cfr_evaluator():
    cfg = Config()
    trainer = _TrainerStub(cfg, cfr_iterations=42)
    assert _resolve_search_iterations(trainer, iterations_override=None, step=0) == 42


def test_resolve_iterations_falls_back_to_cfg_when_no_evaluator():
    cfg = Config()
    cfg.search.iterations = 7
    trainer = _TrainerStub(cfg)
    assert _resolve_search_iterations(trainer, iterations_override=None, step=0) == 7


def test_resolve_iterations_uses_interpolation_when_available():
    cfg = Config()
    cfg.num_steps = 100
    cfg.search.iterations = 10
    cfg.search.iterations_final = 50
    cfg.search.warm_start_iterations = 5
    trainer = _TrainerStub(cfg, initial_iterations=cfg.search.iterations)

    # Midway (step=50) should be roughly halfway between start and final.
    interpolated = _resolve_search_iterations(
        trainer, iterations_override=None, step=50
    )
    assert interpolated >= cfg.search.warm_start_iterations + 1
    assert interpolated == 30
