import torch

from p2.core.structured_config import Config
from p2.utils import training_utils
from p2.utils.training_utils import _resolve_search_iterations


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


def test_print_preflop_grid_uses_model_without_ema(monkeypatch, capsys):
    cfg = Config()
    trainer = type("TrainerStub", (), {})()
    trainer.model = object()
    trainer.ema_context = None
    trainer.cfg = cfg
    trainer.device = None
    trainer.rng = None
    trainer.num_bet_bins = 4

    captured = {}

    class _StubCfrEvaluator:
        def __init__(self, warm_start_iterations: int):
            self.warm_start_iterations = warm_start_iterations
            self.cfr_iterations = warm_start_iterations + 1

    class _StubAnalyzer:
        def __init__(self, model, cfg, button, device, rng, popart_normalizer=None):
            captured["model"] = model
            captured["analyzer"] = self
            self.cfr_evaluator = _StubCfrEvaluator(cfg.search.warm_start_iterations)

        def get_preflop_grids(self):
            ranges = ["fold\n", "call\n", "bet\n", "all-in\n"]
            suited_vs_offsuit = torch.tensor(
                [
                    [0.1, 0.2],
                    [0.3, 0.4],
                    [0.5, 0.6],
                    [0.7, 0.8],
                ],
                dtype=torch.float32,
            )
            return {
                "ranges": ranges,
                "betting": "bet\n",
                "value": "values",
                "suited_vs_offsuit": suited_vs_offsuit,
            }

        def get_preflop_grids_allin_response(self):
            return {"ranges": ["fold\n", "call\n"], "value": "values"}

    monkeypatch.setattr(training_utils, "RebelPreflopAnalyzer", _StubAnalyzer)

    training_utils.print_preflop_range_grid(trainer, step=0, rebel=True)
    captured_out = capsys.readouterr()
    assert "Preflop Range Grid" in captured_out.out
    assert captured["model"] is trainer.model
    analyzer = captured["analyzer"]
    # Warm start guard should not reduce the resolved iteration count
    assert analyzer.cfr_evaluator.cfr_iterations >= cfg.search.iterations
