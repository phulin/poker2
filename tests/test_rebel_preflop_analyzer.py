from __future__ import annotations

import torch

from alphaholdem.core.structured_config import (
    Config,
    EnvConfig,
    ModelConfig,
    SearchConfig,
)
from alphaholdem.env.analyze_tensor_env import RebelPreflopAnalyzer
from alphaholdem.models.mlp.better_ffn import BetterFFN
from alphaholdem.search.rebel_cfr_evaluator import T_WARM


def test_rebel_preflop_analyzer_returns_root_values() -> None:
    torch.manual_seed(0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)

    model = BetterFFN(
        num_actions=5,
        hidden_dim=64,
        range_hidden_dim=32,
        ffn_dim=128,
        num_hidden_layers=2,
        generator=generator,
    )
    model.eval()

    cfg = Config()
    cfg.env = EnvConfig()
    cfg.env.stack = 1000
    cfg.env.sb = 5
    cfg.env.bb = 10
    cfg.env.bet_bins = [0.5, 1.0]
    cfg.model = ModelConfig()
    cfg.search = SearchConfig()
    cfg.search.iterations = T_WARM + 1
    cfg.search.depth = 1

    analyzer = RebelPreflopAnalyzer(
        model=model,
        cfg=cfg,
        button=0,
        device=torch.device("cpu"),
        rng=generator,
    )

    _, values, _ = analyzer.get_probabilities(0)

    root_index = analyzer.current_index - 1
    expected = analyzer.cfr_evaluator.values_avg[root_index, 0]

    torch.testing.assert_close(values, expected, atol=1e-6, rtol=0.0)
    assert values.abs().max().item() <= 1.0 + 1e-6
