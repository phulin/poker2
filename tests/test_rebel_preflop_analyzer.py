import torch
from torch.testing import assert_close

from alphaholdem.core.structured_config import Config
from alphaholdem.env.analyze_tensor_env import RebelPreflopAnalyzer
from alphaholdem.env.card_utils import NUM_HANDS
from alphaholdem.models.mlp.rebel_ffn import RebelFFN


class _FakeCFREvaluator:
    """Minimal stub to exercise action remapping for preflop analyzer."""

    def __init__(self, device: torch.device):
        self.device = device
        self.num_actions = 5
        # depth_offsets: root at 0, four children at indices 1-4.
        self.depth_offsets = [0, 1, 5]

        # Action mapping skips action 2 (0.5x bet) to mirror sparse pruning.
        self.action_from_parent = torch.tensor(
            [-1, 0, 1, 3, 4], device=device, dtype=torch.long
        )

        # Populate policy_probs_avg with distinct masses so ordering is testable.
        self.policy_probs_avg = torch.zeros(
            5, NUM_HANDS, device=device, dtype=torch.float32
        )
        for idx, value in zip(range(1, 5), [0.1, 0.2, 0.3, 0.4]):
            self.policy_probs_avg[idx].fill_(value)

        self.valid_mask = torch.ones(5, device=device, dtype=torch.bool)

        # Root values for both players; only root (index 0) is read.
        self.values_avg = torch.zeros(
            5, 2, NUM_HANDS, device=device, dtype=torch.float32
        )
        self.values_avg[0].fill_(0.123)

    def initialize_subgame(self, *args, **kwargs):
        return None

    def evaluate_cfr(self, training_mode: bool = True):
        return None


def _make_small_config() -> Config:
    cfg = Config()
    cfg.model.hidden_dim = 8
    cfg.model.num_hidden_layers = 1
    cfg.model.detach_value_head = False
    cfg.env.bet_bins = [0.5, 1.0]
    cfg.model.num_actions = len(cfg.env.bet_bins) + 3
    return cfg


def test_rebel_preflop_analyzer_maps_sparse_actions_correctly():
    cfg = _make_small_config()
    device = torch.device("cpu")
    model = RebelFFN(
        input_dim=cfg.model.input_dim,
        num_actions=cfg.model.num_actions,
        hidden_dim=cfg.model.hidden_dim,
        num_hidden_layers=cfg.model.num_hidden_layers,
        detach_value_head=cfg.model.detach_value_head,
    )

    analyzer = RebelPreflopAnalyzer(
        model=model,
        cfg=cfg,
        device=device,
        rng=torch.Generator(device=device),
    )
    analyzer.cfr_evaluator = _FakeCFREvaluator(device)

    probs, values, legal_masks = analyzer.get_probabilities_from_cfr(seat=0)

    expected = torch.tensor([0.1, 0.2, 0.0, 0.3, 0.4])
    assert_close(probs[0], expected)
    assert_close(
        legal_masks[0],
        torch.tensor([True, True, False, True, True]),
    )
    assert_close(values[0], torch.tensor(0.123))
