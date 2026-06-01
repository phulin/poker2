from __future__ import annotations

import hashlib

import torch

from p2.cli import pregenerate_postflop_rebel as pregenerate_cli
from p2.core.structured_config import Config
from p2.env.card_utils import NUM_HANDS
from p2.models.mlp.mlp_features import MLPFeatures
from p2.rl.rebel_batch import RebelBatch


def _batch(stream: str, start: int, count: int) -> RebelBatch:
    rows = torch.arange(start, start + count)
    features = MLPFeatures(
        context=torch.stack(
            [rows.float(), rows.float() + 1.0, rows.float() + 2.0, rows.float() + 3.0],
            dim=1,
        ),
        street=torch.full((count,), 3, dtype=torch.long),
        to_act=rows.remainder(2).long(),
        board=torch.full((count, 5), -1, dtype=torch.long),
        beliefs=torch.full((count, 2 * NUM_HANDS), 1.0 / NUM_HANDS),
    )
    kwargs = {}
    if stream == "value":
        kwargs["value_targets"] = torch.zeros(count, 2, NUM_HANDS)
    else:
        policy_targets = torch.zeros(count, NUM_HANDS, 5)
        policy_targets[..., 1] = 1.0
        kwargs["policy_targets"] = policy_targets
    return RebelBatch(
        features=features,
        legal_masks=torch.ones(count, 5, dtype=torch.bool),
        statistics={"node_depth": torch.zeros(count, dtype=torch.long)},
        **kwargs,
    )


class _FakeGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate_data(self, value_sample_count: int, **kwargs):
        self.calls += 1
        assert value_sample_count == 2
        value = (
            _batch("value", 10 * self.calls, 2)
            if kwargs["return_value_batch"]
            else None
        )
        policy = (
            _batch("policy", 10 * self.calls, 3)
            if kwargs["return_policy_batch"]
            else None
        )
        return value, policy


class _FakeEncoder:
    pass


class _FakeModelComponent(torch.nn.Module):
    def create_feature_encoder(self, env, device=None, dtype=None):
        del env, device, dtype
        return _FakeEncoder()


class _FakeSplitModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.policy_model = _FakeModelComponent()
        self.value_model = _FakeModelComponent()


class _FakeTrainer:
    def __init__(self, cfg: Config, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.env = object()
        self.float_dtype = torch.float32
        self.model = _FakeSplitModel()
        self.data_generator = _FakeGenerator()


def test_pregenerate_postflop_rebel_writes_trimmed_solved_batches(monkeypatch, tmp_path):
    written = {}

    def fake_write(
        output_dir,
        *,
        value_batches,
        policy_batches,
        metadata,
        storage_float_dtype=None,
    ):
        written["output_dir"] = output_dir
        written["value_batches"] = value_batches
        written["policy_batches"] = policy_batches
        written["metadata"] = metadata
        written["storage_float_dtype"] = storage_float_dtype
        return {
            "value_examples": sum(len(batch) for batch in value_batches),
            "policy_examples": sum(len(batch) for batch in policy_batches),
        }

    monkeypatch.setattr(pregenerate_cli, "RebelCFRTrainer", _FakeTrainer)
    monkeypatch.setattr(pregenerate_cli, "write_rebel_solved_dataset", fake_write)
    monkeypatch.setattr(
        pregenerate_cli,
        "_code_version_metadata",
        lambda: {"code_version": "abc123", "code_dirty": False},
    )

    cfg = Config(device="cpu", use_wandb=False)
    cfg.data.mode = "live"
    cfg.rebel_pregenerate.output_dir = str(tmp_path)
    cfg.rebel_pregenerate.root_source = "random_river"
    cfg.rebel_pregenerate.value_target_min = 3
    cfg.rebel_pregenerate.policy_target_min = 4
    cfg.rebel_pregenerate.generation_batch_size = 2
    cfg.rebel_pregenerate.max_generation_batches = 3
    cfg.rebel_pregenerate.storage_dtype = "float16"
    cfg.search.iterations = 17
    checkpoint = tmp_path / "closing.pt"
    checkpoint.write_bytes(b"closing leaf checkpoint")
    cfg.search.closing_leaf_checkpoint = str(checkpoint)

    manifest = pregenerate_cli.pregenerate_postflop_rebel(cfg)

    assert manifest == {"value_examples": 3, "policy_examples": 4}
    assert written["output_dir"] == str(tmp_path)
    assert sum(len(batch) for batch in written["value_batches"]) == 3
    assert sum(len(batch) for batch in written["policy_batches"]) == 4
    spot_config = written["metadata"]["spot_sampler_config"]
    assert written["metadata"]["root_source"] == "random_river"
    assert written["metadata"]["root_source_codes"] == {"2": "random_river"}
    assert written["metadata"]["root_streets"] == ["river"]
    for batch in written["value_batches"]:
        assert torch.equal(
            batch.statistics["root_source"],
            torch.full((len(batch),), 2, dtype=torch.long),
        )
    for batch in written["policy_batches"]:
        assert torch.equal(
            batch.statistics["root_source"],
            torch.full((len(batch),), 2, dtype=torch.long),
        )
    assert spot_config["live_root_source"] == "random_river"
    assert spot_config["board_texture_stratified"] is True
    assert "recursive_strength" in spot_config["belief_mixture_weights"]
    assert "straight_heavy" in spot_config["board_texture_weights"]
    assert written["metadata"]["target_model"] == {
        "role": "closing_leaf",
        "checkpoint": str(checkpoint),
        "sha256": hashlib.sha256(b"closing leaf checkpoint").hexdigest(),
    }
    assert written["metadata"]["feature_encoder"] == {
        "policy": {"model": "_FakeModelComponent", "encoder": "_FakeEncoder"},
        "value": {"model": "_FakeModelComponent", "encoder": "_FakeEncoder"},
    }
    assert written["metadata"]["quality"] == {
        "cfr_iterations": 17,
        "cfr_type": "linear",
        "cfr_plus": True,
        "sparse": True,
        "sparse_fused": False,
        "holdout_value_loss": None,
        "target_model_kl": None,
    }
    assert written["metadata"]["generator"]["code_version"] == "abc123"
    assert written["metadata"]["generator"]["code_dirty"] is False
    assert written["storage_float_dtype"] == "float16"


def test_code_version_metadata_records_commit_and_dirty_state(monkeypatch):
    def fake_git_text(args: list[str]) -> str | None:
        if args == ["rev-parse", "HEAD"]:
            return "commit-sha"
        if args == ["status", "--porcelain", "--untracked-files=no"]:
            return " M src/p2/cli/pregenerate_postflop_rebel.py"
        raise AssertionError(f"unexpected git args: {args}")

    monkeypatch.setattr(pregenerate_cli, "_git_text", fake_git_text)

    assert pregenerate_cli._code_version_metadata() == {
        "code_version": "commit-sha",
        "code_dirty": True,
    }


def test_target_model_metadata_records_distilled_source_checkpoint(tmp_path):
    source_checkpoint = tmp_path / "S_river.pt"
    source_checkpoint.write_bytes(b"frozen start net")
    closing_checkpoint = tmp_path / "E_turn.pt"
    torch.save(
        {
            "model": {},
            "metadata": {
                "curriculum_net": "E_turn",
                "curriculum_from_net": "S_river",
                "curriculum_source_checkpoint": str(source_checkpoint),
            },
        },
        closing_checkpoint,
    )

    cfg = Config(device="cpu", use_wandb=False)
    cfg.search.closing_leaf_checkpoint = str(closing_checkpoint)

    assert pregenerate_cli._target_model_metadata(cfg) == {
        "role": "closing_leaf",
        "checkpoint": str(closing_checkpoint),
        "sha256": hashlib.sha256(closing_checkpoint.read_bytes()).hexdigest(),
        "distilled_from_checkpoint": str(source_checkpoint),
        "distilled_from_sha256": hashlib.sha256(b"frozen start net").hexdigest(),
        "distilled_from_net": "S_river",
        "net": "E_turn",
    }


def test_pregenerate_postflop_rebel_requires_live_mode(tmp_path):
    cfg = Config(device="cpu", use_wandb=False)
    cfg.data.mode = "pregenerated"
    cfg.rebel_pregenerate.output_dir = str(tmp_path)

    try:
        pregenerate_cli.pregenerate_postflop_rebel(cfg)
    except ValueError as exc:
        assert "data.mode=live" in str(exc)
    else:
        raise AssertionError("Expected pregeneration to reject non-live mode")
