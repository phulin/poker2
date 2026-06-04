from __future__ import annotations

import hashlib
import json

import pytest
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
        self.evaluator = object()

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
    def __init__(
        self, cfg: Config, device: torch.device, pregeneration_only: bool = False
    ) -> None:
        assert pregeneration_only is True
        self.cfg = cfg
        self.device = device
        self.env = object()
        self.float_dtype = torch.float32
        self.model = _FakeSplitModel()
        self.data_generator = _FakeGenerator()


def test_pregenerate_postflop_rebel_writes_trimmed_solved_batches(monkeypatch, tmp_path):
    monkeypatch.setattr(pregenerate_cli, "RebelCFRTrainer", _FakeTrainer)
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

    assert manifest["value_examples"] == 3
    assert manifest["policy_examples"] == 4
    assert (tmp_path / "manifest.json").exists()
    disk_manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert disk_manifest["value_examples"] == 3
    assert disk_manifest["policy_examples"] == 4
    assert disk_manifest["shards"]["value"] == [
        {"file": "value/shard_000000.pt", "start": 0, "end": 2},
        {"file": "value/shard_000001.pt", "start": 2, "end": 3},
    ]
    assert disk_manifest["shards"]["policy"] == [
        {"file": "policy/shard_000000.pt", "start": 0, "end": 3},
        {"file": "policy/shard_000001.pt", "start": 3, "end": 4},
    ]
    spot_config = manifest["spot_sampler_config"]
    assert manifest["root_source"] == "random_river"
    assert manifest["root_source_codes"] == {"2": "random_river"}
    assert manifest["root_streets"] == ["river"]
    value_shard = torch.load(
        tmp_path / "value" / "shard_000001.pt",
        map_location="cpu",
        weights_only=True,
    )
    policy_shard = torch.load(
        tmp_path / "policy" / "shard_000001.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert torch.equal(value_shard["statistics.root_source"], torch.full((1,), 2))
    assert torch.equal(policy_shard["statistics.root_source"], torch.full((1,), 2))
    assert spot_config["live_root_source"] == "random_river"
    assert spot_config["board_texture_stratified"] is True
    assert "recursive_strength" in spot_config["belief_mixture_weights"]
    assert "straight_heavy" in spot_config["board_texture_weights"]
    assert manifest["target_model"] == {
        "role": "closing_leaf",
        "checkpoint": str(checkpoint),
        "sha256": hashlib.sha256(b"closing leaf checkpoint").hexdigest(),
    }
    assert manifest["feature_encoder"] == {
        "policy": {"model": "_FakeModelComponent", "encoder": "_FakeEncoder"},
        "value": {"model": "_FakeModelComponent", "encoder": "_FakeEncoder"},
    }
    assert manifest["quality"] == {
        "cfr_iterations": 17,
        "cfr_type": "linear",
        "cfr_plus": True,
        "sparse": True,
        "sparse_fused": False,
        "holdout_value_loss": None,
        "target_model_kl": None,
    }
    assert manifest["generator"]["code_version"] == "abc123"
    assert manifest["generator"]["code_dirty"] is False
    assert manifest["storage_float_dtype"] == "float16"


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


def test_policy_generation_cap_tracks_value_policy_ratio():
    cap = pregenerate_cli._max_policy_samples_for_generation(
        value_examples=0,
        policy_examples=0,
        value_target_min=200_000,
        policy_target_min=500_000,
        generation_batch_size=512,
    )
    assert cap == 1280

    final_cap = pregenerate_cli._max_policy_samples_for_generation(
        value_examples=199_900,
        policy_examples=499_700,
        value_target_min=200_000,
        policy_target_min=500_000,
        generation_batch_size=512,
    )
    assert final_cap == 250

    value_only_cap = pregenerate_cli._max_policy_samples_for_generation(
        value_examples=0,
        policy_examples=0,
        value_target_min=3_000_000,
        policy_target_min=0,
        generation_batch_size=512,
    )
    assert value_only_cap == 0


def test_pregenerate_postflop_rebel_prints_final_avg_exploitability(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(pregenerate_cli, "RebelCFRTrainer", _FakeTrainer)
    monkeypatch.setattr(
        pregenerate_cli,
        "_code_version_metadata",
        lambda: {"code_version": "abc123", "code_dirty": False},
    )
    monkeypatch.setattr(
        pregenerate_cli,
        "_final_average_policy_exploitability_mbbg",
        lambda evaluator: torch.tensor([10.0, 40.0]),
    )

    cfg = Config(device="cpu", use_wandb=False)
    cfg.data.mode = "live"
    cfg.rebel_pregenerate.output_dir = str(tmp_path)
    cfg.rebel_pregenerate.root_source = "random_river"
    cfg.rebel_pregenerate.value_target_min = 2
    cfg.rebel_pregenerate.policy_target_min = 2
    cfg.rebel_pregenerate.generation_batch_size = 2
    cfg.rebel_pregenerate.print_final_average_policy_exploitability = True

    manifest = pregenerate_cli.pregenerate_postflop_rebel(cfg)

    out = capsys.readouterr().out
    assert "Final average-policy exploitability" in out
    summary = manifest["quality"]["final_average_policy_exploitability_mbbg"]
    assert summary["count"] == 2
    assert summary["mean"] == 25.0
    assert summary["geomean"] == pytest.approx(20.0)
    assert summary["min"] == 10.0
    assert summary["max"] == 40.0
    disk_manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert (
        disk_manifest["quality"]["final_average_policy_exploitability_mbbg"]
        == summary
    )


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
