import pytest
import torch

from p2.core.structured_config import (
    Config,
    PregeneratedDatasetConfig,
    PolicyLossType,
    PolicyNodeWeighting,
    ValueHeadType,
)
from p2.env.card_utils import (
    NUM_HANDS,
    combo_suit_permutation_inverse_tensor,
    mask_conflicting_combos,
    suit_permutations_tensor,
)
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.models.mlp.mlp_features import MLPFeatures
from p2.models.mlp.rebel_feature_encoder import RebelFeatureEncoder
from p2.models.model_output import ModelOutput
from p2.rl.cfr_trainer import RebelCFRTrainer, _value_samples_per_step
from p2.rl.losses import RebelSupervisedLoss
from p2.rl.rebel_batch import RebelBatch
from p2.search.rebel_solved_dataset import write_rebel_solved_dataset


def make_env(num_envs: int = 4) -> HUNLTensorEnv:
    env = HUNLTensorEnv(
        num_envs=num_envs,
        starting_stack=1000,
        sb=5,
        bb=10,
        device=torch.device("cpu"),
        float_dtype=torch.float32,
        flop_showdown=False,
    )
    env.reset()
    return env


def test_value_samples_per_step_allows_fractional_reuse_goal():
    assert _value_samples_per_step(batch_size=512, value_reuse_goal=0.5) == 1024
    assert _value_samples_per_step(batch_size=2048, value_reuse_goal=2.0) == 1024

    with pytest.raises(ValueError, match="value_reuse_goal"):
        _value_samples_per_step(batch_size=512, value_reuse_goal=0.0)


def test_rebel_cfr_trainer_rejects_unimplemented_hybrid_mode():
    cfg = Config(device="cpu")
    cfg.data.mode = "hybrid"

    with pytest.raises(NotImplementedError, match="data.mode=live or"):
        RebelCFRTrainer(cfg, torch.device("cpu"))


def _tiny_rebel_cfg() -> Config:
    cfg = Config()
    cfg.num_envs = 1
    cfg.train.batch_size = 1
    cfg.env.bet_bins = [0.5]
    cfg.model.name = "rebel_ffn"
    cfg.model.num_actions = len(cfg.env.bet_bins) + 3
    cfg.model.input_dim = 2661
    cfg.model.hidden_dim = 16
    cfg.model.num_hidden_layers = 1
    cfg.model.value_head_type = ValueHeadType.scalar
    cfg.search.depth = 0
    cfg.search.sparse = True
    cfg.search.iterations = 1
    cfg.search.warm_start_iterations = 0
    cfg.search.dcfr_plus_delay = 0
    return cfg


def _tiny_solved_batch(cfg: Config, *, stream: str, start: int, count: int) -> RebelBatch:
    rows = torch.arange(start, start + count)
    features = MLPFeatures(
        context=torch.stack(
            [rows.float(), rows.float() + 1.0, rows.float() + 2.0, rows.float() + 3.0],
            dim=1,
        ),
        street=rows.remainder(4).long(),
        to_act=rows.remainder(2).long(),
        board=torch.full((count, 5), -1, dtype=torch.long),
        beliefs=torch.full((count, 2 * NUM_HANDS), 1.0 / NUM_HANDS),
    )
    kwargs = {}
    if stream == "value":
        kwargs["value_targets"] = torch.zeros(count, 2, NUM_HANDS)
    else:
        policy_targets = torch.zeros(count, NUM_HANDS, cfg.model.num_actions)
        policy_targets[..., 1] = 1.0
        kwargs["policy_targets"] = policy_targets
    return RebelBatch(
        features=features,
        legal_masks=torch.ones(count, cfg.model.num_actions, dtype=torch.bool),
        statistics={"node_depth": torch.zeros(count, dtype=torch.long)},
        **kwargs,
    )


def test_rebel_cfr_trainer_runs_pregenerated_mode_one_step(tmp_path):
    cfg = _tiny_rebel_cfg()
    cfg.data.mode = "pregenerated"
    cfg.num_steps = 1
    cfg.train.batch_size = 2
    cfg.train.episodes_per_step = 1
    cfg.train.replay_buffer_batches = 2
    cfg.train.value_reuse_goal = 1
    cfg.data.pregenerated.value_batch_size = 2
    cfg.data.pregenerated.policy_batch_size = 2
    cfg.data.pregenerated.datasets = [PregeneratedDatasetConfig(path=str(tmp_path))]
    write_rebel_solved_dataset(
        tmp_path,
        value_batches=[_tiny_solved_batch(cfg, stream="value", start=0, count=4)],
        policy_batches=[_tiny_solved_batch(cfg, stream="policy", start=10, count=4)],
        metadata={
            "model_family": (
                cfg.model.name.value
                if hasattr(cfg.model.name, "value")
                else str(cfg.model.name)
            ),
            "action_schedule": {
                "bet_bins": list(cfg.env.bet_bins),
                "bet_bins_by_depth": cfg.search.bet_bins_by_depth,
                "allin_by_depth": cfg.search.allin_by_depth,
            },
        },
    )

    trainer = RebelCFRTrainer(cfg, torch.device("cpu"))
    metrics = trainer.train_step(0)

    assert metrics["step"] == 1
    assert torch.isfinite(torch.tensor(metrics["value_loss"]))
    assert torch.isfinite(torch.tensor(metrics["policy_loss"]))
    assert trainer.data_generator is None
    assert len(trainer.value_buffer) >= cfg.train.batch_size
    assert len(trainer.policy_buffer) >= cfg.train.batch_size


def test_rebel_cfr_trainer_pregenerated_resume_matches_uninterrupted(tmp_path):
    dataset_path = tmp_path / "dataset"
    cfg = _tiny_rebel_cfg()
    cfg.seed = 123
    cfg.data.mode = "pregenerated"
    cfg.num_steps = 2
    cfg.train.batch_size = 2
    cfg.train.episodes_per_step = 1
    cfg.train.replay_buffer_batches = 2
    cfg.train.value_reuse_goal = 1
    cfg.data.pregenerated.value_batch_size = 2
    cfg.data.pregenerated.policy_batch_size = 2
    cfg.data.pregenerated.datasets = [
        PregeneratedDatasetConfig(path=str(dataset_path))
    ]
    model_family = (
        cfg.model.name.value if hasattr(cfg.model.name, "value") else str(cfg.model.name)
    )
    action_schedule = {
        "bet_bins": list(cfg.env.bet_bins),
        "bet_bins_by_depth": cfg.search.bet_bins_by_depth,
        "allin_by_depth": cfg.search.allin_by_depth,
    }
    write_rebel_solved_dataset(
        dataset_path,
        value_batches=[_tiny_solved_batch(cfg, stream="value", start=0, count=8)],
        policy_batches=[_tiny_solved_batch(cfg, stream="policy", start=10, count=8)],
        metadata={
            "model_family": model_family,
            "action_schedule": action_schedule,
        },
    )

    device = torch.device("cpu")
    uninterrupted = RebelCFRTrainer(cfg, device)
    uninterrupted.train_step(0)
    uninterrupted_metrics = uninterrupted.train_step(1)

    resumed_first = RebelCFRTrainer(cfg, device)
    resumed_first.train_step(0)
    checkpoint_path = tmp_path / "rebel_latest.pt"
    resumed_first.save_checkpoint(str(checkpoint_path), step=0, save_optimizer=True)

    resumed_second = RebelCFRTrainer(cfg, device)
    loaded_step = resumed_second.load_checkpoint(str(checkpoint_path))
    resumed_metrics = resumed_second.train_step(loaded_step + 1)

    assert loaded_step == 0
    assert resumed_second.data_source.state_dict()["value_cursors"] == (
        uninterrupted.data_source.state_dict()["value_cursors"]
    )
    assert resumed_second.data_source.state_dict()["policy_cursors"] == (
        uninterrupted.data_source.state_dict()["policy_cursors"]
    )
    for key in ("loss", "value_loss", "policy_loss", "policy_target_model_kl"):
        assert resumed_metrics[key] == pytest.approx(uninterrupted_metrics[key])
    for lhs, rhs in zip(
        resumed_second.model.parameters(), uninterrupted.model.parameters()
    ):
        torch.testing.assert_close(lhs, rhs)


def test_rebel_cfr_trainer_wires_random_postflop_root_sources():
    expected_streets = {
        "random_flop": 1,
        "random_turn": 2,
        "random_river": 3,
        "random_flop_prefix": 1,
        "random_turn_prefix": 2,
        "random_river_prefix": 3,
    }

    for root_source, street in expected_streets.items():
        cfg = _tiny_rebel_cfg()
        cfg.data.live_root_source = root_source

        trainer = RebelCFRTrainer(cfg, torch.device("cpu"))

        assert trainer.data_generator.root_sampler is not None
        pbs = trainer.data_generator.root_sampler(2)
        assert pbs.env.N == 2
        assert torch.equal(
            pbs.env.street, torch.full((2,), street, dtype=torch.long)
        )
        if root_source.endswith("_prefix"):
            assert torch.equal(pbs.env.actions_this_round, torch.ones(2, dtype=torch.long))


def test_rebel_cfr_trainer_loads_closing_leaf_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "closing.pt"
    source_cfg = _tiny_rebel_cfg()
    source_trainer = RebelCFRTrainer(source_cfg, torch.device("cpu"))
    source_trainer.save_checkpoint(
        str(checkpoint_path), step=0, save_optimizer=False
    )

    cfg = _tiny_rebel_cfg()
    cfg.search.closing_leaf_checkpoint = str(checkpoint_path)
    trainer = RebelCFRTrainer(cfg, torch.device("cpu"))

    assert trainer.cfr_evaluator.closing_leaf_value_model is not None
    assert all(
        not param.requires_grad
        for param in trainer.cfr_evaluator.closing_leaf_value_model.parameters()
    )


def test_rebel_cfr_trainer_constructs_multiway_pbs_env():
    cfg = Config()
    cfg.num_envs = 1
    cfg.env.num_players = 3
    cfg.env.bet_bins = [0.5]
    cfg.search.depth = 1
    cfg.search.iterations = 1
    cfg.search.warm_start_iterations = 0
    cfg.search.sparse = True
    cfg.search.sparse_fused = False
    cfg.search.allin_call_terminal_abstraction = False
    cfg.train.batch_size = 1
    cfg.train.replay_buffer_batches = 1
    cfg.train.value_reuse_goal = 1.0
    cfg.train.policy_capacity_factor = 1.0
    cfg.model.hidden_dim = 16
    cfg.model.range_hidden_dim = 8
    cfg.model.ffn_dim = 48
    cfg.model.num_hidden_layers = 1
    cfg.model.num_policy_layers = 1
    cfg.model.num_value_layers = 1
    cfg.model.policy_rank = 8
    cfg.model.policy_hand_bias_rank = 4
    cfg.model.board_interaction_dim = 4
    cfg.model.num_actions = len(cfg.env.bet_bins) + 3
    cfg.model.enforce_zero_sum = True

    trainer = RebelCFRTrainer(cfg, torch.device("cpu"))

    assert trainer.num_players == 3
    assert trainer.env.num_players == 3
    assert cfg.model.enforce_zero_sum is False


def test_rebel_feature_encoder_shapes():
    env = make_env(2)
    encoder = RebelFeatureEncoder(env, device=env.device, dtype=torch.float32)
    idxs = torch.tensor([0, 1], device=env.device)
    beliefs = torch.full(
        (2, 2, NUM_HANDS), 1.0 / NUM_HANDS, dtype=torch.float32, device=env.device
    )
    mlp_features = encoder.encode(beliefs)
    features = mlp_features[idxs]

    # Verify features structure
    assert features.context.shape == (
        2,
        4,
    )  # to_act, position, pot_fraction, has_bet_flag
    assert features.street.shape == (2,)  # street indices
    assert features.board.shape == (2, 5)  # board indices
    assert features.beliefs.shape == (2, 2 * NUM_HANDS)  # beliefs

    # Verify beliefs sum to 1.0
    hero = features.beliefs[:, :NUM_HANDS]
    opp = features.beliefs[:, NUM_HANDS:]
    torch.testing.assert_close(hero.sum(dim=1), torch.ones(2, device=env.device))
    torch.testing.assert_close(opp.sum(dim=1), torch.ones(2, device=env.device))


def test_rebel_supervised_loss_finite():
    loss_fn = RebelSupervisedLoss()
    batch_size, num_actions = 3, 5
    logits = torch.randn(batch_size, NUM_HANDS, num_actions, requires_grad=True)
    policy_targets = torch.softmax(
        torch.randn(batch_size, NUM_HANDS, num_actions), dim=-1
    )
    legal_masks = torch.ones(batch_size, num_actions, dtype=torch.bool)
    values = torch.randn(batch_size, 2, NUM_HANDS)
    # Beliefs must be normalized probabilities
    beliefs_raw = torch.rand(batch_size, 2 * NUM_HANDS)
    beliefs = beliefs_raw / beliefs_raw.sum(dim=1, keepdim=True)
    mlp_features = MLPFeatures(
        context=torch.randn(batch_size, 4),
        street=torch.zeros(batch_size, dtype=torch.long),
        to_act=torch.zeros(batch_size, dtype=torch.long),
        board=torch.zeros(batch_size, 5),
        beliefs=beliefs,
    )
    batch = RebelBatch(
        features=mlp_features,
        policy_targets=policy_targets,
        value_targets=values,
        legal_masks=legal_masks,
    )
    output = ModelOutput(
        policy_logits=logits, value=torch.zeros(batch_size), hand_values=values
    )
    loss_dict = loss_fn(output, batch)
    assert torch.isfinite(loss_dict["total_loss"]).all()
    assert torch.isfinite(loss_dict["target_entropy"]).all()
    assert torch.isfinite(loss_dict["model_entropy"]).all()
    assert torch.isfinite(loss_dict["entropy_gap"]).all()
    assert loss_dict["target_entropy_all"].shape == (batch_size,)
    assert loss_dict["model_entropy_all"].shape == (batch_size,)
    assert loss_dict["entropy_gap_all"].shape == (batch_size,)
    assert torch.isfinite(loss_dict["target_model_kl"]).all()
    assert loss_dict["target_model_kl_all"].shape == (batch_size,)
    assert torch.isfinite(loss_dict["target_model_kl_all"]).all()
    torch.testing.assert_close(
        loss_dict["target_model_kl"],
        loss_dict["policy_loss"] - loss_dict["target_entropy"],
    )
    torch.testing.assert_close(
        loss_dict["entropy_gap"],
        loss_dict["model_entropy"] - loss_dict["target_entropy"],
    )
    loss_dict["total_loss"].backward()


def test_rebel_supervised_loss_multiway_finite():
    num_players = 3
    loss_fn = RebelSupervisedLoss(num_players=num_players)
    batch_size, num_actions = 2, 5
    logits = torch.randn(batch_size, NUM_HANDS, num_actions, requires_grad=True)
    policy_targets = torch.softmax(
        torch.randn(batch_size, NUM_HANDS, num_actions), dim=-1
    )
    legal_masks = torch.ones(batch_size, num_actions, dtype=torch.bool)
    value_targets = torch.randn(batch_size, num_players, NUM_HANDS)
    beliefs = torch.rand(batch_size, num_players, NUM_HANDS)
    beliefs = beliefs / beliefs.sum(dim=-1, keepdim=True)
    mlp_features = MLPFeatures(
        context=torch.randn(batch_size, 4),
        street=torch.zeros(batch_size, dtype=torch.long),
        to_act=torch.arange(batch_size, dtype=torch.long) % num_players,
        board=torch.full((batch_size, 5), -1, dtype=torch.long),
        beliefs=beliefs.reshape(batch_size, -1),
    )
    batch = RebelBatch(
        features=mlp_features,
        policy_targets=policy_targets,
        value_targets=value_targets,
        legal_masks=legal_masks,
    )
    output = ModelOutput(
        policy_logits=logits,
        value=torch.zeros(batch_size, num_players),
        hand_values=torch.randn(batch_size, num_players, NUM_HANDS),
    )

    loss_dict = loss_fn(output, batch)

    assert torch.isfinite(loss_dict["total_loss"])
    assert loss_dict["policy_weights"].shape == (batch_size, NUM_HANDS)
    assert loss_dict["value_weights"].shape == (batch_size, num_players, NUM_HANDS)
    loss_dict["total_loss"].backward()


def test_rebel_supervised_loss_zeros_board_blocked_weights():
    loss_fn = RebelSupervisedLoss()
    batch_size, num_actions = 1, 5
    board = torch.tensor([[0, 14, 28, -1, -1]])
    allowed = mask_conflicting_combos(board[0])
    beliefs = torch.zeros(batch_size, 2, NUM_HANDS)
    beliefs[:, :, allowed] = 1.0 / allowed.sum()
    mlp_features = MLPFeatures(
        context=torch.randn(batch_size, 4),
        street=torch.ones(batch_size, dtype=torch.long),
        to_act=torch.zeros(batch_size, dtype=torch.long),
        board=board,
        beliefs=beliefs.view(batch_size, -1),
    )
    policy_targets = torch.zeros(batch_size, NUM_HANDS, num_actions)
    policy_targets[..., 1] = 1.0
    legal_masks = torch.ones(batch_size, num_actions, dtype=torch.bool)
    values = torch.zeros(batch_size, 2, NUM_HANDS)
    output = ModelOutput(
        policy_logits=torch.zeros(batch_size, NUM_HANDS, num_actions),
        value=torch.zeros(batch_size),
        hand_values=values,
    )
    batch = RebelBatch(
        features=mlp_features,
        policy_targets=policy_targets,
        value_targets=torch.ones_like(values),
        legal_masks=legal_masks,
    )

    loss_dict = loss_fn(output, batch)

    assert torch.all(loss_dict["policy_weights"][0, ~allowed] == 0)
    assert torch.all(loss_dict["value_weights"][0, :, ~allowed] == 0)
    assert torch.all(loss_dict["policy_weights"][0, allowed] > 0)
    assert torch.all(loss_dict["value_weights"][0, :, allowed] > 0)


def test_rebel_policy_loss_has_saturated_wrong_action_gradient():
    loss_fn = RebelSupervisedLoss()
    batch_size, num_actions = 1, 2
    beliefs = torch.full((batch_size, 2, NUM_HANDS), 1.0 / NUM_HANDS)
    logits = torch.zeros(batch_size, NUM_HANDS, num_actions, requires_grad=True)
    logits.data[..., 0] = 10.0
    policy_targets = torch.zeros(batch_size, NUM_HANDS, num_actions)
    policy_targets[..., 1] = 1.0
    mlp_features = MLPFeatures(
        context=torch.randn(batch_size, 4),
        street=torch.zeros(batch_size, dtype=torch.long),
        to_act=torch.zeros(batch_size, dtype=torch.long),
        board=torch.full((batch_size, 5), -1, dtype=torch.long),
        beliefs=beliefs.view(batch_size, -1),
    )
    batch = RebelBatch(
        features=mlp_features,
        policy_targets=policy_targets,
        value_targets=None,
        legal_masks=torch.ones(batch_size, num_actions, dtype=torch.bool),
    )
    output = ModelOutput(
        policy_logits=logits,
        value=torch.zeros(batch_size),
        hand_values=None,
    )

    loss_dict = loss_fn(output, batch)
    loss_dict["total_loss"].backward()

    assert logits.grad[..., 0].mean() > 1e-4


def test_rebel_policy_logit_l2_is_centered_and_legal_only():
    batch_size, num_actions = 1, 3
    beliefs = torch.full((batch_size, 2, NUM_HANDS), 1.0 / NUM_HANDS)
    features = MLPFeatures(
        context=torch.randn(batch_size, 4),
        street=torch.zeros(batch_size, dtype=torch.long),
        to_act=torch.zeros(batch_size, dtype=torch.long),
        board=torch.full((batch_size, 5), -1, dtype=torch.long),
        beliefs=beliefs.view(batch_size, -1),
    )
    policy_targets = torch.zeros(batch_size, NUM_HANDS, num_actions)
    policy_targets[..., 0] = 1.0
    batch = RebelBatch(
        features=features,
        policy_targets=policy_targets,
        value_targets=None,
        legal_masks=torch.tensor([[True, True, False]]),
    )
    logits = torch.zeros(batch_size, NUM_HANDS, num_actions)
    logits[..., 0] = 12.0
    logits[..., 1] = 8.0
    logits[..., 2] = 1_000_000.0
    shifted_logits = logits.clone()
    shifted_logits[..., :2] += 123.0

    loss_fn = RebelSupervisedLoss(policy_logit_l2_coef=0.5)
    output = ModelOutput(policy_logits=logits, value=None, hand_values=None)
    shifted_output = ModelOutput(
        policy_logits=shifted_logits, value=None, hand_values=None
    )

    loss_dict = loss_fn.forward_policy(output, batch)
    shifted_loss_dict = loss_fn.forward_policy(shifted_output, batch)

    torch.testing.assert_close(loss_dict["policy_logit_l2"], torch.tensor(4.0))
    torch.testing.assert_close(
        shifted_loss_dict["policy_logit_l2"], loss_dict["policy_logit_l2"]
    )
    torch.testing.assert_close(
        loss_dict["total_loss"],
        loss_dict["policy_loss"] + 0.5 * loss_dict["policy_logit_l2"],
    )


def test_rebel_policy_loss_supports_node_reach_weighting_modes():
    batch_size, num_actions = 2, 2
    beliefs = torch.full((batch_size, 2, NUM_HANDS), 1.0 / NUM_HANDS)
    logits = torch.zeros(batch_size, NUM_HANDS, num_actions)
    logits[0, :, 0] = 10.0
    logits[1, :, 0] = 10.0
    policy_targets = torch.zeros(batch_size, NUM_HANDS, num_actions)
    policy_targets[0, :, 0] = 1.0
    policy_targets[1, :, 1] = 1.0
    features = MLPFeatures(
        context=torch.randn(batch_size, 4),
        street=torch.zeros(batch_size, dtype=torch.long),
        to_act=torch.zeros(batch_size, dtype=torch.long),
        board=torch.full((batch_size, 5), -1, dtype=torch.long),
        beliefs=beliefs.view(batch_size, -1),
    )
    batch = RebelBatch(
        features=features,
        policy_targets=policy_targets,
        value_targets=None,
        legal_masks=torch.ones(batch_size, num_actions, dtype=torch.bool),
        statistics={"policy_node_reach": torch.tensor([1.0, 0.01])},
    )
    output = ModelOutput(policy_logits=logits, value=None, hand_values=None)

    losses = {
        mode: RebelSupervisedLoss(policy_node_weighting=mode).forward_policy(
            output, batch
        )["policy_loss"]
        for mode in PolicyNodeWeighting
    }

    assert losses[PolicyNodeWeighting.reach] < losses[PolicyNodeWeighting.clipped_reach]
    assert (
        losses[PolicyNodeWeighting.clipped_reach]
        < losses[PolicyNodeWeighting.sqrt_reach]
    )
    assert losses[PolicyNodeWeighting.sqrt_reach] < losses[PolicyNodeWeighting.uniform]


def test_rebel_policy_loss_type_switches_objective_without_changing_kl_metric():
    batch_size, num_actions = 1, 2
    beliefs = torch.full((batch_size, 2, NUM_HANDS), 1.0 / NUM_HANDS)
    logits = torch.zeros(batch_size, NUM_HANDS, num_actions)
    policy_targets = torch.zeros(batch_size, NUM_HANDS, num_actions)
    policy_targets[..., 1] = 1.0
    features = MLPFeatures(
        context=torch.randn(batch_size, 4),
        street=torch.zeros(batch_size, dtype=torch.long),
        to_act=torch.zeros(batch_size, dtype=torch.long),
        board=torch.full((batch_size, 5), -1, dtype=torch.long),
        beliefs=beliefs.view(batch_size, -1),
    )
    batch = RebelBatch(
        features=features,
        policy_targets=policy_targets,
        value_targets=None,
        legal_masks=torch.ones(batch_size, num_actions, dtype=torch.bool),
    )
    output = ModelOutput(policy_logits=logits, value=None, hand_values=None)

    ce_out = RebelSupervisedLoss(
        policy_loss_type=PolicyLossType.cross_entropy
    ).forward_policy(output, batch)
    mse_out = RebelSupervisedLoss(policy_loss_type=PolicyLossType.mse).forward_policy(
        output, batch
    )

    torch.testing.assert_close(ce_out["target_model_kl"], mse_out["target_model_kl"])
    assert mse_out["policy_loss"] < ce_out["policy_loss"]


def test_cfr_policy_argmax_metrics_split_correct_and_wrong_top1():
    trainer = RebelCFRTrainer.__new__(RebelCFRTrainer)
    trainer.loss_fn = RebelSupervisedLoss()
    trainer.cfg = type("_Cfg", (), {})()
    trainer.cfg.search = type("_Search", (), {})()
    trainer.cfg.search.depth = 1

    metrics_fn = trainer._policy_argmax_metrics
    batch_size, num_actions = 2, 2
    beliefs = torch.full((batch_size, 2, NUM_HANDS), 1.0 / NUM_HANDS)
    features = MLPFeatures(
        context=torch.randn(batch_size, 4),
        street=torch.zeros(batch_size, dtype=torch.long),
        to_act=torch.zeros(batch_size, dtype=torch.long),
        board=torch.full((batch_size, 5), -1, dtype=torch.long),
        beliefs=beliefs.view(batch_size, -1),
    )
    legal_masks = torch.ones(batch_size, num_actions, dtype=torch.bool)
    targets = torch.zeros(batch_size, NUM_HANDS, num_actions)
    targets[0, :, 0] = 1.0
    targets[1, :, 1] = 1.0
    batch = RebelBatch(
        features=features,
        policy_targets=targets,
        legal_masks=legal_masks,
        statistics={"policy_node_reach": torch.tensor([9.0, 1.0])},
    )
    logits = torch.zeros(batch_size, NUM_HANDS, num_actions)
    logits[0, :, 0] = 2.0
    logits[1, :, 0] = 2.0
    output = ModelOutput(policy_logits=logits, value=None, hand_values=None)
    kl_all = torch.tensor([0.25, 1.25])

    metrics = metrics_fn(output, batch, kl_all)

    assert metrics["weighted_argmax_accuracy"] == pytest.approx(0.5)
    assert metrics["train_weighted_argmax_accuracy"] == pytest.approx(0.5)
    assert metrics["node_argmax_accuracy"] == pytest.approx(0.5)
    assert metrics["reach_weighted_argmax_accuracy"] == pytest.approx(0.9)
    assert metrics["mass_correct_top1"] == pytest.approx(0.5)
    assert metrics["mass_wrong_top1"] == pytest.approx(0.5)
    assert metrics["train_weight_frac_correct_top1"] == pytest.approx(0.5)
    assert metrics["train_weight_frac_wrong_top1"] == pytest.approx(0.5)
    assert metrics["node_frac_correct_top1"] == pytest.approx(0.5)
    assert metrics["node_frac_wrong_top1"] == pytest.approx(0.5)
    assert metrics["reach_mass_correct_top1"] == pytest.approx(0.9)
    assert metrics["reach_mass_wrong_top1"] == pytest.approx(0.1)
    assert metrics["kl_correct_top1"] == pytest.approx(0.25)
    assert metrics["kl_wrong_top1"] == pytest.approx(1.25)
    assert metrics["reach_kl_correct_top1"] == pytest.approx(0.25)
    assert metrics["reach_kl_wrong_top1"] == pytest.approx(1.25)


def test_cfr_policy_argmax_metrics_use_plurality_node_top1():
    trainer = RebelCFRTrainer.__new__(RebelCFRTrainer)
    trainer.loss_fn = RebelSupervisedLoss()

    batch_size, num_actions = 1, 3
    beliefs = torch.zeros(batch_size, 2, NUM_HANDS)
    # Actor hand weights are 0.45, 0.35, 0.20. Opponent is uniform, so
    # matchup blocking applies the same multiplier to each actor combo.
    beliefs[:, 0, 0] = 0.45
    beliefs[:, 0, 1] = 0.35
    beliefs[:, 0, 2] = 0.20
    beliefs[:, 1, :] = 1.0 / NUM_HANDS
    features = MLPFeatures(
        context=torch.randn(batch_size, 4),
        street=torch.zeros(batch_size, dtype=torch.long),
        to_act=torch.zeros(batch_size, dtype=torch.long),
        board=torch.full((batch_size, 5), -1, dtype=torch.long),
        beliefs=beliefs.view(batch_size, -1),
    )
    targets = torch.zeros(batch_size, NUM_HANDS, num_actions)
    targets[:, 0, 0] = 1.0
    targets[:, 1, 1] = 1.0
    targets[:, 2, 2] = 1.0
    targets[:, 3:, 0] = 1.0
    logits = torch.zeros(batch_size, NUM_HANDS, num_actions)
    logits[:, 0, 0] = 2.0
    logits[:, 1, 2] = 2.0
    logits[:, 2, 1] = 2.0
    logits[:, 3:, 0] = 2.0
    batch = RebelBatch(
        features=features,
        policy_targets=targets,
        legal_masks=torch.ones(batch_size, num_actions, dtype=torch.bool),
    )
    output = ModelOutput(policy_logits=logits, value=None, hand_values=None)

    metrics = trainer._policy_argmax_metrics(output, batch, torch.tensor([1.0]))

    assert metrics["weighted_argmax_accuracy"] == pytest.approx(0.45)
    assert metrics["node_argmax_accuracy"] == pytest.approx(0.45)
    assert metrics["node_frac_correct_top1"] == pytest.approx(1.0)
    assert "node_frac_wrong_top1" not in metrics


def test_rebel_cfr_trainer_single_step_cpu():
    cfg = Config()
    cfg.num_steps = 1
    cfg.num_envs = 1
    cfg.train.batch_size = 1
    cfg.train.episodes_per_step = 1
    cfg.train.replay_buffer_batches = 1
    cfg.train.value_reuse_goal = 2
    cfg.train.learning_rate = 3e-4
    cfg.train.value_coef = 1.0
    cfg.train.entropy_coef = 0.0
    cfg.train.grad_clip = 1.0
    cfg.train.max_sequence_length = 16
    cfg.env.bet_bins = [0.5, 1.0]
    cfg.env.stack = 1000
    cfg.env.sb = 5
    cfg.env.bb = 10
    cfg.env.flop_showdown = False
    cfg.model.name = "rebel_ffn"
    cfg.model.num_actions = len(cfg.env.bet_bins) + 3
    cfg.model.input_dim = 2661
    cfg.model.hidden_dim = 16
    cfg.model.num_hidden_layers = 1
    cfg.model.value_head_type = ValueHeadType.scalar
    cfg.model.detach_value_head = True
    cfg.search.enabled = True
    cfg.search.sparse = True
    cfg.search.depth = 0
    cfg.search.iterations = 1
    cfg.search.warm_start_iterations = 0
    cfg.search.dcfr_plus_delay = 0

    trainer = RebelCFRTrainer(cfg, torch.device("cpu"))
    batch_size = 1
    beliefs = torch.full((batch_size, 2, NUM_HANDS), 1.0 / NUM_HANDS)
    features = MLPFeatures(
        context=torch.zeros(batch_size, 4),
        street=torch.zeros(batch_size, dtype=torch.long),
        to_act=torch.zeros(batch_size, dtype=torch.long),
        board=torch.full((batch_size, 5), -1, dtype=torch.long),
        beliefs=beliefs.view(batch_size, -1),
    )
    legal_masks = torch.ones(batch_size, cfg.model.num_actions, dtype=torch.bool)
    policy_targets = torch.zeros(batch_size, NUM_HANDS, cfg.model.num_actions)
    policy_targets[..., 1] = 1.0
    value_targets = torch.zeros(batch_size, 2, NUM_HANDS)
    value_batch = RebelBatch(
        features=features,
        value_targets=value_targets,
        legal_masks=legal_masks,
    )
    policy_batch = RebelBatch(
        features=features.clone(),
        policy_targets=policy_targets,
        legal_masks=legal_masks,
    )
    suit_permutations = suit_permutations_tensor(device=torch.device("cpu"))[:1]
    permuted_batch, suit_permutation_idxs = value_batch.with_permuted_targets(
        suit_permutations=suit_permutations,
        num_players=trainer.num_players,
    )
    before = next(trainer.model.parameters()).detach().clone()

    stats, *_ = trainer._supervise(
        value_batch,
        policy_batch,
        permuted_batch,
        suit_permutation_idxs,
        value_latent=None,
        policy_latent=None,
        permuted_latent=None,
    )

    assert torch.isfinite(stats["policy_loss"])
    assert torch.isfinite(stats["value_loss"])
    assert torch.isfinite(stats["total_loss"])
    assert not torch.equal(before, next(trainer.model.parameters()).detach())

    before_policy_only = next(trainer.model.parameters()).detach().clone()
    policy_only_stats = trainer._supervise_policy_only(
        policy_batch,
        policy_latent=None,
    )

    assert torch.isfinite(policy_only_stats["policy_loss"])
    assert torch.isfinite(policy_only_stats["policy_target_model_kl"])
    assert torch.isfinite(policy_only_stats["total_loss"])
    assert not torch.equal(
        before_policy_only, next(trainer.model.parameters()).detach()
    )

    value_only_batch = RebelBatch(
        features=features.clone(),
        value_targets=torch.full_like(value_targets, 0.1),
        legal_masks=legal_masks,
    )
    before_value_only = [
        param.detach().clone() for param in trainer.model.parameters()
    ]
    value_only_stats = trainer.train_value_batch(value_only_batch, step=0)

    assert value_only_stats["policy_loss"] is None
    assert torch.isfinite(torch.tensor(value_only_stats["value_loss"]))
    assert torch.isfinite(torch.tensor(value_only_stats["total_loss"]))
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(before_value_only, trainer.model.parameters())
    )


def test_rebel_cfr_trainer_load_checkpoint_respects_non_strict(tmp_path):
    cfg = Config()
    cfg.num_envs = 1
    cfg.train.batch_size = 1
    cfg.env.bet_bins = [0.5, 1.0]
    cfg.model.name = "rebel_ffn"
    cfg.model.num_actions = len(cfg.env.bet_bins) + 3
    cfg.model.input_dim = 2661
    cfg.model.hidden_dim = 16
    cfg.model.num_hidden_layers = 1
    cfg.model.value_head_type = ValueHeadType.scalar
    cfg.search.depth = 0
    cfg.search.sparse = True
    cfg.search.iterations = 1
    cfg.search.warm_start_iterations = 0
    cfg.search.dcfr_plus_delay = 0
    cfg.strict_model_loading = False

    ckpt_path = tmp_path / "rebel.pt"
    trainer = RebelCFRTrainer(cfg, torch.device("cpu"))
    value_batch = RebelBatch(
        features=MLPFeatures(
            context=torch.arange(8, dtype=torch.float32).view(2, 4),
            street=torch.zeros(2, dtype=torch.long),
            to_act=torch.zeros(2, dtype=torch.long),
            board=torch.full((2, 5), -1, dtype=torch.long),
            beliefs=torch.full((2, 2 * NUM_HANDS), 1.0 / NUM_HANDS),
        ),
        policy_targets=None,
        value_targets=torch.arange(2 * 2 * NUM_HANDS, dtype=torch.float32).view(
            2, 2, NUM_HANDS
        ),
        legal_masks=torch.ones(2, cfg.model.num_actions, dtype=torch.bool),
    )
    policy_batch = RebelBatch(
        features=value_batch.features.clone(),
        policy_targets=torch.full(
            (2, NUM_HANDS, cfg.model.num_actions),
            1.0 / cfg.model.num_actions,
        ),
        value_targets=None,
        legal_masks=torch.ones(2, cfg.model.num_actions, dtype=torch.bool),
        statistics={"node_depth": torch.arange(2, dtype=torch.long)},
    )
    trainer.value_buffer.add_batch(value_batch)
    trainer.policy_buffer.add_batch(policy_batch)
    trainer.data_generator.current_pbs.env.street.fill_(2)
    trainer.data_generator.current_pbs.beliefs.fill_(0.25)
    trainer.data_generator.last_extra = 5
    trainer.save_checkpoint(
        str(ckpt_path),
        step=3,
        save_optimizer=False,
        metadata={"curriculum_substep": "river", "stage_step": 7},
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert checkpoint["metadata"] == {"curriculum_substep": "river", "stage_step": 7}
    checkpoint["model"]["unexpected.compat_key"] = torch.zeros(1)
    torch.save(checkpoint, ckpt_path)

    new_trainer = RebelCFRTrainer(cfg, torch.device("cpu"))
    loaded_step = new_trainer.load_checkpoint(str(ckpt_path))

    assert loaded_step == 3
    assert new_trainer.checkpoint_metadata == {
        "curriculum_substep": "river",
        "stage_step": 7,
    }
    replay_path = RebelCFRTrainer.replay_buffer_checkpoint_path(str(ckpt_path))
    assert (tmp_path / "rebel_replay_buffers.pt").samefile(replay_path)
    assert len(new_trainer.value_buffer) == len(trainer.value_buffer)
    assert len(new_trainer.policy_buffer) == len(trainer.policy_buffer)
    assert new_trainer.data_generator.last_extra == 5
    torch.testing.assert_close(
        new_trainer.data_generator.current_pbs.env.street,
        trainer.data_generator.current_pbs.env.street,
    )
    torch.testing.assert_close(
        new_trainer.data_generator.current_pbs.beliefs,
        trainer.data_generator.current_pbs.beliefs,
    )
    torch.testing.assert_close(
        new_trainer.value_buffer.features.context,
        trainer.value_buffer.features.context,
    )
    torch.testing.assert_close(
        new_trainer.value_buffer.value_targets,
        trainer.value_buffer.value_targets,
    )
    torch.testing.assert_close(
        new_trainer.policy_buffer.policy_targets,
        trainer.policy_buffer.policy_targets,
    )
    torch.testing.assert_close(
        new_trainer.policy_buffer.statistics["node_depth"],
        trainer.policy_buffer.statistics["node_depth"],
    )


def test_permutation_loss_echo_model():
    """
    Test that permutation loss is 0 for an "echo" model that outputs
    hand values = beliefs * torch.arange(1326).

    This verifies that the permutation loss logic correctly handles
    the case where the model commutes with suit permutations.

    The echo model outputs: hand_values[b, p, i] = beliefs[b, p, i] * i
    For the model to commute with permutations, when we permute the input beliefs,
    the permuted output should be: hand_values_permuted[b, p, j] = beliefs_permuted[b, p, j] * original_combo_index_for_j
    where original_combo_index_for_j is obtained via combo_suit_permutation_inverse_tensor.
    """
    device = torch.device("cpu")
    batch_size = 4
    num_actions = 5

    # Create an "echo" model: hand_values = beliefs * torch.arange(1326)
    combo_indices = torch.arange(NUM_HANDS, device=device, dtype=torch.float32)

    # Create features with beliefs = normalized torch.arange(1326)
    beliefs_flat = combo_indices.unsqueeze(0).expand(batch_size, -1).repeat(1, 2)
    beliefs_p0 = beliefs_flat[:, :NUM_HANDS]
    beliefs_p1 = beliefs_flat[:, NUM_HANDS:]
    beliefs_p0 = beliefs_p0 / beliefs_p0.sum(dim=1, keepdim=True)
    beliefs_p1 = beliefs_p1 / beliefs_p1.sum(dim=1, keepdim=True)
    beliefs = torch.cat([beliefs_p0, beliefs_p1], dim=1)

    features = MLPFeatures(
        context=torch.randn(batch_size, 4),
        street=torch.zeros(batch_size, dtype=torch.long),
        to_act=torch.zeros(batch_size, dtype=torch.long),
        board=torch.zeros(batch_size, 5, dtype=torch.long),
        beliefs=beliefs,
    )

    # Create "echo" model output: hand_values = beliefs * combo_indices
    beliefs_reshaped = beliefs.view(batch_size, 2, NUM_HANDS)
    hand_values = beliefs_reshaped * combo_indices.unsqueeze(0).unsqueeze(0)
    policy_logits = torch.randn(batch_size, NUM_HANDS, num_actions)
    value = hand_values.mean(dim=-1).mean(dim=-1)

    output = ModelOutput(
        policy_logits=policy_logits,
        value=value,
        hand_values=hand_values,
    )

    # Permute the features
    suit_permutations_idxs = torch.randint(0, 24, (batch_size,), device=device)
    suit_permutations = suit_permutations_tensor(device=device)[suit_permutations_idxs]
    features_permuted = features.clone()
    features_permuted.permute_suits(suit_permutations)

    # For the echo model to commute, permuted output should use original combo indices
    beliefs_permuted_reshaped = features_permuted.beliefs.view(batch_size, 2, NUM_HANDS)
    combo_permutations_inverse = combo_suit_permutation_inverse_tensor(device=device)[
        suit_permutations_idxs
    ]  # (batch_size, 1326)
    # combo_permutations_inverse[b, j] = original combo index that maps to permuted combo j

    # Echo model output for permuted features: use original combo indices
    hand_values_permuted = (
        beliefs_permuted_reshaped
        * combo_permutations_inverse.unsqueeze(1).expand(-1, 2, -1)
    )

    output_permuted = ModelOutput(
        policy_logits=policy_logits,
        value=hand_values_permuted.mean(dim=-1).mean(dim=-1),
        hand_values=hand_values_permuted,
    )

    # Create batch and compute loss
    # Need to provide dummy targets (not used in permutation loss computation)
    legal_masks = torch.ones(batch_size, num_actions, dtype=torch.bool)
    dummy_value_targets = torch.zeros(batch_size, 2, NUM_HANDS)
    batch = RebelBatch(
        features=features,
        policy_targets=None,
        value_targets=dummy_value_targets,
        legal_masks=legal_masks,
    )

    loss_fn = RebelSupervisedLoss(permutation_weight=1.0)
    loss_dict = loss_fn(
        output,
        batch,
        output_permuted=output_permuted,
        suit_permutation_idxs=suit_permutations_idxs,
    )

    # The permutation loss should be 0 (or very close to 0)
    permutation_loss = loss_dict["permutation_loss"]
    assert permutation_loss < 1e-6, (
        f"Permutation loss should be ~0, got {permutation_loss}"
    )
