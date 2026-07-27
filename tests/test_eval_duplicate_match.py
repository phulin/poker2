"""CPU-only tests for real-hand duplicate evaluation.

No GPU and no trained model: everything here uses the scripted agents, which by
design need neither a model, an evaluator, nor beliefs.
"""

from __future__ import annotations

import json

import pytest
import torch

from p2.env.card_utils import combo_index, combo_lookup_tensor
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.eval.agents import AgentIdentity, CallAgent, FoldAgent, RandomAgent
from p2.eval.duplicate_match import (
    DECK_CARDS,
    MAX_DECISIONS,
    build_mirrored_decks,
    play_duplicate_match,
)
from p2.eval.records import (
    GameBatchTensors,
    GameRecord,
    RecordWriter,
    load_manifest,
    load_records,
    pair_differences,
)

DEVICE = torch.device("cpu")


def make_env(num_envs: int = 1, **kwargs) -> HUNLTensorEnv:
    rng = torch.Generator(device=DEVICE)
    rng.manual_seed(0)
    return HUNLTensorEnv(
        num_envs=num_envs,
        starting_stack=10_000,
        sb=50,
        bb=100,
        device=DEVICE,
        rng=rng,
        **kwargs,
    )


# ------------------------------------------------------------- mirrored decks


def test_build_mirrored_decks_swaps_holes_and_keeps_board():
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(7)
    decks = build_mirrored_decks(6, generator, DEVICE)

    assert decks.shape == (12, DECK_CARDS)
    first = decks[0::2]
    second = decks[1::2]
    # Hole cards swapped between the seats.
    assert torch.equal(first[:, 0:2], second[:, 2:4])
    assert torch.equal(first[:, 2:4], second[:, 0:2])
    # Board (flop/turn/river) identical.
    assert torch.equal(first[:, 4:9], second[:, 4:9])
    # Every game deals 9 distinct cards.
    for row in decks:
        assert len(set(row.tolist())) == DECK_CARDS


def test_env_reset_realizes_mirrored_pairs():
    """Game 2i and 2i+1 really have swapped holes and an identical board."""
    num_pairs = 8
    env = make_env(2 * num_pairs)
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(11)
    decks = build_mirrored_decks(num_pairs, generator, DEVICE)
    buttons = torch.arange(2 * num_pairs, device=DEVICE) % 2
    env.reset(force_button=buttons, force_deck=decks)

    holes = env.hole_indices  # [G, 2 players, 2 cards]
    even, odd = holes[0::2], holes[1::2]
    assert torch.equal(even[:, 0, :], odd[:, 1, :])
    assert torch.equal(even[:, 1, :], odd[:, 0, :])
    # The 5 board cards live in deck slots 4..8 and are the same for both halves.
    assert torch.equal(env.deck[0::2, 4:9], env.deck[1::2, 4:9])
    # And the forced deck is exactly what we asked for.
    assert torch.equal(env.deck, decks)


def test_mirrored_stacks_match_within_pair_for_random_stacks():
    env_proto = make_env(1, stack_mode="fixed_total_random_split")
    result = play_duplicate_match(
        CallAgent(), CallAgent(), env_proto, num_pairs=8, seed=3, device=DEVICE
    )
    stacks = torch.tensor(
        [[r.effective_stack for r in result.records]], dtype=torch.long
    ).view(-1)
    assert torch.equal(stacks[0::2], stacks[1::2])


# -------------------------------------------------------------------- agents


def test_fold_agent_folds_when_legal_and_checks_otherwise():
    env = make_env(2)
    env.reset()
    _, legal = env.legal_bins_amounts_and_mask()
    agent = FoldAgent()
    active = torch.arange(2)
    hands = torch.zeros(2, dtype=torch.long)

    legal_facing_bet = legal.clone()
    legal_facing_bet[:, 0] = True
    probs = agent.action_probs(env, active, hands, legal_facing_bet)
    assert torch.equal(probs.argmax(dim=1), torch.zeros(2, dtype=torch.long))

    legal_no_bet = legal.clone()
    legal_no_bet[:, 0] = False
    probs = agent.action_probs(env, active, hands, legal_no_bet)
    assert torch.equal(probs.argmax(dim=1), torch.ones(2, dtype=torch.long))


def test_call_and_random_agents_are_legal_and_model_free():
    env = make_env(4)
    env.reset()
    _, legal = env.legal_bins_amounts_and_mask()
    active = torch.arange(4)
    hands = torch.zeros(4, dtype=torch.long)

    call_probs = CallAgent().action_probs(env, active, hands, legal)
    assert torch.equal(call_probs.argmax(dim=1), torch.ones(4, dtype=torch.long))

    random_probs = RandomAgent().action_probs(env, active, hands, legal)
    # Mass only on legal bins, and it is uniform over them.
    assert (random_probs[~legal] == 0).all()
    assert (random_probs[legal] == 1).all()

    for agent in (FoldAgent(), CallAgent(), RandomAgent()):
        assert agent.search_fidelity() == {}
        assert agent.identity.kind == "scripted"
        assert not hasattr(agent, "evaluator")


def test_fold_agent_always_loses_the_blinds():
    """A folding agent vs a caller: real chips, no belief scoring."""
    env_proto = make_env(1)
    result = play_duplicate_match(
        FoldAgent(), CallAgent(), env_proto, num_pairs=16, seed=5, device=DEVICE
    )
    assert result.num_games == 32
    folds = [r for r in result.records if r.terminal_type == "fold"]
    # The fold bot is the only one that ever folds, and it only folds when it
    # faces a bet: as the SB that costs 0.5bb, as the BB 1bb.
    assert folds
    assert all(r.street == 0 for r in folds)
    assert all(
        min(abs(r.reward_a_bb + 0.5), abs(r.reward_a_bb + 1.0)) < 1e-5 for r in folds
    )
    assert all(r.reward_a_chips == pytest.approx(r.reward_a_bb * 100.0) for r in folds)
    # Facing no bet folding is illegal, so those hands get checked down; the
    # bot still cannot win chips it never put in, so the edge is negative.
    assert result.mean_bb_per_100 < 0.0


def test_identical_agents_have_bounded_duplicate_score():
    """Two call-bots: the pair difference cancels exactly (deterministic play)."""
    env_proto = make_env(1)
    result = play_duplicate_match(
        CallAgent(), CallAgent(), env_proto, num_pairs=12, seed=9, device=DEVICE
    )
    assert torch.allclose(result.pair_diff_bb, torch.zeros(12), atol=1e-5)
    assert result.mean_bb_per_100 == pytest.approx(0.0, abs=1e-4)
    # All-call reaches showdown every time.
    assert all(r.terminal_type == "showdown" for r in result.records)
    assert all(r.street == 4 for r in result.records)


def test_random_agents_terminate_and_produce_real_outcomes():
    env_proto = make_env(1)
    result = play_duplicate_match(
        RandomAgent("rand_a"),
        RandomAgent("rand_b"),
        env_proto,
        num_pairs=24,
        seed=17,
        device=DEVICE,
    )
    assert result.num_games == 48
    assert all(r.terminal_type in {"fold", "showdown"} for r in result.records)
    assert all(1 <= r.num_decisions <= MAX_DECISIONS for r in result.records)
    # Chips are conserved: the pot is what both players put in, and rewards are
    # bounded by the effective stack.
    assert all(abs(r.reward_a_stacks) <= 1.0 + 1e-6 for r in result.records)


def test_common_random_numbers_make_matches_reproducible():
    env_proto = make_env(1)
    kwargs = dict(num_pairs=8, seed=123, device=DEVICE)
    a = play_duplicate_match(RandomAgent(), RandomAgent(), env_proto, **kwargs)
    b = play_duplicate_match(RandomAgent(), RandomAgent(), env_proto, **kwargs)
    assert torch.equal(a.reward_bb, b.reward_bb)
    assert [r.to_dict() for r in a.records] == [r.to_dict() for r in b.records]


# ------------------------------------------------------------------- records


def _fake_batch(num_pairs: int = 2) -> GameBatchTensors:
    g = 2 * num_pairs
    arange = torch.arange(g, dtype=torch.long)
    return GameBatchTensors(
        pair_id=arange // 2,
        pair_half=arange % 2,
        seat_a=torch.zeros(g, dtype=torch.long),
        button=arange % 2,
        terminal_type=torch.tensor([0, 1] * num_pairs, dtype=torch.long),
        street=torch.tensor([0, 4] * num_pairs, dtype=torch.long),
        final_pot=torch.full((g,), 300, dtype=torch.long),
        num_decisions=torch.full((g,), 3, dtype=torch.long),
        winner=arange % 2,
        effective_stack=torch.full((g,), 10_000, dtype=torch.long),
        hole_a_combo=torch.full((g,), 5, dtype=torch.long),
        hole_b_combo=torch.full((g,), 9, dtype=torch.long),
        reward_a_stacks=torch.linspace(-0.5, 0.5, g),
        board=torch.arange(5 * g, dtype=torch.long).view(g, 5) % 52,
        hole_a=torch.tensor([[0, 1]] * g, dtype=torch.long),
        hole_b=torch.tensor([[2, 3]] * g, dtype=torch.long),
    )


def test_records_round_trip(tmp_path):
    batch = _fake_batch(3)
    records = batch.to_records(
        eval_id="eval-1",
        agent_a_id="ckpt/a.pt|step=100",
        agent_b_id="call_bot",
        big_blind=100,
        seed=42,
        agent_a_fidelity={"cfr_iterations": 32},
        agent_b_fidelity={},
    )
    path = tmp_path / "games.jsonl"
    writer = RecordWriter(path, manifest={"eval_id": "eval-1", "num_pairs": 3})
    assert writer.append(records) == 6

    loaded = load_records(path)
    assert loaded == records
    assert loaded[0].agent_a_fidelity == {"cfr_iterations": 32}
    assert loaded[0].terminal_type == "fold"
    assert loaded[1].terminal_type == "showdown"
    assert loaded[0].street_name == "preflop"
    assert loaded[0].seed == 42

    manifest = load_manifest(path)
    assert manifest["eval_id"] == "eval-1"
    assert manifest["num_pairs"] == 3

    # Appending is additive, never rewriting.
    writer.append(records)
    assert len(load_records(path)) == 12

    # Rows are plain JSON objects with all required fields.
    row = json.loads(path.read_text().splitlines()[0])
    for key in (
        "eval_id",
        "pair_id",
        "pair_half",
        "agent_a_id",
        "agent_b_id",
        "seat_a",
        "button",
        "reward_a_chips",
        "reward_a_bb",
        "reward_a_stacks",
        "terminal_type",
        "street",
        "final_pot",
        "num_decisions",
        "board",
        "hole_a",
        "hole_b",
        "seed",
        "agent_a_fidelity",
        "agent_b_fidelity",
    ):
        assert key in row


def test_record_units_are_consistent():
    records = _fake_batch(2).to_records(
        eval_id="e", agent_a_id="a", agent_b_id="b", big_blind=100, seed=0
    )
    for record in records:
        assert record.reward_a_chips == pytest.approx(
            record.reward_a_stacks * record.effective_stack
        )
        assert record.reward_a_bb == pytest.approx(record.reward_a_chips / 100.0)


def test_pair_differences_math():
    records = _fake_batch(2).to_records(
        eval_id="e", agent_a_id="a", agent_b_id="b", big_blind=100, seed=0
    )
    pair_ids, diffs = pair_differences(records, unit="stacks")
    assert pair_ids == [0, 1]
    expected = [
        0.5 * (records[0].reward_a_stacks + records[1].reward_a_stacks),
        0.5 * (records[2].reward_a_stacks + records[3].reward_a_stacks),
    ]
    assert diffs == pytest.approx(expected)

    # Incomplete pairs are dropped rather than half-counted.
    partial = [r for r in records if not (r.pair_id == 1 and r.pair_half == 1)]
    assert pair_differences(partial)[0] == [0]


def test_pair_differences_match_the_live_result():
    env_proto = make_env(1)
    result = play_duplicate_match(
        RandomAgent(), FoldAgent(), env_proto, num_pairs=10, seed=31, device=DEVICE
    )
    _, diffs = pair_differences(result.records, unit="bb")
    assert torch.allclose(
        torch.tensor(diffs, dtype=torch.float32), result.pair_diff_bb, atol=1e-4
    )


def test_recorded_hole_cards_are_the_real_dealt_cards():
    env_proto = make_env(1)
    writer_records = play_duplicate_match(
        CallAgent(), CallAgent(), env_proto, num_pairs=4, seed=77, device=DEVICE
    ).records
    for record in writer_records:
        assert record.hole_a_combo == combo_index(*record.hole_a)
        assert record.hole_b_combo == combo_index(*record.hole_b)
        assert len(set(record.hole_a + record.hole_b + record.board)) == 9
    # Mirrored halves: agent A holds in half 1 what agent B held in half 0.
    for pair in range(4):
        half0 = writer_records[2 * pair]
        half1 = writer_records[2 * pair + 1]
        assert half0.hole_a == half1.hole_b
        assert half0.hole_b == half1.hole_a
        assert half0.board == half1.board
        assert half0.button == 1 - half1.button


def test_recorder_writes_manifest_and_rows(tmp_path):
    env_proto = make_env(1)
    path = tmp_path / "match.jsonl"
    recorder = RecordWriter(path)
    result = play_duplicate_match(
        CallAgent("caller"),
        RandomAgent("rando"),
        env_proto,
        num_pairs=5,
        seed=13,
        device=DEVICE,
        recorder=recorder,
        eval_id="run-7",
    )
    loaded = load_records(path)
    assert len(loaded) == result.num_games == 10
    assert loaded == result.records
    manifest = load_manifest(path)
    assert manifest["eval_id"] == "run-7"
    assert manifest["agent_a"]["name"] == "caller"
    assert manifest["agent_b"]["name"] == "rando"
    assert manifest["env"]["bb"] == 100
    assert manifest["scoring"] == "duplicate_real_hand_chips"


def test_agent_identity_id_includes_checkpoint_and_step():
    identity = AgentIdentity(
        name="candidate", kind="search", checkpoint="/ckpt/x.pt", step=1234
    )
    assert identity.id == "candidate|/ckpt/x.pt|step=1234"
    assert AgentIdentity(name="call_bot").id == "call_bot"
    assert identity.to_dict()["step"] == 1234


def test_game_record_from_dict_ignores_unknown_keys():
    record = _fake_batch(1).to_records(
        eval_id="e", agent_a_id="a", agent_b_id="b", big_blind=100, seed=0
    )[0]
    row = record.to_dict()
    row["future_field"] = "ignored"
    assert GameRecord.from_dict(row) == record


class _StubEvaluator:
    """Minimal stand-in for a CFR evaluator: no model, no real search.

    Node layout matches the sparse evaluators' contract closely enough to
    exercise `SearchAgent`'s action->child mapping and belief propagation:
    roots are ``0..A-1`` and root ``i``'s child for action ``a`` is
    ``A + i * num_actions + a``.
    """

    def __init__(self, num_actions: int = 8, hand_dim: int = 1326):
        self.device = DEVICE
        self.num_actions = num_actions
        self.hand_dim = hand_dim
        self.cfr_iterations = 4
        self.init_calls: list[torch.Tensor | None] = []
        self.cfr_calls = 0

    def initialize_subgame(self, env, src_indices, initial_beliefs=None):
        self.init_calls.append(
            None if initial_beliefs is None else initial_beliefs.clone()
        )
        a = int(src_indices.numel())
        total = a + a * self.num_actions
        self.parent_index = torch.full((total,), -1, dtype=torch.long)
        self.action_from_parent = torch.full((total,), -1, dtype=torch.long)
        child_ids = torch.arange(a * self.num_actions)
        self.parent_index[a:] = child_ids // self.num_actions
        self.action_from_parent[a:] = child_ids % self.num_actions
        self.policy_probs_avg = torch.zeros(total, self.hand_dim)
        # Every hand always wants the check/call bin except hand 0, which wants
        # the last bin. Roots keep zero probability.
        self.policy_probs_avg[a + torch.arange(a) * self.num_actions + 1, :] = 1.0
        self.policy_probs_avg[a + torch.arange(a) * self.num_actions + 1, 0] = 0.0
        self.policy_probs_avg[
            a + torch.arange(a) * self.num_actions + self.num_actions - 1, 0
        ] = 1.0
        # Distinct, identifiable belief per node.
        self.beliefs = (
            torch.arange(total, dtype=torch.float32)[:, None, None]
            .expand(total, 2, self.hand_dim)
            .contiguous()
        )

    def evaluate_cfr(self, training_mode: bool = False):
        self.cfr_calls += 1


def test_search_agent_uses_its_real_hand_and_propagates_beliefs():
    from p2.eval.agents import SearchAgent

    env = make_env(3)
    env.reset()
    evaluator = _StubEvaluator()
    agent = SearchAgent(evaluator, AgentIdentity(name="stub", kind="search"))
    agent.begin_match(env, seat=0)

    active = torch.arange(3)
    agent.observe(env, active)
    assert evaluator.init_calls == [None]  # uniform beliefs on the first round
    assert evaluator.cfr_calls == 1

    legal = torch.ones(3, evaluator.num_actions, dtype=torch.bool)
    hands = torch.tensor([0, 5, 5])  # hand 0 shoves, the others check/call
    probs = agent.action_probs(env, active, hands, legal)
    assert probs.shape == (3, evaluator.num_actions)
    assert probs.argmax(dim=1).tolist() == [evaluator.num_actions - 1, 1, 1]

    # Beliefs propagate through the chosen *public* child node only.
    actions = torch.tensor([7, 1, 1])
    keep = torch.tensor([True, False, True])
    agent.commit(actions, keep)
    agent.observe(env, active[keep])
    propagated = evaluator.init_calls[1]
    expected_children = torch.tensor(
        [3 + 0 * 8 + 7, 3 + 2 * 8 + 1], dtype=torch.float32
    )
    assert torch.equal(propagated[:, 0, 0], expected_children)

    fidelity = agent.search_fidelity()
    assert fidelity["cfr_iterations"] == 4
    assert fidelity["evaluator"] == "_StubEvaluator"
    assert agent.identity.kind == "search"


def test_search_agent_plays_a_full_duplicate_match():
    from p2.eval.agents import SearchAgent

    evaluator = _StubEvaluator()
    agent = SearchAgent(evaluator, AgentIdentity(name="stub", kind="search"))
    result = play_duplicate_match(
        agent, CallAgent(), make_env(1), num_pairs=4, seed=21, device=DEVICE
    )
    assert result.num_games == 8
    assert len(result.records) == 8
    # The evaluator was re-initialized and re-solved once per decision round.
    assert evaluator.cfr_calls == len(evaluator.init_calls) > 1
    assert result.records[0].agent_a_fidelity["cfr_iterations"] == 4


def test_search_agent_maps_combos_to_preflop_classes():
    from p2.env.card_utils import combo_to_preflop_class_tensor
    from p2.eval.agents import SearchAgent

    env = make_env(2)
    env.reset()
    evaluator = _StubEvaluator(hand_dim=169)
    agent = SearchAgent(evaluator)
    agent.begin_match(env, seat=0)
    agent.observe(env, torch.arange(2))
    hands = torch.tensor([0, 1325])
    rows = agent._hand_rows(hands)
    assert torch.equal(rows, combo_to_preflop_class_tensor()[hands])
    probs = agent.action_probs(
        env, torch.arange(2), hands, torch.ones(2, 8, dtype=torch.bool)
    )
    assert probs.shape == (2, 8)


def test_combo_lookup_matches_env_dealt_holes():
    env = make_env(4)
    env.reset()
    lookup = combo_lookup_tensor(device=DEVICE)
    combos = lookup[env.hole_indices[:, :, 0], env.hole_indices[:, :, 1]]
    assert combos.shape == (4, 2)
    assert (combos >= 0).all()
