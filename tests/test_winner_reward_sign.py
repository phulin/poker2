import torch

from p2.core.structured_config import Config
from p2.models.transformer.structured_embedding_data import StructuredEmbeddingData
from p2.rl.vectorized_replay import VectorizedReplayBuffer


def _embedding(batch_size: int, sequence_length: int, num_bet_bins: int):
    return StructuredEmbeddingData(
        token_ids=torch.zeros(batch_size, sequence_length, dtype=torch.long),
        token_streets=torch.zeros(batch_size, sequence_length, dtype=torch.long),
        card_ranks=torch.zeros(batch_size, sequence_length, dtype=torch.long),
        card_suits=torch.zeros(batch_size, sequence_length, dtype=torch.long),
        action_actors=torch.zeros(batch_size, sequence_length, dtype=torch.long),
        action_legal_masks=torch.ones(
            batch_size, sequence_length, num_bet_bins, dtype=torch.bool
        ),
        action_amounts=torch.zeros(batch_size, sequence_length, dtype=torch.int32),
        context_features=torch.zeros(batch_size, sequence_length, 9),
        lengths=torch.ones(batch_size, dtype=torch.long),
    )


def test_winner_reward_sign_and_replay_buffer_alignment():
    cfg = Config()
    cfg.env.bet_bins = [0.5, 1.0]
    cfg.train.max_trajectory_length = 4
    cfg.train.max_sequence_length = 8
    cfg.model.num_bet_bins = len(cfg.env.bet_bins) + 3

    buffer = VectorizedReplayBuffer(
        capacity=3,
        cfg=cfg,
        device=torch.device("cpu"),
        float_dtype=torch.float32,
        is_transformer=True,
    )

    winners = torch.tensor([0, 1, 2])
    rewards = torch.tensor([1.25, -0.75, 0.0])
    buffer.start_adding_trajectory_batches(3, model_age=0)
    buffer.add_transitions(
        embedding_data=_embedding(3, buffer.max_sequence_length, buffer.num_bet_bins),
        trajectory_indices=torch.arange(3),
        action_indices=torch.ones(3, dtype=torch.long),
        logits=torch.zeros(3, buffer.num_bet_bins),
        rewards=rewards,
        dones=torch.ones(3, dtype=torch.bool),
        legal_masks=torch.ones(3, buffer.num_bet_bins, dtype=torch.bool),
        delta2=torch.zeros(3),
        delta3=torch.zeros(3),
        values=torch.zeros(3),
    )
    added, steps = buffer.finish_adding_trajectory_batches()

    assert added == 3
    assert steps == 3
    final_rewards = buffer.rewards[torch.arange(3), buffer.trajectory_lengths[:3] - 1]
    torch.testing.assert_close(final_rewards, rewards)
    assert torch.all(final_rewards[winners == 0] > 0)
    assert torch.all(final_rewards[winners == 1] < 0)
    assert torch.all(final_rewards[winners == 2] == 0)
