## Directory summary
Reinforcement learning and ReBeL training support: PPO losses, replay buffers, opponent pools, self-play orchestration, ratings, and public-belief game rollout helpers.

### Source files
- `__init__.py`: Package marker.
- `agent_snapshot.py`: Snapshot wrapper for frozen agents.
- `cfr_trainer.py`: ReBeL CFR supervised trainer.
- `checkpoint_io.py`: Shared ReBeL checkpoint helpers.
- `rebel_loop.py`: Shared ReBeL training loop runner.
- `losses.py`: PPO, CFR distillation, and ReBeL supervised losses.
- `optimizers.py`: Optimizer construction helpers.
- `self_play.py`: SelfPlayTrainer and model-history support.
- `replay.py`: Scalar trajectory storage and GAE/PPO batch preparation.
- `vectorized_replay.py`: Batched replay buffer for tensorized environments.
- `rebel_replay.py`: ReBeL policy/value replay buffers with serializable buffer state.
- `rebel_batch.py`: ReBeL batch dataclass and permutation helpers.
- `pbs_games.py`: Public-belief game rollout utilities.
- `opponent_pool.py`: Abstract opponent-pool interface.
- `fixed_opponent_pool.py`: Fixed opponent pool.
- `k_best_pool.py`: K-best opponent selection and updates.
- `dred_pool.py`: DReD opponent pool.
- `kmedoids.py`: PyTorch k-medoids helper for pool diversity.
- `elo_calculator.py`: Elo update helper.
- `trueskill_tracker.py`: TrueSkill and reward-trend tracking.
- `target_provenance.py`: Integer codes and names for value-target provenance recorded in ReBeL batch statistics.
- `popart_normalizer.py`: PopArt normalization module.
- `exponential_controller.py`: Generic exponential schedule/controller.
- `validation_set.py`: Pregenerated solved-dataset value-loss evaluator used for periodic ReBeL validation during training and standalone checkpoint benchmarking.

### Subdirectories
There are no child source directories.
