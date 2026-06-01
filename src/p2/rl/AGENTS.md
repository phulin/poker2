## Directory summary
Reinforcement learning and ReBeL training support: PPO losses, replay buffers, opponent pools, self-play orchestration, ratings, and public-belief game rollout helpers.

### Source files
- `__init__.py`: Package marker.
- `agent_snapshot.py`: Snapshot wrapper for frozen agents.
- `cfr_trainer.py`: ReBeL CFR supervised trainer, including split BetterFFN policy/value ownership, value-only distillation updates, PBSEnv-backed multiway setup, non-fused closing-leaf checkpoint routing, model checkpointing, and replay-buffer sidecar save/load.
- `rebel_loop.py`: Shared ReBeL training loop runner for step execution, metric printing, checkpoint cadence/cleanup, final checkpointing, and TrueSkill snapshots.
- `losses.py`: PPO variants, CFR distillation loss, and ReBeL supervised loss.
- `optimizers.py`: Optimizer construction helpers, including optional Muon or eager NorMuon matrix optimization with AdamW fallback splitting for ReBeL/all-in training and split BetterFFN policy-head grouping.
- `self_play.py`: SelfPlayTrainer and model-history support.
- `replay.py`: Scalar trajectory storage and GAE/PPO batch preparation.
- `vectorized_replay.py`: Batched replay buffer for tensorized environments.
- `rebel_replay.py`: ReBeL policy/value replay buffers with serializable buffer state.
- `rebel_batch.py`: ReBeL batch dataclass.
- `pbs_games.py`: Public-belief game rollout utilities.
- `opponent_pool.py`: Abstract opponent-pool interface.
- `fixed_opponent_pool.py`: Fixed opponent pool.
- `k_best_pool.py`: K-best opponent selection and updates.
- `dred_pool.py`: DReD opponent pool.
- `kmedoids.py`: PyTorch k-medoids helper for pool diversity.
- `elo_calculator.py`: Elo update helper.
- `trueskill_tracker.py`: TrueSkill plus Gaussian reward tracking, snapshot trend metrics, and weight binding helpers.
- `target_provenance.py`: Integer codes and names for value-target provenance recorded in ReBeL batch statistics.
- `popart_normalizer.py`: PopArt normalization module.
- `exponential_controller.py`: Generic exponential schedule/controller.

### Subdirectories
There are no child source directories.
