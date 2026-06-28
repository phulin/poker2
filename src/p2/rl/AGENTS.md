## Directory summary
Reinforcement learning and ReBeL training support: PPO losses, replay buffers, opponent pools, self-play orchestration, ratings, and public-belief game rollout helpers.

### Source files
- `__init__.py`: Package marker.
- `agent_snapshot.py`: Snapshot wrapper for frozen agents.
- `cfr_trainer.py`: ReBeL CFR supervised trainer, including split BetterFFN policy/value ownership, compact 169-hand preflop model routing with BF16 CUDA inference twins, optional block-frozen CFR target model promotion, actor-only backup-consistency auxiliary training from stored child features, value-only distillation updates and checkpoints, PBSEnv-backed multiway setup with dedicated preflop sparse evaluator routing, non-fused closing-leaf checkpoint routing, pregeneration-only construction without replay buffers, data-source state checkpointing, and optional replay-buffer sidecar save/load.
- `checkpoint_io.py`: Shared ReBeL checkpoint helpers for metadata reads, floating tensor dtype restoration, and staged model-weight loading.
- `rebel_loop.py`: Shared ReBeL training loop runner for step execution, metric printing, checkpoint cadence/cleanup, optional preflop analyzer printing, final checkpointing, and TrueSkill snapshots.
- `losses.py`: PPO variants, CFR distillation loss, and ReBeL supervised loss with compact 169-hand preflop value/policy branches, including live-player masks for folded-player-aware policy/value weighting.
- `optimizers.py`: Optimizer construction helpers, including PyTorch Muon with an optional compiled CUDA step, eager NorMuon matrix optimization, AdamW fallback splitting for ReBeL/all-in training, and split BetterFFN policy-head grouping.
- `self_play.py`: SelfPlayTrainer and model-history support.
- `replay.py`: Scalar trajectory storage and GAE/PPO batch preparation.
- `vectorized_replay.py`: Batched replay buffer for tensorized environments.
- `rebel_replay.py`: ReBeL policy/value replay buffers with serializable buffer state.
- `rebel_batch.py`: ReBeL batch dataclass with suit-permutation handling for 1326-combo and compact 169-class targets.
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
- `validation_set.py`: Pregenerated solved-dataset value-loss evaluator used for periodic ReBeL validation during training and standalone checkpoint benchmarking.

### Subdirectories
There are no child source directories.
