## AlphaHoldem (HUNL) — PyTorch Implementation Plan

- **Goal**: Reproduce a practical, efficient Heads-Up No-Limit Texas Hold’em (HUNL) agent following the AlphaHoldem paper using PyTorch, with end-to-end self-play RL, pseudo-siamese network, Trinal-Clip PPO, and K-Best self-play.
- **Success criteria**:
  - Trains from self-play only; no card/betting abstraction features or CFR.
  - Produces actions via a single forward pass at decision time.
  - Reports evaluation in mbb/h and ELO; provides policy heatmaps comparable to the paper.
  - Extensible to multi-player no-limit using the architecture in `ChatGPT Advice.md`.

### Scope and Non-goals
- **In scope**: Environment for HUNL heads-up, tensor state encoders, pseudo-siamese model, Trinal-Clip PPO, replay buffer, self-play with K-best opponent pool, distributed training option, evaluation tooling, checkpoints, and experiment configs.
- **Out of scope (initially)**: Multi-player poker, full-scale massive GPU training, exact reproduction of paper’s wall-clock performance numbers.

### Milestones
1) Project bootstrap
2) State encoders (cards/actions)
3) Pseudo-siamese model (policy/value heads)
4) Trinal-Clip PPO loss
5) Replay buffer + data pipeline
6) Self-play loop with K-Best pool
7) Training loop (+ optional distributed)
8) Evaluation (ELO, mbb/h, heatmaps)
9) Experiments, configs, docs

### Proposed Repository Layout
- `alphaholdem/`
  - `core/`
    - `interfaces.py` — `Env`, `Encoder`, `Model`, `Policy`, `OpponentPool`, `League` ABCs.
    - `registry.py` — registries for encoders/models/policies via string keys.
    - `config.py` — dataclasses and YAML loading; build-by-config utilities.
  - `env/`
    - `hunl_env.py` — HUNL engine (two players, blinds, rounds, legality, terminal states, pot, stacks, rake optional).
    - `nl_env.py` — N-player NL base with table, positions, and seat abstraction (extends HUNL core).
    - `rules.py` — hand evaluation, showdown, deck utilities, shuffling, blinds posting.
    - `types.py` — dataclasses for `GameState`, `Action`, `Observation`.
  - `encoding/`
    - `cards_encoder.py` — pluggable card encoders (baseline 4×13 planes; alt embeddings).
    - `actions_encoder.py` — pluggable action-history encoders (heads-up and N-player variants).
    - `betting_bins.py` — map continuous bet sizes to nb discrete options; snapping utilities.
  - `models/`
    - `siamese_convnet.py` — pseudo-siamese ConvNets; fusion MLP; policy/value heads.
    - `heads.py` — policy/value head abstractions; centralized-critic optional module.
    - `builder.py` — model factory from registry/config.
    - `init.py` — init utilities.
  - `rl/`
    - `losses.py` — Trinal-Clip PPO: policy clip with δ1 for A<0; value clip with δ2, δ3.
    - `replay.py` — replay buffers; GAE; on-policy batching.
    - `self_play.py` — heads-up K-Best self-play driver.
    - `league.py` — PSRO-lite/league framework (League, MetaMix, BR learners) per `ChatGPT Advice.md`.
    - `train.py` — training loop, optimizer, LR sched, gradient clip, checkpointing, model selection.
    - `distributed.py` — optional DDP/MP for self-play workers and learners.
  - `eval/`
    - `elo.py` — ELO tracking and opponent selection utilities.
    - `mbb.py` — mbb/h computation, confidence intervals.
    - `policy_viz.py` — first-action frequency heatmaps for hole cards (suited/offsuit grids).
  - `configs/`
    - `default.yaml` — default hyperparameters and module choices.
    - `small.yaml` — CPU-friendly toy run.
    - `gpu.yaml` — 1–2 GPU training.
  - `cli/`
    - `play.py` — run matches between agents/humans.
    - `train.py` — launch self-play + learning.
  - `utils/`
    - `checkpoint.py`, `logging.py`, `seed.py`, `timer.py`, `profiling.py`.
- `tests/` — unit tests for encoders, env transitions, model shapes, loss math, and hot-swapping.
- `scripts/` — reproducible runners for experiments.
- `README.md` — setup, usage, experiment notes (to be authored after MVP runs).

### Core Technical Design
- **Pluggability and Config-Driven Composition**
  - Define ABCs for `Encoder`, `Model`, and `Policy` with minimal, stable method signatures.
  - Use a lightweight registry to map string names in YAML configs to classes; all components are built from config without code changes.
  - Encoders and model parts accept shapes and `nb` from config, enabling drop-in replacements (e.g., learned card embeddings vs 4×13 planes; ResNet trunk vs shallow CNN).
- **State representation** (per paper):
  - Cards tensor: multiple 4×13 binary planes for hole, flop, turn, river, public, all-cards summary (baseline encoder). Alternative: learned token embeddings over ranks/suits+positional planes.
  - Actions tensor: 24 channels (4 rounds × up to 6 actions). Each channel is 4×nb sparse binary: planes for player1, player2, sum, legal. `nb` configurable. N-player variant: N per-player planes + sum + legal.
  - Include history over actions to aid reasoning with hidden information.
- **Action space**: Discretized bet sizes (fold, check/call, 1/2, 3/4, pot, 1.5×, 2×pot, all-in). Legality masks and nearest-legal snapping.
- **Model (pseudo-siamese)**:
  - Independent ConvNets for cards/actions; fusion MLP; outputs policy logits and value.
  - Heads abstracted to allow centralized critic option for N-player (never uses private cards of others).
  - Keep model size modest (<10M params target) for fast inference.
- **Loss: Trinal-Clip PPO**:
  - Policy: PPO clip ϵ plus δ1 upper bracket when A<0.
  - Value: clip target returns by [−δ2, δ3] derived dynamically from chips placed in replay.
  - GAE(λ), entropy bonus, gradient norm clip.
- **Self-play**:
  - Heads-up: K-Best pool of historical snapshots; opponent sampling by ELO.
  - Multi-player-ready: league/PSRO-lite with `League`, `MetaMix`, and optional best-response learners per `ChatGPT Advice.md`.
- **Replay/data**:
  - Trajectories with obs, actions, log-probs, value targets, advantages, legal masks, betting context; on-policy PPO batching.

### Training Pipeline
1) Launch self-play workers generating rollouts against opponents from K-Best (HU) or league lineups (N-player).
2) Add rollouts to replay; compute GAE, returns, and δ2/δ3 from betting context.
3) Update policy/value via Trinal-Clip PPO for E epochs over mini-batches.
4) Periodically evaluate vs pool/league; update ELO/meta-mix; checkpoint best; refresh pool.

### Key Hyperparameters (initial defaults)
- `nb` (bet options): 9
- PPO: ϵ=0.2, δ1=3.0; δ2/δ3 dynamic from chips placed
- GAE λ=0.95; γ=0.999
- LR=3e-4 Adam; batch size per update ~16k samples (scale for device)
- Entropy coef=0.01; value coef=0.5; grad clip=1.0
- K (opponent pool size): 5–10; league size: 1 main + 3 snapshots to start

### Extensibility to Multi-player (from `ChatGPT Advice.md`)
- Observation: keep card encoder unchanged; generalize action-history encoder to N per-player planes + sum/legal.
- Networks: shared actor per seat with pseudo-siamese trunk; optional centralized critic that receives only public features from other seats.
- Population self-play: maintain league with snapshots and specialists; sample N-player lineups via meta-mix; periodic PSRO-lite meta updates and best-response additions.
- Evaluation: round-robin multi-table metrics; checkpoint selection via average payoff and variance; track exploitability proxy via fresh BR probes.

### Risks and Mitigations
- Sparse rewards, high variance → GAE, value clipping, large batches, entropy regularization.
- Discretization mismatch → legality masking, nearest-legal snapping, curriculum on bet bins.
- Strategy cycling → K-Best/league with diversity sampling and regular evaluation.
- Performance constraints → CPU/small GPU configs; profiling; compact models.

### Step-by-step Checklist (execution order)
- Initialize project, configs, registry, and ABC interfaces.
- Implement HUNL environment and tests; sketch N-player base.
- Implement card/actions encoders (HU first, N-player ready) and legal-action masking.
- Build pseudo-siamese ConvNet with pluggable heads; unit tests for shapes and swaps.
- Implement Trinal-Clip PPO loss and verify numerics on synthetic data.
- Implement replay buffer, GAE, and batching.
- Implement self-play loop (HU K-Best) and league scaffold (N-player PSRO-lite).
- Wire training loop; add checkpointing and logging.
- Run small CPU-only HU experiment; validate learning signals.
- Scale to single-GPU; produce evaluation figures and policy heatmaps.
- Document results and usage in README.

### Deliverables
- Minimal working HU agent; league scaffold for N-player.
- Scripts for small and GPU runs; configs with pluggable components.
- Plots: learning curves, heatmaps, ELO trajectories.
- Checkpoints and best-config files.
