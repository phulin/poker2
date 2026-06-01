# Random-Spot ReBeL Postflop Curriculum Plan

## Goal
Add a backward-bootstrapped postflop curriculum to heads-up ReBeL training, driven by a single orchestrator script. Production training remains live/in-loop CFR, but the live root source changes from self-play continuation to random legal postflop spots sampled on the fly. The curriculum bootstraps backward:

1. Train `S_river` on random river beliefs and river public spots (exact river terminals).
2. Distill `E_turn` from frozen `S_river` (expectation over the river card), then train `S_turn` with `E_turn` as turn-closing terminal values.
3. Distill `E_flop` from frozen `S_turn`, then train `S_flop` with `E_flop` as flop-closing terminal values.
4. Distill `E_preflop` from frozen `S_flop` to feed the preflop handoff; keep preflop itself separate per `preflop_multiway_pbs_bootstrap_plan.md`.

### Model factoring: start-of-street and end-of-street nets per street
Use **distinct networks per postflop street**, not one cumulative model conditioned on a street embedding. This is the DeepStack factoring and is what makes the backward curriculum clean. Per street `X` there are two nets with different roles:

- **Start-of-street net `S_X`** — post-chance (the street-`X` card(s) are dealt, betting about to begin). Predicts counterfactual values **and** policy. Used to resolve street `X` at play time and to value within-street depth-limited leaves during street-`X` CFR (normal ReBeL self-bootstrap on the net being trained).
- **End-of-street net `E_X`** — pre-chance (street-`X` betting closed, the next card not yet dealt). Predicts the **chance-averaged** counterfactual value only (no policy). Used as the **terminal leaf value at street-`X`-closing nodes inside street-`X`'s CFR iterations**.

Distillation chain (backward):

- `S_X` is trained by running street-`X` CFR, whose closing leaves are valued by the frozen `E_X`, and whose terminal leaves use exact fold/showdown/all-in values.
- `E_X` is **distilled from the frozen `S_{X+1}`**: sample end-of-`X` (pre-chance) public belief states and regress `E_X` onto the chance expectation over the dealt card of `S_{X+1}`. `ChanceNodeHelper` runs the 44-card enumeration / 256-flop sample **once, to build `E_X`'s targets** — never inside prior-street CFR iterations. Street-`X` CFR then costs one `E_X` eval per closing leaf instead of `enumerate × S_{X+1}` evals per leaf per iteration.

Why two nets, not one frozen next-street net at the boundary: the inner CFR loop must not enumerate the chance node every iteration. `E_X` amortizes that enumeration into a single cheap eval. (Mechanically this isolates the pre-chance head the current single net already approximates via the `pre_chance_node` flag in `_street_for_phase`.)

Properties:

- Once promoted, `S_X` and `E_X` are **frozen forever**. No self-bootstrap feedback loop across streets and no forgetting to defend against.
- During a stage's solve the evaluator holds the **current-street `S_X`** (training) plus the **frozen `E_X`** (closing leaves). Distilling `E_X` additionally loads the frozen `S_{X+1}`.
- At play/resolve time, search dispatches by street through a small `street → net` registry.

Costs / caveats:

- The evaluator and `ChanceNodeHelper` currently hold a single `model` (`sparse_cfr_evaluator.py`, `chance_node_helper.py`); they must take an `(S_X, E_X)` pair with leaf routing by phase (within-street → `S_X`, street-closing → `E_X`).
- Boundary approximation compounds: `S_{X+1}` error → distilled into `E_X` → consumed by `S_X` training. Keep exact terminals mixed in and validate `E_X` against fresh chance-enumerated `S_{X+1}` values on a holdout.
- Cross-street weight sharing is given up; transfer flows through `E_X` targets instead.

Net inventory: `S_river` (exact river terminals, no `E_river`), `E_turn`←`S_river`, `S_turn`, `E_flop`←`S_turn`, `S_flop`, `E_preflop`←`S_flop` (feeds the preflop handoff).

### Data path: live is the main path
Live, in-loop CFR data generation stays the **production** training path for every postflop street. In live mode, roots are generated on the fly by random legal street-start / later legal-prefix postflop spot samplers, not by advancing a self-play `current_pbs` trajectory. CFR still runs inside the optimizer loop for those sampled roots, and the replay buffer still amortizes recent solves. On-disk solved-example datasets are too large to be the backbone (see "Storage Reality" below), so the trainer supports two data modes:

- `live` — used for the real, full-scale runs. All streets sample random legal postflop roots in-loop, solve them with CFR, and feed the existing replay buffers.
- `pregenerated` — used for **small, fast, fixed-data hyperparameter sweeps only**. Solved examples are written to disk at experiment scale (not the 50M+ production scale) so HP runs see identical data and skip the solver. This mode must never be assumed to fit a full run on disk.

The optional "pregeneration" being added is therefore only the bounded offline dataset path for HP testing and holdouts. The staged train/distill curriculum itself is live by default and uses random spot generation at production scale.

### Storage Reality
On-disk solved datasets do not scale to a full run. Per postflop example the belief vector and value target each cost `2 × 1326 ≈ 2652` floats:

- a value example (`beliefs` + `value_targets` + small features) ≈ 5.3K floats ≈ 21 KB fp32;
- a policy example (`policy_targets[1326, A]` + beliefs) ≈ 13.5K floats ≈ 54 KB fp32;
- river alone wants ~50M value examples ≈ ~1 TB fp32 / ~530 GB fp16 (the value-target half is the ~132B floats).

Crucially, storing *roots* does not help: a root carries `beliefs[2, 1326]`, the same 2652-float cost as a value target, so a 50M-root set is as infeasible as 50M solved examples. River belief and spot sampling must happen on the fly. Turn and flop have the identical 1326-hand footprint, so they are live at full scale too. On-disk datasets are reserved for bounded artifacts: HP-sweep example sets, per-street holdout/validation sets, and preflop-handoff flop roots.

## Non-Goals
- Do not make full multiway postflop ReBeL part of this conversion.
- Do not train the postflop model on preflop examples.
- Do not attempt to store full-run solved datasets on disk. The `pregenerated` mode is for small HP-sweep runs only.
- Do not start with arbitrary incoherent mid-street state mutation. Random public spots must be generated by either a legal state sampler or constrained street-start state constructors.
- Do not enumerate/sample the chance node inside prior-street CFR iterations. The chance expectation is amortized once into a frozen end-of-street net (`E_X`); CFR reads `E_X` at closing leaves. `ChanceNodeHelper` is used to build `E_X`'s training targets, not in the inner loop.
- Do not use a cumulative single net conditioned on street. Each `S_X`/`E_X` is its own net, frozen once promoted.

## Current Starting Point
The current trainer is tightly coupled:

- `RebelCFRTrainer._update_model()` calls `self.data_generator.generate_data()` each step.
- `RebelDataGenerator.generate_data()` maintains `current_pbs`, solves with `evaluator.evaluate_cfr()`, calls `evaluator.training_data()`, writes to replay buffers, then returns fresh batches.
- `CFREvaluator.training_data()` already materializes the core offline object: `RebelBatch` with `MLPFeatures`, legal masks, policy targets, value targets, and statistics.
- Replay buffers already support street/depth sampling, suit permutation, CPU/GPU storage, and sample counters.
- `p2.allin.training_data` already has a useful pregeneration pattern: sharded `.pt` tensors plus a versioned `manifest.json`, deterministic row access, wrapped reads, pinned memory, and async shard prefetch.

The conversion should reuse these pieces instead of replacing the model, loss, or feature encoders first.

## Core Design Decision
The two things being built are independent and should not be conflated:

1. **A curriculum orchestrator** that owns the staged river→turn→flop process, reusing the live trainer. This is the primary deliverable and works entirely in `live` mode.
2. **A bounded offline dataset path** (`pregenerated` mode) for small, reproducible HP sweeps and for holdout/validation sets.

When offline datasets are used (HP mode or holdouts), store already-solved `RebelBatch` examples, not just roots, because re-solving roots would put CFR back in the loop and roots are nearly as expensive to store as solved examples:

- value examples: features, legal mask, `[B, 2, 1326]` value targets, statistics;
- policy examples: features, legal mask, `[B, 1326, A]` policy targets, statistics;
- optional root snapshots: serialized public env state and beliefs used to create those examples.

These bounded datasets give reproducible HP comparisons and fixed-provenance holdouts; they are not the path for a full run.

## Target Architecture

### Training Loop Refactor (prerequisite for the orchestrator)
The loop currently lives in `train_rebel.py` (`src/p2/cli/train_rebel.py`): wandb init with resume-run-id extraction (`_init_wandb`), `RebelCFRTrainer` construction, the `for step in range(...)` loop calling `trainer.train_step`, `print_training_stats`, periodic checkpointing + `_cleanup_old_checkpoints`, trueskill snapshots, and the final checkpoint.

Extract these guts into a reusable runner so both the single-run CLI and the curriculum orchestrator share one code path:

- New `src/p2/rl/rebel_loop.py` (or similar) with a runner such as:

  ```python
  def run_training_loop(
      trainer: RebelCFRTrainer,
      cfg: Config,
      run,                      # wandb run or nullcontext
      *,
      start_step: int,
      stop_step: int,
      stage_tag: str | None = None,
  ) -> int:                     # returns last step completed
      ...
  ```

  This owns the step loop, stat printing, checkpoint cadence/cleanup, and trueskill snapshots.

- `train_rebel.py` becomes a thin hydra wrapper: build `Config`, `_init_wandb`, build `RebelCFRTrainer`, call `run_training_loop` for `[start_step, cfg.num_steps)`. Behavior must be byte-for-byte unchanged (covered by an existing-behavior test).

### Curriculum Orchestrator
Add a new script (e.g. `src/p2/cli/train_rebel_curriculum.py`) that owns the river→turn→flop process and reuses `run_training_loop`:

- Accepts a **superset** of train_rebel's `Config` (every option train_rebel takes) plus a `curriculum:` subtree (the ordered train/distill sub-steps, per-sub-step step budgets, per-sub-step data/search overrides, and the frozen-net handoff paths `S_X`/`E_X`).
- For each sub-step: `train` sub-steps configure the trainer for street `X` (load frozen `E_X` as the closing-leaf net, train `S_X` to its step budget, promote it); `distill` sub-steps regress `E_X` onto the chance expectation of a frozen `S_{X+1}` and promote `E_X`. Each promoted net is frozen and handed to the next sub-step.
- **Same wandb level as train_rebel**, structured as **one run per stage tied by a wandb `group`** (matches per-stage promotion gates and clean intra-stage resume).
- **Resume must be sub-step-aware.** Checkpoints already carry metadata via `save_checkpoint`; add a `curriculum_substep` marker so a resumed orchestrator restarts the correct train/distill sub-step at the correct intra-sub-step step. Extend the existing run-id extraction in `_init_wandb` (it already reads checkpoint metadata) to recover the per-sub-step wandb run.

### Offline Data Components (for `pregenerated` mode + holdouts only)
Add a postflop offline data package, likely under `src/p2/rebel_data/` or `src/p2/search/offline/`:

- `postflop_spot_sampler.py`
  - Generates legal heads-up public roots for river, turn, and flop.
  - Produces `PublicBeliefState` objects with `[B, 2, 1326]` beliefs.
  - Shared by both live generation (on-the-fly roots) and bounded offline writing.

- `pregenerate_postflop_rebel.py`
  - CLI/script that samples roots, runs CFR, calls `training_data()`, and writes **bounded** sharded solved-example datasets for HP sweeps and holdouts. Not used for full-run data.

- `rebel_solved_dataset.py`
  - Reads manifests and shards.
  - Serves value and policy batches separately.
  - Supports wrapped reads, random row sampling, pinned memory, and async prefetch.

- `rebel_data_source.py`
  - Small abstraction used by `RebelCFRTrainer`:
    - `LiveRebelDataSource`: current `RebelDataGenerator` behavior (the default, production path).
    - `PregeneratedRebelDataSource`: bounded offline datasets for HP sweeps.
    - optional `HybridRebelDataSource`: live training plus a fixed offline holdout for validation metrics.

### Trainer Boundary
Refactor the trainer so `_update_model()` does not know whether data came from live CFR or disk.

The minimal interface:

```python
class RebelDataSource:
    def prepare_step(self, step: int) -> tuple[RebelBatch | None, RebelBatch | None]:
        ...

    def sample_value(self, batch_size: int, stratify_streets: list[float] | None) -> RebelBatch:
        ...

    def sample_policy(self, batch_size: int, stratify_streets: list[float] | None) -> RebelBatch:
        ...

    def state_dict(self) -> dict:
        ...
```

First implementation can keep the existing replay buffers by loading offline chunks into them. That is the lowest-risk trainer change. After parity, add a direct dataset sampler to remove the ring-buffer copy for large datasets.

## Dataset Format

### Layout
Use separate value and policy shard streams because policy examples are much more numerous than value examples.

```text
outputs/rebel_postflop/river_v1/
  manifest.json
  value/
    shard_000000.pt
    shard_000001.pt
  policy/
    shard_000000.pt
    shard_000001.pt
  roots/                 # optional
    shard_000000.pt
```

Each value shard stores:

```python
{
    "features.context": Tensor[B, C],
    "features.street": Tensor[B],
    "features.to_act": Tensor[B],
    "features.board": Tensor[B, 5],
    "features.beliefs": Tensor[B, 2 * 1326],
    "legal_masks": Tensor[B, A],
    "value_targets": Tensor[B, 2, 1326],
    "statistics.<key>": Tensor[B, ...],
}
```

Each policy shard stores the same feature/statistics fields plus:

```python
{
    "policy_targets": Tensor[B, 1326, A],
}
```

Do not pickle whole `RebelBatch` objects in the dataset. Store plain tensors so format migrations and partial reads stay manageable.

### Manifest Fields
The manifest is the guardrail against stale or mixed-quality data:

```json
{
  "format": "p2.rebel.solved_postflop.v1",
  "num_players": 2,
  "hands": 1326,
  "num_actions": 8,
  "stage": "river",
  "root_streets": ["river"],
  "included_streets": ["river"],
  "feature_encoder": "BetterStreetValueFFN/BetterPolicyFFN",
  "model_config": {...},
  "env_config": {...},
  "search_config": {...},
  "action_schedule": {...},
  "spot_sampler_config": {...},
  "target_model": {
    "role": "none|river_chance_leaf|turn_chance_leaf",
    "checkpoint": "...",
    "sha256": "...",
    "step": 0
  },
  "generator": {
    "seed": 0,
    "code_version": "...",
    "device": "cuda"
  },
  "value_examples": 1000000,
  "policy_examples": 8000000,
  "street_counts": {...},
  "node_depth_counts": {...},
  "quality": {
    "cfr_iterations": 400,
    "holdout_value_loss": null,
    "target_model_kl": null
  },
  "shards": {
    "value": [...],
    "policy": [...],
    "roots": [...]
  }
}
```

For bounded turn/flop datasets, `target_model` records the frozen `E_X` used at closing leaves (and the `S_{X+1}` it was distilled from), so the dataset can be regenerated reproducibly.

## Random Spot Generation

### DeepStack Procedure (baseline)
Follow the DeepStack training-situation procedure (paper SOM) as the baseline spot generator, since `S_X`/`E_X` here are the same kind of counterfactual-value nets. A situation at the start of a street is fully specified by **pot size, both players' ranges, and the dealt public cards** — the betting history is not needed because pot + ranges are a sufficient statistic. This is exactly the start-of-street (post-chance) root for `S_X` and the pre-chance state for `E_X`.

**Pot size.** Sample pot from a fixed mixture of intervals, each chosen with uniform probability, then a uniform integer within the chosen interval. DeepStack's intervals (100bb = 20000 chips, in chips) are approximately `{[100,200), [200,400), [400,2000), [2000,6000), [6000,19950]}`. Re-bucket these to our stack/blind scale. The net's actual input is **pot as a fraction of total stacks**, and **counterfactual value targets are expressed as fractions of the pot** — both normalizations are important for generalization and should be kept.

**Ranges — recursive coverage generator `R(S, p)`.** Ranges must cover the space of ranges CFR could encounter *during re-solving*, not just equilibrium ranges. DeepStack's recursive procedure assigns probability mass `p` across a hand set `S`:

1. If `|S| == 1`, assign `Pr(s) = p`.
2. Otherwise:
   a. draw `p1 ~ Uniform(0, p)`, set `p2 = p - p1`;
   b. split `S` into `S1` (weaker half) and `S2` (stronger half) by **hand strength** (probability of beating a uniformly random hand given the current board), with `|S1| = floor(|S|/2)`;
   c. recurse `R(S1, p1)` and `R(S2, p2)`.

A full range is `R(all legal hands, 1)`. This produces ranges polarized/condensed across the strength ordering at random granularities, covering the polarized/capped/asymmetric cases by construction. Generate ranges per player independently, then zero board-blocked hands and renormalize.

**Targets.** Solve each situation with CFR⁺-style iterations restricted to fold / call / pot-sized bet / all-in (no card abstraction); `S_X` targets come from these solves, `E_X` targets from the frozen `S_{X+1}` chance expectation. DeepStack used 10M turn and 1M flop situations; the auxiliary end-of-preflop net used 10M situations with targets from enumerating all 22,100 flops — our `E_preflop` distillation mirrors this (sampled rather than fully enumerated).

The `R(S, p)` generator must be tensorized over `[B, 2, 1326]` (precompute per-board hand-strength orderings; do the recursive split as a batched bisection, no per-hand Python loops). The mixture sampler below is a **complement** to `R(S, p)`, not a replacement — keep both and stratify.

### Start With Street-Start Roots
For the first river implementation, generate roots at the beginning of river betting:

- `street == 3`
- five public board cards set
- `actions_this_round == 0`
- legal `to_act`, `button`, stack, committed, pot, and min-raise fields
- no folded or all-in players unless the spot type explicitly supports it
- beliefs zeroed on board-blocked hands and independently normalized per player

This avoids incoherent mid-street state construction while still producing policy examples at many depths because the CFR tree expands betting actions from the root.

After street-start roots pass validation, add a legal mid-street spot sampler by replaying random legal action prefixes from a street-start state. Do not mutate pot/stacks/actions fields independently.

### River Belief Distribution
Random river beliefs should be broad enough that the network learns public-belief reasoning, not just uniform-range showdowns. The DeepStack `R(S, p)` generator above is the primary source; add this mixture sampler alongside it for explicit coverage of named range shapes:

- uniform board-legal ranges;
- Dirichlet or exponential random ranges with concentration buckets;
- hand-strength-biased ranges using river hand rank vs board;
- polarized ranges with high mass on strong hands and missed hands;
- capped ranges with low mass on nutted hands;
- asymmetric ranges where one player is narrow and the other is diffuse.

Every sampler must be tensorized:

- sample scores for `[B, 2, 1326]`;
- mask board-blocked hands;
- clamp and normalize on device;
- avoid per-hand Python loops.

### Public Spot Distribution
Stratify river roots by:

- board texture: paired, monotone, two-tone, straight-heavy, dry;
- pot size in big blinds;
- SPR bucket;
- stack asymmetry;
- actor/button position;
- previous street aggression proxy through `actions_last_round`;
- min-raise pressure and all-in availability.

The first version can use conservative legal templates:

- single-raised pot style;
- three-bet pot style;
- short-stack low-SPR style;
- deep-stack high-SPR style.

Later versions should source roots from solved turn/flop handoffs and preflop handoffs.

## Curriculum

### Stage 0: Loop Refactor + Orchestrator Plumbing
Before training any curriculum stage:

1. Extract `run_training_loop` from `train_rebel.py` and reduce `train_rebel.py` to a thin wrapper (no behavior change; gate with an existing-behavior test).
2. Add the curriculum orchestrator script that drives train/distill sub-steps via `run_training_loop`, with sub-step-aware resume and one-wandb-run-per-sub-step grouping.
3. Add the `data.mode: live|pregenerated|hybrid` switch and the `RebelDataSource` abstraction with `live` as the default.

Optional offline plumbing (only needed before the first HP sweep, not before live curriculum training):

4. Add the bounded solved dataset writer/reader.
5. Make `pregenerated` mode run one trainer step using a tiny synthetic `RebelBatch` dataset.
6. Add validation that offline batches produce identical loss values to the same batches inserted into `RebelReplayBuffer`.
7. Ensure checkpointing stores dataset cursors and manifests for `pregenerated` mode, and the live generator state otherwise.

This stage should not change model quality. It only moves the loop/data boundaries.

### Stage 1: River (`S_river`)
Train `S_river` live. River CFR is the cheapest street: leaves are exact terminal fold/showdown/all-in values, no chance node ahead, no learned leaf net. There is no `E_river`. This is why river can stay live at full scale even though it cannot be pregenerated to disk.

Data generation (live):

1. Sample river street-start roots on the fly with random beliefs and legal public spots (the spot/belief samplers run tensorized inside the live generator).
2. Run sparse CFR to terminal river leaves.
3. Use exact fold/showdown/all-in terminal values from the evaluator.
4. Feed value and policy examples into the existing replay buffers via `training_data()`.

Holdout / HP mode (bounded, optional):

5. Write a small fixed `D_river_val` holdout (different seeds and spot templates) for validation metrics.
6. Optionally write a bounded `D_river_train` for `pregenerated`-mode HP sweeps only.

Training:

1. Train the postflop `BetterFFN`/`BetterTRM` net from river data.
2. Use suit permutation exactly as live training does.
3. Track value loss by river spot bucket and policy KL by node depth/reach.
4. Promote `S_river` only when the holdout improves and the net is stable across board textures.

Output:

- `S_river`: promoted, frozen river start-of-street net.
- `D_river_val`: bounded holdout; optional bounded `D_river_train` for HP sweeps.

### Stage 1.5: Distill `E_turn` from frozen `S_river`
Build the turn-closing terminal-value net so turn CFR never enumerates the river card in its inner loop.

1. Sample end-of-turn (pre-chance) public belief states: `street == 2`, four board cards, turn betting closed, board-legal beliefs.
2. For each, compute the target as the river-card chance expectation of frozen `S_river` via `single_card_chance_values` (one enumeration per state, done here once).
3. Regress the value-only `E_turn` net onto those targets.
4. Validate `E_turn` against freshly enumerated `S_river` expectations on a holdout; keep exact terminals mixed in where the turn-closing node is already terminal (e.g. all-in).

Output: `E_turn`, frozen, consumed as turn-closing leaf values in Stage 2.

### Stage 2: Turn (`S_turn`)
Train `S_turn` live. Turn-closing leaves are valued by the frozen `E_turn` (one eval per leaf); within-turn depth-limited leaves are valued by `S_turn` itself (normal ReBeL bootstrap). No chance enumeration happens inside the CFR loop.

Data generation (live by default):

1. Sample turn roots:
   - `street == 2`;
   - four public board cards;
   - random but legal turn public spots;
   - board-legal random beliefs.
2. Run CFR through bounded turn betting.
3. At terminal leaves, use exact fold/showdown/all-in values.
4. At turn-closing (round-closed) leaves, read values directly from frozen `E_turn`.
5. Record target provenance:
   - `target_source=terminal_exact`;
   - `target_source=E_turn`;
   - `target_source=turn_cfr_backup`.

Training:

1. Train `S_turn` (a fresh net) on turn spots only — no need to mix river or guard against forgetting, since `S_river`/`E_turn` are frozen.
2. Validate on:
   - a bounded turn holdout;
   - `E_turn` boundary agreement with fresh `S_river` enumerations;
   - a small fresh live-solve probe set for sanity only.
3. Promote `S_turn` when the turn holdout improves and boundary values are stable.

Because each street net trains on its own street only, there is no cross-street spot-mix to tune for the turn stage.

Output:

- `S_turn`: promoted, frozen turn start-of-street net.
- `D_turn_val`: bounded holdout; the manifest records `E_turn` (and the `S_river` it was distilled from) as the boundary value source.

### Stage 2.5: Distill `E_flop` from frozen `S_turn`
Same pattern as Stage 1.5, one street earlier:

1. Sample end-of-flop (pre-chance) public belief states: `street == 1`, three board cards, flop betting closed.
2. Compute targets as the turn-card chance expectation of frozen `S_turn` via `single_card_chance_values` (single-card turn chance, enumerated once here).
3. Regress `E_flop` onto those targets; validate against fresh enumerations; mix in exact terminals where applicable.

Output: `E_flop`, frozen, consumed as flop-closing leaf values in Stage 3.

### Stage 3: Flop (`S_flop`)
Train `S_flop` live. Flop-closing leaves are valued by the frozen `E_flop` (one eval per leaf); within-flop depth-limited leaves are valued by `S_flop` itself. The turn card at the flop boundary is a single-card chance node already amortized into `E_flop`, so flop CFR does no chance enumeration in its inner loop.

Data generation (live by default):

1. Sample flop roots:
   - `street == 1`;
   - three public board cards;
   - random legal flop public spots;
   - board-legal random beliefs.
2. Run bounded flop CFR.
3. At terminal leaves, use exact fold/showdown/all-in values.
4. At flop-closing (round-closed) leaves, read values directly from frozen `E_flop`.
5. Record target provenance and depth/street coverage.
6. Add an additional flop-root source from the multiway preflop handoff builder once that path exists.

Training:

1. Train `S_flop` (a fresh net) on flop spots only.
2. Validate on a bounded flop holdout and on `E_flop` boundary agreement with fresh `S_turn` enumerations.
3. Add real handoff flop roots only after random flop roots are stable.
4. Promote `S_flop` when the flop holdout improves and boundary values are stable.

Output:

- `S_flop`: promoted, frozen flop start-of-street net. Together with `S_turn`, `S_river` it is the heads-up postflop model set used at play time (`street → net` dispatch).
- `D_flop_val`: bounded holdout; the manifest records `E_flop` (and its `S_turn` source) as the boundary value source.

### Stage 3.5: Distill `E_preflop` from frozen `S_flop`
The flop chance node deals **three** cards, so this distillation **samples** flops (`ChanceNodeHelper.flop_chance_values`, `FLOP_SAMPLE_SIZE` flops) rather than enumerating all 22,100. `E_preflop` provides flop-boundary terminal values to the multiway preflop CFR via the handoff in `preflop_multiway_pbs_bootstrap_plan.md`.

## Preflop Handoff
Preflop remains a separate multiway project.

**Multiway preflop, forced fold via a legal-action invariant.** Preflop is solved multiway, and the postflop handoff is always heads-up. The forced fold is implemented **inside** the preflop CFR as a constraint on the legal-action set, not as a post-hoc reduction (see "Forced Fold Via A Legal-Action Invariant" in `preflop_multiway_pbs_bootstrap_plan.md`): a player facing the action when two others are already matched at the current bet level may only re-raise / all-in / fold, never flat-call. So a non-all-in round can only close with ≤ 2 matched seats and the flop is always reached heads-up.

Why this is the unbiased choice: every fold is a voluntary, EV-maximizing player decision, so there is no artificial payoff to assign at the boundary and no incentive distortion to correct. The only closed-preflop boundary is heads-up, valued by `E_preflop` (distilled from `S_flop`). The cost is an explicit, logged abstraction — squeeze-or-fold removes multiway limped/called pots and inflates 3-bet frequencies relative to real poker. Multiway all-in showdowns are still allowed and resolved by the side-pot equity resolver.

Use the plan in `preflop_multiway_pbs_bootstrap_plan.md`:

1. Multiway preflop model solves preflop `PBSEnv` states under the legal-action invariant.
2. `PreflopHandoffBuilder` compacts the two live seats of a closed preflop row into a heads-up flop `PublicBeliefState`.
3. Those flop roots become an additional root source for the flop stage.
4. The postflop nets train only on flop/turn/river examples, never direct preflop examples.

The postflop dataset manifest should tag these rows:

```text
root_source=random_flop
root_source=multiway_preflop_handoff_natural_hu
root_source=multiway_preflop_handoff_forced_fold
```

This lets validation distinguish synthetic random flop skill from preflop-distribution handoff skill.

## Config Sketch

Add a `curriculum` subtree consumed by the orchestrator (a superset of the train_rebel config):

```yaml
curriculum:
  # alternating train / distill sub-steps, run in order; resume re-enters the active sub-step
  stages: [river, distill_E_turn, turn, distill_E_flop, flop, distill_E_preflop]
  wandb_group: rebel_postflop_curriculum   # one run per sub-step, shared group
  river:           { kind: train,   net: S_river, num_steps: 200000 }
  distill_E_turn:  { kind: distill, net: E_turn,  from: S_river, chance: single_card, num_steps: 20000 }
  turn:            { kind: train,   net: S_turn,  closing_net: E_turn, num_steps: 150000 }
  distill_E_flop:  { kind: distill, net: E_flop,  from: S_turn,  chance: single_card, num_steps: 20000 }
  flop:            { kind: train,   net: S_flop,  closing_net: E_flop, num_steps: 150000 }
  distill_E_preflop: { kind: distill, net: E_preflop, from: S_flop, chance: sample_flops, num_steps: 30000 }
```

`train` sub-steps run street CFR + supervised training (the `S_X` nets need value+policy); `distill` sub-steps are value-only supervised regression of `E_X` onto chance expectations of a frozen `from` net, with `chance` selecting single-card enumeration vs flop sampling.

Add a data subtree to the ReBeL config. `live` is the production default; `pregenerated` is for bounded HP sweeps only:

```yaml
data:
  mode: live              # live (production) | pregenerated (HP sweeps) | hybrid (live + holdout)
  pregenerated:
    value_batch_size: ${train.batch_size}
    policy_batch_size: ${train.batch_size}
    shuffle: true
    pin_memory: true
    async_shard_prefetch: true
    validate_manifest: true
    datasets:
      - path: outputs/rebel_postflop/river_v1
        value_weight: 1.0
        policy_weight: 1.0
        min_step: 0
        max_step: null
```

Add a pregeneration config (bounded, HP-sweep/holdout sizes — not full-run scale):

```yaml
rebel_pregenerate:
  output_dir: outputs/rebel_postflop/river_v1
  stage: river
  examples:
    roots: 100000
    value_target_min: 100000
    policy_target_min: 1000000
  shard_size:
    value: 8192
    policy: 32768
  generation_batch_size: 512
  seed: 0
  device: cuda
  env:
    num_players: 2
    stack_mode: weighted_uniform_bb
    min_stack_bb: 10
    mid_stack_bb: 200
    max_stack_bb: 400
  spots:
    street: river
    street_start_only: true
    board_texture_weights: balanced
    spr_buckets: [0.25, 0.75, 1.5, 3.0, 8.0]
    belief_mixture:
      uniform: 0.10
      exponential: 0.35
      strength_biased: 0.25
      polarized: 0.20
      capped: 0.10
  search:
    depth: 5
    iterations: 400
    sparse: true
    sparse_fused: false
  target_model:
    checkpoint: null
    use_model_avg: true
```

Start pregeneration with non-fused sparse CFR for correctness. Add fused sparse generation after parity tests show identical shapes, target ranges, and close-enough policy/value targets.

## Implementation Phases

### Phase 1: Loop Refactor + Orchestrator
1. Extract `run_training_loop` from `train_rebel.py`; reduce `train_rebel.py` to a thin wrapper.
2. Add a test that the wrapper trains/checkpoints/resumes identically to the old loop.
3. Add the curriculum orchestrator script (ordered train/distill sub-steps, per-sub-step budgets, frozen `S_X`/`E_X` handoff).
4. Implement sub-step-aware resume (a `curriculum_substep` marker in checkpoint metadata) and one-wandb-run-per-sub-step grouping.

### Phase 2: Data Source Refactor
1. Extract a `RebelDataSource` interface.
2. Wrap current `RebelDataGenerator` as `LiveRebelDataSource` (the default).
3. Move trainer checkpoint handling behind the data source.
4. Keep existing live behavior as the default.
5. Add tests that live mode still trains and checkpoints.

### Phase 3: Solved Dataset IO (bounded, for HP sweeps/holdouts)
1. Add tensor-only serialization helpers for `MLPFeatures` and `RebelBatch`.
2. Add `RebelSolvedDataset` with separate value/policy streams.
3. Add manifest validation:
   - number of players;
   - number of actions;
   - feature context length;
   - model family;
   - action schedule;
   - street support.
4. Add wrapped sequential reads and random row sampling.
5. Add pinned-memory and async prefetch after the basic reader passes tests.

### Phase 4: River Spot Sampling + Live River Stage
1. Implement board-legal belief mixture sampling (tensorized, on-the-fly).
2. Implement conservative legal river street-start public spot templates.
3. Materialize `PublicBeliefState` roots inside the live generator.
4. Run the live river stage via the orchestrator to produce `S_river`.
5. Add bounded holdout/HP-sweep dataset generation support.

### Phase 5: Offline Trainer (bounded mode)
1. Add `PregeneratedRebelDataSource`.
2. In the first version, fill existing `RebelValueBuffer` and `RebelPolicyBuffer` from disk and let `_update_model()` sample as before.
3. Add validation-batch metrics from held-out offline shards.
4. Make `pregenerated`-mode checkpoints not require the live generator state.
5. Add a tiny end-to-end offline train test.

### Phase 6: Turn (distill `E_turn`, then train `S_turn`)
1. Add the `E_X` distiller: sample end-of-street pre-chance PBS, build targets via `ChanceNodeHelper` over a frozen `S_{X+1}`, regress a value-only net.
2. Distill `E_turn` from frozen `S_river` (single-card river enumeration).
3. Train `S_turn` live: turn-closing leaves read frozen `E_turn`, within-turn depth leaves use `S_turn`; exact terminals where applicable.
4. Record target provenance per row; promote `S_turn`.

### Phase 7: Flop (distill `E_flop`, then train `S_flop`) + `E_preflop`
1. Distill `E_flop` from frozen `S_turn` (single-card turn enumeration).
2. Train `S_flop` live: flop-closing leaves read frozen `E_flop`; promote `S_flop`.
3. Distill `E_preflop` from frozen `S_flop` by **sampling flops** (`flop_chance_values`) for the preflop handoff.
4. Add preflop-handoff flop roots to the flop spot sampler after random flop validation is stable.

### Phase 8: Optimization
1. Add fused sparse CFR support for live and offline generation.
2. Add multi-process or multi-GPU sharding by seed/range partition.
3. Add direct dataset sampling to skip replay-buffer staging.
4. Add dataset compaction or dtype reduction:
   - beliefs and targets can be stored as `float16`/`bfloat16` after accuracy checks;
   - policy targets may need `float16` plus renormalization on load;
   - statistics can remain `float32` or integer.

## Correctness Tests

### Dataset IO
- Round-trip `RebelBatch` tensor serialization.
- Manifest rejects incompatible `num_actions`, `num_players`, context length, and street support.
- Wrapped reads match concatenated direct reads.
- Pinned/prefetched reads match non-prefetched reads.

### Spot Sampling
- Boards contain unique cards and the right number of known cards per street.
- Beliefs are zero on board-blocked hands.
- Beliefs normalize per player without CPU sync in the hot generation path.
- Public state invariants hold after root construction.
- Random mid-street sampler, when added, reaches states only through legal action replay.

### River Targets
- Tiny river supports match explicit showdown/fold EV calculations.
- CFR-generated policy targets are normalized over legal actions per hand.
- Value targets are finite, clipped only where intended, and scale-consistent.

### End-of-Street Distillation (`E_X`)
- `E_X` targets match the chance expectation of frozen `S_{X+1}`: single-card enumeration covers all legal next cards; flop sampling draws `FLOP_SAMPLE_SIZE` distinct legal flops.
- A trained `E_X` agrees with freshly enumerated `S_{X+1}` expectations on a holdout within tolerance.
- During street-`X` CFR, closing-leaf values come from `E_X` only (no chance enumeration in the inner loop); within-street depth leaves come from `S_X`.
- For bounded holdout/HP datasets, the manifest records the frozen `E_X` (and its `S_{X+1}` source); same-seed + same-net regeneration reproduces close numerical targets.

### Trainer / Loop Refactor
- The thin `train_rebel.py` wrapper trains, checkpoints, and resumes identically to the pre-refactor loop (golden-run comparison).
- The orchestrator resumes mid-curriculum into the correct sub-step at the correct intra-sub-step step, recovering the per-sub-step wandb run.
- Existing live mode remains unchanged.
- `pregenerated`-mode training on a fixed tiny dataset gives deterministic loss for a fixed seed and resumes with the same dataset cursor and optimizer state.
- Suit permutation changes board, beliefs, value targets, and policy targets consistently.

## Performance Gates
Track these before scaling dataset size:

- roots solved per second by street;
- policy examples written per second;
- value examples written per second;
- average CFR tree nodes per root;
- GPU memory per generation batch;
- shard write bandwidth;
- dataset load seconds per training step;
- trainer samples/sec in offline mode vs live mode;
- CPU-GPU transfer time with and without pinned prefetch.

Do not optimize storage dtype or fused generation until the non-fused sparse pipeline has stable target correctness.

## Quality Gates
Promote a stage checkpoint only when:

- validation value loss improves on that stage's holdout;
- policy target KL improves or remains stable by depth/reach bucket;
- previous-stage holdout does not regress beyond a configured tolerance;
- fresh small live-solve probes agree with validation trends;
- action mix is not collapsing in major spot buckets;
- target mean/std by street and bucket is stable across generation batches.

For the backward curriculum (each net is frozen once promoted, so there is no prior-street regression to guard against — only its own holdout and its boundary net):

- `S_river` is promoted on river holdout.
- `E_turn` is promoted on boundary agreement with `S_river`; `S_turn` on turn holdout.
- `E_flop` is promoted on boundary agreement with `S_turn`; `S_flop` on flop holdout.
- `E_preflop` is promoted on boundary agreement with `S_flop` (flop-sampled).

## Main Risks

### Dataset Distribution Drift
Random roots can become unlike real poker roots. Mitigation:

- start broad but stratified;
- track bucket metrics;
- add roots from sampled prior-stage continuations;
- eventually mix preflop-handoff flop roots.

### Boundary Approximation Compounding
Each `S_X` trains against `E_X`, which was distilled from `S_{X+1}`, which itself approximated `S_{X+2}` — errors compound backward (river → turn → flop → preflop). Because every net is frozen once promoted, there is no self-bootstrap feedback loop, but stale or biased `E_X` still propagates. Mitigation:

- gate each `E_X` on holdout agreement with fresh chance-enumerated `S_{X+1}` values before training the street that consumes it;
- keep exact terminal rows mixed in (they are bias-free);
- re-distill `E_X` and retrain downstream streets if an upstream net is later improved;
- validate `S_X` closing-leaf reads against small fresh solves that enumerate the chance node directly.

(There is no separate "forgetting" risk: a promoted net is never trained again, so earlier streets cannot be degraded by later training.)

### Storage Scale
Full-run solved datasets do not fit on disk (see Storage Reality): beliefs and value targets are each `2 × 1326` floats, and policy targets are `[B, 1326, A]`. Mitigation:

- treat `live` as the only full-scale data path; never size a run to disk;
- keep `pregenerated` datasets bounded to HP-sweep/holdout sizes;
- for those bounded datasets, separate policy/value shards and support dtype compression after tests;
- keep bounded datasets independently removable/regenerable.

### Main-Loop / Orchestrator Complexity
Refactoring the loop and adding stage orchestration can disturb live training. Mitigation:

- extract `run_training_loop` with a golden-run equivalence test before adding the orchestrator;
- introduce `RebelDataSource` with live mode as a wrapper around current code;
- keep `data.mode=live` as the default;
- avoid changing loss/model code in the same phase as the loop refactor.

## Suggested Milestones
1. `run_training_loop` extraction + thin `train_rebel.py` wrapper (golden-run equivalence).
2. Curriculum orchestrator with sub-step-aware resume and per-sub-step grouped wandb runs.
3. `RebelDataSource` abstraction with no behavior change.
4. River spot/belief samplers and the live river stage → `S_river`.
5. Bounded `RebelBatch` writer/reader + manifest validation (for HP sweeps/holdouts).
6. Tiny `pregenerated`-mode trainer test using synthetic batches.
7. `E_X` distiller; distill `E_turn` from `S_river`, then train `S_turn`.
8. Distill `E_flop` from `S_turn`, then train `S_flop`.
9. Distill `E_preflop` from `S_flop` (flop-sampled) for the preflop handoff.
10. Add multiway preflop handoff flop roots as an additional flop root source.
