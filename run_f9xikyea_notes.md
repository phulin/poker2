# Notes: Run f9xikyea Value Loss Jump

## Sources

### Local W&B directories
- `wandb/run-20260706_020603-f9xikyea`
  - Started: `2026-07-06T02:06:03.565861Z`
  - Program: `-m p2.cli.train_rebel_curriculum`
  - Commit: `e6f4aa7acda55d3136ed642079ae7c65ec50240a`
  - Command included:
    - `--config-name config_rebel_curriculum_turn`
    - `curriculum.stages=[turn]`
    - `num_steps=5000`
    - `curriculum.substeps.turn.num_steps=5000`
    - `model.value_turn_range_equity_baseline=false`
    - `checkpoint_dir=checkpoints-rebel-curriculum-sturn-5k-no-turneq-8192reg-lr004-cosine-b2048-from-eturn100k-wandb`
    - `use_wandb=true`
    - `+wandb_name=sturn_5k_no_turneq_8192reg_lr004_cosine_b2048_from_eturn100k`
  - Last printed region:
    - step 574: value `0.00082`
    - step 599: value `0.00078`
    - saved checkpoint at public step 600
    - step 624: value `0.00081`
    - step 649: value `0.00084`
    - step 674: value `0.00081`
  - Crashed with CUDA OOM while computing postflop all-in payoff table.
- `wandb/run-20260706_094542-f9xikyea`
  - Started: `2026-07-06T09:45:42.189609Z`
  - Commit: same `e6f4aa7acda55d3136ed642079ae7c65ec50240a`
  - Same command plus `+resume_from=.../turn/rebel_latest.pt`.
  - Interrupted during torch compile; no useful resumed training metrics.
- `wandb/run-20260706_094730-f9xikyea`
  - Started: `2026-07-06T09:47:30.093438Z`
  - Commit: same `e6f4aa7acda55d3136ed642079ae7c65ec50240a`
  - Same command plus `+resume_from=.../turn/rebel_latest.pt`.
  - Restored replay buffers and resumed public step 600.
  - W&B warned that steps 601-680 were below current step 681 and were ignored.
  - Resumed values:
    - step 600: value `0.00085`
    - step 624: value `0.00133`
    - step 649: value `0.00128`
    - step 674: value `0.00124`
    - step 699: value `0.00128`
    - step 724: value `0.00120`

### Config and dependency comparisons
- Hydra config diff between original `outputs/2026-07-06/02-06-03/.hydra/config.yaml` and successful resume `outputs/2026-07-06/09-47-29/.hydra/config.yaml` only adds `resume_from`.
- W&B requirements diff between original and resume is empty.
- Local W&B code snapshots under `tmp/code` are empty, so there is no saved uncommitted diff/code artifact.

### Commit history comparison
- Original run start: `2026-07-06T02:06:03Z`.
- Successful resume: `2026-07-06T09:47:30Z`.
- W&B-reported commit for both: `e6f4aa7a` from `2026-07-05 15:06:24 +0000`.
- Local `git log --all` shows no commits between `2026-07-05 15:06:24 +0000` and `2026-07-07 18:15:16 +0000`.
- The next commits after the resume were on `2026-07-07`, starting with `59ad6ef7 Initialize BetterFFN embeddings deterministically`, about 32.5 hours after the successful resume.

### Relevant code behavior
- `src/p2/runtime/training_run.py` resumes W&B with `resume="must"` when a checkpoint contains a `wandb_run_id`.
- `src/p2/rl/rebel_loop.py` logs every training step to W&B but prints only at `log_interval`.
- `src/p2/rl/rebel_loop.py` saves checkpoints only when `(step + 1) % checkpoint_interval == 0`; this run used interval 200.
- `src/p2/rl/cfr_trainer.py` saves model, optimizer, trainer RNG, buffer RNG, data-source state, and a shared replay-buffer sidecar for latest checkpoints.
- `src/p2/stages/curriculum.py` loads a checkpoint and starts at `loaded_step + 1`.

## Synthesized Findings

### Code/config change assessment
- There is no local evidence of a committed code change between the original start and resume: all three local W&B directories report commit `e6f4aa7a`.
- Local commit history supports this: no commits landed on July 6 between the original start and resume.
- There is no local evidence of dependency drift: W&B `requirements.txt` files match.
- There is no config drift except adding `resume_from`.
- Uncommitted dirty changes at the time cannot be ruled out because W&B did not save code snapshots or diffs for this run.

### Most likely cause of the visible value-loss jump
- The process crashed after W&B had advanced to public step 681, but the last checkpoint was public step 600.
- On resume, training restarted from step 600 and recomputed steps 601-680.
- W&B rejected the recomputed metrics for steps 601-680 because the run's current step was already 681.
- The graph therefore jumps from the original trajectory around step 680 (`value_loss` around `0.00077` in W&B summary / `0.00081` printed at step 674) to the resumed trajectory after 681 (`~0.0012-0.0013`).
- The resumed trajectory is not bit-identical to the pre-crash trajectory, likely because live CFR data generation and GPU kernels are not fully deterministic across process restarts even when checkpoint state is restored.
