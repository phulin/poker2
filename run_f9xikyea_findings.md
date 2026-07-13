# Findings: Run f9xikyea Value Loss Jump

The local evidence does not support a committed codebase change between the start and resume of run `f9xikyea`.

All three W&B directories for this run report the same commit, `e6f4aa7acda55d3136ed642079ae7c65ec50240a`:

- `2026-07-06T02:06:03Z`: original start
- `2026-07-06T09:45:42Z`: first resume attempt, interrupted during compile
- `2026-07-06T09:47:30Z`: successful resume

The successful resume used the same Hydra config except for adding:

```text
resume_from=checkpoints-rebel-curriculum-sturn-5k-no-turneq-8192reg-lr004-cosine-b2048-from-eturn100k-wandb/turn/rebel_latest.pt
```

The value-loss discontinuity is real locally:

- Original run:
  - step 599: `value=0.00078`
  - checkpoint saved at public step 600
  - step 624: `value=0.00081`
  - step 649: `value=0.00084`
  - step 674: `value=0.00081`
  - crashed with CUDA OOM after W&B had logged through step 681
- Successful resume:
  - restored replay buffers
  - resumed from public step 600
  - W&B rejected logs for steps 601-680 as out of order
  - step 624: `value=0.00133`
  - step 649: `value=0.00128`
  - step 674: `value=0.00124`
  - step 699: `value=0.00128`

Most likely explanation: the run lost about 80 completed-but-uncheckpointed training steps when the process OOMed. On resume, it replayed from checkpoint step 600, but W&B had already advanced to step 681, so the duplicate 601-680 metrics were discarded. The first visible resumed points are from a recomputed trajectory, where the live CFR/generated-data stream was not bit-identical to the crashed trajectory.

What cannot be fully ruled out: uncommitted dirty source changes at the time. W&B did not save a code snapshot or patch for this run (`tmp/code` is empty), so only committed-code/config/dependency drift can be checked from local artifacts.
