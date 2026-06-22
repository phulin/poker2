# Notes: All-In 3P Optimization

## Baseline
- Exact canonical resident CUDA path stores all canonical board representatives and ranks on GPU.
- Previous profiler: Triton accumulation kernel dominated runtime; launch overhead was not material.
- Warmed subset benchmark after current work-batch refactor:
  - `class_ids=range(16)`: about 0.82s
  - `class_ids=range(32)`: about 7.95s

## Current Bottleneck
- The Triton kernel still scans board representatives for each concrete hand triple chunk.
- This is proportional to `compatible concrete triples * canonical boards`.

## Ambitious Direction
- For each board and hero combo, aggregate opponent combo counts by class and rank relation to hero rank.
- Use those aggregates to compute all caller-class pairs for one hero combo without iterating every concrete caller tuple.
- Need handle caller combo overlap exactly; class histograms alone overcount pairs where `h1` and `h2` share a card.

## Dense Gram Prototype
- Formula matched the existing implementation exactly on sampled boards.
- Best quick benchmark for `class_ids=range(32)` was 32 canonical boards in 0.172s at batch 16, about 186 board reps/sec.
- This is far slower than the current Triton tuple-scan path for the same subset, so the dense exact overlap-correction path is not viable as implemented.

## Next Candidate
- Benchmark tuple suit-orbit reduction with exhaustive board streaming (`canonical_boards=False`, `tuple_orbits=True`). It processes more boards but can reduce private tuple multiplicity by suit symmetry.

## Pair-CDF Triton Prototype
- Implemented throwaway `/tmp/pair_cdf_proto.py` with a fused kernel over ordered `(h0,h1)` pairs and `h2` class tiles.
- It uses sorted rank prefix histograms plus exact four-card blocker subtraction and six double-subtraction add-backs.
- Best `class_ids=range(32)` sweep: about 2,000 canonical reps/sec (`BP=16, BB=2, BC=16`).
- Current tuple kernel is about 16,900 canonical reps/sec for the same range, so this pair-CDF form is about 8x slower despite lower asymptotic pair count. Prefix/card-incidence memory traffic dominates.
