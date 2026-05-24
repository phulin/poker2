# Depth-4 100-Iteration Dawn Benchmark Goal

Command:

```bash
/usr/bin/env WEBGPU_BACKEND=metal node --import tsx src/benchSpots.ts --manifest public/models/rebel_latest/model.json --spots bench_spots.json --depth 4 --iterations 100 --warmups 1 --runs 3 --no-cfr-avg
```

Metric: mean of the 12 per-spot `meanMs` values from `benchSpots.ts`, covering all streets.

Baseline measured on current committed solver:

- original_mean_ms_all_streets: 581.35
- goal_mean_ms_all_streets_2x: 290.68

Street means from baseline:

- preflop_mean_ms: 681.29
- flop_mean_ms: 667.87
- turn_mean_ms: 657.40
- river_mean_ms: 318.85
