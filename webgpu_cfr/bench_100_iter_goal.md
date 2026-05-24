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

Winning opt-in command:

```bash
/usr/bin/env WEBGPU_BACKEND=metal node --import tsx src/benchSpots.ts --manifest public/models/rebel_latest/model.json --spots bench_spots.json --depth 4 --iterations 100 --warmups 1 --runs 3 --no-cfr-avg --leaf-refresh-interval 8
```

Winning measurement:

- optimized_mean_ms_all_streets: 192.41
- speedup_vs_original: 3.02x
- preflop_mean_ms: 205.78
- flop_mean_ms: 194.78
- turn_mean_ms: 208.31
- river_mean_ms: 160.78

Note: `--leaf-refresh-interval 8` refreshes neural leaf values every 8 CFR
iterations while still running value backup each iteration. The default interval
is 1, preserving exact existing solver behavior unless the option is enabled.
