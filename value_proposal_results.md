# Value Proposal Results

All runs use the requested pregenerated dataset:
`outputs/rebel_postflop/river_value_100steps_102400_300it_20260630/manifest.json`.

Config caveat: the runs currently listed below were produced before
`scripts/run_value_arch_proposal.py` was fixed to Hydra-compose
`conf/config_rebel_curriculum_river.yaml`. They used structured-config defaults
(`hidden_dim=1536`, `ffn_dim=1024`, `range_hidden_dim=128`,
`num_hidden_layers=3`, `num_value_layers=3` unless overridden), not the intended
overnight baseline (`hidden_dim=384`, `ffn_dim=768`, `range_hidden_dim=192`,
`num_hidden_layers=0`, `num_value_layers=7`). Treat these as pre-fix exploratory
results, not the final comparable sweep.

Per-step timing is persisted from `flops_hand_basis_r256` onward via
`step_time_s` in `metrics.jsonl` and `step_timing` in `summary.json`; the first
two FLOP-reduction runs only have wall-clock totals plus console output.

| Proposal | Status | Final validation value loss | Elapsed |
| --- | --- | ---: | ---: |
| `residual` | done | 0.0232757590 | 153.67s |
| `board_embed` | done | 0.0232025281 | 160.02s |
| `cross_range` | done | 0.0232474527 | 130.82s |
| `card_token` | done | 0.0284049879 | 165.43s |
| `range_stats` | done | 0.0234437757 | 137.57s |
| `multi_token` | done | 0.0263532971 | 176.07s |
| `second_moment` | done | 0.0232858253 | 134.19s |
| `bet_strat` | done | 0.0231579263 | 148.10s |
| `bet_strat_film` | done | 0.0231169730 | 146.54s |
| `bet_strat_relative` | done | 0.0231851922 | 142.65s |
| `bet_strat_k8` | done | 0.0231903816 | 143.66s |
| `bet_strat_film_relative` | done | 0.0231209791 | 144.98s |
| `flops_value_lr512` | done | 0.0230421013 | 129.48s |
| `flops_value_lr192` | done | 0.0231472338 | 138.01s; mean step excl. first 1.288s |
| `flops_value_lr256_bet_film` | done | 0.0229515126 | 152.65s |
| `flops_hand_basis_r256` | done | 0.0257893550 | 138.37s; mean step excl. first 1.306s |
| `flops_belief_low256` | done | 0.0232921875 | 127.85s; mean step excl. first 1.198s |
| `flops_thin_value_tower` | done | 0.0232354183 | 128.72s; mean step excl. first 1.214s |
| `flops_hidden512` | pending | pending | pending |
| `flops_value_layers5` | pending | pending | pending |
