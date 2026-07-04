# River Equity FiLM Experiment

Implemented a low-rank equity-conditioned FiLM residual for river value models.

The branch is enabled with:
- `value_river_range_equity_film_rank > 0`
- `value_river_range_equity_baseline = true`

For the tested proposal, the branch used rank `8` and hidden dim `16`, alongside the existing learned equity feature head and trunk histogram context.

Result for the 500-step pregenerated run:
- Validation value loss: `0.0028294236371011446`
- Pot-relative RMSE: `0.548653397846647`
- Training elapsed: `101.09s`
- Mean training step excluding first: `0.0441s`
