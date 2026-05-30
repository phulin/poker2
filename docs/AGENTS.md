## Directory summary
Design notes and implementation plans for larger architecture changes that span multiple package areas.

### Source files
- `preflop_multiway_pbs_bootstrap_plan.md`: Detailed plan for using multiway `PBSEnv` preflop-only solving, arbitrary-depth preflop value targets, forced heads-up flop handoff, separate preflop/postflop models, and multiway all-in showdown resolution without `H^P` payoff tables.
- `preflop_showdown_equity_approximation_design.md`: Design for fast preflop multiway all-in/showdown equity approximation by streaming sampled boards through the optimized tier-2 by-hand showdown evaluator and accumulating numerator/denominator vectors without `H^P` tables.
- `multiway_rebel_plan.md`: Plan for extending the ReBeL trainer, CFR evaluator, and MLP value/policy models from heads-up play to 3- and 4-way public-belief training with sampled-depth value targets.
- `showdown_per_hand_equity_vector_plan.md`: Plan for converting exact, approximate, and Monte Carlo showdown evaluators to return per-hand equity vectors in `R^H`.

### Subdirectories
There are no child source directories.
