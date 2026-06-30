## Directory summary
Design notes and implementation plans for larger architecture changes that span multiple package areas.

### Source files
- `preflop_multiway_pbs_bootstrap_plan.md`: Detailed plan for using multiway `PBSEnv` preflop-only solving, arbitrary-depth preflop value targets, forced heads-up flop handoff, separate preflop/postflop models, and multiway all-in showdown resolution without `H^P` payoff tables.
- `preflop_showdown_equity_approximation_design.md`: Design for fast preflop multiway all-in/showdown equity approximation by streaming sampled boards through the optimized tier-2 by-hand showdown evaluator and accumulating numerator/denominator vectors without `H^P` tables.
- `multiway_rebel_plan.md`: Plan for extending the ReBeL trainer, CFR evaluator, and MLP value/policy models from heads-up play to 3- and 4-way public-belief training with sampled-depth value targets.
- `rebel_pregenerated_postflop_curriculum_plan.md`: Plan for converting heads-up postflop ReBeL production training to live random legal postflop spot generation, with optional bounded offline solved-example datasets for sweeps/holdouts, bootstrapping river to turn to flop and connecting later to multiway preflop handoff roots.
- `showdown_per_hand_equity_vector_plan.md`: Plan for converting exact, approximate, and Monte Carlo showdown evaluators to return per-hand equity vectors in `R^H`.
- `flop_turn_allin_training_examples_plan.md`: Plan for extending `p2.allin` pregeneration, targets, datasets, and training from preflop all-in states to board-aware flop and turn all-in examples.
- `notes_eos_leaf_investigation.md`: Completed notes from the BTN end-of-street leaf value investigation, including EOS model wiring, target construction, dataset coverage, in-domain belief probes, and ReBeL policy/value target consistency.
- `task_plan_eos_leaf_investigation.md`: Completed task plan for the EOS leaf value investigation.
- `notes_preflop_4_7_restart.md`: Operational notes from monitoring and restarting the preflop actions_4_7 backward-induction run with updated learning-rate settings.
- `task_plan_preflop_4_7_restart.md`: Completed task plan for the preflop actions_4_7 restart monitor.
- `task_plan_preflop_continuation_beliefs.md`: Completed task plan for generating and saving preflop continuation belief cascades.

### Subdirectories
There are no child source directories.
