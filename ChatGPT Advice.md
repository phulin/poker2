0) What we’re carrying over from AlphaHoldem

Representation & model shape. Keep the pseudo-Siamese CNN that ingests separate tensors for cards and betting history, then fuses them to predict an action (AlphaHoldem encodes card history and action history as 3-D tensors and feeds them through a “pseudo-Siamese” stack).

1) Observation & action spaces (N players)

Observation per seat

Cards tensor: unchanged (private hole cards for the seat + public board channels).

Actions tensor: extend “p1/p2/sum/legal” channels to N per-player action planes + a sum/legal plane (linear growth with N). AlphaHoldem’s split card/action tensors and history encoding still apply.

2) Networks

Shared actor per seat (parameter-sharing across seats):

Trunk = AlphaHoldem’s pseudo-Siamese CNN over (card tensor, action-history tensor).

Heads:

Policy logits over discretized actions.

Value head: scalar seat-value (expected chip EV). (Keep value clipping from Trinal-Clip PPO.)

Optional: a centralized critic that also gets other seats’ public features (never private cards) can reduce variance while preserving imperfect-info integrity.

3) Population self-play (league/PSRO-lite)

Maintain a league:

Main learner (θ).

Historical snapshots of θ (frozen).

Specialists (periodic best-response learners) trained against the current meta.

Lineup sampling

For each self-play episode, sample an N-player lineup from the league using a meta-mix (e.g., softmax over ELO, capped participation so one type doesn’t dominate). AlphaHoldem’s K-Best pool idea provides the curation mechanism; here you sample teams/lineups instead of pairs.

Meta update

Every K steps, compute a small empirical payoff table among league members (few thousand hands per pairing suffices early). Update the meta-mix via projected replicator dynamics or a smoothed ELO-style sampler.

Population growth

Periodically add a BR specialist: copy θ and train it to best-respond to the current meta-mix for M steps; then freeze and add to league.

4) Training loop (pseudocode)

```python
# Initialize
league = League()
theta = init_actor_critic()  # pseudo-Siamese trunk + policy/value heads
league.add("main", theta.snapshot())

replay = FIFOBuffer(capacity=…)
optimizer = Adam(lr=3e-4)

for epoch in range(E):
    # ------ Self-play data collection ------
    batches = []
    for _ in range(episodes_per_epoch):
        lineup = league.sample_lineup(N)  # meta-mix over snapshots/specialists
        traj = run_multiplayer_episode(lineup, action_discretization)
        batches.append(extract_PPO_rollouts(traj))  # obs, act, logp_old, R, Â, legal mask
    replay.add(batches)

    # ------ PPO updates with Trinal-Clip ------
    for _ in range(ppo_updates):
        minibatch = replay.sample(B)
        pi, V = theta(minibatch.obs)

        # Ratio
        r = exp(pi.logp(minibatch.act) - minibatch.logp_old)

        # Trinal-Clip policy loss (αH’s extra δ1 upper bound when Â<0)
        L_policy = mean( clip_three_way(r, 1-ε, 1+ε, δ1) * minibatch.A_hat )

        # Trinal-Clip value loss (clip targets to [−δ2, δ3])
        R_clip = clip(minibatch.returns, -δ2(minibatch), δ3(minibatch))
        L_value = mean( (V - R_clip)**2 )

        # Entropy/reg
        L_entropy = mean(entropy(pi))
        loss = -L_policy + c_v*L_value - c_e*L_entropy

        optimizer.zero_grad(); loss.backward(); optimizer.step()

    # ------ League maintenance ------
    if epoch % eval_every == 0:
        scores = league.round_robin_eval(eval_hands)
        league.update_meta_mix(scores)      # PSRO-lite update
        league.maybe_add_snapshot(theta)    # K-Best style curation
        if need_BR(scores):
            league.spawn_best_response(theta, meta_mix, BR_steps)
```

Notes linking to AlphaHoldem:

The Trinal-Clip PPO policy+value clipping is the same stability trick they used.
The K-Best idea appears in the “compete with K best historical versions” pattern we use to curate league members.

5) Evaluation & model selection

Replace heads-up “mbb/h vs Slumbot/DeepStack” with round-robin, multi-table evaluation across league members; select checkpoints by lowest variance + best average payoff vs the current meta (AlphaHoldem used ELO/mbb/h for selection; same spirit here).
Track population exploitability proxy: add a fresh BR probe periodically; big jumps indicate meta overfitting.

6) Practical settings to start
Seats: 6-max.

Action set: {fold, check/call, 25%, 50%, 75%, pot, 150%, jam}.

Optimizer: Adam(3e-4), GAE λ=0.95, γ=0.999 (same ballpark as AlphaHoldem).

Clips: ε=0.2; δ1≈3 (policy upper bound when Â<0); value clips δ2/δ3 computed from chips in pot/stack as in the paper.

League: start with 1 main + 3 snapshots; eval every ~100k hands; add a BR every ~3–5 evals.

7) Pitfalls & how this setup handles them

Cyclic dynamics in multiplayer: league + BR (PSRO-lite) breaks cycles; K-Best curation avoids data hunger of full population methods.

Noisy returns: Trinal-Clip PPO clamps extreme targets and ratios, stabilizing updates.

Scaling observations to N: linear channel growth with N in the action-history tensor keeps memory sane; seat embeddings absorb per-seat context (position, stacks).