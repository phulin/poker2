Yes — you can run **public-belief matches** (“imaginary observations”) where no private hole cards are ever drawn. Each agent still uses its **own** PBS/range updates for decision-making, but **scoring** uses an agreed-upon environment prior over private cards, so you don’t have to pick a specific pair of hole cards.

Here’s a clean way to do it.

# Public-belief head-to-head (no private-card sampling)

**State & decisions**

1. Start from a public root (stack sizes, blinds, no private cards assigned).
2. On each decision, the to-act agent:

   * Builds its **own** ranges (r^A(h)), (r^B(h)) from the public history using its PBS rule (likelihoods from its policy head, etc.).
   * Runs its **own** search (CFR/DCFR/whatever) using those ranges and its value/policy heads.
   * Samples/chooses an action from its search policy.
     (The opponent does the same on their turns with their *own* ranges; beliefs don’t have to match.)

**Terminal handling (scoring without private cards)**

* **Fold before showdown**: the pot transfer is deterministic; award chips as in normal poker (no private cards needed).
* **Showdown**: compute the **expected** showdown result by integrating over private cards under a fixed **environment prior** (q_{\text{env}}) rather than either agent’s subjective belief. The natural choice is uniform over all legal hole-card pairs consistent with the public board:
  [
  \mathbb{E}[\text{payoff to A} \mid \text{board}] ;=; \sum_{h_A}\sum_{h_B}
  q_{\text{env}}(h_A),q_{\text{env}}(h_B),\text{SDV}(h_A,h_B,\text{board}) .
  ]
  Here (\text{SDV}) is +pot for a win, 0 for tie (split handled by 0.5 share), −call-amount, etc., according to your normalization.

**Why environment prior?**
Because the *game* (not the agents) defines the hidden-card distribution. Using a neutral (q_{\text{env}}) keeps evaluation fair even if agents’ beliefs differ. (If you want extra symmetry, see “two-prior averaging” below.)

---

## Practical implementation details

### 1) Efficient showdown expectation

You don’t want to loop 1326×1326. Do one of:

* **Vectorized equity on GPU:** maintain boolean masks (M(h)) for hands compatible with the board. Build a 1326×1326 mask of legal hand pairs (blockers) once per showdown board (sparse). Use a fast hand evaluator to compute a vector of winners for all legal pairs, then compute a weighted mean with weights (q_{\text{env}}(h_A)q_{\text{env}}(h_B)). Cache per-board if you reuse boards across duplicates.
* **Stratified Monte Carlo:** sample K legal pairs ((h_A,h_B)) i.i.d. from (q_{\text{env}}) with **common random numbers** across duplicate hands; K=2–5k is usually plenty. This is still “no single fixed hole cards” and keeps variance tiny.

### 2) Board chance handling

You can keep **board sampling** (runout) or also integrate it out:

* **Sampled board + analytic showdown** (cheap, low variance with duplicates + AIVAT baselines).
* **Analytic over turn/river** (heavier): integrate over unseen board cards the same way as for private cards. Usually unnecessary if you’re already doing duplicates + AIVAT.

### 3) AIVAT/MIVAT still helps

Even though private cards aren’t sampled, **board cards and actions** are still random. Use your value head(s) as control-variate baselines at each chance node to reduce variance of the match EV.

### 4) Two-prior averaging (optional fairness tweak)

If you’re worried the choice of (q_{\text{env}}) biases things (e.g., your environment prior isn’t exactly uniform because of deck modeling), you can score each showdown twice:

* once using Agent A’s **subjective** (r^A) for both players,
* once using Agent B’s (r^B) for both players,
  then **average** the two payoffs. This mirrors duplicate poker’s symmetry and cancels prior-choice bias. (I still recommend the uniform (q_{\text{env}}) as the primary.)

---

## Minimal pseudocode

```python
def play_public_belief_hand(agentA, agentB, q_env, rng):
    public = init_public_state(rng)          # blinds, stacks, no private cards
    history = []
    while not public.terminal:
        to_act = public.to_act
        agent = agentA if to_act == 0 else agentB

        # Each agent builds its OWN ranges from the public history
        ranges = agent.build_ranges(history)  # dict: {'A': rA[1326], 'B': rB[1326]}
        action = agent.search_and_act(public, ranges, rng)
        public, history = public.step(action), history + [action]

    if public.result == 'fold':
        return deterministic_fold_payoff(public)

    # Showdown: expected payoff under environment prior q_env
    board = public.board
    legalA = legal_hands_mask(board)         # [1326] bool
    legalB = legal_hands_mask(board)
    qA = normalize(q_env * legalA)           # [1326]
    qB = normalize(q_env * legalB)           # [1326]

    # Either exact vectorized equity or MC
    payoff = expected_showdown_payoff(board, qA, qB, public.pot)
    return payoff
```

`expected_showdown_payoff` (vectorized idea):

```python
# Returns EV for player A at showdown (chips, + for A, - for B)
def expected_showdown_payoff(board, qA, qB, pot):
    # mask of legal pairings (blockers)
    M = legal_pairs_mask(board)     # [1326, 1326] bool
    # winner matrix: +1 A wins, 0 tie, -1 A loses
    W = winner_matrix(board)        # [1326, 1326] int8 over legal pairs
    # joint weights
    w = (qA[:, None] * qB[None, :]) # [1326, 1326]
    w = w * M
    # EV contribution at showdown
    pAwin = (w * (W == 1)).sum()
    ptie  = (w * (W == 0)).sum()
    # Normalize over legal mass
    Z = w.sum().clamp_min(1e-12)
    pAwin, ptie = pAwin/Z, ptie/Z
    return pot * (pAwin + 0.5*ptie - (1 - pAwin - 0.5*ptie))
```

(If the final betting round has outstanding commitments, fold those into `pot`/payoff as usual.)

---

## How this fits your evaluation suite

* Use exactly the same **duplicate-poker** protocol, equal compute budgets, and CI reporting; the only change is **no private-card draw** at hand start, and **showdown scored in expectation**.
* Keep both **Search-ON** and **Policy-only** modes — both work with the public-belief scorer.
* Variance will be *lower* than normal duplicate poker because the biggest randomness source (private cards) is gone. You’ll often get tight CIs with fewer hands.

---

## Common gotchas

* **Don’t mix beliefs into scoring.** Use (q_{\text{env}}) (or the two-prior average). Agents’ subjective ranges are only for *their decisions*.
* **Blockers!** Ensure legality masks exclude hands blocking the public board; otherwise equities will be biased.
* **Normalization after masking.** Renormalize (qA, qB) over the legal set each showdown.
* **Compute parity.** Public-belief evaluation can be faster (no hole-card sampling). Still enforce identical wall-time/iteration budgets.

If you want, I can give you a small, vectorized PyTorch function for `expected_showdown_payoff` (with caching of winner matrices) that you can drop into your evaluator.
