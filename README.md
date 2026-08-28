# ShuffleLab

**Computational experiments for probabilistic card-shuffling models and optimal guessing strategies.**

ShuffleLab implements and visualises recent results on the **single-shelf shuffle**
and the **complete-feedback card-guessing game**: the exact optimal strategy,
the explicit distribution of correct guesses, its mean, variance, central limit
behaviour, large deviations and phase transitions.

```text
                 ShuffleLab
                    │
        ┌───────────┴───────────┐
        │                       │
   Shuffle models         Guessing agents
        │                       │
  ┌─────┴─────┐          ┌──────┴─────────┐
  │           │          │                │
Single-shelf  Random-  Random   MostLikely   GreedyBayes   Optimal
             to-top   Guess    Position      (generic)     CompleteFeedback
        │                       │
        └───────────┬───────────┘
                    │
               Experiments
        ┌───────────┼────────────┐
   Distribution   CLT tests   Phase transitions
```

## Credit and provenance

The mathematics implemented here is due to:

> Alexander Clay, Markus Kuba, Raghavendra Tripathi,
> *On card guessing after an asymmetric single-shelf shuffle*,
> arXiv:2607.10418 (2026). https://arxiv.org/abs/2607.10418

with the underlying single-shelf shuffle and position matrix from

> Markus Kuba, *On Card guessing after a single shelf shuffle*, arXiv:2602.12928 (2026)
> Raghavendra Tripathi, *On the position matrix of single-shelf shuffle and card guessing*, arXiv:2602.07920 (2026)

**This repository is an independent reference implementation, simulator, visualisation
and experimentation framework. It is not authored by, nor an official companion
implementation of, the paper's authors.** The algorithmic, software and experimental
contributions are original; the underlying theory is theirs and is cited throughout.
If you build on this work, please cite the papers above (see `CITATION.cff`).

## What it does

- Simulates the asymmetric single-shelf shuffle (parameter `p ∈ (0,1)`) exactly.
- Implements the optimal complete-feedback guessing strategy (closed form) and
  three baselines behind a common `GuessingStrategy` interface.
- Computes the *exact* score distribution, mean and variance (no simulation needed).
- Monte-Carlo the strategies and compares them across `n` and `p`.
- Ships an interactive web demo where you play against the optimal algorithm.

## Installation

```bash
git clone https://github.com/example/shelfshuffle
cd shelfshuffle
pip install -e .            # core is pure Python (no required deps)
pip install -e ".[dev]"     # numpy + matplotlib for the experiments
pip install -e ".[demo]"    # streamlit (optional richer demo)
```

## Command-line

```bash
# Monte-Carlo the optimal strategy and compare with the exact theory
shelfshuffle simulate --cards 52 --p 0.65 --games 100000

# Print the exact score distribution
shelfshuffle distribution --cards 20 --p 0.5

# Play interactively against the algorithm (text mode)
shelfshuffle play --cards 52 --p 0.65
```

Example `simulate` output (the optimal strategy tracks the exact theory to within
Monte-Carlo error; for p = 0.65 the shuffle is highly predictable, so ~40/52
correct is expected):

```text
Optimal complete-feedback strategy
────────────────────────────────────
n                 52
p                 0.650
trials            100,000
empirical mean     40.276
theoretical mean   40.275
empirical variance 3.799
theoretical var    3.783
```

For comparison, the symmetric shuffle (p = 1/2) gives E[X_n] = 3n/4 = 39 and
Var(X_n) = n/16 = 3.25 at n = 52.

## Library

```python
from shelfshuffle import (
    SingleShelfShuffle, OptimalCompleteFeedbackStrategy,
    simulate, expected_correct, variance_correct, theoretical_pmf,
)

n, p = 52, 0.65
deck = SingleShelfShuffle(n, p).shuffle()
scores = simulate(OptimalCompleteFeedbackStrategy(), n, p, games=20000, seed=0)

print(expected_correct(n, p))    # 40.275...
print(variance_correct(n, p))    # 3.783...
print(sum(scores) / len(scores)) # ~ empirical mean
```

## Interactive web demo

Open **`web/interactive-demo/index.html`** in any browser (no build, no server).
Pick a deck size and `p`, simulate one shuffle, and play against the optimal
algorithm. After every revealed card the app shows the mathematically optimal
next guess, `P(correct | observations)`, the running scores, and a bar chart of
the posterior over possible next cards.

You can also run the Streamlit version (optional): `streamlit run web/interactive-demo/app.py`.

## Research playground

The library is structured around two interfaces so you can ask experimental
questions directly:

```python
class ShuffleModel:            class GuessingStrategy:
    def shuffle(rng): ...          def guess(state): ...
                                  def probabilities(state): ...
```

Implemented strategies: `RandomStrategy`, `MostLikelyPositionStrategy`,
`GreedyBayesStrategy`, `OptimalCompleteFeedbackStrategy`. Implemented models:
`SingleShelfShuffle`, `RandomToTop`, `UniformShuffle`.

Run the experiments:

```bash
python experiments/symmetric_vs_asymmetric.py   # E[X_n] curves, incl. 3n/4
python experiments/phase_transition.py          # (n, p) -> E[X_n] heat map
python experiments/strategy_comparison.py        # how much better is optimal?
```

## Status

Early but functional: the core mathematics is unit-tested against the paper's
exact mean, variance and distribution. A natural next step is to extend the
experiments with the local CLT, large-deviation rate function and to fold in
the other TA preprints — which is exactly the kind of numerical work not present
in the paper.

## License

MIT — see `LICENSE`.
