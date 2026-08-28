"""
Strategy comparison: how much better is the mathematically optimal strategy?

Monte-Carlo the four guessing strategies over a grid of (n, p) and report mean
scores. This directly answers the research question posed for the project:
the optimal (and greedy-Bayes, which coincides with it for this model)
strategy dominates the naive baselines.

Run:
    python experiments/strategy_comparison.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shelfshuffle.simulator import simulate
from shelfshuffle.strategy import (
    GreedyBayesStrategy,
    MostLikelyPositionStrategy,
    OptimalCompleteFeedbackStrategy,
    RandomStrategy,
)

STRATEGIES = {
    "optimal": OptimalCompleteFeedbackStrategy,
    "greedy-bayes": GreedyBayesStrategy,
    "most-likely-position": MostLikelyPositionStrategy,
    "random": RandomStrategy,
}

GRID = [
    (10, 0.5),
    (20, 0.5),
    (52, 0.5),
    (20, 0.65),
    (52, 0.65),
    (20, 0.2),
    (30, 0.2),
    (12, 0.2),
    (10, 0.9),
]


def main():
    games = 4000
    results = {name: [] for name in STRATEGIES}
    for (n, p) in GRID:
        for name, cls in STRATEGIES.items():
            rng = None if name == "random" else None
            strat = cls() if name != "random" else cls()
            scores = simulate(strat, n, p, games=games, seed=hash((n, p, name)) % (2**31))
            results[name].append(sum(scores) / len(scores))

    # Table
    print(f"{'n':>4} {'p':<5}  " + "  ".join(f"{name:<18}" for name in STRATEGIES))
    for idx, (n, p) in enumerate(GRID):
        row = "  ".join(f"{results[name][idx]:<18.3f}" for name in STRATEGIES)
        print(f"{n:>4} {p:<5}  {row}")

    # Bar chart for a couple of representative settings
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (n, p) in zip(axes, [(52, 0.65), (30, 0.2)]):
        idx = GRID.index((n, p))
        vals = [results[name][idx] for name in STRATEGIES]
        ax.bar(list(STRATEGIES.keys()), vals, color=["#4ade80", "#6ea8fe", "#f59e0b", "#f87171"])
        ax.set_title(f"n = {n}, p = {p}")
        ax.set_ylabel("mean correct guesses")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "strategy_comparison.png")
    fig.savefig(out, dpi=130)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
