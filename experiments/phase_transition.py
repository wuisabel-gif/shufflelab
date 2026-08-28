"""
Phase transition: heat map of E[X_n(p)] over (n, p).

The theory (arXiv:2607.10418, Section 1.1.4) predicts a sharp change in
behaviour of the expected score as p -> 0 and p -> 1. This script visualises
E[X_n(p)] on a grid and saves a heat map.

Run:
    python experiments/phase_transition.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shelfshuffle.distribution import expected_correct, nu

NS = list(range(2, 61))
PS = np.linspace(0.02, 0.98, 48)


def main():
    grid = np.zeros((len(NS), len(PS)))
    for i, n in enumerate(NS):
        for j, p in enumerate(PS):
            grid[i, j] = expected_correct(n, p)

    fig, ax = plt.subplots(figsize=(9, 6))
    # normalise per-row (per n) so the phase structure is visible
    im = ax.imshow(
        grid,
        aspect="auto",
        origin="lower",
        extent=[float(PS.min()), float(PS.max()), min(NS), max(NS)],
        cmap="viridis",
    )
    ax.set_xlabel("p")
    ax.set_ylabel("deck size n")
    ax.set_title("E[correct guesses] = E[X_n(p)]  (Clay-Kuba-Tripathi)")
    fig.colorbar(im, ax=ax, label="E[X_n(p)]")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "phase_transition.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)

    # Show the phase-transition index nu(p) as a function of p.
    print("\nnu(p) = phase-transition index (1 for p >= 1/2):")
    for p in [0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.9]:
        print(f"  p = {p:<5}  nu = {nu(p)}")


if __name__ == "__main__":
    main()
