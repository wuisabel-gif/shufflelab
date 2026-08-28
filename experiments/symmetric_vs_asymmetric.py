"""
Symmetric vs asymmetric: expected number of correct guesses.

Reproduces the symmetric result E[X_n] = 3n/4 (p = 1/2, arXiv:2602.12928 /
Corollary 1 of arXiv:2607.10418) and compares it with the asymmetric theory
for other values of p.

Run:
    python experiments/symmetric_vs_asymmetric.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shelfshuffle.distribution import expected_correct

NS = list(range(2, 53))


def main():
    curves = {
        "p = 0.50 (symmetric)": [expected_correct(n, 0.5) for n in NS],
        "p = 0.65": [expected_correct(n, 0.65) for n in NS],
        "p = 0.20": [expected_correct(n, 0.20) for n in NS],
        "p = 0.90": [expected_correct(n, 0.90) for n in NS],
    }
    plt.figure(figsize=(8, 5))
    for label, vals in curves.items():
        plt.plot(NS, vals, label=label)
    plt.xlabel("deck size n")
    plt.ylabel("E[correct guesses]")
    plt.title("Expected correct guesses under optimal strategy")
    plt.legend()
    plt.grid(alpha=0.3)
    out = os.path.join(os.path.dirname(__file__), "symmetric_vs_asymmetric.png")
    plt.savefig(out, dpi=130)
    print("wrote", out)

    # Print a small table for the largest n.
    print("\nE[X_n] by (n, p):")
    print(f"{'n':>4}  " + "  ".join(f"{p:<8}" for p in [0.5, 0.65, 0.2, 0.9]))
    for n in [5, 10, 20, 52]:
        row = "  ".join(f"{expected_correct(n, p):<8.3f}" for p in [0.5, 0.65, 0.2, 0.9])
        print(f"{n:>4}  {row}")


if __name__ == "__main__":
    main()
