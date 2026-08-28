"""
Empirical statistics and central-limit approximations used by the experiments
and the CLI.
"""

from __future__ import annotations

from typing import List, Tuple

from .distribution import expected_correct, variance_correct


def empirical_mean(scores: List[float]) -> float:
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def empirical_variance(scores: List[float]) -> float:
    if len(scores) < 2:
        return 0.0
    m = empirical_mean(scores)
    return sum((x - m) ** 2 for x in scores) / (len(scores) - 1)


def clt_approximation(n: int, p: float) -> Tuple[float, float]:
    """Mean and standard deviation of X_n under the optimal strategy (the
    (local) central limit theorem of arXiv:2607.10418, Section 1.1.3)."""
    return expected_correct(n, p), variance_correct(n, p) ** 0.5
