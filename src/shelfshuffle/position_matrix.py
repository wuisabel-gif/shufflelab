"""
Position matrix of the asymmetric single-shelf shuffle.

For an n-card asymmetric single-shelf shuffle with parameter p, let
m[i][j] = P(card (i+1) lands in position (j+1)), 0 <= i, j < n.
arXiv:2607.10418, Proposition 1 gives the closed form

    m[i][j] = C(i, j) p^{j+1} (1-p)^{i-j}
            + C(i, n-1-j) p^{i - (n-1-j)} (1-p)^{n-j}

using 0-based indices i = card-1, j = position-1, and the convention
C(a, b) = 0 for b < 0 or b > a.

The matrix is doubly stochastic, and m[i][n-1-j] = m[i][j].
"""

from __future__ import annotations

import math
from typing import List


def position_matrix(n: int, p: float) -> List[List[float]]:
    """Return the n x n position matrix for the asymmetric single-shelf shuffle."""
    if n < 1:
        raise ValueError("deck size n must be a positive integer")
    if not (0.0 < p < 1.0):
        raise ValueError("parameter p must lie in (0, 1)")
    q = 1.0 - p
    M: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):  # i = card - 1
        card = i + 1
        for j in range(n):  # j = position - 1
            # term 1: card i+1 placed "up"
            t1 = 0.0
            a = j  # choose j of the i smaller cards to also go up
            if 0 <= a <= i:
                t1 = math.comb(i, a) * (p ** (j + 1)) * (q ** (i - j))
            # term 2: card i+1 placed "down"
            t2 = 0.0
            b = n - 1 - j  # cards placed below among the i smaller
            if 0 <= b <= i:
                t2 = math.comb(i, b) * (p ** (i - b)) * (q ** (n - j))
            M[i][j] = t1 + t2
    return M
