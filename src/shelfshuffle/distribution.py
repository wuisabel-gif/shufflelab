"""
Exact distribution, expectation and variance of the number of correct guesses
under the optimal strategy (Clay-Kuba-Tripathi, arXiv:2607.10418, Theorems 1-2).
"""

from __future__ import annotations

import math
from typing import Dict, List


def nu(p: float) -> int:
    """The phase-transition index nu(p) from arXiv:2607.10418, eq. (1).

    nu = 1 for p >= 1/2;  floor(log p / log(1-p)) + 1 for p < 1/2.
    """
    if p >= 0.5:
        return 1
    return int(math.floor(math.log(p) / math.log(1.0 - p))) + 1


def expected_correct(n: int, p: float) -> float:
    """E[X_n], the expected number of correct guesses (Theorem 1)."""
    q = 1.0 - p
    v = nu(p)
    if 1 <= n <= v:
        return q * n + p
    return (1.0 - p * q) * n + 2.0 * p - (v + 1) * p * p - (q ** v)


def variance_correct(n: int, p: float) -> float:
    """Var(X_n), the variance of the number of correct guesses (Theorem 1)."""
    q = 1.0 - p
    v = nu(p)
    pq = p * q
    if 1 <= n <= v:
        return (n - 1) * pq
    if n == v + 1:
        qv = q ** v
        return (v - 1) * pq + (1.0 - 2.0 * (v - 1) * p) * qv - qv * qv
    qv = q ** v
    return (
        pq * (1.0 - 3.0 * pq) * n
        - 2.0 * pq
        + (3.0 * v + 5.0) * (p ** 2) * (q ** 2)
        + (1.0 - 2.0 * v * p + 2.0 * p * p) * qv
        - qv * qv
    )


def theoretical_pmf(n: int, p: float) -> List[float]:
    """Probability mass function pi_{n,k} = P(X_n = k), returned as a list
    indexed by k in 0..n (Theorem 2).

    For p >= 1/2 the closed sum is used; for p < 1/2 and n > nu the recurrence
        pi_{n,k} = p pi_{n-1,k-1}
                 + sum_{j=2}^{n-1} p q^{j-1} pi_{n-j,k-j+1}
                 + q^{n-1} 1_{k = n-1}
    is evaluated by dynamic programming, seeded by the binomial regime
    pi_{m,k} = C(m-1, k-1) q^{k-1} p^{m-k} for 1 <= m <= nu.
    """
    if n == 0:
        return [1.0]
    if p >= 0.5:
        q = 1.0 - p
        pmf = [0.0] * (n + 1)
        pmf[n] = p ** (n - 1)
        lo = math.ceil(n / 2.0)
        for k in range(lo, n):
            s = 0.0
            for m in range(n - k, k + 1):
                if m - 1 < 0 or n - k - 1 < 0:
                    continue
                s += (
                    math.comb(m - 1, n - k - 1)
                    * math.comb(n - m, n - k)
                    * (q ** m)
                    * (p ** (n - 1 - m))
                )
            pmf[k] = s
        return pmf

    q = 1.0 - p
    v = nu(p)
    pi: Dict[int, List[float]] = {}
    for m in range(1, v + 1):
        pm = [0.0] * (m + 1)
        for k in range(1, m + 1):
            pm[k] = math.comb(m - 1, k - 1) * (q ** (k - 1)) * (p ** (m - k))
        pi[m] = pm
    for m in range(v + 1, n + 1):
        pm = [0.0] * (m + 1)
        for k in range(1, m + 1):
            val = 0.0
            if 1 <= k - 1 <= m - 1:
                val += p * pi[m - 1][k - 1]
            for j in range(2, m):
                kk = k - j + 1
                if 1 <= kk <= m - j:
                    val += p * (q ** (j - 1)) * pi[m - j][kk]
            if k == m - 1:
                val += q ** (m - 1)
            pm[k] = val
        pi[m] = pm
    return pi[n]
