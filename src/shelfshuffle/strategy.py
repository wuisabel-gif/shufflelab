"""
Guessing strategies for the complete-feedback card-guessing game.

All strategies implement a common interface:

    class GuessingStrategy:
        name: str
        def guess(self, state: GameState) -> int: ...
        def probabilities(self, state: GameState) -> dict[int, float]: ...

``guess`` returns the card the player guesses next; ``probabilities`` returns a
distribution (over the still-unrevealed cards) for the *next* revealed card,
used by the interactive demo to draw its bar chart.

The mathematically optimal strategy for the asymmetric single-shelf shuffle is
due to Clay, Kuba and Tripathi (arXiv:2607.10418, Proposition 1).  Because the
shuffle always yields a unimodal deck and the player receives complete feedback,
the optimal next guess at any state depends only on the most recently revealed
card ``last``:

  * once card n (or n-1) has been revealed, the remaining cards are known and
    must be guessed in decreasing order (probability 1 of being correct);
  * otherwise the next card is the smallest still-unseen card above ``last``
    that was sent "up", or n if none are.  Each unseen card j > last is "up"
    independently with probability p, so
        P(next = j) = (1-p)^{j-last-1} p   (for last < j <= n-1)
        P(next = n) = (1-p)^{n-1-last}.
    The optimal guess is the mode of this geometric law, i.e. argmax over
    {last+1, n} of {p, (1-p)^{n-1-last}} (ties broken toward the larger label).

Implemented strategies:

  * RandomStrategy            -- guess uniformly among remaining cards (baseline).
  * MostLikelyPositionStrategy -- ignore feedback; at position j guess the card
        with the largest single-entry position-matrix probability m[i][j].
  * GreedyBayesStrategy       -- generic Bayesian argmax of P(next | history);
        for the single-shelf model this coincides with the optimal strategy.
  * OptimalCompleteFeedbackStrategy -- the closed-form optimal rule above.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .position_matrix import position_matrix


@dataclass
class GameState:
    """State of a complete-feedback guessing game observed so far.

    ``revealed`` lists the cards shown (top -> bottom) up to the current point.
    The next card to be revealed is at ``position`` = len(revealed) + 1.
    """

    n: int
    p: float
    revealed: List[int] = field(default_factory=list)

    @property
    def position(self) -> int:
        return len(self.revealed) + 1

    @property
    def remaining(self) -> List[int]:
        seen = set(self.revealed)
        return [c for c in range(1, self.n + 1) if c not in seen]


def single_shelf_posterior(state: GameState) -> Dict[int, float]:
    """Posterior P(next revealed card = k | history) for the single-shelf model.

    Returns a dict over the still-unrevealed cards.  This is the exact law
    derived from the unimodal structure of the shuffle; see the module docstring.
    """
    n, p = state.n, state.p
    q = 1.0 - p
    rem = set(state.remaining)
    if not rem:
        return {}

    # Once the peak (n) or n-1 is revealed the rest is deterministic.
    if n in state.revealed or (n - 1) in state.revealed:
        return {max(rem): 1.0}

    last = state.revealed[-1] if state.revealed else 0
    probs: Dict[int, float] = {}
    for j in rem:
        if j > last and j <= n - 1:
            probs[j] = (q ** (j - last - 1)) * p
    if n in rem:
        probs[n] = q ** (n - 1 - last)

    total = sum(probs.values())
    if total <= 0:
        return {max(rem): 1.0}
    return {k: v / total for k, v in probs.items()}


class GuessingStrategy(ABC):
    """Abstract base class for a card-guessing strategy."""

    name: str = "strategy"

    @abstractmethod
    def guess(self, state: GameState) -> int:
        """Return the next card to guess given the observed history."""
        raise NotImplementedError

    def probabilities(self, state: GameState) -> Dict[int, float]:
        """Return P(next card = k | history) over remaining cards (for display)."""
        g = self.guess(state)
        rem = state.remaining
        if not rem:
            return {}
        return {c: (1.0 if c == g else 0.0) for c in rem}


class RandomStrategy(GuessingStrategy):
    """Guess uniformly at random among the remaining cards."""

    name = "random"

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()

    def guess(self, state: GameState) -> int:
        rem = state.remaining
        if not rem:
            raise ValueError("no cards remaining")
        return self.rng.choice(rem)

    def probabilities(self, state: GameState) -> Dict[int, float]:
        rem = state.remaining
        if not rem:
            return {}
        return {c: 1.0 / len(rem) for c in rem}


class MostLikelyPositionStrategy(GuessingStrategy):
    """Guess the card most likely to occupy the current position, ignoring
    feedback.  Uses the position-matrix marginal m[i][position-1]."""

    name = "most-likely-position"

    def __init__(self):
        self._cache: Dict[tuple, List[List[float]]] = {}

    def _matrix(self, n, p):
        key = (n, round(p, 12))
        if key not in self._cache:
            self._cache[key] = position_matrix(n, p)
        return self._cache[key]

    def guess(self, state: GameState) -> int:
        M = self._matrix(state.n, state.p)
        pos = state.position - 1
        rem = set(state.remaining)
        best = None
        best_val = -1.0
        for i in range(state.n):
            if (i + 1) not in rem:
                continue
            if M[i][pos] > best_val:
                best_val = M[i][pos]
                best = i + 1
        if best is None:
            raise ValueError("no cards remaining")
        return best

    def probabilities(self, state: GameState) -> Dict[int, float]:
        M = self._matrix(state.n, state.p)
        pos = state.position - 1
        rem = state.remaining
        raw = {c: M[c - 1][pos] for c in rem}
        total = sum(raw.values())
        if total <= 0:
            return {c: 1.0 / len(rem) for c in rem}
        return {c: v / total for c, v in raw.items()}


class GreedyBayesStrategy(GuessingStrategy):
    """Generic Bayesian strategy: guess argmax_k P(next = k | history).

    For the single-shelf model this equals the optimal strategy; the
    implementation is written generically against the posterior so it can be
    reused with other shuffle models that expose a posterior.
    """

    name = "greedy-bayes"

    def __init__(self, posterior=None):
        self._posterior = posterior or single_shelf_posterior

    def guess(self, state: GameState) -> int:
        post = self._posterior(state)
        if not post:
            raise ValueError("no cards remaining")
        # argmax with tie-break toward the larger label (matches the paper).
        return max(post, key=lambda k: (post[k], k))

    def probabilities(self, state: GameState) -> Dict[int, float]:
        return self._posterior(state)


class OptimalCompleteFeedbackStrategy(GuessingStrategy):
    """The optimal complete-feedback strategy (Clay-Kuba-Tripathi,
    arXiv:2607.10418, Proposition 1)."""

    name = "optimal-complete-feedback"

    def guess(self, state: GameState) -> int:
        post = single_shelf_posterior(state)
        if not post:
            raise ValueError("no cards remaining")
        return max(post, key=lambda k: (post[k], k))

    def probabilities(self, state: GameState) -> Dict[int, float]:
        return single_shelf_posterior(state)
