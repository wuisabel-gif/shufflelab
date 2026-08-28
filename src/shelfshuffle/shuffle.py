"""
Shuffle models.

The flagship model is the *asymmetric single-shelf shuffle* with parameter
p in (0, 1) (arXiv:2607.10418, Definition 1).  Cards labelled 1..n are
processed from the bottom of the ordered deck; each card is placed at the top
of a growing pile with probability p and at the bottom with probability 1-p.
Equivalently, for j = 1..n-1 let eps_j = 1 (card j sent "up") with probability
p and 0 ("down") otherwise, independently.  The shuffled deck (read top to
bottom) is then

    [cards with eps = 1, in increasing order] , n ,
    [cards with eps = 0, in decreasing order].

This always produces a unimodal permutation: increasing up to n, then
decreasing.  The helper ``single_shelf_shuffle`` samples such a deck exactly.

We also provide two further models to expose the library's common interface:
``RandomToTop`` (a standard nearby-transposition / random-to-top shuffle) and
``UniformShuffle`` (uniform random permutation).
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence


def single_shelf_shuffle(n: int, p: float, rng: Optional[random.Random] = None) -> List[int]:
    """Sample one deck from the asymmetric single-shelf shuffle.

    Returns the deck as a list of card labels ordered top -> bottom.
    """
    if n < 1:
        raise ValueError("deck size n must be a positive integer")
    if not (0.0 < p < 1.0):
        raise ValueError("parameter p must lie in (0, 1)")
    rng = rng or random.Random()
    eps = [rng.random() < p for _ in range(n - 1)]  # eps[j] is for card j+1
    top = [j + 1 for j in range(n - 1) if eps[j]]
    bottom = [j + 1 for j in range(n - 1) if not eps[j]]
    return sorted(top) + [n] + sorted(bottom, reverse=True)


def random_to_top(n: int, steps: Optional[int] = None, rng: Optional[random.Random] = None) -> List[int]:
    """Sample one deck from the random-to-top shuffle.

    Starting from the identity deck [1, 2, ..., n], repeat `steps` times: pick a
    uniformly random position and move that card to the top.  With steps=None the
    deck is mixed for n * log2(n) + 1 steps (a common cutoff heuristic).
    """
    if n < 1:
        raise ValueError("deck size n must be a positive integer")
    rng = rng or random.Random()
    if steps is None:
        steps = int(n * (n).bit_length()) + 1 if n > 1 else 1
    deck = list(range(1, n + 1))
    for _ in range(steps):
        k = rng.randrange(n)
        card = deck.pop(k)
        deck.insert(0, card)
    return deck


def uniform_shuffle(n: int, rng: Optional[random.Random] = None) -> List[int]:
    """Sample one deck from the uniform distribution over all permutations."""
    if n < 1:
        raise ValueError("deck size n must be a positive integer")
    rng = rng or random.Random()
    deck = list(range(1, n + 1))
    rng.shuffle(deck)
    return deck


class ShuffleModel(ABC):
    """Common interface for a shuffle model."""

    name: str = "shuffle"

    def __init__(self, n: int, **kwargs):
        self.n = n
        self.params = kwargs

    @abstractmethod
    def shuffle(self, rng: Optional[random.Random] = None) -> List[int]:
        """Return one sampled deck, ordered top -> bottom."""
        raise NotImplementedError


class SingleShelfShuffle(ShuffleModel):
    """Asymmetric single-shelf shuffle with parameter p."""

    name = "single-shelf"

    def __init__(self, n: int, p: float = 0.5):
        super().__init__(n, p=p)
        self.p = p

    def shuffle(self, rng: Optional[random.Random] = None) -> List[int]:
        return single_shelf_shuffle(self.n, self.p, rng)


class RandomToTop(ShuffleModel):
    """Random-to-top shuffle."""

    name = "random-to-top"

    def __init__(self, n: int, steps: Optional[int] = None):
        super().__init__(n, steps=steps)
        self.steps = steps

    def shuffle(self, rng: Optional[random.Random] = None) -> List[int]:
        return random_to_top(self.n, self.steps, rng)


class UniformShuffle(ShuffleModel):
    """Uniform random permutation."""

    name = "uniform"

    def shuffle(self, rng: Optional[random.Random] = None) -> List[int]:
        return uniform_shuffle(self.n, rng)
