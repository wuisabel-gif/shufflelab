"""
Simulator for the complete-feedback card-guessing game.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from .shuffle import ShuffleModel, single_shelf_shuffle, SingleShelfShuffle
from .strategy import GameState, GuessingStrategy


def play_game(
    deck: List[int], strategy: GuessingStrategy, n: int, p: float
) -> Tuple[int, List[int], List[int]]:
    """Play one complete-feedback game against ``deck`` (top -> bottom).

    Returns ``(score, guesses, revealed)`` where ``score`` is the number of
    correct guesses, ``guesses`` is the sequence of guesses, and ``revealed`` is
    the sequence of actually shown cards.
    """
    state = GameState(n=n, p=p)
    guesses: List[int] = []
    correct = 0
    for top_card in deck:
        guess = strategy.guess(state)
        guesses.append(guess)
        if guess == top_card:
            correct += 1
        state.revealed.append(top_card)
    return correct, guesses, list(state.revealed)


def simulate(
    strategy: GuessingStrategy,
    n: int,
    p: float,
    games: int = 10000,
    seed: Optional[int] = None,
    model: Optional[ShuffleModel] = None,
) -> List[int]:
    """Monte-Carlo the strategy over ``games`` single-shelf shuffles.

    Returns the list of scores (correct-guess counts), one per game.
    """
    rng = random.Random(seed)
    model = model or SingleShelfShuffle(n=n, p=p)
    scores: List[int] = []
    for _ in range(games):
        deck = model.shuffle(rng)
        score, _, _ = play_game(deck, strategy, n, p)
        scores.append(score)
    return scores
