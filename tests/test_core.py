import math
import random

import pytest

from shelfshuffle.distribution import (
    expected_correct,
    nu,
    theoretical_pmf,
    variance_correct,
)
from shelfshuffle.position_matrix import position_matrix
from shelfshuffle.shuffle import single_shelf_shuffle
from shelfshuffle.simulator import play_game, simulate
from shelfshuffle.strategy import (
    GameState,
    GreedyBayesStrategy,
    MostLikelyPositionStrategy,
    OptimalCompleteFeedbackStrategy,
    RandomStrategy,
    single_shelf_posterior,
)


def test_position_matrix_doubly_stochastic():
    for n in [1, 2, 5, 20]:
        for p in [0.2, 0.5, 0.8]:
            M = position_matrix(n, p)
            assert len(M) == n and all(len(row) == n for row in M)
            for i in range(n):
                assert math.isclose(sum(M[i]), 1.0, rel_tol=1e-9), (n, p, i)
            for j in range(n):
                col = sum(M[i][j] for i in range(n))
                assert math.isclose(col, 1.0, rel_tol=1e-9), (n, p, j)


def test_single_shelf_unimodal():
    rng = random.Random(0)
    for _ in range(200):
        deck = single_shelf_shuffle(30, 0.7, rng)
        # increasing up to n then decreasing
        peak = deck.index(30)
        assert deck[:peak] == sorted(deck[:peak])
        assert deck[peak + 1:] == sorted(deck[peak + 1:], reverse=True)


def test_optimal_beats_random_and_mostlikely():
    rng = random.Random(1)
    n, p = 40, 0.6
    opt = simulate(OptimalCompleteFeedbackStrategy(), n, p, games=4000, seed=1)
    rnd = simulate(RandomStrategy(rng), n, p, games=4000, seed=2)
    mlp = simulate(MostLikelyPositionStrategy(), n, p, games=4000, seed=3)
    assert expected_correct(n, p) > 5  # sanity on the deck size
    assert abs(sum(opt) / len(opt) - expected_correct(n, p)) < 0.2
    assert sum(opt) / len(opt) > sum(rnd) / len(rnd)
    assert sum(opt) / len(opt) >= sum(mlp) / len(mlp) - 1e-9


def test_symmetric_expectation():
    # p = 1/2 should give E[X_n] = 3n/4 for n >= 2 (Corollary 1).
    for n in [2, 10, 52, 100]:
        assert math.isclose(expected_correct(n, 0.5), 0.75 * n, rel_tol=1e-9)


def test_optimal_matches_theoretical_mean():
    cases = [
        (10, 0.5),
        (52, 0.5),
        (20, 0.9),
        (30, 0.2),
        (12, 0.2),
        (8, 0.35),
    ]
    for n, p in cases:
        scores = simulate(OptimalCompleteFeedbackStrategy(), n, p, games=20000, seed=7)
        emp = sum(scores) / len(scores)
        theo = expected_correct(n, p)
        assert abs(emp - theo) < max(0.15, 0.04 * theo), (n, p, emp, theo)


def test_variance_matches_monte_carlo():
    n, p = 52, 0.5
    scores = simulate(OptimalCompleteFeedbackStrategy(), n, p, games=40000, seed=11)
    emp_var = sum((x - sum(scores) / len(scores)) ** 2 for x in scores) / (len(scores) - 1)
    theo_var = variance_correct(n, p)
    assert abs(emp_var - theo_var) < 0.1, (emp_var, theo_var)


def test_pmf_sums_to_one_and_matches_empirical():
    for n, p in [(8, 0.5), (6, 0.2), (10, 0.3)]:
        pmf = theoretical_pmf(n, p)
        assert math.isclose(sum(pmf), 1.0, rel_tol=1e-9)
        # empirical histogram from simulation
        scores = simulate(OptimalCompleteFeedbackStrategy(), n, p, games=60000, seed=5)
        counts = [0] * (n + 1)
        for s in scores:
            counts[s] += 1
        for k in range(n + 1):
            emp = counts[k] / len(scores)
            assert abs(emp - pmf[k]) < 0.02, (n, p, k, emp, pmf[k])


def test_posterior_deterministic_after_peak():
    # Reveal n -> remaining guessed in descending order with probability 1.
    state = GameState(n=10, p=0.6, revealed=[1, 2, 10])
    post = single_shelf_posterior(state)
    assert post == {9: 1.0}

    # Start of game: geometric law, sums to 1.
    state0 = GameState(n=10, p=0.6)
    post0 = single_shelf_posterior(state0)
    assert math.isclose(sum(post0.values()), 1.0)
    assert post0[1] == pytest.approx(0.6)
    assert post0[10] == pytest.approx(0.4 ** 9)


def test_greedy_bayes_equals_optimal():
    # For the single-shelf model the generic Bayesian argmax coincides with the
    # closed-form optimal strategy.
    rng = random.Random(3)
    for _ in range(50):
        n = rng.randint(2, 30)
        p = rng.random()
        if p == 0 or p == 1:
            continue
        deck = single_shelf_shuffle(n, p, rng)
        a = play_game(deck, OptimalCompleteFeedbackStrategy(), n, p)[0]
        b = play_game(deck, GreedyBayesStrategy(), n, p)[0]
        assert a == b


def test_nu_phase_transition_index():
    assert nu(0.5) == 1
    assert nu(0.9) == 1
    assert nu(0.2) == 8  # floor(log(0.2)/log(0.8)) + 1
    assert nu(0.1) == 22  # floor(log(0.1)/log(0.9)) + 1
