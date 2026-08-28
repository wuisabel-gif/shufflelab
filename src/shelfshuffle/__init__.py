"""
ShuffleLab -- computational experiments for probabilistic card-shuffling models
and optimal guessing strategies.

This package implements the mathematics of the *asymmetric single-shelf shuffle*
and the complete-feedback card-guessing game studied in:

    Alexander Clay, Markus Kuba, Raghavendra Tripathi,
    "On card guessing after an asymmetric single-shelf shuffle",
    arXiv:2607.10418 (2026).

and the underlying single-shelf shuffle / position matrix from:

    Markus Kuba, "On Card guessing after a single shelf shuffle", arXiv:2602.12928 (2026)
    Raghavendra Tripathi, "On the position matrix of single-shelf shuffle and card
    guessing", arXiv:2602.07920 (2026)

The algorithmic/implementation contribution here (the library, simulator,
visualisations, reproducibility framework and experiments) is original; the
underlying mathematics is due to Clay, Kuba and Tripathi and is credited in the
README and CITATION.cff.
"""

from .shuffle import (
    ShuffleModel,
    SingleShelfShuffle,
    RandomToTop,
    UniformShuffle,
    single_shelf_shuffle,
    random_to_top,
    uniform_shuffle,
)
from .position_matrix import position_matrix
from .strategy import (
    GameState,
    GuessingStrategy,
    RandomStrategy,
    MostLikelyPositionStrategy,
    GreedyBayesStrategy,
    OptimalCompleteFeedbackStrategy,
)
from .simulator import play_game, simulate
from .distribution import expected_correct, variance_correct, theoretical_pmf, nu
from .statistics import empirical_mean, empirical_variance

__version__ = "0.1.0"

__all__ = [
    "ShuffleModel",
    "SingleShelfShuffle",
    "RandomToTop",
    "UniformShuffle",
    "single_shelf_shuffle",
    "random_to_top",
    "uniform_shuffle",
    "position_matrix",
    "GameState",
    "GuessingStrategy",
    "RandomStrategy",
    "MostLikelyPositionStrategy",
    "GreedyBayesStrategy",
    "OptimalCompleteFeedbackStrategy",
    "play_game",
    "simulate",
    "expected_correct",
    "variance_correct",
    "theoretical_pmf",
    "nu",
    "empirical_mean",
    "empirical_variance",
]
