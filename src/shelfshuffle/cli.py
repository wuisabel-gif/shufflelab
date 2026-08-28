"""
Command-line interface for ShuffleLab.

Examples
--------
    shelfshuffle simulate --cards 52 --p 0.65 --games 100000

    shelfshuffle play --cards 52 --p 0.65
"""

from __future__ import annotations

import argparse
import random
import sys
from typing import List

from .distribution import expected_correct, theoretical_pmf, variance_correct
from .shuffle import SingleShelfShuffle, single_shelf_shuffle
from .simulator import play_game, simulate
from .statistics import empirical_mean, empirical_variance
from .strategy import (
    GameState,
    GreedyBayesStrategy,
    MostLikelyPositionStrategy,
    OptimalCompleteFeedbackStrategy,
    RandomStrategy,
)

STRATEGIES = {
    "optimal": OptimalCompleteFeedbackStrategy,
    "greedy-bayes": GreedyBayesStrategy,
    "most-likely-position": MostLikelyPositionStrategy,
    "random": RandomStrategy,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shelfshuffle",
        description="ShuffleLab: simulate and analyse optimal card guessing "
        "after an asymmetric single-shelf shuffle (arXiv:2607.10418).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="Monte-Carlo the guessing strategy")
    sim.add_argument("--cards", "-n", type=int, required=True)
    sim.add_argument("--p", type=float, required=True)
    sim.add_argument("--games", "-g", type=int, default=10000)
    sim.add_argument(
        "--strategy",
        "-s",
        choices=list(STRATEGIES.keys()),
        default="optimal",
    )
    sim.add_argument("--seed", type=int, default=None)

    play = sub.add_parser("play", help="Play interactively against the optimal algorithm")
    play.add_argument("--cards", "-n", type=int, default=52)
    play.add_argument("--p", type=float, default=0.65)
    play.add_argument("--seed", type=int, default=None)

    pmf = sub.add_parser("distribution", help="Print the exact score distribution")
    pmf.add_argument("--cards", "-n", type=int, required=True)
    pmf.add_argument("--p", type=float, required=True)
    return parser


def cmd_simulate(args) -> int:
    strategy = STRATEGIES[args.strategy]()
    scores = simulate(strategy, args.cards, args.p, games=args.games, seed=args.seed)
    emp_mean = empirical_mean(scores)
    emp_var = empirical_variance(scores)
    theo_mean = expected_correct(args.cards, args.p)
    theo_var = variance_correct(args.cards, args.p)

    print("Optimal complete-feedback strategy" if args.strategy == "optimal"
          else f"Strategy: {args.strategy}")
    print("─" * 42)
    print(f"{'n':<18}{args.cards}")
    print(f"{'p':<18}{args.p:.3f}")
    print(f"{'trials':<18}{args.games:,}")
    print(f"{'empirical mean':<18}{emp_mean:.3f}")
    print(f"{'theoretical mean':<18}{theo_mean:.3f}")
    print(f"{'empirical variance':<18}{emp_var:.3f}")
    print(f"{'theoretical var':<18}{theo_var:.3f}")
    return 0


def cmd_distribution(args) -> int:
    pmf = theoretical_pmf(args.cards, args.p)
    total = sum(pmf)
    print(f"P(X_{args.cards} = k) for p = {args.p}  (sum = {total:.4f})")
    print("─" * 30)
    for k, prob in enumerate(pmf):
        if prob > 1e-12:
            print(f"k = {k:<4} {prob:.6f}  {'#' * int(round(prob * 60))}")
    print("─" * 30)
    print(f"E[X] = {expected_correct(args.cards, args.p):.4f}")
    return 0


def cmd_play(args) -> int:
    n, p = args.cards, args.p
    rng = random.Random(args.seed)
    deck = single_shelf_shuffle(n, p, rng)
    algo = OptimalCompleteFeedbackStrategy()
    state = GameState(n=n, p=p)
    algo_score = 0
    your_score = 0
    revealed_so_far: List[int] = []

    print(f"Asymmetric Shelf Shuffle")
    print(f"Deck size: {n}")
    print(f"p: {p}")
    print()

    for step, top_card in enumerate(deck, start=1):
        post = algo.probabilities(state)
        guess = algo.guess(state)
        p_correct = post.get(guess, 0.0)
        correct = guess == top_card

        print(f"Step {step}")
        print("Cards observed:", ", ".join(str(c) for c in revealed_so_far) or "(none)")
        print(f"Optimal next guess: → {guess}")
        print(f"P(correct | observations): {p_correct:.3f}")
        algo_score += int(correct)
        print(f"Algorithm score: {algo_score} / {step}")

        your_input = input("Your guess (Enter to auto-skip): ").strip()
        if your_input == "":
            print("(skipped your turn)")
        else:
            try:
                you = int(your_input)
                your_score += int(you == top_card)
            except ValueError:
                print("(invalid guess, counted as miss)")
        print(f"Your score:      {your_score} / {step}")
        print(f"Revealed card:   {top_card}")
        print()
        state.revealed.append(top_card)
        revealed_so_far.append(top_card)

    print("=" * 40)
    print(f"Final — Algorithm: {algo_score}/{n}   You: {your_score}/{n}")
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "simulate":
        return cmd_simulate(args)
    if args.command == "distribution":
        return cmd_distribution(args)
    if args.command == "play":
        return cmd_play(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
