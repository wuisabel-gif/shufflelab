"""
Optional Streamlit demo for ShuffleLab.

Run with:
    pip install streamlit
    streamlit run web/interactive-demo/app.py

The core maths is delegated to the shelfshuffle library; this file only provides
the UI. (The dependency-free HTML demo in the same folder needs no install.)
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import streamlit as st

from shelfshuffle.shuffle import single_shelf_shuffle
from shelfshuffle.strategy import GameState, OptimalCompleteFeedbackStrategy

st.set_page_config(page_title="ShuffleLab", layout="wide")
st.title("ShuffleLab — play against the optimal card-guessing algorithm")

col1, col2, col3 = st.columns(3)
n = col1.number_input("Deck size n", min_value=2, max_value=200, value=20)
p = col2.number_input("p", min_value=0.01, max_value=0.99, value=0.65)
seed = col3.number_input("seed", min_value=0, value=1)

if "deck" not in st.session_state or st.session_state.get("cfg") != (n, p, seed):
    st.session_state.deck = single_shelf_shuffle(int(n), float(p), random.Random(int(seed)))
    st.session_state.revealed = []
    st.session_state.algo = 0
    st.session_state.cfg = (n, p, seed)

state = GameState(n=int(n), p=float(p), revealed=list(st.session_state.revealed))
algo = OptimalCompleteFeedbackStrategy()
post = algo.probabilities(state)
guess = algo.guess(state)
p_correct = post.get(guess, 0.0)

st.markdown(f"**Optimal next guess:** → {guess}  "
            f"(P(correct | observations) = {p_correct:.3f})")
st.markdown(f"**Algorithm score:** {st.session_state.algo} / {int(n)}")

st.subheader("Cards observed")
st.write(st.session_state.revealed)

st.subheader("Posterior over possible next cards")
rows = sorted(post.items(), key=lambda kv: -kv[1])[:10]
st.bar_chart({str(k): v for k, v in rows})

if st.button("Reveal next card") and len(st.session_state.revealed) < int(n):
    top = st.session_state.deck[len(st.session_state.revealed)]
    if guess == top:
        st.session_state.algo += 1
    st.session_state.revealed.append(top)
    st.experimental_rerun()
