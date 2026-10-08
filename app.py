import re
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

import ptp_pipeline as pp  # must be importable so joblib can unpickle the engine

st.set_page_config(page_title="PTP Credibility Engine", page_icon="📊", layout="wide")

ENGINE_PATH = Path(__file__).parent / "ptp_engine.joblib"


@st.cache_resource
def load_engine():
    engine = joblib.load(ENGINE_PATH)
    # warm-up so the first real prediction isn't flagged as slow
    engine.predict_live({"transcript": "AGENT: hello\nBORROWER: haan"})
    return engine


def clean_transcript(raw):
    """The model only reads lines starting with AGENT: or BORROWER:. Fix casing, drop everything else."""
    lines = []
    for ln in raw.splitlines():
        m = re.match(r"\s*(agent|borrower)\s*:\s*(.*)", ln, re.I)
        if m and m.group(2).strip():
            lines.append(f"{m.group(1).upper()}: {m.group(2).strip()}")
    return "\n".join(lines)


if not ENGINE_PATH.exists():
    st.error("ptp_engine.joblib not found. Run generate_synthetic_data.py and train_model.py first.")
    st.stop()
engine = load_engine()

# Use the exact categories the model was trained on (a wrong label is silently ignored by the encoder).
cats = dict(zip(pp.CATEGORICAL, engine.pre.named_transformers_["cat"].categories_))

st.title("PTP Credibility Engine")
st.write("Estimate whether a Promise to Pay is likely to be kept, and what kind of promise it is.")
st.divider()

with st.form("ptp_form"):
    st.subheader("PTP details")
    c1, c2, c3 = st.columns(3)
    with c1:
        portfolio = st.selectbox("Portfolio", list(cats["portfolio"]))
        dpd_bucket = st.selectbox("DPD bucket", list(cats["dpd_bucket"]), index=min(2, len(cats["dpd_bucket"]) - 1))
        channel = st.selectbox("Channel", list(cats["channel"]))
    with c2:
        emi_amount = st.number_input("EMI amount (Rs)", min_value=500.0, value=5000.0, step=500.0)
        promised_amount = st.number_input("Promised amount (Rs)", min_value=0.0, value=5000.0, step=500.0)
        ability_to_pay = st.slider("Ability to pay estimate (0 = none, 1 = full)", 0.0, 1.0, 0.5, 0.05)
    with c3:
        n_past_ptps = st.number_input("Previous PTPs", min_value=0, value=5, step=1)
        past_keep_rate = st.slider("Past PTP keep rate", 0.0, 1.0, 0.7, 0.05,
                                   help="Ignored when there are no previous PTPs.")
        days_from_salary = st.slider("Promised date vs salary credit (days apart)", 0, 15, 5)

    st.subheader("Conversation")
    transcript = st.text_area(
        "Paste the call. Every line must start with AGENT: or BORROWER:", height=180,
        placeholder="AGENT: Aapki EMI pending hai, kab tak payment karenge?\n"
                    "BORROWER: Salary aayi toh 15 tarikh ko kar dunga.\nAGENT: Theek hai.")

    with st.expander("Call and agent signals (neutral defaults)"):
        a1, a2, a3 = st.columns(3)
        with a1:
            days_to_promise = st.slider("Days until promised date", 1, 30, 7)
            negotiation_turns = st.slider("Negotiation turns", 0, 6, 1)
            agree_latency_sec = st.slider("Seconds to agree", 1, 30, 10)
        with a2:
            promise_before_hangup = st.checkbox("Promise came right before hangup")
            voice_stress = st.slider("Voice stress (0-1)", 0.0, 1.0, 0.4, 0.05)
            agent_vs_peers = st.slider("Agent keep rate vs peers (0 = average)", -0.3, 0.3, 0.0, 0.05)
        with a3:
            mins_to_shift_end = st.slider("Minutes to shift end", 0, 480, 240, 10)
            remark_copy_score = st.slider("Remark copy score (0-1)", 0.0, 1.0, 0.15, 0.05)

    submitted = st.form_submit_button("Analyze PTP", type="primary")

if submitted:
    clean = clean_transcript(transcript)
    if not clean:
        st.warning("No usable lines found. Start each line with AGENT: or BORROWER:.")
        st.stop()
    if "BORROWER:" not in clean:
        st.warning("No BORROWER lines found, so there is nothing to read the promise from.")

    record = dict(
        portfolio=portfolio, dpd_bucket=dpd_bucket, channel=channel,
        past_keep_rate=past_keep_rate if n_past_ptps > 0 else np.nan,
        n_past_ptps=n_past_ptps, past_broken_ptps=round(n_past_ptps * (1 - past_keep_rate)),
        emi_amount=emi_amount, promised_amount=promised_amount,
        promised_amount_ratio=float(np.clip(promised_amount / emi_amount, 0, 1)),
        ability_to_pay=ability_to_pay, promised_days_from_salary=days_from_salary,
        days_to_promise=days_to_promise, negotiation_turns=negotiation_turns,
        agree_latency_sec=agree_latency_sec, promise_before_hangup=int(promise_before_hangup),
        voice_stress=voice_stress, agent_keep_rate_vs_peers=agent_vs_peers,
        mins_to_shift_end=mins_to_shift_end, remark_copy_score=remark_copy_score,
        transcript=clean)
    result = engine.predict_live(record)

    st.divider()
    st.subheader("Prediction")
    m1, m2, m3 = st.columns(3)
    m1.metric("PTP credibility (kept probability)", f"{result['keep_probability']:.1%}")
    m2.metric("PTP type", result["ptp_type"].replace("_", " ").title())
    m3.metric("Recommended route", result["route"].replace("_", " ").title())

    if result["hardship_guard_triggered"]:
        st.error("Hardship signals heard. No pressure prompts. Route to a human or restructuring.")
    elif result["weak_promise"]:
        st.warning("Weak promise: firm it up with the prompt below, within Fair Practices Code limits.")
    else:
        st.success("Promise looks credible.")

    d1, d2 = st.columns(2)
    with d1:
        st.write("**Agent prompt**")
        st.info(result["agent_prompt"])
        st.write(f"**Parking:** {result['parking']}")
    with d2:
        st.write("**Signals detected**")
        if result["reason_codes"]:
            for r in result["reason_codes"]:
                st.write(f"• {r}")
        else:
            st.write("None of the tracked weak-promise signals were found.")
        ms = result["latency_ms"]
        st.write(f"**Model latency:** {ms:.1f} ms (excludes speech-to-text)")
        if result["over_hard_limit"]:
            st.error("Over the 1,500 ms hard limit")
        elif not result["within_budget"]:
            st.warning("Above the 300 ms target")

    st.subheader("PTP type probabilities")
    for name, p in sorted(result["type_probabilities"].items(), key=lambda x: -x[1]):
        st.progress(float(min(max(p, 0.0), 1.0)), text=f"{name.replace('_', ' ').title()}: {p:.1%}")

st.divider()
st.caption("Prototype trained on synthetic data. Predictions should support human decisions, not replace them.")