#!/usr/bin/env python3
"""
Live PTP credibility demo (Streamlit).

Shows, utterance by utterance, how the model reads a call:
  * transcript with detected signals highlighted (hedges, conditions, vague dates, hardship, ...)
  * keep probability over time, PTP type probabilities, agent prompt, route, reason codes
  * latency split into ASR time vs model time against the 300 ms target / 1500 ms hard limit

Input modes (tabs):
  1. Replay      : built-in demo calls or calls from ptp_synthetic.csv (no mic or ASR needed)
  2. Mic         : record one utterance at a time, pick who spoke, faster-whisper transcribes
  3. Recorded    : upload a STEREO call (left = agent, right = borrower), channels transcribed
                   separately, so no diarization is needed, then replayed in time order
  4. Type        : type an utterance by hand (fallback when ASR fails)

Catches handled:
  * speaker labels   : manual speaker choice (mic) or channel split (stereo); mono uploads are rejected
  * ASR latency      : measured and shown separately from model latency
  * Hinglish noise   : normalize() maps Devanagari and spelling variants to the phrases the model knows,
                       plus fuzzy token repair; raw vs normalised text is shown
  * ASR confidence   : low-confidence utterances are excluded from scoring (slider)
  * partial calls    : model rescored after every utterance; "enough heard" badge
  * hardship guard   : banner + no-pressure prompt, same rule as the engine

Run:
    pip install streamlit faster-whisper pandas numpy scikit-learn joblib
    streamlit run app.py   (this file is the 'Live Call' page, in the pages/ folder)
"""
import difflib
import html
import os
import re
from pathlib import Path
import tempfile
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import ptp_pipeline as pp
from ptp_pipeline import (COND_RE, FIRM_RE, HARDSHIP_RE, HEDGE_RE, PRESSURE_RE, THIRD_RE, VAGUE_RE)

ROOT = Path(__file__).resolve().parent.parent
ENGINE_PATH = str(ROOT / "ptp_engine.joblib")
DATA_PATH = str(ROOT / "ptp_synthetic.csv")
WHISPER_PROMPT = ("EMI payment, salary aayi toh, dekhta hoon, try karunga, kar dunga, "
                  "tarikh, mera beta dega, naukri chali gayi.")

# ------------------------------------------------------------------ ASR text normalisation
DEVA = {"देखता हूँ": "dekhta hoon", "देखता हूं": "dekhta hoon", "देखते हैं": "dekhte hain",
        "कोशिश करूँगा": "try karunga", "कोशिश करूंगा": "try karunga", "पता नहीं": "pata nahi",
        "शायद": "shayad", "तारीख": "tarikh", "सैलरी आएगी तो": "salary aayi toh",
        "सैलरी आई तो": "salary aayi toh", "पैसे आए तो": "paise aaye toh", "अगले हफ्ते": "agle hafte",
        "महीने के आखिर तक": "mahine ke end tak", "नौकरी चली गई": "naukri chali gayi",
        "मेरा बेटा": "mera beta", "पत्नी": "patni", "पापा": "papa", "मुश्किल है": "mushkil hai",
        "ठीक है": "theek hai", "हाँ": "haan", "रुपये": "rupaye"}
VARIANTS = [
    (re.compile(r"dekh\s*(?:ta|tha|taa)\s*(?:hu|hoon|hun|hoo)\b"), "dekhta hoon"),
    (re.compile(r"dekh\s*(?:te|the)\s*(?:hai|hain|hein)\b"), "dekhte hain"),
    (re.compile(r"try\s*kar(?:unga|oonga|ta hu|ta hoon)"), "try karunga"),
    (re.compile(r"\b(salary|paise|paisa|bonus|payment)\s+(?:aa|aayi|ayi|ai|aaye|aye|aaya|aya|mila)\s+(?:toh|to|tho)\b"),
     r"\1 aayi toh"),
    (re.compile(r"\b(\d{1,2})\s*(?:th|st|nd|rd)\b"), r"\1 tarikh"),
    (re.compile(r"\b(\d{1,2})\s*(?:tareekh|tarik|taarikh|date)\b"), r"\1 tarikh"),
    (re.compile(r"pata\s+nai\b|pata\s+nahi\b"), "pata nahi"),
    (re.compile(r"month\s*end"), "month end"),
]
VOCAB = ["dekhta", "dekhte", "karunga", "koshish", "shayad", "tarikh", "salary", "mushkil",
         "hopefully", "naukri", "hospital", "medical"]


def normalize(text):
    s = text.strip()
    for k, v in DEVA.items():
        s = s.replace(k, v)
    s = s.lower()
    for rx, rep in VARIANTS:
        s = rx.sub(rep, s)
    out = []
    for w in s.split():
        core = w.strip(".,?!;:")
        if len(core) >= 5 and core not in VOCAB:
            m = difflib.get_close_matches(core, VOCAB, n=1, cutoff=0.82)
            if m:
                w = w.replace(core, m[0])
        out.append(w)
    return " ".join(out)


# ------------------------------------------------------------------ highlighting
BORROWER_PATTERNS = [("hedge", HEDGE_RE, "#f5a623"), ("condition", COND_RE, "#f8e71c"),
                     ("vague date", VAGUE_RE, "#bd10e0"), ("firm date", FIRM_RE, "#7ed321"),
                     ("third party", THIRD_RE, "#4a90e2"), ("hardship", HARDSHIP_RE, "#d0021b")]
AGENT_PATTERNS = [("pressure", PRESSURE_RE, "#9b9b9b")]


def highlight(text, speaker):
    spans = []
    for name, rx, color in (BORROWER_PATTERNS if speaker == "BORROWER" else AGENT_PATTERNS):
        spans += [(m.start(), m.end(), name, color) for m in rx.finditer(text)]
    spans.sort()
    out, pos = [], 0
    for s, e, name, color in spans:
        if s < pos:
            continue
        out.append(html.escape(text[pos:s]))
        out.append(f'<span title="{name}" style="background:{color};color:#000;padding:1px 5px;'
                   f'border-radius:4px">{html.escape(text[s:e])}</span>')
        pos = e
    out.append(html.escape(text[pos:]))
    return "".join(out)


LEGEND = " ".join(f'<span style="background:{c};color:#000;padding:1px 6px;border-radius:4px;font-size:12px">{n}</span>'
                  for n, _, c in BORROWER_PATTERNS + AGENT_PATTERNS)

# ------------------------------------------------------------------ demo calls
DEMOS = {
    "Genuine and feasible": (0.9, [
        ("AGENT", "Namaste, aapki EMI Rs 4000 pending hai."), ("BORROWER", "ji haan pata hai"),
        ("AGENT", "Kab tak payment kar paayenge?"), ("BORROWER", "10 tarikh ko pakka kar dunga"),
        ("BORROWER", "Rs 4000 UPI se bhej dunga"), ("BORROWER", "theek hai, dhanyavaad")]),
    "Escape promise (noisy ASR spelling)": (0.2, [
        ("AGENT", "Aapki EMI Rs 6000 pending hai, aaj hi payment karna padega."),
        ("BORROWER", "Dekhta hu, try karta hoon"), ("AGENT", "Kab tak kar denge?"),
        ("BORROWER", "next week dekh te hai"), ("BORROWER", "haan haan ok ab rakhta hoon mujhe jaana hai")]),
    "Genuine but not feasible (hardship)": (0.6, [
        ("AGENT", "Namaste, aapki EMI Rs 5000 pending hai."),
        ("BORROWER", "sir meri naukri chali gayi hai, salary nahi mili 2 mahine se"),
        ("AGENT", "Samajh sakta hoon, kab tak possible hoga?"),
        ("BORROWER", "salary aayi to 25 tarikh tak kuch kar paunga")]),
    "Third-party promise": (0.5, [
        ("AGENT", "Namaste, aapki EMI Rs 3500 pending hai."),
        ("BORROWER", "main nahi, mera beta dega paise"), ("AGENT", "Kya aap confirm kar sakte hain?"),
        ("BORROWER", "patni se baat karni padegi, office wala bhi dega shayad")]),
    "Agent-pushed PTP": (0.5, [
        ("AGENT", "Namaste, aapki EMI Rs 8000 pending hai."), ("BORROWER", "hmm"),
        ("AGENT", "Maine 12 tarikh ki PTP note kar di hai."), ("BORROWER", "hmm")]),
}


# ------------------------------------------------------------------ resources
@st.cache_resource
def load_engine():
    return joblib.load(ENGINE_PATH)


@st.cache_resource
def load_whisper(size):
    from faster_whisper import WhisperModel
    return WhisperModel(size, device="cpu", compute_type="int8")


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else None


def asr_available():
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


# ------------------------------------------------------------------ state
def reset_call():
    st.session_state.update(lines=[], scores=[], ctx_override=None, ptr=0, replay_id=None, last_audio=None)


def sidebar_ctx():
    with st.sidebar:
        st.header("Account context")
        c = dict(portfolio=st.selectbox("Portfolio", ["salaried_pl", "microfinance", "msme", "two_wheeler"]),
                 dpd_bucket=st.selectbox("DPD bucket", ["X", "1-30", "31-60", "61-90", "90+"], index=2),
                 channel=st.selectbox("Channel", ["tele_caller", "voice_bot"]))
        rate = st.slider("Past PTP keep rate", 0.0, 1.0, 0.5, 0.05)
        n_past = st.number_input("Past PTPs", 0, 20, 4)
        emi = st.number_input("EMI (Rs)", 500, 60000, 5000, 500)
        ratio = st.selectbox("Promised share of EMI", [1.0, 0.5, 0.25])
        c.update(past_keep_rate=rate if n_past else np.nan, n_past_ptps=n_past,
                 past_broken_ptps=round(n_past * (1 - rate)), emi_amount=emi,
                 promised_amount=int(emi * ratio), promised_amount_ratio=ratio,
                 ability_to_pay=st.slider("Ability to pay estimate", 0.0, 1.0, 0.5, 0.05),
                 promised_days_from_salary=st.slider("Promised date vs salary credit (days apart)", 0, 15, 5))
        with st.expander("Audio and agent signals (neutral defaults)"):
            c.update(days_to_promise=7, negotiation_turns=st.slider("Negotiation turns", 0, 6, 1),
                     agree_latency_sec=st.slider("Seconds to agree", 1, 30, 10),
                     promise_before_hangup=int(st.checkbox("Promise right before hangup")),
                     voice_stress=st.slider("Voice stress (0-1)", 0.0, 1.0, 0.4),
                     agent_keep_rate_vs_peers=st.slider("Agent keep rate vs peers", -0.3, 0.3, 0.0),
                     mins_to_shift_end=st.slider("Minutes to shift end", 0, 480, 240),
                     remark_copy_score=st.slider("Remark copy score", 0.0, 1.0, 0.15))
        st.divider()
        min_conf = st.slider("Min ASR confidence (avg logprob)", -3.0, 0.0, -1.2, 0.1,
                             help="Utterances below this are shown but not scored.")
        if st.button("Reset call"):
            reset_call()
            st.rerun()
    return c, min_conf


def add_line(engine, base_ctx, min_conf, speaker, raw, conf=None, asr_ms=None):
    norm = normalize(raw)
    st.session_state.lines.append(dict(speaker=speaker, raw=raw, text=norm, conf=conf))
    usable = [l for l in st.session_state.lines if l["conf"] is None or l["conf"] >= min_conf]
    transcript = "\n".join(f"{l['speaker']}: {l['text']}" for l in usable)
    ctx = st.session_state.ctx_override or base_ctx
    res = engine.predict_live(dict(ctx, transcript=transcript))
    n_b = sum(l["speaker"] == "BORROWER" for l in usable)
    st.session_state.scores.append(dict(res=res, n_b=n_b, asr_ms=asr_ms, n_lines=len(st.session_state.lines)))


# ------------------------------------------------------------------ dashboard
def render(slot):
    lines, scores = st.session_state.lines, st.session_state.scores
    with slot.container():
        left, right = st.columns([3, 2])
        with left:
            st.subheader("Live transcript")
            st.markdown(LEGEND, unsafe_allow_html=True)
            if not lines:
                st.caption("No speech yet. Use one of the input tabs below.")
            for l in lines:
                who = "Agent" if l["speaker"] == "AGENT" else "Borrower"
                low = l["conf"] is not None and l["conf"] < st.session_state.get("min_conf", -1.2)
                style = "opacity:0.45;" if low else ""
                st.markdown(f'<div style="{style}margin:4px 0"><b>{who}:</b> {highlight(l["text"], l["speaker"])}'
                            f'{" <i>(low ASR confidence, not scored)</i>" if low else ""}</div>',
                            unsafe_allow_html=True)
                if l["raw"].strip().lower() != l["text"]:
                    st.caption(f"heard as: {l['raw']}")
        with right:
            st.subheader("Model reading")
            if not scores:
                st.caption("Waiting for the first utterance.")
                return
            s = scores[-1]
            r = s["res"]
            m1, m2 = st.columns(2)
            m1.metric("Keep probability", f"{r['keep_probability']:.0%}",
                      f"{(r['keep_probability'] - scores[-2]['res']['keep_probability']) * 100:+.0f} pts" if len(scores) > 1 else None)
            m2.metric("Likely PTP type", r["ptp_type"].replace("_", " "))
            top = max(r["type_probabilities"].values())
            ready = (top >= 0.8 and s["n_b"] >= 2) or r["hardship_guard_triggered"]
            st.markdown("**Status:** " + (":green[enough heard, act now]" if ready else ":orange[still listening]"))
            if r["hardship_guard_triggered"]:
                st.error("Hardship signals heard. No pressure prompts. Route to human / restructuring.")
            (st.warning if r["weak_promise"] else st.success)(f"**Agent prompt:** {r['agent_prompt']}")
            st.caption(f"Route: {r['route']} | Parking: {r['parking']}")
            if r["reason_codes"]:
                st.write("Why: " + ", ".join(r["reason_codes"]))
            asr = s["asr_ms"]
            total = r["latency_ms"] + (asr or 0)
            l1, l2, l3 = st.columns(3)
            l1.metric("ASR (last)", f"{asr:.0f} ms" if asr else "n/a")
            l2.metric("Model", f"{r['latency_ms']:.0f} ms")
            l3.metric("Total", f"{total:.0f} ms", "within 300 ms" if total < 300 else "over 300 ms target",
                      delta_color="normal" if total < 300 else "inverse")
            if total > 1500:
                st.error("Over the 1,500 ms hard limit: the prompt would arrive too late.")
        c1, c2 = st.columns(2)
        c1.caption("Keep probability after each utterance")
        c1.line_chart(pd.DataFrame({"keep_prob": [x["res"]["keep_probability"] for x in scores]}), height=200)
        c2.caption("PTP type probabilities (latest)")
        c2.bar_chart(pd.Series(scores[-1]["res"]["type_probabilities"]), height=200)


# ------------------------------------------------------------------ ASR helpers
def transcribe(model, source, lang):
    t0 = time.perf_counter()
    segs, _ = model.transcribe(source, language=None if lang == "auto" else lang,
                               initial_prompt=WHISPER_PROMPT, vad_filter=True, beam_size=1)
    segs = list(segs)
    return segs, (time.perf_counter() - t0) * 1000


def play(engine, ctx, min_conf, items, delay, slot):
    """items: list of (speaker, text, conf, asr_ms)"""
    for sp, tx, conf, ams in items:
        add_line(engine, ctx, min_conf, sp, tx, conf, ams)
        render(slot)
        time.sleep(delay)


# ------------------------------------------------------------------ app
st.set_page_config(page_title="PTP live credibility", layout="wide")
st.title("Real-time PTP credibility: how the model reads the call")

if not os.path.exists(ENGINE_PATH):
    st.error(f"{ENGINE_PATH} not found. Run generate_synthetic_data.py and train_model.py first.")
    st.stop()
engine = load_engine()
if "lines" not in st.session_state:
    reset_call()
base_ctx, min_conf = sidebar_ctx()
st.session_state.min_conf = min_conf

slot = st.empty()
tab_replay, tab_mic, tab_rec, tab_type = st.tabs(["Replay", "Mic (one utterance at a time)",
                                                  "Recorded stereo call", "Type"])

with tab_replay:
    ds = load_dataset()
    src = st.radio("Source", ["Built-in demo calls"] + (["Calls from ptp_synthetic.csv"] if ds is not None else []),
                   horizontal=True)
    if src == "Built-in demo calls":
        name = st.selectbox("Demo call", list(DEMOS))
        rate, script = DEMOS[name]
        items = [(s, t, None, None) for s, t in script]
        override = dict(base_ctx, past_keep_rate=rate)
        call_id = f"demo:{name}"
    else:
        ptype = st.selectbox("PTP type", pp.TYPES)
        if st.button("Pick another random call") or "ds_row" not in st.session_state:
            st.session_state.ds_row = int(np.random.choice(ds.index[ds["ptp_type"] == ptype]))
        row = ds.loc[st.session_state.ds_row]
        items = []
        for ln in str(row["transcript"]).split("\n"):
            sp, _, tx = ln.partition(":")
            items.append((sp.strip(), tx.strip(), None, None))
        override = {c: row[c] for c in pp.FEATURE_COLUMNS if c != pp.TEXT}
        call_id = f"ds:{st.session_state.ds_row}"
        st.caption(f"Dataset call {st.session_state.ds_row}: true type = {row['ptp_type']}, kept = {int(row['kept'])}. "
                   "Account context comes from the dataset row, not the sidebar.")
    delay = st.slider("Seconds per utterance", 0.0, 3.0, 0.8, 0.1)
    b1, b2 = st.columns(2)
    if b1.button("Play call", type="primary"):
        reset_call()
        st.session_state.ctx_override = override
        play(engine, base_ctx, min_conf, items, delay, slot)
    if b2.button("Step one utterance"):
        if st.session_state.replay_id != call_id:
            reset_call()
            st.session_state.ctx_override, st.session_state.replay_id = override, call_id
        if st.session_state.ptr < len(items):
            sp, tx, cf, am = items[st.session_state.ptr]
            add_line(engine, base_ctx, min_conf, sp, tx, cf, am)
            st.session_state.ptr += 1

with tab_mic:
    if not asr_available():
        st.info("Install the ASR engine to use this tab:  pip install faster-whisper")
    else:
        st.caption("Single mic means no automatic speaker labels. Pick who is speaking before each clip.")
        a, b, c = st.columns(3)
        who = a.radio("Who is speaking?", ["BORROWER", "AGENT"], horizontal=True)
        size = b.selectbox("Whisper model", ["base", "small", "medium"], index=1,
                           help="Smaller is faster. ASR time is the main latency cost.")
        lang = c.selectbox("Language hint", ["hi", "en", "auto"],
                           help="Hindi hint can return Devanagari; normalize() maps the key phrases back to Roman.")
        clip = st.audio_input("Record the next utterance")
        if clip is not None:
            data = clip.getvalue()
            if hash(data) != st.session_state.last_audio:
                st.session_state.last_audio = hash(data)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(data)
                with st.spinner("Transcribing..."):
                    segs, ms = transcribe(load_whisper(size), f.name, lang)
                os.unlink(f.name)
                if segs:
                    text = " ".join(s.text.strip() for s in segs)
                    add_line(engine, base_ctx, min_conf, who, text, float(np.mean([s.avg_logprob for s in segs])), ms)
                else:
                    st.warning("No speech detected in that clip.")

with tab_rec:
    if not asr_available():
        st.info("Install the ASR engine to use this tab:  pip install faster-whisper")
    else:
        st.caption("Upload a 2-channel recording: LEFT = agent, RIGHT = borrower (how most telephony stacks record).")
        up = st.file_uploader("Stereo call recording", type=["wav", "mp3", "m4a", "flac"])
        c1, c2, c3 = st.columns(3)
        size2 = c1.selectbox("Whisper model ", ["base", "small", "medium"], index=1)
        lang2 = c2.selectbox("Language hint ", ["hi", "en", "auto"])
        speed = c3.slider("Replay speed (seconds per utterance)", 0.0, 3.0, 0.8, 0.1)
        if up is not None and st.button("Transcribe and replay", type="primary"):
            from faster_whisper.audio import decode_audio
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(up.name)[1], delete=False) as f:
                f.write(up.getvalue())
            left, right = decode_audio(f.name, sampling_rate=16000, split_stereo=True)
            os.unlink(f.name)
            if np.corrcoef(left[:16000 * 30], right[:16000 * 30])[0, 1] > 0.98:
                st.error("Both channels are almost identical, so this looks like a mono recording. "
                         "Speakers cannot be separated without diarization. Use stereo or the Mic tab.")
            else:
                model = load_whisper(size2)
                items = []
                with st.spinner("Transcribing both channels..."):
                    for arr, sp in ((left, "AGENT"), (right, "BORROWER")):
                        segs, ms = transcribe(model, arr, lang2)
                        items += [(s.start, sp, s.text.strip(), s.avg_logprob, ms / max(len(segs), 1)) for s in segs]
                items.sort()
                reset_call()
                play(engine, base_ctx, min_conf, [(sp, tx, cf, am) for _, sp, tx, cf, am in items], speed, slot)

with tab_type:
    t1, t2 = st.columns([1, 4])
    sp = t1.radio("Speaker", ["BORROWER", "AGENT"])
    with t2.form("typed", clear_on_submit=True):
        txt = st.text_input("Utterance (Hinglish, Roman or Devanagari)")
        if st.form_submit_button("Add") and txt.strip():
            add_line(engine, base_ctx, min_conf, sp, txt)

render(slot)