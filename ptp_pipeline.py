#!/usr/bin/env python3
"""
Script 2: ML pipeline for real-time PTP credibility, built strictly around the financial transcript.

What from the transcript is encoded here
----------------------------------------
* Six PTP categories (feasible, infeasible, escape, third-party, repeat, agent-pushed).
* Net P&L = Avoided Escalation + Incremental Cash - AI Compute - Cost of AI Errors.
* AI compute cost ~ Rs 0.30 per call (Rs 0.20-0.50 range), Field visit Rs 150-400,
  tele call Rs 15-30, manual audit Rs 20-50.
* Type I error  (escape/repeat predicted as hardship): ~15-25% recovery drop from ageing.
* Type II error (hardship predicted as escape / pressure): legal + RBI Fair Practices risk. Weighted heavily.
* Third-party miss: RBI / DPDP disclosure fine. False agent accusation: audit cost + morale.
* Latency: hard ceiling 1,500 ms, target a few hundred ms.
* Target gains: ~30% of field visits saved, ~15% more cash on escape promises.
* Hardship rule from the problem statement: a low score must never lead to harsher treatment
  when hardship signals are present; those go to restructuring / a human.

Rupee figures that the transcript does NOT quantify (legal/RBI cost, disclosure fine, churn) are
marked ASSUMPTION in PnLConfig. Replace them with CN's real numbers.

This module is imported by train_model.py. It has no training code.
"""
import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

TYPES = ["genuine_feasible", "genuine_infeasible", "escape",
         "third_party", "repeat_promiser", "agent_pushed"]
TYPE_TO_ID = {t: i for i, t in enumerate(TYPES)}
N_TYPES = len(TYPES)
FEASIBLE, INFEASIBLE, ESCAPE, THIRD, REPEAT, PUSHED = range(6)

NUMERIC = ["past_keep_rate", "n_past_ptps", "past_broken_ptps", "emi_amount", "promised_amount",
           "promised_amount_ratio", "ability_to_pay", "promised_days_from_salary",
           "days_to_promise", "negotiation_turns", "agree_latency_sec", "promise_before_hangup",
           "voice_stress", "agent_keep_rate_vs_peers", "mins_to_shift_end", "remark_copy_score"]
CATEGORICAL = ["portfolio", "dpd_bucket", "channel"]
TEXT = "transcript"
FEATURE_COLUMNS = NUMERIC + CATEGORICAL + [TEXT]
# Deliberately NOT features: ptp_type*, kept, post_call_* (not available during the call), agent_id.

# ------------------------------------------------------------------ transcript features
HEDGE_RE = re.compile(r"dekh(?:ta|te) (?:hoon|hain)|try karunga|koshish|pata nahi|shayad|"
                      r"let me see|hopefully|maybe", re.I)
COND_RE = re.compile(r"(?:salary|paise|payment|bonus|paisa) (?:aayi|aaye|aaya|mila) toh|"
                     r"agar ho paya|jab paisa aayega", re.I)
VAGUE_RE = re.compile(r"next week|month end|kuch din|jaldi|agle hafte|mahine ke end|baad mein", re.I)
DATE_RE = re.compile(r"\b\d{1,2}\s*tarikh", re.I)
FIRM_RE = re.compile(r"\b\d{1,2}\s*tarikh ko pakka|aaj shaam|pakka kal|abhi link", re.I)
THIRD_RE = re.compile(r"mera beta|patni|papa|bhai|office wala|husband|wife|mummy|ghar wale", re.I)
HARDSHIP_RE = re.compile(r"naukri chali|medical|hospital|dukaan band|business band|"
                         r"salary nahi mili|job nahi", re.I)
PRESSURE_RE = re.compile(r"legal notice|abhi karo|aaj hi|last warning|case file", re.I)
AMT_RE = re.compile(r"(?:rs\.?|₹)\s*\d[\d,]*|\d+\s*(?:rupaye|hazar)", re.I)
AGENT_NOTE_RE = re.compile(r"note kar di|record kar liya", re.I)
NEGATIVE_RE = re.compile(r"mushkil|thoda time", re.I)


def split_speakers(text):
    b, a, n = [], [], 0
    for line in str(text).split("\n"):
        n += 1
        if line.startswith("BORROWER:"):
            b.append(line[9:].strip())
        elif line.startswith("AGENT:"):
            a.append(line[6:].strip())
    return " . ".join(b), " . ".join(a), n


class TranscriptFeatureExtractor(BaseEstimator, TransformerMixin):
    """Works on partial transcripts too (live calls)."""
    FEATURES = ["n_hedge", "hedge_density", "n_condition", "n_vague_date", "borrower_firm_date",
                "borrower_stated_date", "agent_stated_date", "borrower_stated_amount",
                "agent_stated_amount", "n_third_party", "n_hardship", "n_agent_pressure",
                "agent_noted_ptp", "n_pushback", "borrower_words", "agent_words", "n_lines",
                "borrower_word_share"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in np.asarray(X).ravel():
            b, a, n = split_speakers(text)
            bw, aw = len(b.split()), len(a.split())
            n_h = len(HEDGE_RE.findall(b))
            rows.append([
                n_h, n_h / max(bw, 1), len(COND_RE.findall(b)), len(VAGUE_RE.findall(b)),
                int(bool(FIRM_RE.search(b))),
                int(bool(DATE_RE.search(b) or VAGUE_RE.search(b))), int(bool(DATE_RE.search(a))),
                int(bool(AMT_RE.search(b))), int(bool(AMT_RE.search(a.split(" . ", 1)[-1]))
                                                 if " . " in a else 0),
                len(THIRD_RE.findall(b)), len(HARDSHIP_RE.findall(b)),
                len(PRESSURE_RE.findall(a)), int(bool(AGENT_NOTE_RE.search(a))),
                len(NEGATIVE_RE.findall(b)), bw, aw, n, bw / max(bw + aw, 1)])
        return np.asarray(rows, dtype=np.float32)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.FEATURES)


def hardship_counts(transcripts):
    return np.array([len(HARDSHIP_RE.findall(split_speakers(t)[0])) for t in transcripts])


def build_preprocessor():
    """Dense output so HistGradientBoosting can use it (NaNs in past_keep_rate are handled natively)."""
    return ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("txt", TranscriptFeatureExtractor(), TEXT),
    ])


# ------------------------------------------------------------------ P&L model (financial transcript)
@dataclass
class PnLConfig:
    ai_compute_cost: float = 0.30          # transcript: Rs 0.20-0.50 per call
    call_cost: float = 22.5                # transcript: Rs 15-30 per connect (midpoint)
    field_visit_cost: float = 275.0        # transcript: Rs 150-400 (midpoint)
    field_visit_saved_share: float = 0.30  # transcript: saving 30% of field visits
    incremental_cash_share: float = 0.15   # transcript: recovering 15% more on escape promises
    audit_cost: float = 35.0               # transcript: Rs 20-50 per manual review (midpoint)
    aging_recovery_drop: float = 0.20      # transcript: 15-25% drop when bucket rolls over
    agent_morale_cost: float = 10.0        # ASSUMPTION
    legal_rbi_cost: float = 3000.0         # ASSUMPTION: legal handling + RBI FPC complaint risk
    disclosure_fine_expected: float = 2500.0   # ASSUMPTION: expected RBI/DPDP cost of a 3rd-party miss
    churn_fixed: float = 150.0             # ASSUMPTION: LTV loss from friction
    churn_share: float = 0.05              # share of promised amount lost to friction


def gain_tables(cfg: PnLConfig):
    """
    gain[true, pred] = F[true, pred] + V[true, pred] * promised_amount   (Rs, vs the rule-based baseline).
    Baseline = predicting 'feasible' (park and remind), which is what the rule-based path does.
    AI compute cost is charged separately per call.
    """
    F = np.zeros((N_TYPES, N_TYPES))
    V = np.zeros((N_TYPES, N_TYPES))
    c = cfg
    # correct detections = the "pros" columns of the transcript
    F[INFEASIBLE, INFEASIBLE] = c.field_visit_cost * c.field_visit_saved_share   # avoided escalation
    V[ESCAPE, ESCAPE] = c.incremental_cash_share                                  # incremental cash
    F[THIRD, THIRD] = c.disclosure_fine_expected                                  # compliance avoided
    F[REPEAT, REPEAT] = 2 * c.call_cost                                           # fewer re-contacts
    F[PUSHED, PUSHED] = c.call_cost                                               # wage saving
    # feasible borrower mistreated
    F[FEASIBLE, ESCAPE] = -c.churn_fixed; V[FEASIBLE, ESCAPE] = -c.churn_share
    V[FEASIBLE, INFEASIBLE] = -0.03
    F[FEASIBLE, THIRD] = -c.call_cost
    F[FEASIBLE, REPEAT] = -50.0
    F[FEASIBLE, PUSHED] = -(c.audit_cost + c.agent_morale_cost)
    # hardship borrower pressured = Type II error, weighted heavily
    for p in (ESCAPE, REPEAT):
        F[INFEASIBLE, p] = -c.legal_rbi_cost; V[INFEASIBLE, p] = -c.churn_share * 2
    F[INFEASIBLE, THIRD] = -c.call_cost
    F[INFEASIBLE, PUSHED] = -(c.audit_cost + c.agent_morale_cost)
    # Type I error: insincere borrower given easy extension
    for t in (ESCAPE, REPEAT, PUSHED):
        V[t, INFEASIBLE] = -c.aging_recovery_drop
    # third-party miss = disclosure breach
    for p in range(N_TYPES):
        if p != THIRD:
            F[THIRD, p] = -c.disclosure_fine_expected
    F[THIRD, PUSHED] -= (c.audit_cost + c.agent_morale_cost)
    # unnecessary confirmation calls / false accusations
    for t in (ESCAPE, REPEAT, PUSHED):
        F[t, THIRD] = -c.call_cost
    for t in (ESCAPE, REPEAT):
        F[t, PUSHED] = -(c.audit_cost + c.agent_morale_cost)
    F[REPEAT, ESCAPE] = 0.0
    return F, V


def evaluate_pnl(y_true, y_pred, amounts, cfg: PnLConfig = None):
    """Net P&L = Avoided Escalation + Incremental Cash + other avoided costs - AI Compute - Error Costs."""
    cfg = cfg or PnLConfig()
    F, V = gain_tables(cfg)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    amounts = np.asarray(amounts, dtype=float)
    fixed, var = F[y_true, y_pred], V[y_true, y_pred] * amounts
    gain = fixed + var
    avoided = fixed[(y_true == INFEASIBLE) & (y_pred == INFEASIBLE)].sum()
    cash = var[(y_true == ESCAPE) & (y_pred == ESCAPE)].sum()
    diag = y_true == y_pred
    other = gain[diag].sum() - avoided - cash
    errors = gain[~diag].sum()           # negative number
    compute = cfg.ai_compute_cost * len(y_true)
    out = dict(avoided_escalation=avoided, incremental_cash=cash, other_avoided_costs=other,
               ai_compute_cost=-compute, error_losses=errors,
               net_pnl=avoided + cash + other + errors - compute,
               n_calls=len(y_true), error_rate=float((~diag).mean()))
    out["net_pnl_per_call"] = out["net_pnl"] / max(out["n_calls"], 1)
    by_type = pd.DataFrame({"true": [TYPES[i] for i in y_true], "gain": gain - cfg.ai_compute_cost})
    out["by_true_type"] = by_type.groupby("true")["gain"].agg(["count", "sum", "mean"]).round(2)
    return out


# ------------------------------------------------------------------ actions per type (problem statement table)
ACTIONS = {
    FEASIBLE: dict(action="park_and_remind", parking_days="until due date",
                   prompt="Confirm the date and send a reminder a day before."),
    INFEASIBLE: dict(action="restructure_or_reschedule", parking_days="until post-salary date",
                     prompt="Offer a post-salary date or a smaller amount; consider restructuring."),
    ESCAPE: dict(action="part_payment_now", parking_days="short",
                 prompt="Politely invite a part payment now via payment link; follow up sooner. No pressure."),
    THIRD: dict(action="confirm_with_borrower", parking_days="do not park",
                prompt="Do not discuss the debt with the third party. Ask for the borrower directly."),
    REPEAT: dict(action="stronger_treatment", parking_days="short",
                 prompt="Move to stronger treatment with a shorter parking window (within Fair Practices Code)."),
    PUSHED: dict(action="integrity_review", parking_days="do not park",
                 prompt="Call the borrower to confirm the promise; send to integrity review."),
}
HUMAN_ROUTE = dict(action="human_or_restructuring", parking_days="hold",
                   prompt="Hardship signs detected. No pressure prompts. Route to a human / restructuring.")


# ------------------------------------------------------------------ decision engine
class PTPDecisionEngine:
    """Bundles preprocessor + keep model + type model + cost-sensitive decision + hardship guard."""
    HARD_LATENCY_MS = 1500   # transcript: past this the call ends before the agent sees the prompt
    TARGET_LATENCY_MS = 300  # problem statement: a few hundred ms

    def __init__(self, preprocessor, keep_model, type_model, cfg: PnLConfig = None, keep_flag_threshold=0.5):
        self.pre, self.keep_model, self.type_model = preprocessor, keep_model, type_model
        self.cfg = cfg or PnLConfig()
        self.keep_flag_threshold = keep_flag_threshold

    def _frame(self, records):
        df = pd.DataFrame(records) if not isinstance(records, pd.DataFrame) else records
        for col in FEATURE_COLUMNS:
            if col not in df:
                df[col] = np.nan if col in NUMERIC else ("" if col == TEXT else "unknown")
        return df[FEATURE_COLUMNS]

    def type_proba(self, X):
        raw = self.type_model.predict_proba(X)
        full = np.zeros((len(raw), N_TYPES))
        full[:, self.type_model.classes_] = raw
        return full

    def decide_types(self, proba, amounts, hardship):
        """Pick the type with the highest EXPECTED P&L gain (not argmax probability), then apply the guard."""
        F, V = gain_tables(self.cfg)
        exp_gain = proba @ F + (proba @ V) * np.asarray(amounts, dtype=float)[:, None]
        pred = exp_gain.argmax(1)
        # guard: never apply pressure-type treatment where hardship signs exist
        pred = np.where(np.isin(pred, (ESCAPE, REPEAT)) & (np.asarray(hardship) > 0), INFEASIBLE, pred)
        return pred

    def predict_batch(self, records):
        df = self._frame(records)
        X = self.pre.transform(df)
        keep = self.keep_model.predict_proba(X)[:, 1]
        proba = self.type_proba(X)
        hard = hardship_counts(df[TEXT].values)
        amounts = df["promised_amount"].fillna(0).values
        return dict(keep_prob=keep, type_proba=proba, hardship=hard,
                    pred_type=self.decide_types(proba, amounts, hard))

    def predict_live(self, record):
        """One call, one partial transcript. Returns the agent-desktop / voice-bot payload."""
        t0 = time.perf_counter()
        out = self.predict_batch([record])
        pred, keep, proba, hard = int(out["pred_type"][0]), float(out["keep_prob"][0]), \
            out["type_proba"][0], int(out["hardship"][0])
        route = HUMAN_ROUTE if (hard > 0 and keep < self.keep_flag_threshold) else ACTIONS[pred]
        feats = TranscriptFeatureExtractor().transform([record.get(TEXT, "")])[0]
        f = dict(zip(TranscriptFeatureExtractor.FEATURES, feats))
        reasons = [r for r, cond in [
            ("hedging phrases heard", f["n_hedge"] > 0), ("conditional promise", f["n_condition"] > 0),
            ("vague date", f["n_vague_date"] > 0), ("borrower gave no amount", f["borrower_stated_amount"] == 0),
            ("agent proposed the date", f["agent_stated_date"] and not f["borrower_stated_date"]),
            ("third-party context", f["n_third_party"] > 0), ("hardship signals", f["n_hardship"] > 0),
            ("weak past keep rate", (record.get("past_keep_rate") or 1) < 0.4)] if cond]
        ms = (time.perf_counter() - t0) * 1000
        return dict(keep_probability=round(keep, 3), weak_promise=keep < self.keep_flag_threshold,
                    ptp_type=TYPES[pred], type_probabilities={TYPES[i]: round(float(p), 3) for i, p in enumerate(proba)},
                    hardship_guard_triggered=hard > 0, route=route["action"], parking=route["parking_days"],
                    agent_prompt=route["prompt"], reason_codes=reasons, latency_ms=round(ms, 1),
                    within_budget=ms < self.TARGET_LATENCY_MS, over_hard_limit=ms > self.HARD_LATENCY_MS)
