#!/usr/bin/env python3
"""
Script 3: Model trainer for real-time PTP credibility.

Model choice: sklearn HistGradientBoostingClassifier
  * handles NaN natively (past_keep_rate is missing for first-time promisers)
  * captures non-linear interactions (hedging x past keep rate x salary alignment)
  * inference on one row is a few ms, so the whole live path stays well under a few hundred ms
  * keep model is wrapped in isotonic calibration so a score of 0.6 means ~60%
Two models share one preprocessor:
  1. keep model  : P(PTP is kept), calibrated
  2. type model  : 6-way PTP type, trained with extra weight on hardship / third-party classes
The final type decision is cost-sensitive (max expected P&L, see ptp_pipeline.gain_tables).

Usage:
    python generate_synthetic_data.py --n 20000 --out ptp_synthetic.csv
    python train_model.py --data ptp_synthetic.csv --out ptp_engine.joblib
"""
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from ptp_pipeline import (ESCAPE, FEATURE_COLUMNS, INFEASIBLE, REPEAT, TEXT, THIRD, TYPES,
                          PnLConfig, PTPDecisionEngine, TranscriptFeatureExtractor,
                          build_preprocessor, evaluate_pnl)


def expected_calibration_error(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
               for b in range(bins) if (idx == b).any())


def truncate(transcripts, frac):
    """Simulate a live call: keep only the first `frac` of the lines."""
    out = []
    for t in transcripts:
        lines = t.split("\n")
        out.append("\n".join(lines[:max(2, int(round(len(lines) * frac)))]))
    return out


def hgb(seed, **kw):
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=24,
                                          l2_regularization=1.0, early_stopping=True,
                                          validation_fraction=0.1, random_state=seed, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ptp_synthetic.csv")
    ap.add_argument("--out", default="ptp_engine.joblib")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bench_calls", type=int, default=300)
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    df[TEXT] = df[TEXT].fillna("")
    y_type = df["ptp_type_id"].values
    y_keep = df["kept"].values
    tr, te = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y_type, random_state=a.seed)
    Xtr_df, Xte_df = df.iloc[tr][FEATURE_COLUMNS], df.iloc[te][FEATURE_COLUMNS]

    pre = build_preprocessor().fit(Xtr_df)
    Xtr, Xte = pre.transform(Xtr_df), pre.transform(Xte_df)

    # sanity check: hedging must correlate negatively with past keep rate
    f_names = list(TranscriptFeatureExtractor.FEATURES)
    hedge = Xtr_df.assign(h=TranscriptFeatureExtractor().transform(Xtr_df[TEXT])[:, f_names.index("n_hedge")])
    c = hedge["past_keep_rate"].corr(hedge["h"], method="spearman")
    print(f"[sanity] Spearman(extracted hedges, past keep rate) = {c:.3f}")

    # ---- keep model (calibrated)
    keep_model = CalibratedClassifierCV(hgb(a.seed), method="isotonic", cv=3).fit(Xtr, y_keep[tr])
    # ---- type model (hardship misclassification weighted heavily)
    w = compute_sample_weight("balanced", y_type[tr])
    w = w * np.where(y_type[tr] == INFEASIBLE, 3.0, 1.0) * np.where(y_type[tr] == THIRD, 2.0, 1.0)
    type_model = hgb(a.seed).fit(Xtr, y_type[tr], sample_weight=w)

    engine = PTPDecisionEngine(pre, keep_model, type_model, PnLConfig())

    # ---------------- evaluation on held-out calls (full transcript)
    out = engine.predict_batch(Xte_df)
    k, yk = out["keep_prob"], y_keep[te]
    print("\n=== Keep prediction (end of call) ===")
    print(f"AUC {roc_auc_score(yk, k):.3f} | Brier {brier_score_loss(yk, k):.3f} | ECE {expected_calibration_error(yk, k):.3f}")
    broken = yk == 0
    print(f"Share of broken PTPs flagged (keep_prob < {engine.keep_flag_threshold}): {(k[broken] < engine.keep_flag_threshold).mean():.3f}")
    for p, g in Xte_df.groupby("portfolio").groups.items():
        pos = Xte_df.index.get_indexer(g)
        print(f"  calibration gap [{p}]: mean pred {k[pos].mean():.3f} vs actual {yk[pos].mean():.3f}")

    yt = y_type[te]
    pred_argmax = out["type_proba"].argmax(1)
    pred = out["pred_type"]
    print("\n=== PTP type ===")
    print(f"argmax    : acc {accuracy_score(yt, pred_argmax):.3f} | macro-F1 {f1_score(yt, pred_argmax, average='macro'):.3f}")
    print(f"cost-aware: acc {accuracy_score(yt, pred):.3f} | macro-F1 {f1_score(yt, pred, average='macro'):.3f}")
    hardship_mask = yt == INFEASIBLE
    for name, pr in (("argmax", pred_argmax), ("cost-aware+guard", pred)):
        rate = np.isin(pr[hardship_mask], (ESCAPE, REPEAT)).mean()
        print(f"hardship treated as escape/repeat [{name}]: {rate:.3%}")

    # ---------------- early decision: only part of the transcript heard
    print("\n=== Partial-transcript keep AUC (live call) ===")
    for frac in (0.4, 0.6, 0.8, 1.0):
        part = Xte_df.copy()
        part[TEXT] = truncate(part[TEXT].values, frac)
        kp = engine.keep_model.predict_proba(pre.transform(part))[:, 1]
        print(f"  {int(frac * 100):>3}% of call heard: AUC {roc_auc_score(yk, kp):.3f}")

    # ---------------- P&L per the financial transcript
    amounts = Xte_df["promised_amount"].values
    pnl = evaluate_pnl(yt, pred, amounts, engine.cfg)
    base = evaluate_pnl(yt, np.zeros_like(yt), amounts, engine.cfg)
    print("\n=== Net P&L (Rs) on test calls ===")
    for key in ("avoided_escalation", "incremental_cash", "other_avoided_costs",
                "ai_compute_cost", "error_losses", "net_pnl"):
        print(f"  {key:<22}{pnl[key]:>14,.0f}")
    print(f"  net per call: Rs {pnl['net_pnl_per_call']:.2f} | type error rate {pnl['error_rate']:.1%} "
          f"(transcript assumes 10-15%)")
    print(f"  rule-based baseline net (compute cost only, no AI gains): Rs {base['net_pnl'] - base['error_losses']:,.0f} "
          f"before its own misses")
    print(pnl["by_true_type"].to_string())

    # ---------------- latency benchmark on single live records
    recs = Xte_df.head(a.bench_calls).to_dict("records")
    for r in recs:
        r[TEXT] = truncate([r[TEXT]], 0.6)[0]
    engine.predict_live(recs[0])  # warm-up
    ms = np.array([engine.predict_live(r)["latency_ms"] for r in recs])
    print("\n=== Live latency per call (ms) ===")
    print(f"p50 {np.percentile(ms, 50):.1f} | p95 {np.percentile(ms, 95):.1f} | p99 {np.percentile(ms, 99):.1f} | max {ms.max():.1f}")
    print(f"target < {engine.TARGET_LATENCY_MS} ms: {'PASS' if np.percentile(ms, 95) < engine.TARGET_LATENCY_MS else 'FAIL'} | "
          f"hard limit {engine.HARD_LATENCY_MS} ms: {'PASS' if ms.max() < engine.HARD_LATENCY_MS else 'FAIL'}")

    print("\nExample live payload:")
    for k_, v in engine.predict_live(recs[1]).items():
        print(f"  {k_}: {v}")

    joblib.dump(engine, a.out)
    print(f"\nSaved engine -> {a.out}")


if __name__ == "__main__":
    main()
