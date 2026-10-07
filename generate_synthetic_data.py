#!/usr/bin/env python3
"""
Script 1: Synthetic data generator for Problem Statement 1 (Real-Time Fake PTP Detection).

Generates one row per call with a code-mixed (Hinglish) transcript plus account, agent
and audio-derived signals.

Key design rule (as requested): hedging language is CORRELATED with past PTP keep rate.
Borrowers with a poor history hedge more ("dekhta hoon", "try karunga"), so
    lambda_hedge = 0.25 + 2.0 * (1 - latent_past_keep_rate) + type_bump
and the number of hedges per call is Poisson(lambda_hedge).

The six PTP types follow the financial transcript / problem statement:
    0 genuine_feasible, 1 genuine_infeasible, 2 escape,
    3 third_party, 4 repeat_promiser, 5 agent_pushed

"kept" definition used here (agree this with the business in real life):
    borrower paid >= the promised amount by the promised date + 2 days grace.

Post-call columns (link opened, SMS reply) are written for completeness but must NOT be
used by the live model, because they do not exist while the borrower is still on the line.

Usage:
    python generate_synthetic_data.py --n 20000 --out ptp_synthetic.csv --seed 42
"""
import argparse

import numpy as np
import pandas as pd

TYPES = ["genuine_feasible", "genuine_infeasible", "escape",
         "third_party", "repeat_promiser", "agent_pushed"]

PORTFOLIOS = {  # mean keep rate differs by portfolio (so calibration per portfolio matters)
    "salaried_pl": dict(mean_keep=0.55, p=0.35),
    "microfinance": dict(mean_keep=0.70, p=0.25),
    "msme": dict(mean_keep=0.45, p=0.20),
    "two_wheeler": dict(mean_keep=0.50, p=0.20),
}

HEDGE = ["dekhta hoon", "dekhte hain", "try karunga", "koshish karta hoon", "pata nahi",
         "shayad ho jayega", "let me see", "hopefully", "maybe"]
CONDITION = ["salary aayi toh", "paise aaye toh", "client ka payment aaya toh",
             "agar ho paya toh", "bonus mila toh", "jab paisa aayega tab"]
VAGUE_DATE = ["next week", "month end", "kuch din mein", "jaldi", "agle hafte",
              "mahine ke end tak", "baad mein"]
THIRD = ["mera beta dega", "patni se baat karni padegi", "papa bhej denge",
         "bhai pay kar dega", "office wala dega", "husband dega"]
HARDSHIP = ["naukri chali gayi", "medical emergency hai ghar mein", "dukaan band hai",
            "hospital ka kharcha hai", "salary nahi mili abhi tak", "business band chal raha hai"]
PRESSURE = ["legal notice bhej denge", "abhi karo warna case file hoga",
            "aaj hi payment karna padega", "last warning hai"]
AGREE = ["haan theek hai", "ok ji", "haan kar dunga", "chalo theek hai"]

# per-type behaviour tables (index = type id)
HEDGE_BUMP = [-0.15, 0.4, 1.2, 0.3, 1.0, 0.2]
COND_LAM = [0.10, 0.90, 0.25, 0.15, 0.30, 0.05]
VAGUE_LAM = [0.10, 0.30, 1.00, 0.40, 0.70, 0.10]
PRESS_LAM = [0.10, 0.20, 0.80, 0.30, 0.80, 1.20]
P_STATED_AMT = [0.85, 0.60, 0.20, 0.30, 0.35, 0.03]
P_STATED_DATE = [0.90, 0.70, 0.30, 0.35, 0.40, 0.03]
P_FIRM_DATE = [0.85, 0.50, 0.10, 0.20, 0.20, 0.0]
NEG_LAM = [1.0, 2.5, 0.3, 0.6, 0.5, 0.1]
P_HANGUP = [0.15, 0.20, 0.70, 0.30, 0.50, 0.50]
STRESS_MU = [0.30, 0.60, 0.50, 0.35, 0.50, 0.30]
LATENCY_MU = [14, 20, 4, 9, 6, 2]
BASE_KEEP = [0.88, 0.30, 0.10, 0.25, 0.12, 0.06]
SALARY_DAYS = [1, 5, 7, 10, 25, 28]


def make_agents(rng, n_agents=200):
    gaming = rng.beta(1, 10, n_agents)  # most agents honest, a few push PTPs
    keep = np.clip(0.55 - 0.5 * gaming + rng.normal(0, 0.04, n_agents), 0.05, 0.9)
    return pd.DataFrame({"agent_id": np.arange(n_agents), "gaming": gaming, "keep_rate": keep})


def circ_diff(a, b, period=30):
    d = abs(a - b)
    return min(d, period - d)


def build_transcript(rng, t, d, amt, emi, n_hedge, n_cond, n_vague, n_press, neg_turns,
                     stated_amt, stated_date, firm_date, hardship, third, hangup):
    L = [f"AGENT: Namaste, main bank se bol raha hoon, aapki EMI Rs {emi} pending hai."]
    for _ in range(n_press):
        L.append(f"AGENT: {rng.choice(PRESSURE)}.")

    parts = []
    if hardship:
        parts.append(rng.choice(HARDSHIP))
    if third:
        parts.append(rng.choice(THIRD))
    parts += list(rng.choice(HEDGE, n_hedge)) if n_hedge else []
    parts += list(rng.choice(CONDITION, n_cond)) if n_cond else []
    if not parts:
        parts.append(rng.choice(["haan bataiye", "ji mujhe pata hai", "haan boliye"]))
    L.append("BORROWER: " + ", ".join(parts))

    for _ in range(neg_turns):
        L.append(f"AGENT: Kya {rng.integers(1, 29)} tarikh tak possible hai?")
        L.append("BORROWER: mushkil hai, thoda time chahiye")

    if t == 5:  # agent pushed: borrower never gives a date or amount
        if rng.random() < 0.4:
            L.append(f"AGENT: Maine {d} tarikh ki PTP note kar di hai.")
        else:
            L.append(f"AGENT: {d} tarikh tak ho jayega na?")
        L.append("BORROWER: hmm")
    else:
        if stated_date:
            phrase = f"{d} tarikh ko pakka" if firm_date else str(rng.choice(VAGUE_DATE))
            L.append(f"BORROWER: {phrase} kar dunga")
        else:
            L.append(f"AGENT: {d} tarikh tak kar sakte hain?")
            L.append(f"BORROWER: {rng.choice(AGREE)}")
        if stated_amt:
            L.append(f"BORROWER: Rs {amt} bhej dunga")
        else:
            L.append(f"AGENT: Rs {amt} ka payment karna hoga.")
            L.append(f"BORROWER: {rng.choice(AGREE)}")
        # extra vague date talk from the borrower
        for _ in range(n_vague):
            L.append(f"BORROWER: {rng.choice(VAGUE_DATE)} tak")
    L.append("BORROWER: ab rakhta hoon, mujhe jaana hai" if hangup
             else "BORROWER: theek hai, dhanyavaad")
    return "\n".join(L)


def make_call(rng, i, agents):
    names = list(PORTFOLIOS)
    portfolio = rng.choice(names, p=[PORTFOLIOS[k]["p"] for k in names])
    m = PORTFOLIOS[portfolio]["mean_keep"]
    r = rng.beta(m * 6, (1 - m) * 6)                    # latent true keep propensity
    n_past = int(rng.poisson(3.0))
    broken = int(rng.binomial(n_past, 1 - r))
    obs_rate = np.nan if n_past == 0 else 1 - broken / n_past

    ag = agents.iloc[int(rng.integers(len(agents)))]
    w = np.array([0.15 + 0.6 * r, 0.14 + 0.05 * (1 - r), 0.04 + 0.35 * (1 - r), 0.07,
                  0.35 * (1 - r) ** 2 if n_past >= 3 else 0.01, 0.03 + 0.5 * ag.gaming])
    t = int(rng.choice(6, p=w / w.sum()))

    emi = int(np.clip(rng.lognormal(8.9, 0.7), 1500, 60000) // 100 * 100)
    ratio = float(rng.choice([1.0, 0.5, 0.25], p=[0.7, 0.2, 0.1]))
    amt = int(emi * ratio)

    sd = int(rng.choice(SALARY_DAYS))
    if t == 0:
        d = (sd + int(rng.integers(0, 4)) - 1) % 28 + 1
    elif t == 1:
        d = (sd - int(rng.integers(5, 16)) - 1) % 28 + 1
    else:
        d = int(rng.integers(1, 29))
    days_from_salary = circ_diff(d, sd)
    days_to_promise = int(rng.integers(1, 13) if t == 0 else rng.integers(5, 31))

    ability = float(np.clip(rng.normal([0.75, 0.25, 0.5, 0.5, 0.45, 0.5][t],
                                       0.12 if t < 2 else 0.2), 0, 1))

    # ---- hedging is correlated with past keep rate (the key requirement) ----
    lam_hedge = max(0.05, 0.25 + 2.0 * (1 - r) + HEDGE_BUMP[t])
    n_hedge = int(rng.poisson(lam_hedge))
    n_cond = int(rng.poisson(COND_LAM[t]))
    n_vague = int(rng.poisson(VAGUE_LAM[t]))
    n_press = int(rng.poisson(PRESS_LAM[t]))
    neg = int(rng.poisson(NEG_LAM[t]))
    stated_amt = rng.random() < P_STATED_AMT[t]
    stated_date = rng.random() < P_STATED_DATE[t]
    firm = stated_date and rng.random() < P_FIRM_DATE[t]
    hardship = rng.random() < (0.85 if t == 1 else 0.03)
    third = rng.random() < (1.0 if t == 3 else 0.02)
    hangup = rng.random() < P_HANGUP[t]

    transcript = build_transcript(rng, t, d, amt, emi, n_hedge, n_cond, n_vague, n_press, neg,
                                  stated_amt, stated_date, firm, hardship, third, hangup)

    p_keep = BASE_KEEP[t] + 0.2 * (r - 0.5) + 0.1 * (ability - 0.5)
    if days_from_salary <= 3:
        p_keep += 0.05
    p_keep = float(np.clip(p_keep, 0.02, 0.97))
    kept = int(rng.random() < p_keep)

    return dict(
        call_id=i, agent_id=int(ag.agent_id), transcript=transcript,
        portfolio=portfolio,
        dpd_bucket=str(rng.choice(["X", "1-30", "31-60", "61-90", "90+"], p=[.1, .3, .3, .2, .1])),
        channel=str(rng.choice(["tele_caller", "voice_bot"], p=[0.8, 0.2])),
        past_keep_rate=obs_rate, n_past_ptps=n_past, past_broken_ptps=broken,
        emi_amount=emi, promised_amount=amt, promised_amount_ratio=ratio,
        ability_to_pay=ability, promised_days_from_salary=days_from_salary,
        days_to_promise=days_to_promise, negotiation_turns=neg,
        agree_latency_sec=float(max(0.5, rng.normal(LATENCY_MU[t], LATENCY_MU[t] * 0.4))),
        promise_before_hangup=int(hangup),
        voice_stress=float(np.clip(rng.normal(STRESS_MU[t], 0.15), 0, 1)),
        agent_keep_rate_vs_peers=float(ag.keep_rate - 0.55),
        mins_to_shift_end=float(rng.exponential(40) if t == 5 else rng.uniform(0, 480)),
        remark_copy_score=float(rng.beta(5, 2) if t == 5 else rng.beta(1.5, 6)),
        ptp_type=TYPES[t], ptp_type_id=t, kept=kept,
        post_call_link_opened=int(rng.random() < (0.75 if kept else 0.15)),
        post_call_sms_reply=int(rng.random() < (0.50 if kept else 0.08)),
        _gen_hedge_count=n_hedge,  # dropped before saving, used only for the sanity check
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--out", default="ptp_synthetic.csv")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    agents = make_agents(rng)
    df = pd.DataFrame([make_call(rng, i, agents) for i in range(a.n)])

    sub = df.dropna(subset=["past_keep_rate"])
    corr = sub["past_keep_rate"].corr(sub["_gen_hedge_count"], method="spearman")
    print(f"Rows: {len(df)}")
    print("Type mix:\n", df["ptp_type"].value_counts(normalize=True).round(3).to_string())
    print(f"Overall keep rate: {df['kept'].mean():.3f}")
    print("Keep rate by type:\n", df.groupby("ptp_type")["kept"].mean().round(3).to_string())
    print(f"Spearman(hedge count, past keep rate) = {corr:.3f}  (should be clearly negative)")

    df.drop(columns=["_gen_hedge_count"]).to_csv(a.out, index=False)
    print(f"Saved -> {a.out}")


if __name__ == "__main__":
    main()
