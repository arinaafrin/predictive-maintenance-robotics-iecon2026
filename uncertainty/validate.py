"""Validation checks for the uncertainty study on REAL data.

These checks cannot prove the results are "accurate". They try to catch the
common ways such results go wrong: bad labels, leakage, hidden randomness,
and a pipeline that invents signal.

Run from the repo root:
    python -m uncertainty.validate --csv features.csv --target isCollision --group User_ID
Add --fast for a quick trial run (smaller models, fewer repeats).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from uncertainty.uq_study import make_member, make_synthetic, run_study

RESULTS = []


def report(status, name, detail):
    RESULTS.append(status)
    print(f"[{status}] {name}: {detail}")


def check_data(y, g):
    users = np.unique(g)
    pos_per_user = pd.Series(y).groupby(g).sum()
    n_pos, n_users_pos = int(y.sum()), int((pos_per_user > 0).sum())
    print(f"Data: {len(y)} windows, {len(users)} users, {n_pos} anomalies "
          f"({y.mean() * 100:.2f}%), {n_users_pos} users with anomalies")
    if y.mean() > 0.95:
        report("FAIL", "label rate", "over 95% positive: the label is probably inverted")
    else:
        report("PASS", "label rate", f"{y.mean() * 100:.2f}% positive")
    if n_pos < 60 or n_users_pos < 10:
        report("WARN", "enough anomalies", "fewer than 60 anomalies or 10 users with anomalies: results will be noisy")
    else:
        report("PASS", "enough anomalies", f"{n_pos} anomalies over {n_users_pos} users")


def check_reproducibility(X, y, g, kw):
    a, _ = run_study(X, y, g, out_dir="results/validation/run_a", **kw)
    b, _ = run_study(X, y, g, out_dir="results/validation/run_b", **kw)
    diff = float(np.nanmax(np.abs(a["mean"].values - b["mean"].values)))
    if diff < 1e-4:
        report("PASS", "reproducibility", f"two runs agree (max difference {diff:.2e})")
    else:
        report("FAIL", "reproducibility", f"two runs differ (max difference {diff:.3g}): a seed is missing")
    return a


def check_shuffled_control(X, y, g, kw):
    rng = np.random.RandomState(0)
    y_shuf = y.copy()
    for u in np.unique(g):
        idx = np.where(g == u)[0]
        y_shuf[idx] = rng.permutation(y[idx])
    s, _ = run_study(X, y_shuf, g, out_dir="results/validation/shuffled", **kw)
    auroc, size = s.loc["auroc", "mean"], s.loc["avg_set_size", "mean"]
    if 0.4 <= auroc <= 0.6:
        report("PASS", "shuffled-label AUROC", f"{auroc:.3f} (about 0.5, as expected)")
    else:
        report("FAIL", "shuffled-label AUROC", f"{auroc:.3f}: the pipeline finds signal in random labels, so look for leakage")
    if size >= 1.3:
        report("PASS", "shuffled-label set size", f"{size:.2f} (large sets when the model knows nothing)")
    else:
        report("WARN", "shuffled-label set size", f"{size:.2f}: sets should be large when there is no signal")


def check_leakage_gap(X, y, user_auroc, fast):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    m = make_member(0, fast).fit(Xtr, ytr)
    rand_auroc = float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
    gap = rand_auroc - user_auroc
    detail = f"random-window AUROC {rand_auroc:.3f} vs user-level AUROC {user_auroc:.3f} (gap {gap:+.3f})"
    if gap > 0.03:
        report("WARN", "leakage gap", detail + ": a random split is optimistic, report the user-level number")
    else:
        report("PASS", "leakage gap", detail)


def check_summary(s):
    g = lambda k: float(s.loc[k, "mean"])  # noqa: E731
    sd = lambda k: float(s.loc[k, "std"])  # noqa: E731
    tol = max(2 * sd("coverage_anomaly"), 0.05)
    if g("coverage_anomaly") >= 0.90 - tol:
        report("PASS", "anomaly coverage", f"{g('coverage_anomaly'):.3f} (target 0.90, tolerance {tol:.3f})")
    else:
        report("WARN", "anomaly coverage", f"{g('coverage_anomaly'):.3f} is below target: user shift may break the guarantee, report it")
    report("PASS" if g("ece_cal") < g("ece_raw") else "WARN", "calibration",
           f"ECE raw {g('ece_raw'):.3f} -> calibrated {g('ece_cal'):.3f}")
    a = g("auroc_entropy_vs_error")
    report("PASS" if a > 0.55 else "WARN", "uncertainty predicts errors", f"AUROC {a:.3f} (0.5 means no link)")
    e80, e100 = g("error_at_80pct_coverage"), g("decision_error_rate")
    report("PASS" if e80 < e100 else "WARN", "risk-coverage",
           f"error at 80% coverage {e80:.3f} vs overall {e100:.3f}")
    report("PASS" if g("n_test_pos") >= 30 else "WARN", "test anomalies per repeat", f"{g('n_test_pos'):.0f}")
    report("PASS" if np.isfinite(g("conformal_q1")) else "WARN", "conformal threshold (anomaly class)",
           f"{g('conformal_q1'):.3f}" if np.isfinite(g("conformal_q1")) else "inf: too few calibration anomalies")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--csv")
    ap.add_argument("--demo", action="store_true", help="synthetic data, to test this script")
    ap.add_argument("--target", default="isCollision")
    ap.add_argument("--group", default="User_ID")
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()

    if a.demo:
        X, y, g = make_synthetic()
    elif a.csv:
        df = pd.read_csv(a.csv)
        cols = [c for c in df.columns if c not in (a.target, a.group)]
        X, y, g = df[cols].values.astype(float), df[a.target].values.astype(int), df[a.group].values
    else:
        ap.error("use --csv or --demo")

    kw = dict(n_repeats=3 if a.fast else 5, n_members=3 if a.fast else 5, fast=a.fast)
    check_data(y, g)
    summary = check_reproducibility(X, y, g, kw)
    check_shuffled_control(X, y, g, kw)
    check_leakage_gap(X, y, float(summary.loc["auroc", "mean"]), a.fast)
    check_summary(summary)
    print(f"\nSummary: {RESULTS.count('PASS')} pass, {RESULTS.count('WARN')} warn, {RESULTS.count('FAIL')} fail")
    print("WARN and FAIL lines need an explanation in your write-up, or a fix.")


if __name__ == "__main__":
    main()