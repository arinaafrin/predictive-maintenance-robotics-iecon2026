"""Uncertainty extension for the predictive maintenance pipeline.

Extension after the IECON 2026 paper. It does NOT change the paper's code or results.

What it does, on window-level features (one row = one window, one user per row):
  1. Trains a small ensemble (bootstrap + different seeds) as the base detector.
  2. Splits by USER (train / calibration / test users), so no user appears in two parts.
  3. Measures calibration (Brier, log loss, ECE) before and after Platt recalibration.
  4. Builds class-conditional split-conformal prediction sets (coverage target 1 - alpha).
  5. Checks whether uncertainty predicts mistakes (AUROC, risk-coverage curve).
  6. Links uncertainty to a cost-aware decision: act, or defer to manual inspection.

Run a demo on synthetic data:   python -m uncertainty.uq_study --demo
Run on your feature table:      python -m uncertainty.uq_study --csv features.csv \
                                    --target isCollision --group User_ID
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class CostModel:
    """Illustrative costs per window. Replace them with the values from your
    own cost-aware framework before you report any cost result."""

    c_fn: float = 10.0     # missed anomaly (no action, failure happens)
    c_fp: float = 1.0      # false alarm (unneeded action)
    c_defer: float = 0.5   # manual inspection, which finds the true state

    @property
    def bayes_threshold(self) -> float:
        """Cost-optimal probability threshold if probabilities are calibrated."""
        return self.c_fp / (self.c_fp + self.c_fn)


# --------------------------------------------------------------------------- #
# Base learner
# --------------------------------------------------------------------------- #
def make_member(seed: int, fast: bool = False):
    """XGBoost if installed (same family as the paper), else sklearn boosting."""
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=60 if fast else 300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )
    except ImportError:
        return GradientBoostingClassifier(
            n_estimators=40 if fast else 150,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=seed,
        )


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def split_users(groups, y, seed, fractions=(0.6, 0.2, 0.2), min_pos=3, tries=200):
    """Split rows into train/calibration/test by user. Each part must hold
    at least `min_pos` positive windows, otherwise the split is redrawn."""
    groups = np.asarray(groups)
    y = np.asarray(y)
    users = np.unique(groups)
    rng = np.random.RandomState(seed)
    n_tr = int(round(fractions[0] * len(users)))
    n_ca = int(round(fractions[1] * len(users)))
    for _ in range(tries):
        perm = rng.permutation(users)
        parts = (perm[:n_tr], perm[n_tr:n_tr + n_ca], perm[n_tr + n_ca:])
        idx = [np.where(np.isin(groups, p))[0] for p in parts]
        if all(y[i].sum() >= min_pos for i in idx):
            return idx
    raise RuntimeError(
        "Could not find a user split with enough anomalies in every part. "
        "Lower min_pos or check that the label has enough positives."
    )


# --------------------------------------------------------------------------- #
# Uncertainty and calibration helpers
# --------------------------------------------------------------------------- #
def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def binary_entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def expected_calibration_error(y, p, n_bins=10):
    """ECE on the anomaly probability, equal-width bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def reliability_points(y, p, n_bins=10, min_count=5):
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        m = bins == b
        if m.sum() >= min_count:
            xs.append(p[m].mean())
            ys.append(y[m].mean())
    return np.array(xs), np.array(ys)


def classwise_conformal_thresholds(p_cal, y_cal, alpha):
    """Class-conditional split conformal for a binary problem.
    Score for class k is 1 - p(k). Returns [q0, q1]; inf means 'always include'
    (too few calibration points for that class at this alpha)."""
    p_cal, y_cal = np.asarray(p_cal), np.asarray(y_cal)
    probs = np.stack([1 - p_cal, p_cal], axis=1)
    q = np.zeros(2)
    for k in (0, 1):
        s = 1 - probs[y_cal == k, k]
        n = len(s)
        level = math.ceil((n + 1) * (1 - alpha)) / n if n else 2.0
        q[k] = np.inf if level > 1 else np.quantile(s, level, method="higher")
    return q


def prediction_sets(p, q):
    """Boolean matrix (n, 2): does the set contain class 0 / class 1?"""
    p = np.asarray(p)
    probs = np.stack([1 - p, p], axis=1)
    return (1 - probs) <= q[None, :]


def risk_coverage(u, err):
    """Error rate among the (1 - r) most certain windows, for every coverage."""
    order = np.argsort(u, kind="stable")
    e = np.asarray(err, float)[order]
    n = len(e)
    cum = np.cumsum(e) / np.arange(1, n + 1)
    return np.arange(1, n + 1) / n, cum


def _auroc(err, u):
    err = np.asarray(err)
    if err.min() == err.max():
        return float("nan")
    return float(roc_auc_score(err, u))


def policy_cost(y, act, defer, costs: CostModel):
    """Mean cost per window. `act` = predicted anomaly, `defer` = inspect."""
    y, act, defer = (np.asarray(a, bool) for a in (y, act, defer))
    fn = (~defer) & (~act) & y
    fp = (~defer) & act & (~y)
    return float((costs.c_fn * fn + costs.c_fp * fp + costs.c_defer * defer).mean())


# --------------------------------------------------------------------------- #
# One repeat
# --------------------------------------------------------------------------- #
def _fit_ensemble(X, y, n_members, seed, fast):
    members = []
    n = len(y)
    for k in range(n_members):
        r = np.random.RandomState(seed * 1000 + k)
        for _ in range(50):
            idx = r.choice(n, size=n, replace=True)
            if 0 < y[idx].sum() < n:
                break
        pos = max(int(y[idx].sum()), 1)
        neg = max(int((y[idx] == 0).sum()), 1)
        w = np.where(y[idx] == 1, neg / pos, 1.0)  # same idea as scale_pos_weight
        m = make_member(seed * 1000 + k, fast)
        m.fit(X[idx], y[idx], sample_weight=w)
        members.append(m)
    return members


def _predict(members, X):
    P = np.stack([m.predict_proba(X)[:, 1] for m in members])
    return P.mean(axis=0), P.std(axis=0)


def run_repeat(X, y, groups, seed, alpha, n_members, costs, fast):
    tr, ca, te = split_users(groups, y, seed)
    members = _fit_ensemble(X[tr], y[tr], n_members, seed, fast)
    p_ca, _ = _predict(members, X[ca])
    p_te, std_te = _predict(members, X[te])
    y_ca, y_te = y[ca], y[te]

    # Platt recalibration, fitted on calibration users only
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p_ca)[:, None], y_ca)
    cal = lambda p: platt.predict_proba(_logit(p)[:, None])[:, 1]  # noqa: E731
    pc_ca, pc_te = cal(p_ca), cal(p_te)

    # Class-conditional conformal sets (scores from the ensemble probability)
    q = classwise_conformal_thresholds(p_ca, y_ca, alpha)
    sets = prediction_sets(p_te, q)
    covered = sets[np.arange(len(y_te)), y_te.astype(int)]
    size = sets.sum(axis=1)

    # Decisions
    t = costs.bayes_threshold
    act_te = pc_te >= t
    err_te = act_te != y_te.astype(bool)
    u_ent = binary_entropy(pc_te)

    # Entropy deferral: tau chosen on calibration users only
    u_ca = binary_entropy(pc_ca)
    act_ca = pc_ca >= t
    best_tau, best_cost = np.inf, policy_cost(y_ca, act_ca, np.zeros(len(y_ca)), costs)
    for tau in np.quantile(u_ca, np.linspace(0.5, 0.99, 25)):
        c = policy_cost(y_ca, act_ca, u_ca > tau, costs)
        if c < best_cost:
            best_tau, best_cost = tau, c

    defer_conf = size != 1
    defer_ent = u_ent > best_tau
    none = np.zeros(len(y_te), bool)
    cov_curve, risk_curve = risk_coverage(u_ent, err_te)
    k80 = max(int(0.8 * len(y_te)) - 1, 0)

    row = {
        "n_test": len(y_te), "n_test_pos": int(y_te.sum()),
        "auroc": float(roc_auc_score(y_te, p_te)),
        "avg_precision": float(average_precision_score(y_te, p_te)),
        "brier_raw": float(brier_score_loss(y_te, p_te)),
        "brier_cal": float(brier_score_loss(y_te, pc_te)),
        "logloss_raw": float(log_loss(y_te, np.clip(p_te, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "logloss_cal": float(log_loss(y_te, np.clip(pc_te, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "ece_raw": expected_calibration_error(y_te, p_te),
        "ece_cal": expected_calibration_error(y_te, pc_te),
        "coverage_all": float(covered.mean()),
        "coverage_anomaly": float(covered[y_te == 1].mean()),
        "coverage_normal": float(covered[y_te == 0].mean()),
        "avg_set_size": float(size.mean()),
        "frac_singleton": float((size == 1).mean()),
        "frac_both": float((size == 2).mean()),
        "frac_empty": float((size == 0).mean()),
        "decision_error_rate": float(err_te.mean()),
        "auroc_entropy_vs_error": _auroc(err_te, u_ent),
        "auroc_ensstd_vs_error": _auroc(err_te, std_te),
        "aurc": float(risk_curve.mean()),
        "error_at_80pct_coverage": float(risk_curve[k80]),
        "cost_act_raw_p05": policy_cost(y_te, p_te >= 0.5, none, costs),
        "cost_act_bayes": policy_cost(y_te, act_te, none, costs),
        "cost_defer_conformal": policy_cost(y_te, act_te, defer_conf, costs),
        "cost_defer_entropy": policy_cost(y_te, act_te, defer_ent, costs),
        "defer_rate_conformal": float(defer_conf.mean()),
        "defer_rate_entropy": float(defer_ent.mean()),
        "conformal_q0": float(q[0]), "conformal_q1": float(q[1]),
    }
    pooled = {"y": y_te, "p_raw": p_te, "p_cal": pc_te, "u": u_ent, "err": err_te}
    return row, pooled


# --------------------------------------------------------------------------- #
# Full study and plots
# --------------------------------------------------------------------------- #
def run_study(X, y, groups, alpha=0.1, n_repeats=5, n_members=5, seed=42,
              costs=None, out_dir="results/uncertainty", fast=False):
    costs = costs or CostModel()
    X = np.asarray(X, float)
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    os.makedirs(out_dir, exist_ok=True)

    rows, pool = [], []
    for r in range(n_repeats):
        row, pooled = run_repeat(X, y, groups, seed + r, alpha, n_members, costs, fast)
        row["repeat"] = r
        rows.append(row)
        pool.append(pooled)

    per_repeat = pd.DataFrame(rows)
    metrics = per_repeat.drop(columns=["repeat"])
    summary = pd.DataFrame({"mean": metrics.mean(), "std": metrics.std(ddof=1)})
    per_repeat.to_csv(os.path.join(out_dir, "per_repeat.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "summary.csv"))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"alpha": alpha, "n_repeats": n_repeats, "n_members": n_members,
                   "seed": seed, "n_windows": int(len(y)), "n_users": int(len(np.unique(groups))),
                   "anomaly_rate": float(y.mean()), "costs": asdict(costs)}, f, indent=2)
    _plots(pool, out_dir)
    return summary, per_repeat


def _plots(pool, out_dir):
    y = np.concatenate([p["y"] for p in pool])
    p_raw = np.concatenate([p["p_raw"] for p in pool])
    p_cal = np.concatenate([p["p_cal"] for p in pool])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    for p, lab in ((p_raw, "Ensemble (raw)"), (p_cal, "After Platt")):
        xs, ys = reliability_points(y, p)
        ax.plot(xs, ys, "o-", label=lab)
    ax.set_xlabel("Predicted anomaly probability")
    ax.set_ylabel("Observed anomaly frequency")
    ax.set_title("Reliability diagram (held-out users)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reliability.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for i, p in enumerate(pool):
        cov, risk = risk_coverage(p["u"], p["err"])
        ax.plot(cov, risk, color="tab:blue", alpha=0.35, label="Entropy ordering" if i == 0 else None)
        rcov, rrisk = risk_coverage(np.random.RandomState(i).rand(len(p["u"])), p["err"])
        ax.plot(rcov, rrisk, color="gray", alpha=0.35, label="Random ordering" if i == 0 else None)
    ax.set_xlabel("Coverage (share of windows decided automatically)")
    ax.set_ylabel("Decision error rate")
    ax.set_title("Risk-coverage: does uncertainty predict errors?")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "risk_coverage.png"), dpi=200)
    plt.close(fig)


def plot_cost_bars(per_repeat, out_dir):
    cols = ["cost_act_raw_p05", "cost_act_bayes", "cost_defer_conformal", "cost_defer_entropy"]
    labels = ["Act at 0.5", "Act at cost\nthreshold", "Defer if\nconformal set\nnot single", "Defer if\nentropy high"]
    m, s = per_repeat[cols].mean(), per_repeat[cols].std(ddof=1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(4), m.values, yerr=s.values, capsize=4, color="tab:blue", alpha=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean cost per window (illustrative costs)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cost_comparison.png"), dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def run_from_feature_frame(feat_df, feature_cols, target_col, group_col="User_ID", **kw):
    """Use inside the notebook after the feature matrix `feat_df` is built."""
    summary, per_repeat = run_study(
        feat_df[feature_cols].values, feat_df[target_col].values,
        feat_df[group_col].values, **kw)
    plot_cost_bars(per_repeat, kw.get("out_dir", "results/uncertainty"))
    return summary, per_repeat


def make_synthetic(n_users=30, d=12, seed=0, anomaly_rate=0.06):
    """Synthetic grouped data with a per-user shift. Only for testing the code.
    Results on this data say nothing about the real robot data."""
    r = np.random.RandomState(seed)
    Xs, ys, gs = [], [], []
    for u in range(n_users):
        n = r.randint(150, 300)
        shift = r.normal(0, 0.4, size=d)
        y = (r.rand(n) < anomaly_rate).astype(int)
        X = r.normal(0, 1, size=(n, d)) + shift
        X[:, :3] += y[:, None] * 1.6
        Xs.append(X), ys.append(y), gs.append(np.full(n, f"User_{u + 1}"))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="run on synthetic data")
    ap.add_argument("--csv", help="feature table with one row per window")
    ap.add_argument("--target", default="isCollision")
    ap.add_argument("--group", default="User_ID")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--out", default="results/uncertainty")
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()

    if a.demo:
        X, y, g = make_synthetic()
        feat = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        feat[a.target], feat[a.group] = y, g
    elif a.csv:
        feat = pd.read_csv(a.csv)
    else:
        ap.error("use --demo or --csv")
    cols = [c for c in feat.columns if c not in (a.target, a.group)]
    summary, _ = run_from_feature_frame(
        feat, cols, a.target, a.group, alpha=a.alpha, n_repeats=a.repeats,
        n_members=a.members, out_dir=a.out, fast=a.fast)
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(summary)


if __name__ == "__main__":
    main()