"""Find out what drives the label, using held-out users.

Why: if one or two features separate the classes almost perfectly, the task is
easy by construction and uncertainty has little to do. This script shows that.

Run from the repo root:
    python -m uncertainty.label_drivers --csv Data/features.csv --target non_compliant --group User_ID
Add --fast for a quick trial.

Outputs (results/label_drivers/): importance.csv, single_feature_auroc.csv,
ablation.csv, top_features.png
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from uncertainty.uq_study import make_member  # noqa: E402

STATS = ["mean", "std", "rms", "kurtosis", "skewness", "energy", "dom_freq", "spec_ent", "psd_mean"]


def split_name(feature):
    """Return (sensor group, statistic) from a feature name."""
    if feature.startswith("corr_"):
        return "correlation", "corr"
    stat = next((s for s in STATS if feature.endswith("_" + s)), "other")
    sensor = feature[: -(len(stat) + 1)] if stat != "other" else feature
    if sensor.startswith(("ToolForce", "ToolTorque")):
        return "force_torque", stat
    return "kinematics", stat


def grouped_auroc(X, y, groups, seed=0, fast=False, n_splits=5):
    """Mean and std of AUROC over folds where whole users are held out."""
    scores, importances = [], []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        if y[tr].min() == y[tr].max() or y[te].min() == y[te].max():
            continue
        m = make_member(seed + k, fast).fit(X[tr], y[tr])
        scores.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        importances.append(m.feature_importances_)
    return float(np.mean(scores)), float(np.std(scores)), np.mean(importances, axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", default="non_compliant")
    ap.add_argument("--group", default="User_ID")
    ap.add_argument("--out", default="results/label_drivers")
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    cols = [c for c in df.columns if c not in (a.target, a.group)]
    X, y, g = df[cols].values.astype(float), df[a.target].values.astype(int), df[a.group].values
    os.makedirs(a.out, exist_ok=True)

    # 1. Model importance, with users held out
    base, base_sd, imp = grouped_auroc(X, y, g, fast=a.fast)
    imp_df = pd.DataFrame({"feature": cols, "importance": imp}).sort_values("importance", ascending=False)
    imp_df[["group", "stat"]] = [split_name(f) for f in imp_df["feature"]]
    imp_df.to_csv(os.path.join(a.out, "importance.csv"), index=False)
    print(f"Baseline AUROC on held-out users: {base:.3f} (std {base_sd:.3f}, 5 folds)\n")
    print("Top 10 features by model importance:")
    print(imp_df.head(10)[["feature", "importance"]].to_string(index=False, float_format="{:.3f}".format))
    print("\nImportance share by sensor group:")
    print(imp_df.groupby("group")["importance"].sum().sort_values(ascending=False).round(3).to_string())
    print("\nImportance share by statistic:")
    print(imp_df.groupby("stat")["importance"].sum().sort_values(ascending=False).round(3).to_string())

    # 2. Single-feature AUROC (no model, a feature on its own)
    single = []
    for j, c in enumerate(cols):
        if np.std(X[:, j]) < 1e-12:
            continue
        auc = roc_auc_score(y, X[:, j])
        single.append((c, max(auc, 1 - auc)))
    single_df = pd.DataFrame(single, columns=["feature", "auroc"]).sort_values("auroc", ascending=False)
    single_df.to_csv(os.path.join(a.out, "single_feature_auroc.csv"), index=False)
    print("\nTop 10 single-feature AUROC (one feature, no model):")
    print(single_df.head(10).to_string(index=False, float_format="{:.3f}".format))

    # 3. Ablation: remove features and test again on held-out users
    ranked = list(imp_df["feature"])
    groups = {f: split_name(f)[0] for f in cols}
    plans = {"all features": []}
    for k in (1, 3, 10):
        plans[f"drop top {k}"] = ranked[:k]
    for grp in ("force_torque", "kinematics", "correlation"):
        plans[f"drop all {grp}"] = [f for f in cols if groups[f] == grp]
    rows = []
    for name, drop in plans.items():
        keep = [i for i, c in enumerate(cols) if c not in set(drop)]
        if len(keep) == 0:
            continue
        m, s, _ = grouped_auroc(X[:, keep], y, g, fast=a.fast)
        rows.append({"setting": name, "n_features": len(keep), "auroc_mean": m, "auroc_std": s})
    abl = pd.DataFrame(rows)
    abl.to_csv(os.path.join(a.out, "ablation.csv"), index=False)
    print("\nAblation (AUROC on held-out users):")
    print(abl.to_string(index=False, float_format="{:.3f}".format))

    # 4. Plain-language hints
    print("\nHints:")
    top1 = float(imp_df["importance"].iloc[0])
    best_single = float(single_df["auroc"].iloc[0])
    if best_single > 0.95:
        print(f"- One feature alone gives AUROC {best_single:.3f}: the label is almost given by that feature.")
    elif best_single > 0.85:
        print(f"- One feature alone gives AUROC {best_single:.3f}: the task is fairly easy.")
    else:
        print(f"- Best single feature AUROC is {best_single:.3f}: no single feature gives the label away.")
    if top1 > 0.5:
        print(f"- One feature has {top1 * 100:.0f}% of model importance: the model relies on one signal.")
    drop10 = abl.loc[abl["setting"] == "drop top 10", "auroc_mean"]
    if len(drop10):
        d10 = float(drop10.iloc[0])
        if d10 >= base - 0.02:
            print(f"- AUROC barely changes without the top 10 features ({base:.3f} -> {d10:.3f}): "
                  "the signal is spread over many correlated features.")
        else:
            print(f"- AUROC falls from {base:.3f} to {d10:.3f} without the top 10 features: they carry the signal.")
    for _, r in abl[abl["setting"].str.startswith("drop all")].iterrows():
        if r["auroc_mean"] < base - 0.15:
            print(f"- '{r['setting']}' drops AUROC to {r['auroc_mean']:.3f}: the label depends on this group.")

    # 5. Plot
    top = imp_df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["feature"], top["importance"], color="tab:blue")
    ax.set_xlabel("Model importance (mean over folds)")
    ax.set_title("What drives the label (held-out users)")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "top_features.png"), dpi=200)
    print(f"\nSaved results to {a.out}/")


if __name__ == "__main__":
    main()