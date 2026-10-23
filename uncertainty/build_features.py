"""Build the window-level feature table (features.csv) from the raw Sheffield files.

Uses the same windows and features as the paper notebook (window = 10 rows,
time, frequency and correlation features). It does NOT use Colab.

Step 1, look at the labels first:
    python -m uncertainty.build_features --data-dir Data --diagnose
Step 2, build the table (use the SAME label as your paper):
    python -m uncertainty.build_features --data-dir Data --label isCollision --out Data/features.csv
    python -m uncertainty.build_features --data-dir Data --label isCompliance --invert --out Data/features.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft

warnings.filterwarnings("ignore")

WINDOW = 10
SENSOR_COLS = [
    "ToolPosition_x", "ToolPosition_y", "ToolPosition_z",
    "ToolPosition_A", "ToolPosition_b", "ToolPosition_c",
    "ToolForce_x", "ToolForce_y", "ToolForce_z",
    "ToolTorque_x", "ToolTorque_y", "ToolTorque_z",
    "JointPosition", "JointPosition.1", "JointPosition.2",
    "JointPosition.3", "JointPosition.4", "JointPosition.5", "JointPosition.6",
    "Tool_Velocity", "Tool_Acceleration",
]
CORR_PAIRS = [
    ("ToolPosition_x", "ToolForce_x"), ("ToolPosition_y", "ToolForce_y"),
    ("ToolPosition_z", "ToolForce_z"), ("ToolTorque_x", "ToolForce_x"),
    ("ToolTorque_z", "ToolForce_z"), ("JointPosition", "ToolPosition_y"),
    ("JointPosition.6", "ToolPosition_y"),
]


def to_int_flag(series):
    """TRUE/FALSE/1/0 (text or bool) to 0/1. Same logic as the notebook."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    return (series.astype(str).str.strip().str.upper()
            .map({"TRUE": 1, "FALSE": 0, "1": 1, "0": 0}).fillna(0).astype(int))


def load_users(data_dir):
    files = glob.glob(os.path.join(data_dir, "User_*.csv"))
    files.sort(key=lambda f: int(re.findall(r"User_(\d+)", os.path.basename(f))[0]))
    if not files:
        raise SystemExit(f"No User_*.csv files found in '{data_dir}'.")
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d["User_ID"] = os.path.splitext(os.path.basename(f))[0]
        d = d.ffill()  # fill inside each user only, so values never cross users
        d = d.drop_duplicates()
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def time_features(x, col):
    if np.std(x) < 1e-10:
        kurt, skew = 0.0, 0.0
    else:
        kurt, skew = float(stats.kurtosis(x)), float(stats.skew(x))
    return {
        f"{col}_mean": float(np.mean(x)), f"{col}_std": float(np.std(x)),
        f"{col}_rms": float(np.sqrt(np.mean(x ** 2))),
        f"{col}_kurtosis": kurt, f"{col}_skewness": skew,
        f"{col}_energy": float(np.sum(x ** 2)),
    }


def freq_features(x, col):
    n = len(x)
    yf = np.abs(fft(x)[: n // 2])
    psd = yf ** 2
    psd_norm = psd / (psd.sum() + 1e-10)
    return {
        f"{col}_dom_freq": int(np.argmax(yf)),
        f"{col}_spec_ent": float(-np.sum(psd_norm * np.log(psd_norm + 1e-10))),
        f"{col}_psd_mean": float(np.mean(psd)),
    }


def corr_features(window):
    out = {}
    for c1, c2 in CORR_PAIRS:
        if c1 in window.columns and c2 in window.columns:
            v1, v2 = window[c1].values.astype(float), window[c2].values.astype(float)
            ok = np.std(v1) > 1e-10 and np.std(v2) > 1e-10
            out[f"corr_{c1}_{c2}"] = float(np.corrcoef(v1, v2)[0, 1]) if ok else 0.0
    return out


def extract_features(df, label_col, sensors, window=WINDOW):
    records = []
    for uid, udf in df.groupby("User_ID", sort=False):
        udf = udf.reset_index(drop=True)
        for i in range(0, len(udf) - window + 1, window):
            w = udf.iloc[i:i + window]
            row = {"User_ID": uid, label_col: int(w[label_col].max())}
            for col in sensors:
                x = w[col].values.astype(float)
                row.update(time_features(x, col))
                row.update(freq_features(x, col))
            row.update(corr_features(w))
            records.append(row)
    feat = pd.DataFrame(records)
    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat.fillna(feat.median(numeric_only=True), inplace=True)
    return feat


def diagnose(df):
    print(f"Users: {df['User_ID'].nunique()} | rows: {len(df):,}")
    for col in ("isCollision", "isCompliance"):
        if col not in df.columns:
            print(f"{col}: column not found")
            continue
        flag = to_int_flag(df[col])
        per_user = flag.groupby(df["User_ID"]).sum()
        print(f"{col}: {int(flag.sum()):,} positive rows ({flag.mean() * 100:.2f}%), "
              f"{int((per_user > 0).sum())} users with positives")
    print("\nIf isCollision has (almost) no positives, the paper probably used isCompliance "
          "with --invert (non-compliant = anomaly). Check your notebook before choosing.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", default="Data")
    ap.add_argument("--label", default="isCollision", help="isCollision or isCompliance")
    ap.add_argument("--invert", action="store_true", help="anomaly = label is False (use with isCompliance)")
    ap.add_argument("--out", default="Data/features.csv")
    ap.add_argument("--diagnose", action="store_true", help="only print label counts")
    a = ap.parse_args()

    df = load_users(a.data_dir)
    if a.diagnose:
        diagnose(df)
        return
    if a.label not in df.columns:
        raise SystemExit(f"Label column '{a.label}' not found. Columns: {list(df.columns)[:12]} ...")

    flag = to_int_flag(df[a.label])
    target = "non_compliant" if a.invert else a.label
    df[target] = (1 - flag) if a.invert else flag
    sensors = [c for c in SENSOR_COLS if c in df.columns]
    missing = [c for c in SENSOR_COLS if c not in df.columns]
    if missing:
        print("Warning, sensor columns not found:", missing)

    feat = extract_features(df, target, sensors)
    n_pos = int(feat[target].sum())
    print(f"Feature table: {len(feat):,} windows, {feat.shape[1] - 2} features, "
          f"{feat['User_ID'].nunique()} users")
    print(f"Anomaly windows: {n_pos} ({feat[target].mean() * 100:.2f}%)")
    if n_pos == 0:
        raise SystemExit("No anomaly windows. Choose another label (see --diagnose).")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    feat.to_csv(a.out, index=False)
    print(f"Saved: {a.out}")
    print(f"\nNext: python -m uncertainty.validate --csv {a.out} --target {target} --group User_ID --fast")


if __name__ == "__main__":
    main()