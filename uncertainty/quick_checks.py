"""Two quick checks on the label.

1. Direction: how does Tool_Velocity_mean differ between the two classes?
2. Timing: where in each recording do the anomaly windows sit?

Run from the repo root:
    python -m uncertainty.quick_checks --csv Data/features.csv --target non_compliant
"""
import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--csv", default="Data/features.csv")
    ap.add_argument("--target", default="non_compliant")
    ap.add_argument("--group", default="User_ID")
    ap.add_argument("--feature", default="Tool_Velocity_mean")
    a = ap.parse_args()

    df = pd.read_csv(a.csv).copy()
    pd.set_option("display.width", 140)

    print(f"CHECK 1: {a.feature} for each class ({a.target}: 0 = normal, 1 = anomaly)")
    desc = df.groupby(a.target)[a.feature].describe()
    print(desc.round(4).to_string())
    m0, m1 = desc.loc[0, "mean"], desc.loc[1, "mean"]
    print(f"\nMean for anomaly windows is {m1:.4f}, for normal windows {m0:.4f}.")
    if abs(m1) < 0.25 * abs(m0):
        print("-> Anomaly windows have a much LOWER velocity: the robot is almost still.")
    elif abs(m1) > 4 * abs(m0):
        print("-> Anomaly windows have a much HIGHER velocity.")
    else:
        print("-> The two means are of similar size: look at the spread (std) and quartiles too.")

    print("\nCHECK 2: share of anomaly windows in each tenth of the recording")
    n = df.groupby(a.group)[a.group].transform("count")
    df["position"] = df.groupby(a.group).cumcount() / n
    by_pos = df.groupby(pd.cut(df["position"], 10), observed=True)[a.target].mean()
    print((by_pos * 100).round(1).astype(str).add(" %").to_string())
    overall = df[a.target].mean() * 100
    edges = pd.concat([by_pos.iloc[:2], by_pos.iloc[-2:]]).mean() * 100
    middle = by_pos.iloc[2:-2].mean() * 100
    print(f"\nOverall rate {overall:.1f} %. First and last 20% of recordings: {edges:.1f} %. Middle 60%: {middle:.1f} %.")
    if edges > 2 * middle:
        print("-> Anomalies sit mostly at the start or end: the label probably marks a phase of the task.")
    elif middle > 2 * edges:
        print("-> Anomalies sit mostly in the middle of the recordings.")
    else:
        print("-> Anomalies are spread through the recordings.")


if __name__ == "__main__":
    main()