#!/usr/bin/env python3
"""Analyze real single-source / multi-source field-trial logs.

Expected CSV columns (case-insensitive, extra columns ignored):
    method, trial, source, true_x, true_y, est_x, est_y, time_s, path_m, success

- method:  e.g. "Proposed", "Voronoi-SPF"
- trial:   integer trial id (same id across methods => paired comparison)
- source:  source label, e.g. "Co-60", "Cs-137-1"
- success: 1/0, Y/N, True/False

Usage:
    python3 analyze_trials.py --csv real_single_source.csv
    python3 analyze_trials.py --csv real_single_source.csv --latex table_real.tex

The script only summarizes data you provide. It does not generate or impute
measurements.
"""

import argparse
import math
import sys

import numpy as np
import pandas as pd
from scipy import stats


REQUIRED = ["method", "trial", "source", "true_x", "true_y", "est_x", "est_y"]


def parse_success(series):
    if series.dtype == bool:
        return series.astype(float)
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["1", "y", "yes", "true", "t", "success", "ok"]).astype(float)


def ci95(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))
    mean = values.mean()
    sem = stats.sem(values)
    if sem == 0 or math.isnan(sem):
        return (mean, mean)
    lo, hi = stats.t.interval(0.95, n - 1, loc=mean, scale=sem)
    return (float(lo), float(hi))


def summarize(df):
    rows = []
    for (method, source), g in df.groupby(["method", "source"], sort=False):
        err = g["error_m"].to_numpy(dtype=float)
        n = len(err)
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mean = float(np.mean(err))
        sd = float(np.std(err, ddof=1)) if n > 1 else 0.0
        lo, hi = ci95(err)
        success = float(g["success"].mean()) if "success" in g else float("nan")
        time_mean = float(g["time_s"].mean()) if "time_s" in g else float("nan")
        path_mean = float(g["path_m"].mean()) if "path_m" in g else float("nan")
        rows.append(
            dict(
                method=method,
                source=source,
                n=n,
                rmse=rmse,
                mean=mean,
                sd=sd,
                ci_lo=lo,
                ci_hi=hi,
                success=success,
                time_mean=time_mean,
                path_mean=path_mean,
            )
        )

    # Overall per method
    for method, g in df.groupby("method", sort=False):
        err = g["error_m"].to_numpy(dtype=float)
        n = len(err)
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mean = float(np.mean(err))
        sd = float(np.std(err, ddof=1)) if n > 1 else 0.0
        lo, hi = ci95(err)
        success = float(g["success"].mean()) if "success" in g else float("nan")
        time_mean = float(g["time_s"].mean()) if "time_s" in g else float("nan")
        path_mean = float(g["path_m"].mean()) if "path_m" in g else float("nan")
        rows.append(
            dict(
                method=method,
                source="OVERALL",
                n=n,
                rmse=rmse,
                mean=mean,
                sd=sd,
                ci_lo=lo,
                ci_hi=hi,
                success=success,
                time_mean=time_mean,
                path_mean=path_mean,
            )
        )
    return pd.DataFrame(rows)


def paired_tests(df):
    """Wilcoxon signed-rank between every pair of methods on matched trials."""
    if df["method"].nunique() < 2:
        return []
    key = ["trial", "source"]
    pivot = df.pivot_table(index=key, columns="method", values="error_m", aggfunc="mean")
    methods = list(pivot.columns)
    results = []
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            a = pivot[methods[i]]
            b = pivot[methods[j]]
            mask = a.notna() & b.notna()
            if mask.sum() < 5:
                continue
            try:
                stat, p = stats.wilcoxon(a[mask], b[mask])
                results.append((methods[i], methods[j], int(mask.sum()), float(stat), float(p)))
            except ValueError:
                continue
    return results


def latex_table(summary, path):
    def fmt(x, nd=3):
        if isinstance(x, float) and math.isnan(x):
            return "--"
        return f"{x:.{nd}f}"

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Field-trial localization results (mean $\pm$ SD, 95\% CI).}")
    lines.append(r"\label{tab:field_trials}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{llcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Source & $n$ & RMSE (m) & Mean $\pm$ SD (m) & 95\% CI & Success & Time (s) \\")
    lines.append(r"\midrule")
    for _, r in summary.iterrows():
        lines.append(
            "{} & {} & {} & {} & ${} \\pm {}$ & $[{}, {}]$ & {} & {} \\\\".format(
                r["method"],
                r["source"],
                int(r["n"]),
                fmt(r["rmse"]),
                fmt(r["mean"]),
                fmt(r["sd"]),
                fmt(r["ci_lo"]),
                fmt(r["ci_hi"]),
                fmt(r["success"], 2),
                fmt(r["time_mean"], 1),
            )
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="field-trial CSV")
    ap.add_argument("--latex", default=None, help="optional output .tex table")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"Missing columns: {missing}\nExpected: {REQUIRED}")

    for c in ["true_x", "true_y", "est_x", "est_y"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["true_x", "true_y", "est_x", "est_y"]).copy()

    df["error_m"] = np.sqrt(
        (df["est_x"] - df["true_x"]) ** 2 + (df["est_y"] - df["true_y"]) ** 2
    )
    if "success" in df:
        df["success"] = parse_success(df["success"])
    for c in ["time_s", "path_m"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    summary = summarize(df)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    tests = paired_tests(df)
    if tests:
        print("\nPaired Wilcoxon signed-rank tests (matched trial/source):")
        for a, b, n, stat, p in tests:
            print(f"  {a} vs {b}: n={n}, W={stat:.1f}, p={p:.4g}")

    if args.latex:
        latex_table(summary, args.latex)
        print(f"\nLaTeX table written to {args.latex}")


if __name__ == "__main__":
    main()
