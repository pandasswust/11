#!/usr/bin/env python3
"""Build Section 5 tables from the two field-trial CSVs.

Outputs:
  table_real_single_source.tex
  table_hil_multi_source.tex
  table_mission_efficiency.tex
and prints a summary used in the text.
"""

import math
import numpy as np
import pandas as pd
from scipy import stats

REAL = "real_single_source.csv"
HIL = "hil_multi_source.csv"


def ci95(values):
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 2:
        return (float("nan"), float("nan"))
    return tuple(float(x) for x in stats.t.interval(0.95, n - 1, loc=v.mean(), scale=stats.sem(v)))


def summarize(df, mission_level=False):
    rows = []
    for (method, source), g in df.groupby(["method", "source"], sort=False):
        err = g["error_m"].to_numpy(float)
        n = len(err)
        lo, hi = ci95(err)
        rows.append(dict(
            method=method, source=source, n=n,
            rmse=float(np.sqrt(np.mean(err ** 2))),
            mean=float(err.mean()),
            sd=float(err.std(ddof=1)) if n > 1 else 0.0,
            ci_lo=lo, ci_hi=hi,
            success=float(g["success"].mean()),
            time=float(g["time_s"].mean()),
            path=float(g["path_m"].mean()),
        ))
    for method, g in df.groupby("method", sort=False):
        err = g["error_m"].to_numpy(float)
        n = len(err)
        lo, hi = ci95(err)
        rows.append(dict(
            method=method, source="Overall", n=n,
            rmse=float(np.sqrt(np.mean(err ** 2))),
            mean=float(err.mean()),
            sd=float(err.std(ddof=1)) if n > 1 else 0.0,
            ci_lo=lo, ci_hi=hi,
            success=float(g["success"].mean()),
            time=float(g["time_s"].mean()),
            path=float(g["path_m"].mean()),
        ))
    return pd.DataFrame(rows)


def mission_summary(df):
    """One mission time/path per (method, trial)."""
    m = df.groupby(["method", "trial"], as_index=False).agg(time_s=("time_s", "first"), path_m=("path_m", "first"))
    rows = []
    for method, g in m.groupby("method", sort=False):
        t = g["time_s"].to_numpy(float)
        p = g["path_m"].to_numpy(float)
        tlo, thi = ci95(t)
        plo, phi = ci95(p)
        rows.append(dict(method=method, n=len(g),
                         time_mean=float(t.mean()), time_sd=float(t.std(ddof=1)),
                         time_lo=tlo, time_hi=thi,
                         path_mean=float(p.mean()), path_sd=float(p.std(ddof=1)),
                         path_lo=plo, path_hi=phi))
    return pd.DataFrame(rows), m


def welcoxon(a, b):
    mask = a.notna() & b.notna()
    if mask.sum() < 5:
        return float("nan")
    return float(stats.wilcoxon(a[mask], b[mask]).pvalue)


def fmt(x, nd=2):
    return "--" if (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def fmt_p(p):
    if isinstance(p, float) and math.isnan(p):
        return "--"
    if p < 1e-4:
        exp = int(math.floor(math.log10(p)))
        mant = p / (10 ** exp)
        return f"${mant:.1f}\\times10^{{{exp}}}$"
    return f"${p:.3f}$"


def pretty_source(s):
    return {
        "Co-60": "Co-60",
        "Cs-137_High": "Cs-137 (high)",
        "Cs-137_Low": "Cs-137 (low)",
        "Overall": "Overall",
    }.get(s, str(s).replace("_", r"\_"))


def write_real_table(df, path):
    s = summarize(df)
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Real single-source field trials (10 trials per source and method). Errors are Euclidean terminal localization errors; time and path are per run.}",
        r"\label{tab:real_single}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Method & Source & RMSE (m) & Mean $\pm$ SD (m) & 95\% CI (m) & Success & Time (s) & Path (m) \\",
        r"\midrule",
    ]
    for _, r in s.iterrows():
        lines.append(
            f"{r['method']} & {pretty_source(r['source'])} & {fmt(r['rmse'],3)} & ${fmt(r['mean'],3)} \\pm {fmt(r['sd'],3)}$ & "
            f"$[{fmt(r['ci_lo'],3)}, {fmt(r['ci_hi'],3)}]$ & {100*r['success']:.0f}\\% & {fmt(r['time'],1)} & {fmt(r['path'],1)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def write_hil_table(df, path):
    s = summarize(df)
    p = welcoxon(
        df[df.method == "Proposed"].set_index(["trial", "source"])["error_m"],
        df[df.method == "Voronoi-SPF"].set_index(["trial", "source"])["error_m"],
    )
    trial_ok = {}
    for m, g in df.groupby("method"):
        piv = g.pivot_table(index="trial", columns="source", values="success", aggfunc="first")
        trial_ok[m] = (piv.sum(axis=1) == len(piv.columns)).sum()
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{HiL multi-source field trials (three sources, 10 trials per method). Radiation fields were generated from calibrated Co-60/Cs-137 source models; robot kinematics, SLAM, communication, and Voronoi computation were physically executed.}",
        r"\label{tab:hil_multi}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Method & Source & RMSE (m) & Mean $\pm$ SD (m) & 95\% CI (m) & Success \\",
        r"\midrule",
    ]
    for _, r in s.iterrows():
        lines.append(
            f"{r['method']} & {pretty_source(r['source'])} & {fmt(r['rmse'],3)} & ${fmt(r['mean'],3)} \\pm {fmt(r['sd'],3)}$ & "
            f"$[{fmt(r['ci_lo'],3)}, {fmt(r['ci_hi'],3)}]$ & {100*r['success']:.0f}\\% \\\\"
        )
    lines += [
        r"\bottomrule",
        rf"\multicolumn{{6}}{{l}}{{\scriptsize Trial-level success (all three sources in one mission): {trial_ok.get('Voronoi-SPF', 0)}/10 (Voronoi-SPF) vs {trial_ok.get('Proposed', 0)}/10 (proposed).}} \\",
        rf"\multicolumn{{6}}{{l}}{{\scriptsize Paired Wilcoxon signed-rank test on matched trial/source errors: p = {fmt_p(p)}.}} \\",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def write_efficiency_table(real, hil, path):
    # Real single-source: one run per (trial, source) => n = 30 per method.
    rs = summarize(real)
    rs = rs[rs.source == "Overall"].copy()
    # HiL: one mission per trial => n = 10 per method.
    hm, hmt = mission_summary(hil)

    rp = real.pivot_table(index=["trial", "source"], columns="method", values="time_s", aggfunc="first")
    rpp = real.pivot_table(index=["trial", "source"], columns="method", values="path_m", aggfunc="first")
    p_time_real = welcoxon(rp["Proposed"], rp["Voronoi-SPF"])
    p_path_real = welcoxon(rpp["Proposed"], rpp["Voronoi-SPF"])
    p_time_hil = welcoxon(
        hmt[hmt.method == "Proposed"].set_index("trial")["time_s"],
        hmt[hmt.method == "Voronoi-SPF"].set_index("trial")["time_s"],
    )
    p_path_hil = welcoxon(
        hmt[hmt.method == "Proposed"].set_index("trial")["path_m"],
        hmt[hmt.method == "Voronoi-SPF"].set_index("trial")["path_m"],
    )

    def real_sd(method):
        v = real[real.method == method]
        return float(v["time_s"].std(ddof=1)), float(v["path_m"].std(ddof=1))

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Efficiency. Real single-source values are per run ($n=30$ per method); HiL multi-source values are per mission ($n=10$ per method).}",
        r"\label{tab:efficiency}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Experiment & Method & $n$ & Time (s) & Path per robot (m) & $p$ (time / path) \\",
        r"\midrule",
    ]
    for _, r in rs.iterrows():
        tsd, psd = real_sd(r["method"])
        lines.append(f"Real single-source & {r['method']} & {int(r['n'])} & ${fmt(r['time'],1)} \\pm {fmt(tsd,1)}$ & ${fmt(r['path'],1)} \\pm {fmt(psd,1)}$ & ${p_time_real:.2g}$ / ${p_path_real:.2g}$ \\\\")
    for _, r in hm.iterrows():
        lines.append(f"HiL multi-source & {r['method']} & {int(r['n'])} & ${fmt(r['time_mean'],1)} \\pm {fmt(r['time_sd'],1)}$ & ${fmt(r['path_mean'],1)} \\pm {fmt(r['path_sd'],1)}$ & ${p_time_hil:.2g}$ / ${p_path_hil:.2g}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def main():
    real = pd.read_csv(REAL)
    hil = pd.read_csv(HIL)
    for df in (real, hil):
        df["error_m"] = np.sqrt((df.est_x - df.true_x) ** 2 + (df.est_y - df.true_y) ** 2)

    print("=== REAL SINGLE-SOURCE ===")
    print(summarize(real).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n=== HIL MULTI-SOURCE ===")
    print(summarize(hil).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n=== MISSION EFFICIENCY ===")
    print(mission_summary(real)[0].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(mission_summary(hil)[0].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    write_real_table(real, "table_real_single_source.tex")
    write_hil_table(hil, "table_hil_multi_source.tex")
    write_efficiency_table(real, hil, "table_mission_efficiency.tex")
    print("\nWrote table_real_single_source.tex, table_hil_multi_source.tex, table_mission_efficiency.tex")


if __name__ == "__main__":
    main()
