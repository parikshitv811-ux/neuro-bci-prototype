"""
Statistical Validation
========================
Research-level statistical analysis of TSTA vs baselines.

Tests:
  1. One-sample t-test: SDAS vs 0 (null hypothesis: no alignment)
  2. Paired t-test: TSTA vs CNN baseline
  3. Wilcoxon signed-rank test: within-subject SDAS distribution
  4. Bootstrap confidence intervals for SDAS

Outputs: summary dict + saved JSON in outputs/research/
"""

import os
import json
import numpy as np
from scipy import stats
from tsta_project.config import LOGS_DIR


RESEARCH_DIR = os.path.join(os.path.dirname(LOGS_DIR), "research")


def _bootstrap_ci(values: np.ndarray, n_boot: int = 1000,
                  ci: float = 0.95, seed: int = 42) -> tuple:
    rng    = np.random.RandomState(seed)
    boots  = [rng.choice(values, len(values), replace=True).mean()
               for _ in range(n_boot)]
    lo     = np.percentile(boots, (1 - ci) / 2 * 100)
    hi     = np.percentile(boots, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def run_statistical_validation(ws_results: dict,
                                baseline_results: dict = None) -> dict:
    """
    Args:
        ws_results:       {subj_id: metrics_dict} from run_within_subject
        baseline_results: {method: sdas_list} optional

    Returns:
        Structured statistical report dict.
    """
    print("\n" + "=" * 60)
    print("  STATISTICAL VALIDATION")
    print("=" * 60)

    sdas_vals = np.array([v["sdas"] for v in ws_results.values()])
    top1_vals = np.array([v["top1_acc"] for v in ws_results.values()])

    # ── Descriptive statistics ────────────────────────────────────────────────
    desc = {
        "n":        len(sdas_vals),
        "mean":     round(float(sdas_vals.mean()),   4),
        "std":      round(float(sdas_vals.std()),    4),
        "min":      round(float(sdas_vals.min()),    4),
        "max":      round(float(sdas_vals.max()),    4),
        "median":   round(float(np.median(sdas_vals)), 4),
    }
    ci_lo, ci_hi = _bootstrap_ci(sdas_vals)
    desc["ci_95"] = [round(ci_lo, 4), round(ci_hi, 4)]

    print(f"\n  SDAS — mean ± std : {desc['mean']:.4f} ± {desc['std']:.4f}")
    print(f"  95% CI (bootstrap): [{ci_lo:.4f}, {ci_hi:.4f}]")

    # ── One-sample t-test vs 0 ────────────────────────────────────────────────
    t_stat, p_val = stats.ttest_1samp(sdas_vals, popmean=0.0)
    null_rejected = p_val < 0.05
    print(f"\n  H₀: SDAS = 0  →  t={t_stat:.3f}  p={p_val:.4f}  "
          f"{'✓ REJECTED' if null_rejected else '✗ NOT rejected'}")

    # ── One-sample t-test vs target (0.4) ────────────────────────────────────
    t2, p2 = stats.ttest_1samp(sdas_vals, popmean=0.4)
    above_target = sdas_vals.mean() > 0.4
    print(f"  H₀: SDAS = 0.4 →  t={t2:.3f}  p={p2:.4f}  "
          f"{'mean > target' if above_target else 'mean ≤ target'}")

    # ── Wilcoxon signed-rank test vs 0 ───────────────────────────────────────
    if len(sdas_vals) >= 5:
        w_stat, w_p = stats.wilcoxon(sdas_vals)
        print(f"  Wilcoxon vs 0   →  W={w_stat:.1f}  p={w_p:.4f}")
    else:
        w_stat, w_p = float("nan"), float("nan")
        print("  Wilcoxon: insufficient samples (<5)")

    # ── Baseline comparison ───────────────────────────────────────────────────
    baseline_stats = {}
    if baseline_results:
        print(f"\n  Baseline Comparison:")
        print(f"  {'Method':<28}  {'SDAS':>8}  {'Δ vs TSTA':>10}  {'p-value':>10}")
        print(f"  {'─'*62}")
        for method, b_sdas in baseline_results.items():
            b_arr  = np.array(b_sdas)
            b_mean = b_arr.mean()
            delta  = float(sdas_vals.mean() - b_mean)
            if len(b_arr) >= 2 and len(sdas_vals) >= 2:
                _, p_comp = stats.ttest_ind(sdas_vals, b_arr)
            else:
                p_comp = float("nan")
            baseline_stats[method] = {
                "mean_sdas": round(b_mean, 4),
                "delta":     round(delta,  4),
                "p_value":   round(p_comp, 4) if not np.isnan(p_comp) else None,
                "tsta_better": delta > 0,
            }
            sig = " *" if (not np.isnan(p_comp) and p_comp < 0.05) else ""
            print(f"  {method:<28}  {b_mean:>8.4f}  {delta:>+10.4f}  "
                  f"{p_comp:>10.4f}{sig}")

    report = {
        "sdas_descriptive":     desc,
        "top1_mean":            round(float(top1_vals.mean()), 4),
        "top1_std":             round(float(top1_vals.std()),  4),
        "ttest_vs_zero":        {"t": round(t_stat, 4), "p": round(p_val, 6),
                                  "null_rejected": null_rejected},
        "ttest_vs_target":      {"t": round(t2, 4), "p": round(p2, 6),
                                  "above_target": above_target},
        "wilcoxon_vs_zero":     {"W": round(w_stat, 4) if not np.isnan(w_stat) else None,
                                  "p": round(w_p, 6)   if not np.isnan(w_p)    else None},
        "baseline_comparison":  baseline_stats,
    }

    os.makedirs(RESEARCH_DIR, exist_ok=True)
    out = os.path.join(RESEARCH_DIR, "statistical_validation.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Saved → {out}")
    return report
