"""
Final Structured Research Report
===================================
Prints and returns a clean structured summary of all 9 analysis phases,
evaluating each of the 5 deep claims about EEG trajectory representations.
"""

import json
import os
import numpy as np
from tsta_project.config import LOGS_DIR


CLAIMS = {
    "C1": "EEG meaning lies in direction, NOT magnitude (amplitude invariance)",
    "C2": "Direction is consistent across humans (cross-subject invariance)",
    "C3": "Direction persists under noise and domain shift",
    "C4": "Direction encodes temporal causality (time reversal effect)",
    "C5": "Direction forms a structured latent space (geometric structure)",
}


def _check(results: dict, key: str, sub_key: str = "claim_supported",
           default: bool = False) -> bool:
    return results.get(key, {}).get(sub_key, default)


def print_research_report(results: dict) -> dict:
    """
    Args:
        results: dict with keys = analysis phase names

    Returns:
        Structured report dict (also saved to logs).
    """
    W = 62
    print("\n" + "█" * W)
    print("  TSTA RESEARCH REPORT — Deep Trajectory Representation Analysis")
    print("█" * W)

    # ── Claim evaluation ──────────────────────────────────────────────────────
    claim_verdicts = {}

    # C1: Amplitude invariance
    amp = results.get("amplitude_invariance", {})
    c1_ok  = _check(results, "amplitude_invariance")
    ais    = amp.get("ais_random", float("nan"))
    claim_verdicts["C1"] = {
        "verdict":  c1_ok,
        "evidence": f"AIS (random scale) = {ais:.4f}  (threshold > 0.85)",
    }

    # C2: Cross-subject direction invariance
    inv = results.get("direction_invariance", {})
    c2_ok  = _check(results, "direction_invariance")
    cdas   = inv.get("cdas", float("nan"))
    claim_verdicts["C2"] = {
        "verdict":  c2_ok,
        "evidence": f"CDAS = {cdas:.4f}  (threshold > 0.05)",
    }

    # C3: Noise + domain shift
    ds  = results.get("domain_shift", {})
    nr  = results.get("noise_robustness", {})
    c3_ok = _check(results, "domain_shift")
    dt_sdas = ds.get("mean_dt_sdas", float("nan"))
    claim_verdicts["C3"] = {
        "verdict":  c3_ok,
        "evidence": f"Mean domain-transfer SDAS = {dt_sdas:.4f}",
    }

    # C4: Time reversal
    tr   = results.get("time_reversal", {})
    c4_ok = _check(results, "time_reversal")
    tre   = tr.get("tre", float("nan"))
    ang   = tr.get("mean_angular_change", float("nan"))
    claim_verdicts["C4"] = {
        "verdict":  c4_ok,
        "evidence": f"TRE = {tre:.4f}  angular shift = {ang:.1f}°  (threshold TRE > 0.3)",
    }

    # C5: Geometric structure
    geo  = results.get("geometric_structure", {})
    c5_ok = _check(results, "geometric_structure")
    ari   = geo.get("kmeans_ari", float("nan"))
    ang_s = geo.get("inter_class_angle_deg", float("nan"))
    claim_verdicts["C5"] = {
        "verdict":  c5_ok,
        "evidence": f"k-Means ARI = {ari:.4f}  inter-class angle = {ang_s:.1f}°",
    }

    # ── Print claims ──────────────────────────────────────────────────────────
    print("\n  ╔═══ CLAIM VERDICTS " + "═" * 42 + "╗")
    for cid, info in claim_verdicts.items():
        icon = "✓ SUPPORTED" if info["verdict"] else "✗ NOT SUPPORTED"
        print(f"  ║  {cid}: {icon}")
        print(f"  ║     {CLAIMS[cid]}")
        print(f"  ║     Evidence: {info['evidence']}")
        print(f"  ║  {'─' * 56}")
    print("  ╚" + "═" * 58 + "╝")

    # ── Section summaries ─────────────────────────────────────────────────────
    sections = {
        "INVARIANCE": {
            "Cross-subject CDAS": inv.get("cdas"),
            "Amplitude AIS (random)": ais,
        },
        "ROBUSTNESS": {
            "Domain transfer SDAS": dt_sdas,
        },
        "TEMPORAL": {
            "Smoothness (correct)":   results.get("temporal_smoothness", {}).get("smoothness_correct"),
            "Smoothness (incorrect)": results.get("temporal_smoothness", {}).get("smoothness_incorrect"),
            "Smoothness ratio":       results.get("temporal_smoothness", {}).get("smoothness_ratio"),
        },
        "GEOMETRY": {
            "Angular separability (ASS)": geo.get("ass"),
            "Inter-class angle (°)":      ang_s,
            "k-Means ARI":                ari,
            "Cluster purity":             geo.get("kmeans_purity"),
        },
        "CAUSALITY": {
            "Time Reversal Effect (TRE)":   tre,
            "Mean angular change (°)":      ang,
            "Direction flip rate":          tr.get("direction_flips_pct"),
        },
        "EARLY DETECTION": {
            "cos-sim at 25% signal": results.get("partial_signal", {})
                                          .get("per_fraction", {})
                                          .get("0.25", {}).get("cos_sim"),
            "First fraction at SDAS target": results.get("partial_signal", {})
                                                    .get("first_target_fraction"),
        },
        "FAILURE MODES": {
            "Accuracy":              results.get("failure_modes", {}).get("accuracy"),
            "Confidence (correct)":  results.get("failure_modes", {}).get("mean_confidence_correct"),
            "Confidence (incorrect)":results.get("failure_modes", {}).get("mean_confidence_incorrect"),
        },
    }

    for section, metrics in sections.items():
        print(f"\n  ── {section} {'─' * (W - len(section) - 6)}")
        for name, val in metrics.items():
            val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
            print(f"    {name:<35} {val_str}")

    n_supported = sum(1 for v in claim_verdicts.values() if v["verdict"])
    print(f"\n  ══ OVERALL: {n_supported}/{len(CLAIMS)} claims supported ══")
    print("█" * W + "\n")

    # ── Save report ───────────────────────────────────────────────────────────
    report = {
        "claim_verdicts": claim_verdicts,
        "n_claims_supported": n_supported,
        "n_claims_total": len(CLAIMS),
        "sections": {k: {m: v for m, v in vals.items()}
                     for k, vals in sections.items()},
        "raw_results_keys": list(results.keys()),
    }
    os.makedirs(LOGS_DIR, exist_ok=True)
    out_path = os.path.join(LOGS_DIR, "research_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report saved → {out_path}")
    return report
