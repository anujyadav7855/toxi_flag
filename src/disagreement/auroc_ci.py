"""src/disagreement/auroc_ci.py - bootstrap 95% confidence intervals for the error-detection AUROC.

Adds uncertainty estimates to the engine's results WITHOUT changing the engine's design:
same members, same reference member and primary score (both chosen on validation), same scores.
Resampling is paired: every score is evaluated on the same resampled test comments.

Usage (from D:\\TOXIFLAG):
    python src/disagreement/auroc_ci.py
    python src/disagreement/auroc_ci.py --members muril-base-cased_seed42 muril-base-cased_seed43 muril-base-cased_seed44
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from engine import DIS_SCORES, error_auroc, load_members, score_sets, stack


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics-dir", default="reports/metrics")
    p.add_argument("--members", nargs="*", default=None)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-json", default="reports/metrics/disagreement_auroc_ci.json")
    args = p.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    members = load_members(Path(args.metrics_dir), args.members)
    tags = list(members)
    yv, Pv = stack(members, "val")
    yt, Pt = stack(members, "test")

    val_f1 = [f1_score(yv, (Pv[i] >= 0.5).astype(int), average="macro") for i in range(len(tags))]
    ref_idx = int(np.argmax(val_f1))
    _, sv = score_sets(Pv, ref_idx)
    _, st = score_sets(Pt, ref_idx)

    val_auc = {n: error_auroc(yv, sv[n][1], sv[n][0]) for n in DIS_SCORES}
    primary = max(DIS_SCORES, key=lambda n: -1 if np.isnan(val_auc[n]) else val_auc[n])
    print(f"members: {tags}\nreference single member: {tags[ref_idx]} | primary disagreement score: {primary}\n")

    rng = np.random.default_rng(args.seed)
    n = len(yt)
    draws = {name: [] for name in st}
    for _ in range(args.n_boot):
        idx = rng.choice(n, size=n, replace=True)
        for name, (score, pred) in st.items():
            draws[name].append(error_auroc(yt[idx], pred[idx], score[idx]))
    draws = {k: np.array(v) for k, v in draws.items()}

    results = {"members": tags, "primary": primary, "reference_member": tags[ref_idx],
               "n_boot": args.n_boot, "auroc": {}, "primary_minus_reference": {}}
    print(f"Test error-detection AUROC, 95% bootstrap CI ({args.n_boot} resamples)")
    for name in st:
        point = error_auroc(yt, st[name][1], st[name][0])
        lo, hi = np.nanpercentile(draws[name], [2.5, 97.5])
        results["auroc"][name] = {"point": point, "ci95": [float(lo), float(hi)]}
        print(f"  {name:<26} {point:.3f}  [{lo:.3f}, {hi:.3f}]")

    print(f"\nPaired difference in test AUROC: {primary} minus reference score")
    for ref in ("single_confidence_unc", "ensemble_confidence_unc"):
        diff = draws[primary] - draws[ref]
        diff = diff[~np.isnan(diff)]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        results["primary_minus_reference"][ref] = {
            "mean_diff": float(diff.mean()), "ci95": [float(lo), float(hi)],
            "share_diff_le_0": float((diff <= 0).mean())}
        print(f"  vs {ref:<26} mean {diff.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  share <= 0: {(diff <= 0).mean():.3f}")

    Path(args.out_json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.out_json}")


if __name__ == "__main__":
    main()