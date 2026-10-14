"""src/evaluation/compare_paired.py - paired comparison of two models on the SAME test rows.

Each input CSV needs columns: label, prob_hate (row order = test.csv order).
Reports accuracy, macro-F1 and ROC-AUC for both models, a paired bootstrap 95% CI for
the difference (B minus A), and an exact McNemar test on the per-comment errors.

Usage (from D:\\TOXIFLAG):
    python src/evaluation/compare_paired.py
    python src/evaluation/compare_paired.py --a reports/metrics/baseline_test_predictions.csv --b reports/metrics/muril-base-cased_seed42_test_probs.csv
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def scores(y, prob):
    pred = (prob >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "roc_auc": roc_auc_score(y, prob),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", default="reports/metrics/baseline_test_predictions.csv")
    p.add_argument("--b", default="reports/metrics/muril-base-cased_seed42_test_probs.csv")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    a, b = pd.read_csv(args.a), pd.read_csv(args.b)
    for name, frame in (("A", a), ("B", b)):
        if not {"label", "prob_hate"} <= set(frame.columns):
            sys.exit(f"Model {name} file needs columns label and prob_hate.")
    if len(a) != len(b) or not (a["label"].to_numpy() == b["label"].to_numpy()).all():
        sys.exit("The two files do not refer to the same test rows (length or labels differ).")

    y = a["label"].to_numpy()
    pa, pb = a["prob_hate"].to_numpy(), b["prob_hate"].to_numpy()
    sa, sb = scores(y, pa), scores(y, pb)

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(y))
    diffs = {k: [] for k in sa}
    for _ in range(args.n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        ra, rb = scores(y[s], pa[s]), scores(y[s], pb[s])
        for k in diffs:
            diffs[k].append(rb[k] - ra[k])

    print(f"A = {args.a}\nB = {args.b}\nrows = {len(y)}\n")
    print(f"{'metric':<10} {'A':>8} {'B':>8} {'B-A':>8}   paired bootstrap 95% CI of (B-A)   share of resamples with B-A <= 0")
    for k in sa:
        d = np.array(diffs[k])
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"{k:<10} {sa[k]:>8.4f} {sb[k]:>8.4f} {sb[k] - sa[k]:>+8.4f}   [{lo:+.4f}, {hi:+.4f}]   {(d <= 0).mean():.3f}")

    wrong_a = (pa >= 0.5).astype(int) != y
    wrong_b = (pb >= 0.5).astype(int) != y
    only_a_wrong = int((wrong_a & ~wrong_b).sum())
    only_b_wrong = int((~wrong_a & wrong_b).sum())
    n_disc = only_a_wrong + only_b_wrong
    print(f"\nMcNemar (accuracy): A wrong and B right = {only_a_wrong}, A right and B wrong = {only_b_wrong}")
    if n_disc:
        print(f"exact two-sided p-value = {binomtest(only_a_wrong, n_disc, 0.5).pvalue:.4f}")
    else:
        print("the two models make identical errors; no test possible")


if __name__ == "__main__":
    main()