"""src/disagreement/engine.py - Phase 6: multi-model disagreement and uncertainty flagging.

Inputs: for every ensemble member, a validation file and a test file of probabilities
    reports/metrics/<tag>_val_probs.csv   and   reports/metrics/<tag>_test_probs.csv
(columns: label, prob_hate; row order = data/processed/val.csv and test.csv).

What it does
  1. Final label = mean member probability >= 0.5.
  2. Disagreement scores per comment: vote_disagreement, prob_std, jsd (higher = less agreement).
  3. Reference scores to beat: confidence of the best single member, and confidence of the ensemble mean.
  4. Review thresholds are set on VALIDATION only (target review rates); the same thresholds are then
     applied to the test set. Test labels are used only to evaluate.

Usage (from D:\\TOXIFLAG):
    python src/disagreement/engine.py
    python src/disagreement/engine.py --members baseline_tfidf_lr muril-base-cased_seed42 muril-base-cased_seed43
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

EPS = 1e-9
RATES = [0.10, 0.20, 0.30]
TABLE_COVERAGES = [1.0, 0.9, 0.8, 0.7, 0.6]
DIS_SCORES = ["vote_disagreement", "prob_std", "jsd"]


def binary_entropy(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def load_members(metrics_dir: Path, wanted):
    found = {}
    for val_file in sorted(metrics_dir.glob("*_val_probs.csv")):
        tag = val_file.name[: -len("_val_probs.csv")]
        test_file = metrics_dir / f"{tag}_test_probs.csv"
        if "smoke" not in tag and test_file.exists():
            found[tag] = (val_file, test_file)
    if wanted:
        missing = [t for t in wanted if t not in found]
        if missing:
            sys.exit(f"Members not found (need <tag>_val_probs.csv and <tag>_test_probs.csv): {missing}\nAvailable: {sorted(found)}")
        found = {t: found[t] for t in wanted}
    if len(found) < 2:
        sys.exit(f"Need at least 2 members, found {len(found)}: {sorted(found)}")
    return found


def stack(members, which):
    labels, probs = None, []
    for tag, (val_file, test_file) in members.items():
        frame = pd.read_csv(val_file if which == "val" else test_file)
        if not {"label", "prob_hate"} <= set(frame.columns):
            sys.exit(f"{tag}: file needs columns label and prob_hate")
        y = frame["label"].to_numpy()
        if labels is None:
            labels = y
        elif len(y) != len(labels) or not (y == labels).all():
            sys.exit(f"{tag}: {which} labels/rows differ from the first member")
        probs.append(frame["prob_hate"].to_numpy(dtype=float))
    return labels, np.vstack(probs)


def score_sets(P, ref_idx):
    """Returns mean probability and {name: (uncertainty score, predictions that the score is about)}."""
    n_members = P.shape[0]
    mean_p = P.mean(axis=0)
    ens_pred = (mean_p >= 0.5).astype(int)
    k1 = (P >= 0.5).sum(axis=0)
    majority = np.maximum(k1, n_members - k1)
    ref = P[ref_idx]
    return mean_p, {
        "vote_disagreement": (1 - majority / n_members, ens_pred),
        "prob_std": (P.std(axis=0), ens_pred),
        "jsd": (binary_entropy(mean_p) - binary_entropy(P).mean(axis=0), ens_pred),
        "ensemble_confidence_unc": (1 - np.maximum(mean_p, 1 - mean_p), ens_pred),
        "single_confidence_unc": (1 - np.maximum(ref, 1 - ref), (ref >= 0.5).astype(int)),
    }


def error_auroc(y, pred, score):
    err = (pred != y).astype(int)
    return float("nan") if err.min() == err.max() else float(roc_auc_score(err, score))


def coverage_accuracy(y, pred, score, coverage, rng):
    order = np.lexsort((rng.random(len(y)), score))  # lowest uncertainty first, ties broken at random
    k = int(np.ceil(coverage * len(y)))
    return float((pred[order][:k] == y[order][:k]).mean())


def pick_threshold(val_score, rate):
    """Smallest threshold whose flagged share (score >= tau) on validation is <= rate."""
    tau = np.inf
    for value in np.unique(val_score)[::-1]:
        if (val_score >= value).mean() <= rate:
            tau = float(value)
        else:
            break
    return tau


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics-dir", default="reports/metrics")
    p.add_argument("--members", nargs="*", default=None, help="member tags; default = every pair of files found")
    p.add_argument("--rate", type=float, default=0.20, help="review rate used for the per-comment output file")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-json", default="reports/metrics/disagreement_results.json")
    p.add_argument("--out-csv", default="reports/metrics/disagreement_test_outputs.csv")
    p.add_argument("--out-fig", default="reports/figures/coverage_accuracy_test.png")
    args = p.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    members = load_members(Path(args.metrics_dir), args.members)
    tags = list(members)
    yv, Pv = stack(members, "val")
    yt, Pt = stack(members, "test")
    print(f"members ({len(tags)}): {tags}\nval rows {len(yv)} | test rows {len(yt)}\n")

    print(f"{'member':<34} {'val_macroF1':>11} {'test_acc':>9} {'test_macroF1':>13} {'test_auc':>9}")
    val_f1 = []
    for i, tag in enumerate(tags):
        vf1 = f1_score(yv, (Pv[i] >= 0.5).astype(int), average="macro")
        val_f1.append(vf1)
        print(f"{tag:<34} {vf1:>11.4f} {accuracy_score(yt, (Pt[i] >= 0.5).astype(int)):>9.4f} "
              f"{f1_score(yt, (Pt[i] >= 0.5).astype(int), average='macro'):>13.4f} {roc_auc_score(yt, Pt[i]):>9.4f}")
    ref_idx = int(np.argmax(val_f1))
    print(f"\nreference single member (best validation macro-F1): {tags[ref_idx]}")

    mean_v, sv = score_sets(Pv, ref_idx)
    mean_t, st = score_sets(Pt, ref_idx)
    ens_t = (mean_t >= 0.5).astype(int)
    print(f"ENSEMBLE (mean prob)               val_macroF1 {f1_score(yv, (mean_v >= 0.5).astype(int), average='macro'):.4f} | "
          f"test_acc {accuracy_score(yt, ens_t):.4f} | test_macroF1 {f1_score(yt, ens_t, average='macro'):.4f} | test_auc {roc_auc_score(yt, mean_t):.4f}")

    print("\nError-detection AUROC (how well the score ranks wrong predictions above right ones; 0.5 = no signal)")
    print(f"{'score':<26} {'val':>7} {'test':>7}")
    aur = {}
    for name in sv:
        a_val = error_auroc(yv, sv[name][1], sv[name][0])
        a_test = error_auroc(yt, st[name][1], st[name][0])
        aur[name] = {"val": a_val, "test": a_test}
        print(f"{name:<26} {a_val:>7.3f} {a_test:>7.3f}")

    primary = max(DIS_SCORES, key=lambda n: -1 if np.isnan(aur[n]["val"]) else aur[n]["val"])
    print(f"\nprimary disagreement score (best validation error-detection AUROC): {primary}")

    print("\nTest accuracy of the comments kept when the least certain ones are removed (random tie-break)")
    print(f"{'score':<26}" + "".join(f"{f'keep {int(c * 100)}%':>10}" for c in TABLE_COVERAGES))
    curves = {}
    for name in sv:
        row = [coverage_accuracy(yt, st[name][1], st[name][0], c, np.random.default_rng(args.seed)) for c in TABLE_COVERAGES]
        curves[name] = dict(zip(map(str, TABLE_COVERAGES), row))
        print(f"{name:<26}" + "".join(f"{v:>10.3f}" for v in row))

    print("\nThresholds chosen on VALIDATION, applied unchanged to TEST")
    print(f"{'score':<26} {'target':>6} {'tau':>8} {'val_flag':>8} {'test_flag':>9} {'acc_all':>8} {'acc_kept':>9} {'errors_flagged':>15}")
    operating = []
    for name in sv:
        for rate in RATES:
            tau = pick_threshold(sv[name][0], rate)
            val_flag = float((sv[name][0] >= tau).mean())
            flagged = st[name][0] >= tau
            pred = st[name][1]
            kept = ~flagged
            acc_all = float((pred == yt).mean())
            acc_kept = float((pred[kept] == yt[kept]).mean()) if kept.any() else float("nan")
            wrong = pred != yt
            err_share = float((wrong & flagged).sum() / wrong.sum())
            operating.append({"score": name, "target_rate": rate, "tau": tau, "val_flagged": val_flag,
                              "test_flagged": float(flagged.mean()), "acc_all": acc_all,
                              "acc_kept": acc_kept, "share_of_errors_flagged": err_share})
            print(f"{name:<26} {rate:>6.2f} {tau:>8.3f} {val_flag:>8.3f} {flagged.mean():>9.3f} {acc_all:>8.3f} {acc_kept:>9.3f} {err_share:>15.3f}")

    tau_primary = pick_threshold(sv[primary][0], args.rate)
    flagged_primary = st[primary][0] >= tau_primary
    out = pd.DataFrame({
        "label": yt, "mean_prob_hate": mean_t, "pred": ens_t,
        "agreement": 1 - st["vote_disagreement"][0], "prob_std": st["prob_std"][0], "jsd": st["jsd"][0],
        f"flag_{primary}": flagged_primary.astype(int),
    })
    for i, tag in enumerate(tags):
        out[f"p_{tag}"] = Pt[i]
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    grid = np.linspace(1.0, 0.3, 15)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for name in sv:
        ys = [coverage_accuracy(yt, st[name][1], st[name][0], c, np.random.default_rng(args.seed)) for c in grid]
        ax.plot(grid * 100, ys, marker="o", ms=3, label=name)
    ax.invert_xaxis()
    ax.set_xlabel("Comments kept (%), most certain first")
    ax.set_ylabel("Accuracy on kept comments (test)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=150)
    plt.close(fig)

    results = {
        "members": tags, "reference_single_member": tags[ref_idx], "primary_score": primary,
        "error_detection_auroc": aur, "coverage_accuracy_test": curves, "operating_points": operating,
        "sizes": {"val": int(len(yv)), "test": int(len(yt))},
    }
    Path(args.out_json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.out_json}, {args.out_csv}, {args.out_fig}")


if __name__ == "__main__":
    main()