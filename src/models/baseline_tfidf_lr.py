"""src/models/baseline_tfidf_lr.py - Day 4: TF-IDF + Logistic Regression baseline.

Hyperparameters are chosen on the VALIDATION set only (macro-F1). The test set is
evaluated once, after the choice is made. Do not re-run and tweak settings after
looking at test numbers; that would be tuning on the test set.

Usage (from D:\\TOXIFLAG):
    python src/models/baseline_tfidf_lr.py
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

SEED = 42
FEATURES = {
    "word_1_2": dict(analyzer="word", ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b"),
    "char_wb_2_5": dict(analyzer="char_wb", ngram_range=(2, 5)),
}
C_GRID = [0.3, 1, 3, 10]


def build(feature_cfg: dict, c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, sublinear_tf=True, min_df=2, **feature_cfg)),
            ("clf", LogisticRegression(C=c_value, class_weight="balanced", max_iter=2000, random_state=SEED)),
        ]
    )


def compute_metrics(y, prob, threshold=0.5) -> dict:
    y = np.asarray(y)
    pred = (prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", pos_label=1, zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision_hate": float(precision),
        "recall_hate": float(recall),
        "f1_hate": float(f1),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "roc_auc": float(roc_auc_score(y, prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_ci(y, prob, n_boot=1000, seed=SEED) -> dict:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    pred = (prob >= 0.5).astype(int)
    idx = np.arange(len(y))
    accs, f1s = [], []
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        accs.append((pred[s] == y[s]).mean())
        f1s.append(f1_score(y[s], pred[s], average="macro"))
    return {
        "accuracy_95ci": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
        "macro_f1_95ci": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
        "n_boot": n_boot,
    }


def save_confusion_matrix(m: dict, path: Path, title: str):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Normal", "Hate"])
    ax.set_yticks([0, 1], ["Normal", "Hate"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--model-path", default="models/baseline_tfidf_lr.joblib")
    parser.add_argument("--metrics-path", default="reports/metrics/baseline_metrics.json")
    parser.add_argument("--fig-path", default="reports/figures/baseline_confusion_matrix.png")
    parser.add_argument("--pred-path", default="reports/metrics/baseline_test_predictions.csv")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    data_dir = Path(args.data_dir)
    frames = {}
    for name in ("train", "val", "test"):
        path = data_dir / f"{name}.csv"
        if not path.exists():
            sys.exit(f"Missing {path}. Run src/preprocessing/split_data.py first.")
        frames[name] = pd.read_csv(path, encoding="utf-8")
        if not {"clean_text", "label"} <= set(frames[name].columns):
            sys.exit(f"{path} must contain clean_text and label columns.")
    train, val, test = frames["train"], frames["val"], frames["test"]
    print(f"train {len(train)} | val {len(val)} | test {len(test)}")

    print("\nGrid search on VALIDATION set (selection metric: macro-F1)")
    print(f"{'features':<13} {'C':>5} {'val_macro_f1':>13} {'val_acc':>8}")
    results, best = [], None
    for feat_name, feat_cfg in FEATURES.items():
        for c_value in C_GRID:
            model = build(feat_cfg, c_value).fit(train["clean_text"], train["label"])
            prob_val = model.predict_proba(val["clean_text"])[:, 1]
            m = compute_metrics(val["label"], prob_val)
            results.append({"features": feat_name, "C": c_value, "val_macro_f1": m["macro_f1"], "val_accuracy": m["accuracy"]})
            print(f"{feat_name:<13} {c_value:>5} {m['macro_f1']:>13.4f} {m['accuracy']:>8.4f}")
            if best is None or m["macro_f1"] > best["val_macro_f1"]:
                best = {"features": feat_name, "C": c_value, "val_macro_f1": m["macro_f1"], "model": model}

    model = best["model"]
    print(f"\nSelected on validation: features={best['features']}, C={best['C']}")

    prob_val = model.predict_proba(val["clean_text"])[:, 1]
    val_metrics = compute_metrics(val["label"], prob_val)

    start = time.perf_counter()
    prob_test = model.predict_proba(test["clean_text"])[:, 1]
    elapsed = time.perf_counter() - start
    test_metrics = compute_metrics(test["label"], prob_test)
    ci = bootstrap_ci(test["label"], prob_test)

    print("\nFINAL (test evaluated once)")
    print(f"{'metric':<16} {'val':>8} {'test':>8}")
    for key in ("accuracy", "precision_hate", "recall_hate", "f1_hate", "macro_f1", "roc_auc"):
        print(f"{key:<16} {val_metrics[key]:>8.4f} {test_metrics[key]:>8.4f}")
    print(f"test confusion matrix: TN={test_metrics['tn']} FP={test_metrics['fp']} FN={test_metrics['fn']} TP={test_metrics['tp']}")
    print(f"test accuracy 95% CI  : {ci['accuracy_95ci'][0]:.3f} - {ci['accuracy_95ci'][1]:.3f}")
    print(f"test macro-F1 95% CI  : {ci['macro_f1_95ci'][0]:.3f} - {ci['macro_f1_95ci'][1]:.3f}")
    print(f"inference: {elapsed / len(test) * 1000:.3f} ms per comment (batch of {len(test)})")

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    pred_path = Path(args.pred_path)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "label": test["label"], "prob_hate": prob_test, "pred": (prob_test >= 0.5).astype(int)
    }).to_csv(pred_path, index=False)

    save_confusion_matrix(test_metrics, Path(args.fig_path), "Baseline (TF-IDF + LR), test")

    report = {
        "seed": SEED,
        "selected": {"features": best["features"], "C": best["C"], "selected_on": "validation macro-F1"},
        "grid": results,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_bootstrap": ci,
        "test_ms_per_comment_batch": elapsed / len(test) * 1000,
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    metrics_path = Path(args.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved: {model_path}, {metrics_path}, {pred_path}, {args.fig_path}")


if __name__ == "__main__":
    main()