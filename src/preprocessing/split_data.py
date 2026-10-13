"""src/preprocessing/split_data.py - Day 3: near-duplicate handling, light cleaning,
and a stratified, leakage-checked train/val/test split.

Usage (from D:\\TOXIFLAG):
    python src/preprocessing/split_data.py
    python src/preprocessing/split_data.py --seed 42
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

URL_RE = re.compile(r"(https?://\S+|www\.\S+|pic\.twitter\.com/\S+|fb\.me/\S+)", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+")
NON_WORD_RE = re.compile(r"[\W_]+", re.UNICODE)


def clean_text(text: str) -> str:
    """Conservative cleaning. Keeps case, punctuation, negations, emojis and slang."""
    text = URL_RE.sub(" URL ", text)
    text = MENTION_RE.sub(" @user ", text)
    return WS_RE.sub(" ", text).strip()


def dedupe_key(text: str) -> str:
    """Key used ONLY to find near-duplicates (ignores URLs, mentions, case, punctuation, spaces)."""
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    return NON_WORD_RE.sub("", text.lower())


def class_ratio(frame: pd.DataFrame) -> float:
    return float(frame["label"].mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/raw/hinglish_tweets.csv")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--metrics", default="reports/metrics/split_summary.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    src = Path(args.src)
    if not src.exists():
        sys.exit(f"File not found: {src}. Run src/data/extract_hinglish.py first.")

    df = pd.read_csv(src, encoding="utf-8")
    if not {"text", "label"} <= set(df.columns):
        sys.exit(f"Expected columns text,label. Found: {list(df.columns)}")

    summary = {"seed": args.seed, "rows_loaded": int(len(df))}
    print(f"Rows loaded: {len(df)}")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    if not set(df["label"].unique()) <= {0, 1}:
        sys.exit(f"Unexpected labels: {sorted(df['label'].unique())}")

    df["key"] = df["text"].map(dedupe_key)

    empty = df["key"] == ""
    summary["dropped_empty_after_normalisation"] = int(empty.sum())
    df = df[~empty]
    print(f"Dropped (nothing left after removing URLs/mentions/punctuation): {summary['dropped_empty_after_normalisation']}")

    labels_per_key = df.groupby("key")["label"].nunique()
    conflict_keys = set(labels_per_key[labels_per_key > 1].index)
    conflict_rows = df["key"].isin(conflict_keys)
    summary["conflicting_label_groups"] = len(conflict_keys)
    summary["dropped_conflicting_rows"] = int(conflict_rows.sum())
    df = df[~conflict_rows]
    print(f"Near-duplicate groups with conflicting labels: {len(conflict_keys)} (dropped {summary['dropped_conflicting_rows']} rows)")

    before = len(df)
    df = df.drop_duplicates(subset="key", keep="first")
    summary["dropped_same_label_near_duplicates"] = int(before - len(df))
    print(f"Near-duplicates with the same label removed (kept first): {summary['dropped_same_label_near_duplicates']}")

    df["clean_text"] = df["text"].map(clean_text)
    df = df.reset_index(drop=True)
    summary["rows_after_cleaning"] = int(len(df))
    summary["hate_share_after_cleaning"] = round(class_ratio(df), 4)
    print(f"Rows after dedupe: {len(df)} | Hate share: {class_ratio(df):.3f}")

    train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=args.seed)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=args.seed)
    splits = {"train": train, "val": val, "test": test}

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(splits[a]["key"]) & set(splits[b]["key"])
        if overlap:
            sys.exit(f"LEAKAGE: {len(overlap)} near-duplicate keys appear in both {a} and {b}")
    print("Leakage check passed: no near-duplicate key appears in more than one split.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\nSplit    rows   hate_share")
    summary["splits"] = {}
    for name, frame in splits.items():
        frame = frame[["text", "clean_text", "label"]].reset_index(drop=True)
        frame.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")
        summary["splits"][name] = {"rows": int(len(frame)), "hate_share": round(class_ratio(frame), 4)}
        print(f"{name:<6} {len(frame):>6}   {class_ratio(frame):.3f}")

    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved splits to {out_dir}/ and summary to {metrics_path}")


if __name__ == "__main__":
    main()