"""src/data/extract_hinglish.py - keeps only the Roman-script Hinglish tweets
(source == 'hate_speech_tsv') from the combined PRISM file.

Usage (from D:\\TOXIFLAG):
    python src/data/extract_hinglish.py
    python src/data/extract_hinglish.py --src data/raw/combined_hate_speech_dataset.csv --out data/raw/hinglish_tweets.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

KEEP_SOURCE = "hate_speech_tsv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/raw/combined_hate_speech_dataset.csv")
    parser.add_argument("--out", default="data/raw/hinglish_tweets.csv")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    src = Path(args.src)
    if not src.exists():
        sys.exit(f"File not found: {src}. Copy combined_hate_speech_dataset.csv into data/raw/ first.")

    df = pd.read_csv(src, encoding="utf-8")
    needed = {"text", "hate_label", "source"}
    missing = needed - set(df.columns)
    if missing:
        sys.exit(f"Missing columns: {sorted(missing)}. Found: {list(df.columns)}")

    print(f"Rows in combined file : {len(df)}")
    print("Rows per source:")
    print(df["source"].value_counts().to_string())

    out_df = (
        df.loc[df["source"] == KEEP_SOURCE, ["text", "hate_label"]]
        .rename(columns={"hate_label": "label"})
        .reset_index(drop=True)
    )
    if out_df.empty:
        sys.exit(f"No rows with source == '{KEEP_SOURCE}'.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False, encoding="utf-8")

    print(f"\nSaved {len(out_df)} rows to {out}")
    print("Label counts (0 = Normal Speech, 1 = Hate Speech):")
    print(out_df["label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()