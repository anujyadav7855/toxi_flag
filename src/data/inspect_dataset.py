"""src/data/inspect_dataset.py - first look at a raw dataset (read-only).

Usage (from D:\\TOXIFLAG):
    python src/data/inspect_dataset.py data/raw/<file>.csv
    python src/data/inspect_dataset.py data/raw/<file>.csv --text <column> --label <column>
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

DEVANAGARI = re.compile(r"[\u0900-\u097F]")


def load(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return pd.read_json(path)
    sep = "\t" if suffix == ".tsv" else ","
    last_error = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, sep=sep, encoding=enc)
        except UnicodeDecodeError as err:
            last_error = err
    raise last_error


def guess_columns(df: pd.DataFrame):
    string_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
    text_col = None
    if string_cols:
        text_col = max(string_cols, key=lambda c: df[c].dropna().astype(str).str.len().mean())
    label_col = None
    for c in df.columns:
        if c != text_col and df[c].nunique(dropna=True) <= 10:
            label_col = c
            break
    return text_col, label_col


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--text", help="name of the text column")
    parser.add_argument("--label", help="name of the label column")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    path = Path(args.path)
    if not path.exists():
        sys.exit(f"File not found: {path}")

    df = load(path)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 80)

    print(f"FILE   : {path}")
    print(f"SHAPE  : {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nCOLUMNS AND TYPES:")
    print(df.dtypes.to_string())
    print("\nMISSING VALUES PER COLUMN:")
    print(df.isna().sum().to_string())
    print("\nFIRST 5 ROWS:")
    print(df.head().to_string())

    guess_text, guess_label = guess_columns(df)
    text_col = args.text or guess_text
    label_col = args.label or guess_label
    print(f"\nTEXT COLUMN  : {text_col}" + ("" if args.text else "  (guessed)"))
    print(f"LABEL COLUMN : {label_col}" + ("" if args.label else "  (guessed)"))

    if text_col not in df.columns:
        sys.exit("Text column not found. Re-run with --text <column>.")

    text = df[text_col].astype(str)
    norm = text.str.lower().str.strip().str.replace(r"\s+", " ", regex=True)

    print("\nTEXT CHECKS:")
    print(f"  empty/missing text          : {int((norm.isin(['', 'nan'])).sum())}")
    print(f"  exact duplicate texts       : {int(text.duplicated().sum())}")
    print(f"  duplicates after lower/trim : {int(norm.duplicated().sum())}")
    print(f"  rows containing Devanagari  : {text.apply(lambda s: bool(DEVANAGARI.search(s))).mean():.1%}")
    print("  words per text:")
    print(text.str.split().str.len().describe().round(1).to_string())

    if label_col and label_col in df.columns:
        counts = df[label_col].value_counts(dropna=False)
        print(f"\nLABEL DISTRIBUTION ({label_col}):")
        print(pd.DataFrame({"count": counts, "percent": (counts / len(df) * 100).round(1)}).to_string())

        conflicts = df.assign(_norm=norm).groupby("_norm")[label_col].nunique()
        print(f"\nSAME TEXT, DIFFERENT LABELS : {int((conflicts > 1).sum())} texts")

        print("\nSAMPLE TEXTS PER LABEL (random_state=42):")
        for value, group in df.groupby(label_col):
            print(f"\n  label = {value!r}")
            for sample in group[text_col].astype(str).sample(min(3, len(group)), random_state=42):
                print(f"    - {sample[:150]}")
    else:
        print("\nNo label column found. Re-run with --label <column>.")


if __name__ == "__main__":
    main()