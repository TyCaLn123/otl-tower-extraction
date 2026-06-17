from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Merge sensitivity CSV files from multiple batches.")
    parser.add_argument("--root", required=True, help="Root directory containing sensitivity CSV files.")
    parser.add_argument("--pattern", default="**/sensitivity_raw.csv")
    parser.add_argument("--out", default="merged_sensitivity_raw.csv")
    args = parser.parse_args()

    root = Path(args.root)
    files = sorted(root.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {args.pattern} under {root}")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["source_csv"] = str(f)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)

    print(f"Merged {len(files)} files.")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
