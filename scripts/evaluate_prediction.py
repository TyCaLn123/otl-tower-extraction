from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from otl_tower_extraction.data import load_point_cloud
from otl_tower_extraction.metrics import binary_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate extracted tower TXT against labelled ground-truth TXT.")
    parser.add_argument("--gt", required=True, help="Ground-truth TXT with x y z label.")
    parser.add_argument("--pred", required=True, help="Prediction TXT with x y z label. Points in this file are treated as predicted tower points.")
    parser.add_argument("--label-column", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--out", default="evaluation.csv")
    args = parser.parse_args()

    gt = load_point_cloud(args.gt, args.label_column)
    pred = load_point_cloud(args.pred, args.label_column)

    scale = 1 / args.tolerance
    pred_keys = set(map(tuple, np.round(pred.points * scale).astype(np.int64)))
    gt_keys = np.round(gt.points * scale).astype(np.int64)

    y_pred = np.array([tuple(k) in pred_keys for k in gt_keys], dtype=np.int64)
    y_true = gt.labels.astype(np.int64)
    metrics = binary_metrics(y_true, y_pred)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(out, index=False)
    print(metrics)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
