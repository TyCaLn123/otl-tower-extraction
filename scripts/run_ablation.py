from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from otl_tower_extraction.config import load_config
from otl_tower_extraction.experiments import run_ablation


def _resolve_path(path_value: str, config_path: Path) -> str:
    p = Path(path_value)
    if p.is_absolute():
        return str(p)
    # Prefer paths relative to the current working directory/project root. If
    # they do not exist, fall back to paths relative to the ablation config file.
    cwd_path = (Path.cwd() / p).resolve()
    if cwd_path.exists():
        return str(cwd_path)
    return str((config_path.parent / p).resolve())


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments.")
    parser.add_argument("--config", required=True, help="Path to ablation YAML config.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    raw = yaml.safe_load(open(config_path, encoding="utf-8")) or {}

    base_config_path = _resolve_path(raw["base_config"], config_path)
    cfg = load_config(base_config_path)

    input_paths = raw.get("input_paths")
    if input_paths:
        input_paths = [_resolve_path(p, config_path) for p in input_paths]

    output_root = raw.get("output_root", str(Path(cfg.output_dir) / "ablation"))
    output_root = Path(output_root)
    if not output_root.is_absolute():
        output_root = (ROOT / output_root).resolve()

    raw_df, agg_df = run_ablation(
        cfg,
        raw["variants"],
        input_paths=input_paths,
        output_root=output_root,
        save_point_clouds=bool(raw.get("save_point_clouds", False)),
        show_progress=bool(raw.get("show_progress", False)),
        continue_on_error=bool(raw.get("continue_on_error", True)),
    )

    output_root.mkdir(parents=True, exist_ok=True)
    raw_out = output_root / "ablation_raw.csv"
    agg_out = output_root / "ablation_summary.csv"
    raw_df.to_csv(raw_out, index=False)
    agg_df.to_csv(agg_out, index=False)

    print("Raw ablation results:")
    print(raw_df)
    print("\nAggregated ablation results:")
    print(agg_df)
    print(f"Saved raw results: {raw_out}")
    print(f"Saved summary: {agg_out}")


if __name__ == "__main__":
    main()
