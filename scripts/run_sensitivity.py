from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from otl_tower_extraction.config import load_config
from otl_tower_extraction.experiments import run_sensitivity, plot_sensitivity_curves


def _resolve_path(path_str: str, *, config_path: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p

    candidates = [
        Path.cwd() / p,
        ROOT / p,
        config_path.parent / p,
    ]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    # Return ROOT-relative path as a sensible default even if it does not exist yet.
    return (ROOT / p).resolve()


def main():
    parser = argparse.ArgumentParser(description="Run one-factor parameter sensitivity analysis.")
    parser.add_argument("--config", required=True, help="Sensitivity YAML file.")
    parser.add_argument("--strategy", default=None, choices=["one_factor", "grid"], help="Override strategy in YAML.")
    parser.add_argument("--no-plot", action="store_true", help="Disable sensitivity curve plotting.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately when one run fails.")
    parser.add_argument("--no-traceback", action="store_true", help="Do not store full traceback strings in CSV.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    raw = yaml.safe_load(open(config_path, encoding="utf-8")) or {}

    base_config_path = _resolve_path(raw["base_config"], config_path=config_path)
    cfg = load_config(base_config_path)

    input_paths = raw.get("input_paths")
    if input_paths is not None:
        input_paths = [str(_resolve_path(p, config_path=config_path)) for p in input_paths]

    output_root = raw.get("output_dir", str(Path(cfg.output_dir) / "sensitivity"))
    output_root = _resolve_path(output_root, config_path=config_path)

    strategy = args.strategy or raw.get("strategy", "one_factor")
    base_overrides = raw.get("base_overrides", {})

    raw_df, agg_df = run_sensitivity(
        cfg,
        raw["sweep"],
        input_paths=input_paths,
        output_root=output_root,
        base_overrides=base_overrides,
        strategy=strategy,
        include_default=bool(raw.get("include_default", False)),
        continue_on_error=not args.stop_on_error,
        save_traceback=not args.no_traceback,
        raw_csv_path=output_root / "sensitivity_raw_incremental.csv",
    )

    raw_path = output_root / "sensitivity_raw.csv"
    agg_path = output_root / "sensitivity_aggregate.csv"

    raw_df.to_csv(raw_path, index=False)
    agg_df.to_csv(agg_path, index=False)

    print("\nSensitivity raw results:")
    print(raw_df)
    print(f"\nSaved raw results: {raw_path}")

    print("\nSensitivity aggregate results:")
    print(agg_df)
    print(f"\nSaved aggregate results: {agg_path}")

    plot_cfg = raw.get("plot", {})
    plot_enabled = bool(plot_cfg.get("enabled", True)) and not args.no_plot
    if plot_enabled:
        metric = plot_cfg.get("metric", "F1")
        figure_name = plot_cfg.get("filename", f"sensitivity_{metric}.pdf")
        figure_path = output_root / figure_name
        plot_sensitivity_curves(
            agg_df,
            figure_path,
            metric=metric,
            title=plot_cfg.get("title", "Parameter sensitivity analysis"),
        )
        print(f"Saved sensitivity figure: {figure_path}")


if __name__ == "__main__":
    main()
