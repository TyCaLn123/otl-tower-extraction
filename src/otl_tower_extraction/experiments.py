from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import itertools
import time
import math
import traceback
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Use Times New Roman for all figure text.
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False

from .config import PipelineConfig
from .pipeline import TowerExtractionPipeline


def _set_nested(obj: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    target = obj
    for p in parts[:-1]:
        target = getattr(target, p)
    setattr(target, parts[-1], value)


def _safe_value(value: Any) -> str:
    text = str(value)
    for old, new in [(".", "p"), ("-", "m"), ("/", "_"), ("\\", "_"), (" ", ""), (":", "_")]:
        text = text.replace(old, new)
    return text


def _normalize_sweep(sweep: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = []
    for key, spec in sweep.items():
        if isinstance(spec, dict):
            path = spec.get("path", key)
            values = spec.get("values", [])
            name = spec.get("name", key)
            symbol = spec.get("symbol", name)
            unit = spec.get("unit", "")
            description = spec.get("description", "")
        else:
            path = key
            values = spec
            name = key.replace(".", "_")
            symbol = name
            unit = ""
            description = ""

        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(f"Sweep parameter `{key}` must provide a non-empty value list.")

        items.append({
            "name": str(name),
            "path": str(path),
            "values": values,
            "symbol": str(symbol),
            "unit": str(unit),
            "description": str(description),
        })
    return items


def _apply_overrides(cfg: PipelineConfig, overrides: Optional[Dict[str, Any]]) -> None:
    if not overrides:
        return
    for key, value in overrides.items():
        _set_nested(cfg, key, value)


def _failure_metrics() -> Dict[str, float]:
    return {
        "TP": float("nan"),
        "FP": float("nan"),
        "FN": float("nan"),
        "TN": float("nan"),
        "Precision": float("nan"),
        "Recall": float("nan"),
        "F1": float("nan"),
        "IoU": float("nan"),
        "OA": float("nan"),
    }


def _aggregate_ablation(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ablation metrics over all input lines."""
    if raw_df.empty:
        return raw_df.copy()

    metric_cols = ["TP", "FP", "FN", "TN", "Precision", "Recall", "F1", "IoU", "OA"]
    rows = []
    for variant, group in raw_df.groupby("variant", sort=False):
        row = {
            "variant": variant,
            "num_runs": int(len(group)),
            "successful_runs": int(group["success"].astype(bool).sum()) if "success" in group else 0,
            "failed_runs": int((~group["success"].astype(bool)).sum()) if "success" in group else 0,
            "runtime_sec_total": float(pd.to_numeric(group.get("runtime_sec", pd.Series(dtype=float)), errors="coerce").sum(skipna=True)),
            "runtime_sec_mean": float(pd.to_numeric(group.get("runtime_sec", pd.Series(dtype=float)), errors="coerce").mean(skipna=True)),
        }
        for col in metric_cols:
            if col in group.columns:
                values = pd.to_numeric(group[col], errors="coerce")
                row[f"{col}_mean"] = values.mean(skipna=True)
                row[f"{col}_std"] = values.std(ddof=1) if len(values.dropna()) > 1 else 0.0
                row[f"{col}_min"] = values.min(skipna=True)
                row[f"{col}_max"] = values.max(skipna=True)
        rows.append(row)
    return pd.DataFrame(rows)


def run_ablation(
    base_cfg: PipelineConfig,
    variants: Dict[str, Dict[str, Any]],
    *,
    input_paths: Optional[List[str]] = None,
    output_root: Optional[str | Path] = None,
    save_point_clouds: bool = False,
    show_progress: bool = False,
    continue_on_error: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run ablation experiments.

    By default, ablation runs do not save point-cloud outputs. Only metric CSVs
    are written by the pipeline and summarized by this function.
    """
    paths = input_paths or [base_cfg.input_path]
    output_root = Path(output_root) if output_root is not None else Path(base_cfg.output_dir) / "ablation"
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, changes in variants.items():
        for input_path in paths:
            cfg = deepcopy(base_cfg)
            cfg.input_path = str(input_path)
            cfg.output_dir = str(output_root / name / Path(input_path).stem)
            cfg.output_prefix = "auto"
            cfg.save_point_clouds = bool(save_point_clouds)
            cfg.progress.enabled = bool(show_progress)
            cfg.progress.stage_messages = bool(show_progress)
            cfg.progress.open3d_progress = bool(show_progress)
            cfg.continue_on_candidate_error = True

            for key, val in changes.items():
                _set_nested(cfg, key, val)

            start = time.perf_counter()
            metrics: Dict[str, Any] = {}
            success = True
            error_type = ""
            error_message = ""
            error_traceback = ""
            result: Dict[str, Any] = {}

            try:
                result = TowerExtractionPipeline(cfg).run()
                success = bool(result.get("line_success", True))
                metrics = result.get("scene_metrics") or _failure_metrics()
                if not success:
                    error_type = result.get("first_error_type", "PartialCandidateFailure") or "PartialCandidateFailure"
                    error_message = result.get("first_error_message", "At least one candidate tower failed.")
                    error_traceback = result.get("first_error_traceback", "")
            except Exception as exc:
                success = False
                error_type = type(exc).__name__
                error_message = str(exc)
                error_traceback = traceback.format_exc()
                metrics = _failure_metrics()
                if not continue_on_error:
                    raise

            elapsed = time.perf_counter() - start
            row = {
                "variant": name,
                "input_path": str(input_path),
                "input_file": Path(input_path).name,
                "input_stem": Path(input_path).stem,
                "success": success,
                "error_type": error_type,
                "error_message": error_message,
                "error_traceback": error_traceback,
                "runtime_sec": elapsed,
                "num_candidates": result.get("num_candidates", float("nan")) if result else float("nan"),
                "successful_candidates": result.get("successful_candidates", float("nan")) if result else float("nan"),
                "failed_candidates": result.get("failed_candidates", float("nan")) if result else float("nan"),
                "line_success": result.get("line_success", False) if result else False,
                "partial_line": result.get("partial_line", False) if result else False,
                "output_dir": cfg.output_dir,
                "summary_path": result.get("summary_path", "") if result else "",
                "scene_metrics_path": result.get("scene_metrics_path", "") if result else "",
                "save_point_clouds": cfg.save_point_clouds,
                "central_region_use_whole_candidate": cfg.central_region.use_whole_candidate,
                "side_view_enabled": cfg.side_view.enabled,
                "front_view_enabled": cfg.front_view.enabled,
                "base_filter_enabled": cfg.base_filter.enabled,
                "base_filter_height_threshold_only": cfg.base_filter.use_height_threshold_only,
            }
            row.update(metrics)
            rows.append(row)

            # Incremental save for long ablation runs.
            raw_path = output_root / "ablation_raw_incremental.csv"
            pd.DataFrame([row]).to_csv(
                raw_path,
                mode="a",
                index=False,
                header=not raw_path.exists(),
            )

            if not success:
                print(
                    "\n[Warning] Ablation run failed but was recorded.\n"
                    f"  variant: {name}\n"
                    f"  input: {input_path}\n"
                    f"  error: {error_type}: {error_message}",
                    flush=True,
                )

    raw_df = pd.DataFrame(rows)
    agg_df = _aggregate_ablation(raw_df)
    return raw_df, agg_df


def run_sensitivity(
    base_cfg: PipelineConfig,
    sweep: Dict[str, Any],
    *,
    input_paths: Optional[List[str]] = None,
    output_root: Optional[str | Path] = None,
    base_overrides: Optional[Dict[str, Any]] = None,
    strategy: str = "one_factor",
    include_default: bool = False,
    continue_on_error: bool = True,
    save_traceback: bool = True,
    raw_csv_path: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run parameter sensitivity analysis.

    Failed runs are recorded instead of terminating the whole experiment. This
    is important because unreasonable parameter values may make an intermediate
    stage fail, e.g., central-region identification.
    """
    sweep_items = _normalize_sweep(sweep)
    paths = input_paths or [base_cfg.input_path]
    output_root = Path(output_root) if output_root is not None else Path(base_cfg.output_dir) / "sensitivity"
    output_root.mkdir(parents=True, exist_ok=True)

    raw_csv_path = Path(raw_csv_path) if raw_csv_path is not None else output_root / "sensitivity_raw_incremental.csv"
    if raw_csv_path.exists():
        raw_csv_path.unlink()

    raw_rows: List[Dict[str, Any]] = []

    def append_row(row: Dict[str, Any]) -> None:
        raw_rows.append(row)
        out_df = pd.DataFrame([row])
        raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(
            raw_csv_path,
            mode="a",
            index=False,
            header=not raw_csv_path.exists(),
        )

    def run_single(
        *,
        input_path: str,
        parameter: str,
        parameter_path: str,
        parameter_symbol: str,
        parameter_unit: str,
        parameter_value: Any,
        output_dir: Path,
        is_default: bool,
        grid_values: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = deepcopy(base_cfg)
        _apply_overrides(cfg, base_overrides)
        cfg.input_path = input_path
        cfg.output_dir = str(output_dir)
        cfg.output_prefix = "auto"

        if not is_default:
            if grid_values is None:
                _set_nested(cfg, parameter_path, parameter_value)
            else:
                for key, value in grid_values.items():
                    _set_nested(cfg, key, value)

        start = time.perf_counter()
        metrics: Dict[str, Any] = {}
        success = True
        error_type = ""
        error_message = ""
        error_traceback = ""
        num_candidates = float("nan")
        summary_path = ""
        scene_metrics_path = ""

        try:
            result = TowerExtractionPipeline(cfg).run()
            line_success = bool(result.get("line_success", True))
            success = line_success

            metrics = result.get("scene_metrics") or {}
            if not line_success:
                error_type = result.get("first_error_type", "PartialCandidateFailure") or "PartialCandidateFailure"
                error_message = result.get("first_error_message", "At least one candidate tower failed.")
                error_traceback = result.get("first_error_traceback", "") if save_traceback else ""
                # If the pipeline produced partial metrics from successful candidates,
                # keep them in the sensitivity table. Otherwise, fill NaN.
                if not metrics:
                    metrics = _failure_metrics()

            num_candidates = result.get("num_candidates", float("nan"))
            summary_path = result.get("summary_path", "")
            scene_metrics_path = result.get("scene_metrics_path", "")

        except Exception as exc:
            success = False
            error_type = type(exc).__name__
            error_message = str(exc)
            error_traceback = traceback.format_exc() if save_traceback else ""
            metrics = _failure_metrics()

            print(
                "\n[Warning] Sensitivity run failed but will be recorded and skipped.\n"
                f"  input: {input_path}\n"
                f"  parameter: {parameter_path} = {parameter_value}\n"
                f"  error: {error_type}: {error_message}",
                flush=True,
            )

            if not continue_on_error:
                raise

        elapsed = time.perf_counter() - start
        input_stem = Path(input_path).stem

        row = {
            "success": success,
            "error_type": error_type,
            "error_message": error_message,
            "error_traceback": error_traceback,
            "input_path": input_path,
            "input_file": Path(input_path).name,
            "input_stem": input_stem,
            "parameter": parameter,
            "parameter_path": parameter_path,
            "parameter_symbol": parameter_symbol,
            "parameter_unit": parameter_unit,
            "parameter_value": parameter_value,
            "parameter_value_label": _safe_value(parameter_value),
            "is_default": is_default,
            "num_candidates": num_candidates,
            "successful_candidates": result.get("successful_candidates", float("nan")) if 'result' in locals() and result is not None else float("nan"),
            "failed_candidates": result.get("failed_candidates", float("nan")) if 'result' in locals() and result is not None else float("nan"),
            "runtime_sec": elapsed,
            "output_dir": str(output_dir),
            "summary_path": summary_path,
            "scene_metrics_path": scene_metrics_path,
        }
        row.update(metrics)
        append_row(row)

    if include_default:
        for input_path in paths:
            out_dir = output_root / "default" / Path(input_path).stem
            run_single(
                input_path=input_path,
                parameter="default",
                parameter_path="default",
                parameter_symbol="default",
                parameter_unit="",
                parameter_value="default",
                output_dir=out_dir,
                is_default=True,
            )

    if strategy == "one_factor":
        for item in sweep_items:
            name = item["name"]
            path = item["path"]
            symbol = item["symbol"]
            unit = item["unit"]
            for value in item["values"]:
                for input_path in paths:
                    out_dir = output_root / name / f"{name}_{_safe_value(value)}" / Path(input_path).stem
                    run_single(
                        input_path=input_path,
                        parameter=name,
                        parameter_path=path,
                        parameter_symbol=symbol,
                        parameter_unit=unit,
                        parameter_value=value,
                        output_dir=out_dir,
                        is_default=False,
                    )

    elif strategy == "grid":
        keys = [item["path"] for item in sweep_items]
        names = [item["name"] for item in sweep_items]
        values_lists = [item["values"] for item in sweep_items]

        for values in itertools.product(*values_lists):
            tag = "__".join(f"{n}_{_safe_value(v)}" for n, v in zip(names, values))
            grid_values = dict(zip(keys, values))
            for input_path in paths:
                out_dir = output_root / "grid" / tag / Path(input_path).stem
                run_single(
                    input_path=input_path,
                    parameter="grid",
                    parameter_path=",".join(keys),
                    parameter_symbol=",".join(names),
                    parameter_unit="",
                    parameter_value=str(dict(zip(names, values))),
                    output_dir=out_dir,
                    is_default=False,
                    grid_values=grid_values,
                )
    else:
        raise ValueError("strategy must be either 'one_factor' or 'grid'.")

    raw_df = pd.DataFrame(raw_rows)
    agg_df = aggregate_sensitivity(raw_df)
    return raw_df, agg_df


def aggregate_sensitivity(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df.copy()

    metric_cols = [
        c for c in [
            "Precision", "Recall", "F1", "IoU", "OA",
            "TP", "FP", "FN", "TN",
            "runtime_sec", "num_candidates",
        ]
        if c in raw_df.columns
    ]

    group_cols = [
        "parameter",
        "parameter_path",
        "parameter_symbol",
        "parameter_unit",
        "parameter_value",
        "parameter_value_label",
    ]

    rows = []
    for keys, group in raw_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row["n_datasets"] = group["input_stem"].nunique() if "input_stem" in group.columns else len(group)
        row["n_runs"] = len(group)

        if "success" in group.columns:
            success_values = group["success"].astype(bool)
            row["n_success"] = int(success_values.sum())
            row["n_failed"] = int((~success_values).sum())
            row["success_rate"] = float(success_values.mean())
        else:
            row["n_success"] = len(group)
            row["n_failed"] = 0
            row["success_rate"] = 1.0

        for col in metric_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = values.mean(skipna=True)
            row[f"{col}_std"] = values.std(ddof=1) if len(values.dropna()) > 1 else 0.0
            row[f"{col}_min"] = values.min(skipna=True)
            row[f"{col}_max"] = values.max(skipna=True)

        if "error_message" in group.columns:
            failed = group[group["success"].astype(bool) == False]
            errors = failed["error_message"].dropna().astype(str)
            errors = errors[errors.str.len() > 0]
            row["example_error"] = errors.iloc[0] if len(errors) else ""

        rows.append(row)

    agg = pd.DataFrame(rows)

    if "parameter_value" in agg.columns:
        agg["_sort_value"] = pd.to_numeric(agg["parameter_value"], errors="coerce")
        agg = agg.sort_values(["parameter", "_sort_value", "parameter_value_label"]).drop(columns="_sort_value")

    return agg.reset_index(drop=True)


def plot_sensitivity_curves(
    agg_df: pd.DataFrame,
    out_path: str | Path,
    *,
    metric: str = "F1",
    title: str = "Parameter sensitivity analysis",
) -> None:
    """
    Generate a 3 x 3 multi-panel sensitivity figure.

    Failed runs are ignored in mean/std metric columns because their metrics are
    NaN. If all runs fail for a parameter value, the corresponding subplot is
    left blank.
    """
    if agg_df.empty:
        return

    import matplotlib.pyplot as plt
    import numpy as np

    # Use Times New Roman for manuscript figures.
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.unicode_minus"] = False

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    if mean_col not in agg_df.columns:
        raise ValueError(f"`{mean_col}` not found in aggregate dataframe.")

    # Fixed order for the manuscript sensitivity figure.
    parameter_order = [
        "dbscan_min_points",
        "delta_x",
        "delta_y",
        "epsilon_db",
        "h_f",
        "jump_threshold",
        "tau_b",
        "base_top_score_lambda",
        "voxel_size",
    ]

    params = [p for p in parameter_order if p in set(agg_df["parameter"])]

    n = len(params)
    if n == 0:
        return

    subfigure_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"]

    ncols = 3
    nrows = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)

    for idx, (ax, param) in enumerate(zip(axes.ravel(), params)):
        sub = agg_df[agg_df["parameter"] == param].copy()
        sub["_x"] = pd.to_numeric(sub["parameter_value"], errors="coerce")
        sub = sub.sort_values("_x")
        sub = sub[pd.notna(sub[mean_col])]

        if sub.empty:
            ax.set_title(subfigure_labels[idx], fontsize=13, pad=6)
            ax.axis("off")
            continue

        x = sub["_x"].to_numpy()
        y = sub[mean_col].to_numpy(dtype=float)
        ystd = sub[std_col].to_numpy(dtype=float) if std_col in sub.columns else np.zeros_like(y)

        ax.plot(x, y, marker="o", linewidth=1.5)
        if np.any(np.isfinite(ystd)) and np.nanmax(ystd) > 0:
            ax.fill_between(x, y - ystd, y + ystd, alpha=0.2)

        symbol = str(sub["parameter_symbol"].iloc[0])
        unit = str(sub["parameter_unit"].iloc[0])
        xlabel = symbol if not unit else f"{symbol} ({unit})"
        ax.set_xlabel(xlabel)
        ax.set_ylabel(metric)
        ax.set_title(subfigure_labels[idx], fontsize=13, pad=6)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    for ax in axes.ravel()[len(params):]:
        ax.axis("off")

    # No global title. The title and subfigure descriptions are provided in the manuscript text.
    fig.tight_layout(w_pad=2.0, h_pad=2.0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

