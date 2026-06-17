from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_sensitivity_from_aggregate(
    csv_path: str | Path,
    out_pdf: str | Path,
    out_svg: str | Path,
    out_png: str | Path | None = None,
    metric: str = "F1",
):
    csv_path = Path(csv_path)
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # ---------- global font settings ----------
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

    if mean_col not in df.columns:
        raise ValueError(f"{mean_col} not found in {csv_path}")

    # ---------- parameter order ----------
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

    # Only keep the nine manuscript parameters.
    parameter_order = [p for p in parameter_order if p in set(df["parameter"])]

    # ---------- axis labels ----------
    xlabels = {
        "dbscan_min_points": r"$MinPts$",
        "delta_x": r"$\delta_x$ (m)",
        "delta_y": r"$\delta_y$ (m)",
        "epsilon_db": r"$\epsilon_{db}$ (m)",
        "h_f": r"$h_f$ (m)",
        "jump_threshold": r"$T_j$ (m)",
        "tau_b": r"$\tau_b$ (m)",
        "base_top_score_lambda": r"$\lambda$",
        "voxel_size": r"$v_s$ (m)",
        "dbscan_voxel_size": r"$v_{db}$ (m)",
    }

    sub_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"]

    n = len(parameter_order)
    ncols = 3
    nrows = 3

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 3.2 * nrows),
        squeeze=False,
    )

    for idx, param in enumerate(parameter_order):
        ax = axes.ravel()[idx]
        sub = df[df["parameter"] == param].copy()

        sub["_x"] = pd.to_numeric(sub["parameter_value"], errors="coerce")
        sub[mean_col] = pd.to_numeric(sub[mean_col], errors="coerce")

        if std_col in sub.columns:
            sub[std_col] = pd.to_numeric(sub[std_col], errors="coerce")
        else:
            sub[std_col] = 0.0

        sub = sub.sort_values("_x")
        sub = sub[pd.notna(sub["_x"]) & pd.notna(sub[mean_col])]

        if sub.empty:
            ax.set_title(sub_labels[idx], fontsize=13)
            ax.text(
                0.5,
                0.5,
                "No valid data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
            )
            ax.axis("off")
            continue

        x = sub["_x"].to_numpy(dtype=float)
        y = sub[mean_col].to_numpy(dtype=float)
        ystd = sub[std_col].to_numpy(dtype=float)

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.5,
            markersize=4.5,
        )

        if np.any(np.isfinite(ystd)) and np.nanmax(ystd) > 0:
            ax.fill_between(
                x,
                y - ystd,
                y + ystd,
                alpha=0.20,
                linewidth=0,
            )

        # 如果某个参数值存在失败线路，用 x 标记该点
        # if "success_rate" in sub.columns:
        #     success_rate = pd.to_numeric(sub["success_rate"], errors="coerce").to_numpy(dtype=float)
        #     failed = np.isfinite(success_rate) & (success_rate < 1.0)
        #     if failed.any():
        #         ax.scatter(
        #             x[failed],
        #             y[failed],
        #             marker="x",
        #             s=55,
        #             linewidths=1.2,
        #         )

        ax.set_title(sub_labels[idx], fontsize=13, pad=6)
        ax.set_xlabel(xlabels.get(param, param), fontsize=12)
        ax.set_ylabel(metric, fontsize=12)

        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)

        # 让 y 轴稍微留白，避免误差带贴边
        ymin = np.nanmin(y - ystd) if np.any(np.isfinite(ystd)) else np.nanmin(y)
        ymax = np.nanmax(y + ystd) if np.any(np.isfinite(ystd)) else np.nanmax(y)
        margin = max((ymax - ymin) * 0.12, 0.01)
        ax.set_ylim(ymin - margin, ymax + margin)

    # 删除多余空子图
    for ax in axes.ravel()[n:]:
        ax.axis("off")

    # 不设置 fig.suptitle，即去掉大标题
    fig.tight_layout(w_pad=2.0, h_pad=2.0)

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")

    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=600, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved PDF: {out_pdf}")
    if out_png is not None:
        print(f"Saved PNG: {out_png}")


if __name__ == "__main__":
    plot_sensitivity_from_aggregate(
        csv_path="outputs/sensitivity/sensitivity_aggregate.csv",
        out_pdf="outputs/sensitivity/sensitivity_F1_replot.pdf",
        out_svg="outputs/sensitivity/sensitivity_F1_replot.svg",
        out_png="outputs/sensitivity/sensitivity_F1_replot.png",
        metric="F1",
    )