from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Dict, Tuple, Optional
import cv2
import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree

from .geometry import (
    minimum_rotated_rectangle_xy,
    rectangle_angles_deg,
    rectangle_side_lengths,
    fit_z_as_linear_of_x,
    inverse_linear,
    rotate_xy,
    fit_3d_line_xz_yz,
    point_triangular_pyramid,
)


# def debug_plot_front_view_points_and_contour(
#     pts,
#     contours,
#     contour_yz,
#     min_grid,
#     step,
#     raster_margin,
#     out_path="debug_front_view_contour.png",
#     scale=10.0,
#     max_scatter_points=200000,
# ):
#     from pathlib import Path
#     import matplotlib.pyplot as plt
#     import numpy as np
#
#     out_path = Path(out_path)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#
#     if len(pts) > max_scatter_points:
#         idx = np.linspace(0, len(pts) - 1, max_scatter_points).astype(np.int64)
#         plot_pts = pts[idx]
#     else:
#         plot_pts = pts
#
#     plt.rcParams["font.family"] = "Times New Roman"
#     plt.rcParams["mathtext.fontset"] = "stix"
#
#     fig, ax = plt.subplots(figsize=(8, 10))
#
#     ax.scatter(
#         plot_pts[:, 1] / scale,
#         plot_pts[:, 2] / scale,
#         s=0.2,
#         alpha=0.25,
#         label="Front-view points"
#     )
#
#     for k, c in enumerate(contours):
#         c = c[:, 0, :]
#         c_yz = (c[:, [1, 0]] - raster_margin + min_grid).astype(float) * step
#         ax.plot(
#             c_yz[:, 0] / scale,
#             c_yz[:, 1] / scale,
#             linewidth=0.8,
#             alpha=0.35,
#             label="All contours" if k == 0 else None,
#         )
#
#     ax.plot(
#         contour_yz[:, 0] / scale,
#         contour_yz[:, 1] / scale,
#         linewidth=2.0,
#         label="Selected contour"
#     )
#
#     ax.set_xlabel("Y (m)")
#     ax.set_ylabel("Z (m)")
#     ax.set_aspect("equal", adjustable="box")
#     ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
#     ax.legend(markerscale=8)
#
#     fig.savefig(out_path, dpi=300, bbox_inches="tight")
#     plt.close(fig)
#
#     print(f"Saved: {out_path}", flush=True)



@dataclass
class CentralRegion:
    theta: float
    z_values: np.ndarray
    rectangles: np.ndarray
    rotated_points: np.ndarray
    params_x_left: Tuple[float, float]
    params_x_right: Tuple[float, float]
    params_y_left: Tuple[float, float]
    params_y_right: Tuple[float, float]


@dataclass
class ExtractionResult:
    tower_points: np.ndarray
    non_tower_points: np.ndarray
    tower_labels: Optional[np.ndarray]
    non_tower_labels: Optional[np.ndarray]
    mask: np.ndarray
    debug: Dict[str, object]



def _line_angle_distance(alpha: float, beta: float) -> float:
    """
    Angular distance between two unoriented 2D lines.

    alpha and alpha + pi are treated as the same direction.
    The returned value lies in [0, pi/2].
    """
    return abs((alpha - beta + np.pi / 2.0) % np.pi - np.pi / 2.0)


def _pca_main_direction_xy(points_xy: np.ndarray) -> float:
    """
    Estimate the main horizontal direction of a 2D point set using PCA.

    Returns
    -------
    float
        Main direction angle in radians, normalized to [0, pi).
    """
    xy = np.asarray(points_xy, dtype=np.float64)
    xy = xy - xy.mean(axis=0, keepdims=True)
    cov = xy.T @ xy / max(len(xy) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, int(np.argmax(eigvals))]
    return float(np.arctan2(v[1], v[0]) % np.pi)


def _select_mbr_angle_by_top_pca(alpha_mbr: float, points: np.ndarray, top_z: float):
    """
    Resolve the 90-degree ambiguity of the MBR angle using the PCA main
    direction of top-region points.

    The selected angle is chosen from {alpha_mbr, alpha_mbr + pi/2}.
    """
    top_region = points[points[:, 2] >= top_z]
    alpha_0 = alpha_mbr % np.pi
    alpha_1 = (alpha_mbr + np.pi / 2.0) % np.pi

    if len(top_region) < 2:
        return alpha_0, {
            "alpha_mbr": alpha_0,
            "alpha_mbr_plus_90": alpha_1,
            "alpha_pca": None,
            "selected": alpha_0,
            "d0": None,
            "d1": None,
            "used_top_pca": False,
            "top_region_points": len(top_region),
        }

    alpha_pca = _pca_main_direction_xy(top_region[:, :2])
    d0 = _line_angle_distance(alpha_0, alpha_pca)
    d1 = _line_angle_distance(alpha_1, alpha_pca)

    selected = alpha_1 if d1 < d0 else alpha_0
    return selected, {
        "alpha_mbr": alpha_0,
        "alpha_mbr_plus_90": alpha_1,
        "alpha_pca": alpha_pca,
        "selected": selected,
        "d0": d0,
        "d1": d1,
        "used_top_pca": True,
        "top_region_points": len(top_region),
    }



def _hist_low_density_score(values: np.ndarray, bin_width: float = 1.0, low_ratio: float = 0.15) -> tuple[float, dict]:
    """
    Compute a histogram sparsity score for one projected coordinate axis.

    The score is the proportion of non-empty bins whose counts are lower than
    low_ratio * max_bin_count. A larger score indicates that the projected
    distribution contains more low-density non-empty bins.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return 0.0, {
            "num_points": int(len(values)),
            "num_bins": 0,
            "num_nonzero_bins": 0,
            "max_count": 0,
            "low_count_bins": 0,
        }

    bin_width = max(float(bin_width), 1.0e-6)
    v_min = np.floor(values.min() / bin_width) * bin_width
    v_max = np.ceil(values.max() / bin_width) * bin_width

    if v_max <= v_min:
        return 0.0, {
            "num_points": int(len(values)),
            "num_bins": 1,
            "num_nonzero_bins": 1,
            "max_count": int(len(values)),
            "low_count_bins": 0,
        }

    bins = np.arange(v_min, v_max + bin_width * 1.5, bin_width)
    hist, _ = np.histogram(values, bins=bins)
    nonzero = hist[hist > 0]

    if len(nonzero) == 0:
        return 0.0, {
            "num_points": int(len(values)),
            "num_bins": int(len(hist)),
            "num_nonzero_bins": 0,
            "max_count": 0,
            "low_count_bins": 0,
        }

    threshold = max(1.0, float(nonzero.max()) * float(low_ratio))
    low_count_bins = int(np.sum(nonzero <= threshold))
    score = low_count_bins / float(len(nonzero))

    return score, {
        "num_points": int(len(values)),
        "num_bins": int(len(hist)),
        "num_nonzero_bins": int(len(nonzero)),
        "max_count": int(nonzero.max()),
        "low_threshold": float(threshold),
        "low_count_bins": low_count_bins,
    }


def _select_mbr_angle_by_top_density_hist(
    alpha_mbr: float,
    points: np.ndarray,
    top_z: float,
    bin_width: float = 1.0,
    low_ratio: float = 0.15,
) -> tuple[float, dict]:
    """
    Resolve the 90-degree ambiguity of the MBR angle using top-region density
    histograms.

    The points are first rotated according to alpha_mbr. The top-region point
    density is then summarized along the rotated X and Y axes. The axis with a
    larger low-density-bin proportion is treated as the side-view axis. If the
    rotated X axis is sparser, alpha_mbr is kept. If the rotated Y axis is
    sparser, alpha_mbr + pi/2 is selected.
    """
    alpha_0 = alpha_mbr % np.pi
    alpha_1 = (alpha_mbr + np.pi / 2.0) % np.pi

    top_region = points[points[:, 2] >= top_z]
    if len(top_region) < 2:
        return alpha_0, {
            "method": "top_density_hist",
            "selected": alpha_0,
            "alpha_mbr": alpha_0,
            "alpha_mbr_plus_90": alpha_1,
            "used_density_hist": False,
            "reason": "insufficient top-region points",
            "top_region_points": int(len(top_region)),
        }

    theta_0 = np.pi / 2.0 - alpha_0
    top_rot = rotate_xy(top_region.astype(np.float64, copy=False), theta_0)

    score_x, hist_x = _hist_low_density_score(top_rot[:, 0], bin_width=bin_width, low_ratio=low_ratio)
    score_y, hist_y = _hist_low_density_score(top_rot[:, 1], bin_width=bin_width, low_ratio=low_ratio)

    # Larger low-density-bin proportion indicates the side-view-like distribution.
    selected = alpha_1 if score_y > score_x else alpha_0

    return selected, {
        "method": "top_density_hist",
        "selected": selected,
        "alpha_mbr": alpha_0,
        "alpha_mbr_plus_90": alpha_1,
        "score_x": float(score_x),
        "score_y": float(score_y),
        "hist_x": hist_x,
        "hist_y": hist_y,
        "used_density_hist": True,
        "top_region_points": int(len(top_region)),
        "bin_width": float(bin_width),
        "low_ratio": float(low_ratio),
    }


class TowerPreciseExtractor:
    def __init__(self, central_cfg, side_cfg, front_cfg, base_cfg, show_progress: bool = True):
        self.central_cfg = central_cfg
        self.side_cfg = side_cfg
        self.front_cfg = front_cfg
        self.base_cfg = base_cfg
        self.show_progress = show_progress

    def extract(self, candidate_points: np.ndarray, candidate_labels=None) -> ExtractionResult:
        points = np.asarray(candidate_points)
        mask = np.ones(len(points), dtype=bool)
        debug = {}

        if self.show_progress:
            print(f"  - precise_extraction: candidate_points={len(points):,}", flush=True)

        if getattr(self.central_cfg, "use_whole_candidate", False):
            central = self._estimate_whole_candidate_region(points)
            debug["central_region_ablation"] = "whole_candidate_pose_estimation"
        else:
            central = self._estimate_central_region(points)
        rotated = central.rotated_points
        debug["central_region"] = central

        if self.side_cfg.enabled:
            side_mask, side_debug = self._side_view_filter(rotated, central)
            mask &= side_mask
            debug["side_view"] = side_debug
            if self.show_progress:
                print(f"  - side_view_filter: kept={int(side_mask.sum()):,}/{len(side_mask):,}", flush=True)

        if self.front_cfg.enabled:
            front_mask, front_debug = self._front_view_filter(rotated, mask, central)
            mask &= front_mask
            debug["front_view"] = front_debug
            if self.show_progress:
                print(f"  - front_view_filter: kept={int(front_mask.sum()):,}/{len(front_mask):,}", flush=True)

        if self.base_cfg.enabled:
            if getattr(self.base_cfg, "use_height_threshold_only", False):
                base_mask, base_debug = self._base_height_threshold_filter(points, mask)
                debug["base_filter"] = base_debug
                debug["base_filter_ablation"] = "height_threshold_only"
            else:
                base_mask, base_debug = self._base_filter(points, mask, central, candidate_labels)
                debug["base_filter"] = base_debug
            mask &= base_mask
            if self.show_progress:
                print(f"  - base_filter: kept={int(base_mask.sum()):,}/{len(base_mask):,}", flush=True)

        candidate_labels = None if candidate_labels is None else np.asarray(candidate_labels)
        tower_labels = None if candidate_labels is None else candidate_labels[mask]
        non_tower_labels = None if candidate_labels is None else candidate_labels[~mask]
        return ExtractionResult(points[mask], points[~mask], tower_labels, non_tower_labels, mask, debug)

    @staticmethod
    def _effective_slope_from_inverse_fit(params: Tuple[float, float]) -> float:
        """
        Convert z=a*x+b into the effective boundary slope x(z)=k*z+b'.
        """
        a, _ = params
        if abs(a) < 1.0e-12:
            return np.inf
        return 1.0 / a

    def _slope_consistency_debug(
        self,
        params_x_left: Tuple[float, float],
        params_x_right: Tuple[float, float],
        params_y_left: Tuple[float, float],
        params_y_right: Tuple[float, float],
    ) -> Dict[str, object]:
        slopes = np.array([
            abs(self._effective_slope_from_inverse_fit(params_x_left)),
            abs(self._effective_slope_from_inverse_fit(params_x_right)),
            abs(self._effective_slope_from_inverse_fit(params_y_left)),
            abs(self._effective_slope_from_inverse_fit(params_y_right)),
        ], dtype=np.float64)

        finite = slopes[np.isfinite(slopes)]
        slope_abs_diff = np.inf if len(finite) == 0 else float(np.max(finite) - np.min(finite))

        return {
            "abs_slopes": slopes,
            "slope_abs_diff": slope_abs_diff,
            "slope_abs_diff_threshold": float(self.central_cfg.slope_abs_diff_threshold),
            "passed": bool(slope_abs_diff <= self.central_cfg.slope_abs_diff_threshold),
        }

    def _estimate_whole_candidate_region(self, points: np.ndarray) -> CentralRegion:
        """
        Ablation estimator without least-disturbed central-region guidance.

        Instead of selecting the stable central body, this routine uses slice
        descriptors from the entire candidate point cloud. The pose is estimated
        from the minimum-area rectangle of the whole candidate, and the boundary
        trajectories are fitted from all valid slices. Consequently, conductors,
        insulators, ground points, and vegetation can affect the estimated pose
        and structural references.
        """
        cfg = self.central_cfg
        pts = points.astype(np.float64, copy=False)
        z_min, z_max = pts[:, 2].min(), pts[:, 2].max()

        try:
            full_rect = minimum_rotated_rectangle_xy(pts[:, :2])
            full_angles = rectangle_angles_deg(full_rect)
            alpha_mbr = float(np.mean(full_angles)) * np.pi / 180.0
        except Exception:
            alpha_mbr = _pca_main_direction_xy(pts[:, :2])

        theta = np.pi / 2.0 - alpha_mbr
        rotated = rotate_xy(pts, theta)

        starts = np.arange(z_min, z_max - cfg.slice_window_height, cfg.slice_height)
        rectangles, z_values = [], []
        iterator = tqdm(starts, desc="    whole-candidate slices", leave=False) if self.show_progress else starts
        for z0 in iterator:
            layer = pts[(pts[:, 2] >= z0) & (pts[:, 2] <= z0 + cfg.slice_window_height)]
            if len(layer) < 4:
                continue
            try:
                rectangles.append(minimum_rotated_rectangle_xy(layer[:, :2]))
                z_values.append(float(z0))
            except Exception:
                continue

        if len(rectangles) < 2:
            # Fallback: use low and high horizontal slices derived from the full
            # rotated bounding rectangle. A tiny height-dependent perturbation is
            # added to avoid singular inverse-linear fits.
            rot_rect3 = rotate_xy(np.column_stack([full_rect, np.zeros(4)]), theta)
            x0, x1 = rot_rect3[:, 0].min(), rot_rect3[:, 0].max()
            y0, y1 = rot_rect3[:, 1].min(), rot_rect3[:, 1].max()
            eps = max(1.0e-6, 1.0e-4 * max(z_max - z_min, 1.0))
            z_values = np.asarray([z_min, z_max], dtype=np.float64)
            rect_rot = np.asarray([
                [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                [[x0 + eps, y0 + eps], [x1 + eps, y0 + eps], [x1 + eps, y1 + eps], [x0 + eps, y1 + eps]],
            ], dtype=np.float64)
            # Convert rotated rectangles back only for storage/base corner logic.
            # The fitted parameters are computed directly below from rect_rot.
            rectangles = np.repeat(full_rect[None, :, :], 2, axis=0)
        else:
            rectangles = np.asarray(rectangles, dtype=np.float64)
            z_values = np.asarray(z_values, dtype=np.float64)
            rect_points = rectangles.reshape(-1, 2)
            rect_points3 = np.column_stack([rect_points, np.zeros(len(rect_points))])
            rect_rot = rotate_xy(rect_points3, theta)[:, :2].reshape(len(z_values), 4, 2)

        x_left, x_right = rect_rot[:, :, 0].min(axis=1), rect_rot[:, :, 0].max(axis=1)
        y_left, y_right = rect_rot[:, :, 1].min(axis=1), rect_rot[:, :, 1].max(axis=1)

        params_x_left = fit_z_as_linear_of_x(x_left, z_values)
        params_x_right = fit_z_as_linear_of_x(x_right, z_values)
        params_y_left = fit_z_as_linear_of_x(y_left, z_values)
        params_y_right = fit_z_as_linear_of_x(y_right, z_values)

        if self.show_progress:
            print(
                f"  - central_region ablation: whole_candidate=True, "
                f"slices={len(z_values)}, z_range=({z_values.min():.2f}, {z_values.max():.2f})",
                flush=True,
            )

        return CentralRegion(
            theta=theta,
            z_values=z_values,
            rectangles=np.asarray(rectangles, dtype=np.float64),
            rotated_points=rotated,
            params_x_left=params_x_left,
            params_x_right=params_x_right,
            params_y_left=params_y_left,
            params_y_right=params_y_right,
        )

    def _base_height_threshold_filter(self, points: np.ndarray, current_mask: np.ndarray):
        """
        Ablation base filter using only a simple height threshold.

        This replaces geometric base-model reconstruction. Points below
        z_min + height_threshold are removed directly, while all points above
        the threshold are preserved. No transition-height estimation, lower
        tower-foot plane fitting, or pyramid-like base model is used.
        """
        cfg = self.base_cfg
        keep = current_mask.copy()
        active = points[current_mask]
        if len(active) == 0:
            return keep, {"warning": "empty current mask", "mode": "height_threshold_only"}

        z_min = float(active[:, 2].min())
        threshold = z_min + float(getattr(cfg, "height_threshold", cfg.fallback_base_height))
        target = current_mask & (points[:, 2] <= threshold)
        keep[target] = False

        return keep, {
            "mode": "height_threshold_only",
            "z_min": z_min,
            "height_threshold": float(getattr(cfg, "height_threshold", cfg.fallback_base_height)),
            "absolute_threshold_z": float(threshold),
            "removed_points": int(np.count_nonzero(target)),
        }

    def _estimate_central_region(self, points: np.ndarray) -> CentralRegion:
        cfg = self.central_cfg
        pts = points.astype(np.float64, copy=False)
        z_min, z_max = pts[:, 2].min(), pts[:, 2].max()

        def compute_slice_descriptors(slice_height: float):
            starts = np.arange(z_min, z_max - cfg.slice_window_height, slice_height)
            angle_means, angle_ranges, rect_diffs, rectangles, z_values = [], [], [], [], []

            iterator = tqdm(starts, desc="    central-region slices", leave=False) if self.show_progress else starts

            for z0 in iterator:
                layer = pts[(pts[:, 2] >= z0) & (pts[:, 2] <= z0 + cfg.slice_window_height)]
                z_values.append(z0)

                if len(layer) < 4:
                    angle_means.append(np.nan)
                    angle_ranges.append(np.inf)
                    rect_diffs.append(np.inf)
                    rectangles.append(np.full((4, 2), np.nan))
                    continue

                try:
                    rect = minimum_rotated_rectangle_xy(layer[:, :2])
                    lens = rectangle_side_lengths(rect)
                    rect_angles = rectangle_angles_deg(rect)

                    angle_means.append(float(np.mean(rect_angles)))
                    angle_ranges.append(float(np.ptp(rect_angles)))
                    rect_diffs.append(float(abs(lens[0] - lens[1])))
                    rectangles.append(rect)
                except Exception:
                    angle_means.append(np.nan)
                    angle_ranges.append(np.inf)
                    rect_diffs.append(np.inf)
                    rectangles.append(np.full((4, 2), np.nan))

            return (
                np.asarray(angle_means, dtype=np.float64),
                np.asarray(angle_ranges, dtype=np.float64),
                np.asarray(rect_diffs, dtype=np.float64),
                np.asarray(rectangles, dtype=np.float64),
                np.asarray(z_values, dtype=np.float64),
            )

        def select_stable_indices(angle_means, angle_ranges, rect_diffs) -> np.ndarray:
            valid = np.where(
                np.isfinite(angle_means)
                & (angle_ranges <= cfg.angle_range_threshold_deg)
                & (rect_diffs <= cfg.rectangle_diff_threshold)
            )[0]

            # 保留 valid 中最长的连续段
            if valid.size > 1:
                # 找到不连续的位置
                split_positions = np.where(np.diff(valid) != 1)[0] + 1

                # 拆分为多个连续段
                valid_segments = np.split(valid, split_positions)

                # 选择长度最长的连续段
                valid = max(valid_segments, key=lambda seg: seg.size)

            if len(valid) < 2:
                return np.asarray([], dtype=np.int64)

            mean = np.mean(angle_means[valid])
            std = np.std(angle_means[valid])
            stable = valid[
                np.abs(angle_means[valid] - mean)
                <= max(std * cfg.angle_std_factor, 1.0e-6)
            ]

            if len(stable) < 2:
                stable = valid

            return stable.astype(np.int64)

        def build_central_result(
            angle_means,
            rectangles,
            z_values,
            stable: np.ndarray,
            slice_height: float,
            adaptive_debug: Dict[str, object],
        ) -> tuple[CentralRegion, Dict[str, object]]:
            alpha_mbr = np.mean(angle_means[stable]) * np.pi / 180.0

            alpha_selected, orientation_debug = _select_mbr_angle_by_top_density_hist(
                alpha_mbr=alpha_mbr,
                points=pts,
                top_z=z_values[stable].max(),
                bin_width=max(1.0, float(self.side_cfg.vertical_step)),
                low_ratio=0.15,
            )

            theta = np.pi / 2.0 - alpha_selected
            rotated = rotate_xy(pts, theta)

            rect_points = rectangles[stable].reshape(-1, 2)
            rect_points3 = np.column_stack([rect_points, np.zeros(len(rect_points))])
            rect_rot = rotate_xy(rect_points3, theta)[:, :2].reshape(len(stable), 4, 2)
            z_sel = z_values[stable]

            x_left, x_right = rect_rot[:, :, 0].min(axis=1), rect_rot[:, :, 0].max(axis=1)
            y_left, y_right = rect_rot[:, :, 1].min(axis=1), rect_rot[:, :, 1].max(axis=1)

            params_x_left = fit_z_as_linear_of_x(x_left, z_sel)
            params_x_right = fit_z_as_linear_of_x(x_right, z_sel)
            params_y_left = fit_z_as_linear_of_x(y_left, z_sel)
            params_y_right = fit_z_as_linear_of_x(y_right, z_sel)

            slope_debug = self._slope_consistency_debug(
                params_x_left,
                params_x_right,
                params_y_left,
                params_y_right,
            )

            debug = {
                "stable_indices": stable,
                "z_range": (float(z_sel.min()), float(z_sel.max())),
                "num_stable_slices": int(len(stable)),
                "orientation": orientation_debug,
                "slope": slope_debug,
                "adaptive": adaptive_debug,
                "slice_height": float(slice_height),
            }

            central = CentralRegion(
                theta=theta,
                z_values=z_sel,
                rectangles=rectangles[stable],
                rotated_points=rotated,
                params_x_left=params_x_left,
                params_x_right=params_x_right,
                params_y_left=params_y_left,
                params_y_right=params_y_right,
            )
            return central, debug

        def evaluate_slice_height(slice_height: float, used_adaptive: bool, trial_id: int | None):
            angle_means, angle_ranges, rect_diffs, rectangles, z_values = compute_slice_descriptors(slice_height)
            stable = select_stable_indices(angle_means, angle_ranges, rect_diffs)
            if len(stable) < 2:
                return None

            adaptive_debug = {
                "enabled": bool(cfg.adaptive_slope_check),
                "used_adaptive_parameters": bool(used_adaptive),
                "trial_id": trial_id,
                "slice_height": float(slice_height),
                "angle_range_threshold_deg": float(cfg.angle_range_threshold_deg),
                "rectangle_diff_threshold": float(cfg.rectangle_diff_threshold),
                "angle_std_factor": float(cfg.angle_std_factor),
            }
            return build_central_result(
                angle_means,
                rectangles,
                z_values,
                stable,
                slice_height,
                adaptive_debug,
            )

        default_eval = evaluate_slice_height(cfg.slice_height, used_adaptive=False, trial_id=None)
        if default_eval is None:
            raise RuntimeError("Failed to identify central region.")

        best_central, best_debug = default_eval
        best_score = best_debug["slope"]["slope_abs_diff"]

        if cfg.adaptive_slope_check and best_score > cfg.slope_abs_diff_threshold:
            candidates = []
            tried = set()

            slice_heights = getattr(
                cfg,
                "adaptive_slice_heights",
                [0.1, 0.2, 0.5, 1.0, 2.0],
            )

            for trial_id, slice_height in enumerate(slice_heights):
                slice_height = float(slice_height)

                # Avoid invalid or duplicate step sizes.
                if slice_height <= 0:
                    continue
                key = round(slice_height, 6)
                if key in tried:
                    continue
                tried.add(key)

                result = evaluate_slice_height(slice_height, used_adaptive=True, trial_id=trial_id)
                if result is None:
                    continue

                central, debug = result
                slope_diff = debug["slope"]["slope_abs_diff"]
                num_slices = debug["num_stable_slices"]

                # Prioritize slope consistency. If tied, prefer more stable slices.
                candidates.append((slope_diff, -num_slices, central, debug))

            if candidates:
                passing = [c for c in candidates if c[0] <= cfg.slope_abs_diff_threshold]
                if passing:
                    selected = sorted(passing, key=lambda x: (x[0], x[1]))[0]
                else:
                    selected = sorted(candidates, key=lambda x: (x[0], x[1]))[0]

                if selected[0] < best_score:
                    best_central, best_debug = selected[2], selected[3]
                    best_debug["adaptive"]["previous_slope_abs_diff"] = float(best_score)
                    best_debug["adaptive"]["default_slice_height"] = float(cfg.slice_height)

        if self.show_progress:
            slope_msg = (
                f", slope_abs_diff={best_debug['slope']['slope_abs_diff']:.4f}"
                f"/{best_debug['slope']['slope_abs_diff_threshold']:.4f}"
            )

            orientation_debug = best_debug["orientation"]
            orientation_msg = ""
            if orientation_debug.get("used_density_hist", False):
                orientation_msg = (
                    f", density_score_x={orientation_debug['score_x']:.3f}"
                    f", density_score_y={orientation_debug['score_y']:.3f}"
                    f", selected={orientation_debug['selected'] * 180.0 / np.pi:.2f} deg"
                )

            adaptive_msg = ""
            if best_debug["adaptive"].get("used_adaptive_parameters", False):
                adaptive_msg = (
                    f", adaptive_slice_height="
                    f"{best_debug['adaptive']['slice_height']:.3f}"
                )

            print(
                f"  - central_region: stable_slices={best_debug['num_stable_slices']}, "
                f"z_range=({best_debug['z_range'][0]:.2f}, {best_debug['z_range'][1]:.2f})"
                f"{orientation_msg}{slope_msg}{adaptive_msg}",
                flush=True,
            )

        return best_central

    def _side_view_filter(self, rotated: np.ndarray, central: CentralRegion):
        cfg = self.side_cfg
        z_min, z_max = rotated[:, 2].min(), rotated[:, 2].max()
        zc_top = central.z_values.max()
        step = cfg.vertical_step

        ax_l, bx_l = central.params_x_left
        ax_r, bx_r = central.params_x_right
        x_l0 = lambda z: inverse_linear(z, ax_l, bx_l)
        x_r0 = lambda z: inverse_linear(z, ax_r, bx_r)

        high_z = np.arange(zc_top, z_max + step, step)
        left_records = [(zc_top, x_l0(zc_top))]
        right_records = [(zc_top, x_r0(zc_top))]
        top_left_ref = x_l0(zc_top)
        top_right_ref = x_r0(zc_top)

        iterator = tqdm(high_z[1:], desc="    side-view high-region search", leave=False) if self.show_progress else high_z[1:]
        for z in iterator:
            layer = rotated[np.isclose(rotated[:, 2], z, atol=step / 2)]
            if len(layer) == 0:
                left_records.append((z, left_records[-1][1]))
                right_records.append((z, right_records[-1][1]))
                continue

            xs = layer[:, 0]
            x_min, x_max = float(xs.min()), float(xs.max())
            last_left, last_right = left_records[-1][1], right_records[-1][1]

            if np.all(xs >= top_left_ref) and last_left <= x_min <= last_right and x_min < x_l0(z):
                left_records.append((z, x_min))

            last_left = left_records[-1][1]
            if np.all(xs <= top_right_ref) and x_max <= last_right and x_max >= last_left and x_max > x_r0(z):
                right_records.append((z, x_max))

        if left_records[-1][0] < z_max:
            left_records.append((z_max, left_records[-1][1]))
        if right_records[-1][0] < z_max:
            right_records.append((z_max, right_records[-1][1]))

        z_grid = np.arange(z_min, z_max + step, step)
        left = np.empty_like(z_grid)
        right = np.empty_like(z_grid)

        low = z_grid <= zc_top
        left[low], right[low] = x_l0(z_grid[low]), x_r0(z_grid[low])
        zl, xl = np.asarray(left_records).T
        zr, xr = np.asarray(right_records).T
        left[~low] = np.interp(z_grid[~low], zl, xl)
        right[~low] = np.interp(z_grid[~low], zr, xr)

        idx = np.searchsorted(z_grid, rotated[:, 2])
        idx = np.clip(idx, 0, len(z_grid) - 1)
        keep = (rotated[:, 0] >= left[idx] - cfg.tolerance) & (rotated[:, 0] <= right[idx] + cfg.tolerance)
        return keep, {"z_grid": z_grid, "left": left, "right": right}

    def _front_view_filter(self, rotated: np.ndarray, current_mask: np.ndarray, central: CentralRegion):
        cfg = self.front_cfg
        pts = rotated[current_mask]
        keep_all = np.ones(len(rotated), dtype=bool)
        if len(pts) < 3:
            return keep_all, {"warning": "insufficient side-filtered points"}

        z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
        zc_top = central.z_values.max()
        step = cfg.vertical_step

        ay_l, by_l = central.params_y_left
        ay_r, by_r = central.params_y_right
        y_l0 = lambda z: inverse_linear(z, ay_l, by_l)
        y_r0 = lambda z: inverse_linear(z, ay_r, by_r)
        y_center = (y_l0(zc_top) + y_r0(zc_top)) / 2.0

        yz = np.column_stack([pts[:, 1], pts[:, 2]])
        grid = np.unique(np.round(yz / step).astype(int), axis=0)
        min_grid = grid.min(axis=0)
        image_pts = grid - min_grid + cfg.raster_margin
        shape = image_pts.max(axis=0) + cfg.raster_margin + 1
        img_raw = np.zeros((shape[0], shape[1]), dtype=np.uint8)

        # Vectorized rasterization. img is indexed as img[y_index, z_index].
        img_raw[image_pts[:, 0], image_pts[:, 1]] = 255

        def _contour_to_yz(contour_arr: np.ndarray) -> np.ndarray:
            """
            Convert OpenCV contour coordinates to front-view coordinates.

            OpenCV returns contour coordinates as [col, row]. Since the raster
            image is indexed as img[y_index, z_index], [col, row] corresponds to
            [z_index, y_index]. Therefore, it must be converted to [y, z].
            """
            contour_arr = contour_arr[:, 0, :] if contour_arr.ndim == 3 else contour_arr
            return (contour_arr[:, [1, 0]] - cfg.raster_margin + min_grid).astype(float) * step

        def _find_largest_contour(contour_img: np.ndarray):
            contours_, _ = cv2.findContours(
                contour_img,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )
            if not contours_:
                return None, None, None, np.inf

            selected = max(contours_, key=cv2.contourArea)
            selected_yz = _contour_to_yz(selected)

            # The selected contour should reach the top of the front-view point set.
            # If this gap is large, the contour is likely only a local low-region component.
            selected_top_z = selected_yz[:, 1].max()
            top_gap = z_max - selected_top_z
            return contours_, selected, selected_yz, top_gap

        # Step 1: try contour extraction on the original raster image.
        contours, contour_raw, contour_yz, top_gap = _find_largest_contour(img_raw)
        if contours is None:
            return keep_all, {"warning": "no contour"}

        # Step 2: only use morphology when the selected contour is too low.
        # In the current integer-coordinate pipeline, coordinates are usually scaled by 10.
        # Therefore, default 10.0 corresponds to 1.0 m.
        contour_top_gap_threshold = getattr(cfg, "contour_top_gap_threshold", 10.0)

        morph_used = False
        morph_kernel_cells = 0
        morph_iterations = getattr(cfg, "morph_iterations", 1)

        # Kernel sizes are raster-cell counts, not metric distances.
        morph_start_cells = int(getattr(cfg, "morph_start_cells", 7))
        morph_max_cells = int(getattr(cfg, "morph_max_cells", 21))
        morph_step_cells = int(getattr(cfg, "morph_step_cells", 2))

        morph_start_cells = max(3, morph_start_cells)
        morph_max_cells = max(morph_start_cells, morph_max_cells)
        morph_step_cells = max(1, morph_step_cells)

        # Prefer odd kernel sizes: 7, 9, 11, ...
        if morph_start_cells % 2 == 0:
            morph_start_cells += 1
        if morph_max_cells % 2 == 0:
            morph_max_cells += 1
        if morph_step_cells % 2 == 1:
            morph_step_cells += 1

        best_contours = contours
        best_contour_yz = contour_yz
        best_top_gap = top_gap
        best_kernel_cells = 0

        if top_gap > contour_top_gap_threshold:
            for k in range(morph_start_cells, morph_max_cells + 1, morph_step_cells):
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                img_morph = cv2.morphologyEx(
                    img_raw,
                    cv2.MORPH_CLOSE,
                    kernel,
                    iterations=int(morph_iterations),
                )

                test_contours, _, test_contour_yz, test_top_gap = _find_largest_contour(img_morph)
                if test_contours is None:
                    continue

                # Keep the best available result even if the threshold is not fully met.
                if test_top_gap < best_top_gap:
                    best_contours = test_contours
                    best_contour_yz = test_contour_yz
                    best_top_gap = test_top_gap
                    best_kernel_cells = k

                if test_top_gap <= contour_top_gap_threshold:
                    morph_used = True
                    morph_kernel_cells = k
                    contours = test_contours
                    contour_yz = test_contour_yz
                    top_gap = test_top_gap
                    break

            # If no kernel satisfies the threshold, still use the best improved result.
            if top_gap > contour_top_gap_threshold and best_kernel_cells > 0:
                morph_used = True
                morph_kernel_cells = best_kernel_cells
                contours = best_contours
                contour_yz = best_contour_yz
                top_gap = best_top_gap

        if self.show_progress:
            print(
                "    front-view contour: "
                f"top_gap={top_gap:.3f}, "
                f"threshold={contour_top_gap_threshold:.3f}, "
                f"morph_used={morph_used}, "
                f"kernel={morph_kernel_cells}",
                flush=True,
            )

        z_high = np.arange(zc_top, z_max, step)
        yl_min, yl_max, yr_min, yr_max = [], [], [], []

        if self.show_progress:
            print(f"    front-view contour extrema: {len(z_high)} layers", flush=True)

        for i, z in enumerate(z_high):
            ys = contour_yz[np.isclose(contour_yz[:, 1], z, atol=step / 2), 0]
            left_ys, right_ys = ys[ys <= y_center], ys[ys > y_center]

            if i == 0:
                yl0, yr0 = y_l0(zc_top), y_r0(zc_top)
                yl_min.append(yl0)
                yl_max.append(yl0)
                yr_min.append(yr0)
                yr_max.append(yr0)
            elif len(left_ys) == 0 or len(right_ys) == 0:
                yl_min.append(yl_min[-1])
                yl_max.append(yl_max[-1])
                yr_min.append(yr_min[-1])
                yr_max.append(yr_max[-1])
            else:
                yl_min.append(left_ys.min())
                yl_max.append(left_ys.max())
                yr_min.append(right_ys.min())
                yr_max.append(right_ys.max())

        if self.show_progress:
            print("    front-view contour extrema: done", flush=True)

        yl_min, yl_max = np.asarray(yl_min), np.asarray(yl_max)
        yr_min, yr_max = np.asarray(yr_min), np.asarray(yr_max)
        delta = yr_min - yl_max

        # debug_plot_front_view_points_and_contour(
        #     pts=pts,
        #     contours=contours,
        #     contour_yz=contour_yz,
        #     min_grid=min_grid,
        #     step=step,
        #     raster_margin=cfg.raster_margin,
        #     out_path="debug_front_view/Line6_front_view_contour.png",
        #     scale=10.0,
        # )

        if len(delta) < 2:
            return keep_all, {"warning": "insufficient contour difference samples"}

        delta_diff = np.diff(delta)
        jump_set = set(np.where(delta_diff >= cfg.jump_threshold)[0].tolist())
        recessed_idx = int(np.argmin(delta_diff))

        y_left_h, y_right_h = [], []
        for i in range(len(z_high)):
            if i in jump_set and i + 1 < len(yl_min):
                y_left_h.append(yl_min[i + 1])
                y_right_h.append(yr_max[i + 1])
            elif (i + 1) in jump_set and i + 2 < len(yl_min):
                y_left_h.append(yl_min[i + 2])
                y_right_h.append(yr_max[i + 2])
            elif i >= recessed_idx:
                y_left_h.append(yl_min[i])
                y_right_h.append(yr_max[i])
            else:
                y_left_h.append(yl_max[i])
                y_right_h.append(yr_min[i])

        z_low = np.arange(z_min, zc_top, step)
        z_grid = np.hstack([z_low, z_high])
        y_left = np.hstack([y_l0(z_low), np.asarray(y_left_h)])
        y_right = np.hstack([y_r0(z_low), np.asarray(y_right_h)])

        idx = np.searchsorted(z_grid, rotated[:, 2])
        idx = np.clip(idx, 0, len(z_grid) - 1)
        keep = (rotated[:, 1] >= y_left[idx] - cfg.tolerance) & (rotated[:, 1] <= y_right[idx] + cfg.tolerance)

        return keep, {
            "z_grid": z_grid,
            "left": y_left,
            "right": y_right,
            "delta": delta,
            "delta_diff": delta_diff,
            "jump_set": np.asarray(sorted(jump_set)),
            "recessed_idx": recessed_idx,
            "front_contour_top_gap": top_gap,
            "front_contour_top_gap_threshold": contour_top_gap_threshold,
            "front_contour_morph_used": morph_used,
            "front_contour_morph_kernel_cells": morph_kernel_cells,
        }


    @staticmethod
    def _align_mbr_corners(rectangles: np.ndarray) -> np.ndarray:
        """
        Align MBR corners across slices.

        Minimum rotated rectangles do not guarantee that the same corner index
        corresponds to the same physical tower corner across slices. This
        function searches over all 4! permutations at each slice and chooses the
        assignment with the minimum total distance to the previously aligned
        slice.
        """
        rectangles = np.asarray(rectangles, dtype=np.float64)
        if len(rectangles) == 0:
            return rectangles

        aligned = [rectangles[0]]
        perms = list(permutations(range(4)))

        for rect in rectangles[1:]:
            prev = aligned[-1]
            best_perm = perms[0]
            best_cost = np.inf
            for perm in perms:
                candidate = rect[list(perm)]
                cost = np.linalg.norm(candidate - prev, axis=1).sum()
                if cost < best_cost:
                    best_cost = cost
                    best_perm = perm
            aligned.append(rect[list(best_perm)])

        return np.asarray(aligned)

    def _base_filter(
        self,
        points: np.ndarray,
        current_mask: np.ndarray,
        central: CentralRegion,
        labels: Optional[np.ndarray] = None,
    ):
        """
        Base-constrained filtering with label-free transition-candidate selection.

        All transition heights detected from the four inter-axis center lines are
        retained. Transition heights within ``transition_cluster_gap`` are grouped,
        and the median of each group is used as a candidate base-top height.

        For each candidate base-top height, a base model is reconstructed and
        base filtering is performed. The final base-top height is selected by a
        label-free geometric score:

            S = s1 + lambda * s2

        where s1 is the relative transition height normalized by the tower height,
        and s2 is the number of retained points in a slab around the tower-foot
        plane normalized by the maximum slab count among all candidate heights.
        A smaller score indicates a better base-top candidate.
        """
        cfg = self.base_cfg
        default_keep = current_mask.copy()
        kept = points[current_mask].astype(np.float64, copy=False)

        if len(kept) < 10 or len(central.rectangles) < 2:
            return default_keep, {"warning": "insufficient points for base filtering"}

        rects = self._align_mbr_corners(central.rectangles)
        z_vals = central.z_values.astype(np.float64, copy=False)

        rect3 = np.array([
            np.column_stack([rect, np.full(4, z)])
            for rect, z in zip(rects, z_vals)
        ], dtype=np.float64)

        lines = [fit_3d_line_xz_yz(rect3[:, i, :]) for i in range(4)]

        z_min = kept[:, 2].min()
        z_max_kept = kept[:, 2].max()
        tower_height = max(float(z_max_kept - z_min), 1.0e-9)
        z_top = z_vals.max()
        z_grid = np.arange(z_min - cfg.lower_margin, z_top, cfg.search_step, dtype=np.float64)
        if len(z_grid) == 0:
            return default_keep, {"warning": "empty base z_grid"}

        axes = []
        for kx, bx, ky, by in lines:
            axes.append(np.column_stack([kx * z_grid + bx, ky * z_grid + by, z_grid]))
        axes = np.asarray(axes, dtype=np.float64)

        centers = np.asarray([(axes[i] + axes[(i + 1) % 4]) / 2.0 for i in range(4)], dtype=np.float64)

        tree = cKDTree(kept[:, :3])

        def dist_profile(samples: np.ndarray) -> np.ndarray:
            dists, _ = tree.query(samples[:, :3], k=1, workers=-1)
            return dists

        # ---------------------------------------------------------
        # Lower tower-foot vertices from axis contacts.
        # ---------------------------------------------------------
        lower_z = []
        for i in range(4):
            hits = np.where(dist_profile(axes[i]) <= cfg.axis_contact_threshold)[0]
            if len(hits):
                lower_z.append(float(z_grid[hits[0]] - cfg.lower_margin))
            else:
                lower_z.append(float(z_min))

        # ---------------------------------------------------------
        # Collect all transition heights instead of stopping at the
        # first gap on each inter-axis center line.
        # ---------------------------------------------------------
        transitions = []
        transition_sources = []
        gap_steps = max(1, int(round(cfg.center_contact_gap / cfg.search_step)))
        for i in range(4):
            hits = np.where(dist_profile(centers[i]) <= cfg.axis_contact_threshold)[0]
            for j in range(len(hits) - 1):
                if hits[j + 1] - hits[j] >= gap_steps:
                    transitions.append(float(z_grid[hits[j + 1]]))
                    transition_sources.append((i, int(hits[j]), int(hits[j + 1])))

        # ---------------------------------------------------------
        # Cluster transition heights within 1 m by default.
        # In integer-internal mode, transition_cluster_gap is already
        # scaled by make_internal_config().
        # ---------------------------------------------------------
        def cluster_transition_heights(values):
            if len(values) == 0:
                return [], []
            values = np.sort(np.asarray(values, dtype=np.float64))
            cluster_gap = float(getattr(cfg, "transition_cluster_gap", 10.0))
            clusters = []
            current = [values[0]]
            for v in values[1:]:
                if v - current[-1] <= cluster_gap:
                    current.append(v)
                else:
                    clusters.append(np.asarray(current, dtype=np.float64))
                    current = [v]
            clusters.append(np.asarray(current, dtype=np.float64))

            candidates = [float(np.median(c)) for c in clusters]
            return candidates, clusters

        candidate_base_tops, transition_clusters = cluster_transition_heights(transitions)

        if not candidate_base_tops:
            fallback = max(lower_z) + cfg.fallback_base_height
            candidate_base_tops = [float(fallback)]
            transition_clusters = [np.asarray([fallback], dtype=np.float64)]

        # Remove invalid or duplicate candidate heights.
        valid_candidates = []
        for zc in candidate_base_tops:
            if not np.isfinite(zc):
                continue
            zc = float(np.clip(zc, z_min, z_top))
            if not valid_candidates or all(abs(zc - p) > 1.0e-6 for p in valid_candidates):
                valid_candidates.append(zc)
        candidate_base_tops = valid_candidates or [
            float(np.clip(max(lower_z) + cfg.fallback_base_height, z_min, z_top))
        ]

        def build_base_geometry(base_top: float):
            lows, highs, mids = [], [], []
            for axis_id, (kx, bx, ky, by) in enumerate(lines):
                zl = lower_z[axis_id]
                lows.append(np.array([kx * zl + bx, ky * zl + by, zl], dtype=np.float64))
                highs.append(np.array([kx * base_top + bx, ky * base_top + by, base_top], dtype=np.float64))

            lows = np.asarray(lows, dtype=np.float64)
            highs = np.asarray(highs, dtype=np.float64)

            for axis_id in range(4):
                mids.append((highs[axis_id] + highs[(axis_id + 1) % 4]) / 2.0)
            mids = np.asarray(mids, dtype=np.float64)
            return lows, highs, mids

        def apply_base_filter_for_top(base_top: float):
            base_keep = current_mask.copy()
            lows, highs, mids = build_base_geometry(base_top)

            target = np.where(current_mask & (points[:, 2] <= base_top - cfg.lower_margin))[0]
            for point_idx in target:
                p = points[point_idx].astype(np.float64, copy=False)

                d0 = point_triangular_pyramid(lows[0], highs[0], mids[0], mids[3], p)
                if d0 <= cfg.distance_tolerance:
                    continue

                d1 = point_triangular_pyramid(lows[1], highs[1], mids[1], mids[0], p)
                if d1 <= cfg.distance_tolerance:
                    continue

                d2 = point_triangular_pyramid(lows[2], highs[2], mids[2], mids[1], p)
                if d2 <= cfg.distance_tolerance:
                    continue

                d3 = point_triangular_pyramid(lows[3], highs[3], mids[3], mids[2], p)
                if d3 <= cfg.distance_tolerance:
                    continue

                base_keep[point_idx] = False

            return base_keep, lows, highs, mids, target

        def fit_plane_from_points(plane_points: np.ndarray):
            """
            Fit a plane to four lower tower-foot vertices by SVD.

            Returns
            -------
            centroid, normal, valid
            """
            plane_points = np.asarray(plane_points, dtype=np.float64)
            centroid = plane_points.mean(axis=0)
            centered = plane_points - centroid

            if len(plane_points) < 3 or np.linalg.matrix_rank(centered) < 2:
                return centroid, np.array([0.0, 0.0, 1.0], dtype=np.float64), False

            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            normal = vh[-1]
            norm = np.linalg.norm(normal)
            if norm < 1.0e-12:
                return centroid, np.array([0.0, 0.0, 1.0], dtype=np.float64), False

            normal = normal / norm
            return centroid, normal, True

        def count_points_in_base_plane_slab(base_keep: np.ndarray, lows: np.ndarray):
            """
            Count retained points close to the lower-foot plane.

            ``base_plane_thickness`` is interpreted as the full slab thickness.
            Therefore, points with absolute plane distance <= thickness / 2 are counted.
            """
            final_mask = current_mask & base_keep
            candidate_points = points[final_mask].astype(np.float64, copy=False)

            if len(candidate_points) == 0:
                return 0, False, 0.0

            centroid, normal, valid = fit_plane_from_points(lows)
            thickness = float(getattr(cfg, "base_plane_thickness", 10.0))
            half_thickness = max(thickness / 2.0, 1.0e-9)

            signed_dist = np.abs((candidate_points[:, :3] - centroid) @ normal)
            slab_count = int(np.count_nonzero(signed_dist <= half_thickness))
            return slab_count, valid, half_thickness

        # ---------------------------------------------------------
        # Evaluate each candidate using a label-free composite criterion.
        #
        # s1 = relative transition height = (base_top - z_min) / tower_height
        # s2 = normalized slab count = slab_count / max_slab_count
        # S  = s1 + lambda * s2
        #
        # The final base height minimizes S. If tied, the lower base_top is used.
        # ---------------------------------------------------------
        candidate_records = []

        candidate_iter = candidate_base_tops
        if self.show_progress and len(candidate_base_tops) > 1:
            candidate_iter = tqdm(candidate_base_tops, desc="    base-top candidates", leave=False)

        for cand_id, base_top in enumerate(candidate_iter):
            base_keep, lows, highs, mids, target = apply_base_filter_for_top(float(base_top))
            final_mask = current_mask & base_keep

            slab_count, plane_valid, half_thickness = count_points_in_base_plane_slab(base_keep, lows)

            s1 = (float(base_top) - float(z_min)) / tower_height
            s1 = float(max(s1, 0.0))

            record = {
                "candidate_id": cand_id,
                "base_top": float(base_top),
                "s1_relative_height": s1,
                "s2_normalized_slab_count": np.nan,
                "base_top_score": np.nan,
                "base_top_score_lambda": float(getattr(cfg, "base_top_score_lambda", 1.0)),
                "base_plane_slab_count": int(slab_count),
                "base_plane_valid": bool(plane_valid),
                "base_plane_half_thickness": float(half_thickness),
                "tower_height": float(tower_height),
                "target_points": int(len(target)),
                "kept_points": int(final_mask.sum()),
                "removed_points": int(current_mask.sum() - final_mask.sum()),
                "_base_keep": base_keep,
                "_lows": lows,
                "_highs": highs,
                "_mids": mids,
                "_target": target,
            }
            candidate_records.append(record)

        max_slab_count = max((r["base_plane_slab_count"] for r in candidate_records), default=0)
        lambda_score = float(getattr(cfg, "base_top_score_lambda", 1.0))

        best = None
        for record in candidate_records:
            if max_slab_count > 0:
                s2 = float(record["base_plane_slab_count"]) / float(max_slab_count)
            else:
                s2 = 0.0
            score = float(record["s1_relative_height"]) + lambda_score * s2

            record["s2_normalized_slab_count"] = s2
            record["base_top_score"] = score

            key = (-score, -float(record["base_top"]))
            if best is None or key > best["key"]:
                best = {
                    "key": key,
                    "base_top": float(record["base_top"]),
                    "keep": record["_base_keep"],
                    "lows": record["_lows"],
                    "highs": record["_highs"],
                    "mids": record["_mids"],
                    "target": record["_target"],
                    "record": record,
                }

        keep = best["keep"]
        base_top = best["base_top"]
        lows = best["lows"]
        highs = best["highs"]
        mids = best["mids"]

        # Remove large arrays from records before returning debug info.
        clean_candidate_records = []
        for record in candidate_records:
            clean_record = {
                k: v for k, v in record.items()
                if not k.startswith("_")
            }
            clean_candidate_records.append(clean_record)

        if self.show_progress:
            print(
                f"    base_filter: selected base_top={base_top:.3f}, "
                f"score={best['record']['base_top_score']:.6f}, "
                f"s1={best['record']['s1_relative_height']:.6f}, "
                f"s2={best['record']['s2_normalized_slab_count']:.6f}, "
                f"lambda={lambda_score:.3f}, "
                f"candidates={len(clean_candidate_records)}",
                flush=True,
            )

        return keep, {
            "base_top": base_top,
            "lower_z": lower_z,
            "aligned_rectangles": rects,
            "axis_lines": lines,
            "low_vertices": lows,
            "high_vertices": highs,
            "mid_vertices": mids,
            "raw_transitions": np.asarray(transitions, dtype=np.float64),
            "transition_sources": transition_sources,
            "transition_clusters": transition_clusters,
            "candidate_base_tops": np.asarray(candidate_base_tops, dtype=np.float64),
            "candidate_records": clean_candidate_records,
            "selected_candidate_record": {k: v for k, v in best["record"].items() if not k.startswith("_")},
            "selection_metric": "s1_relative_height_plus_lambda_s2_normalized_slab_count",
            "base_top_score_lambda": lambda_score,
            "max_base_plane_slab_count": int(max_slab_count),
        }

