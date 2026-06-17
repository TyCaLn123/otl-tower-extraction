from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import open3d as o3d
from tqdm import tqdm


def _aggregate_binary_labels_fast(
    sorted_labels: np.ndarray,
    start_idx: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """
    Fast majority voting for binary labels {0, 1}.
    """
    label_sum = np.add.reduceat(
        sorted_labels.astype(np.int64, copy=False),
        start_idx
    )
    return (label_sum * 2 >= counts).astype(sorted_labels.dtype, copy=False)


def _aggregate_labels(labels: np.ndarray, index_groups, method: str = "majority") -> np.ndarray:
    labels = np.asarray(labels)
    out = []
    for group in index_groups:
        idx = np.asarray(group, dtype=np.int64)
        idx = idx[idx >= 0]
        if len(idx) == 0:
            out.append(0)
            continue
        group_labels = labels[idx]
        if method == "first":
            out.append(group_labels[0])
        elif method == "majority":
            values, counts = np.unique(group_labels, return_counts=True)
            out.append(values[np.argmax(counts)])
        else:
            raise ValueError(f"Unsupported label aggregation method: {method}")
    return np.asarray(out, dtype=labels.dtype)


def voxel_downsample(
    points: np.ndarray,
    voxel_size: float,
    labels: Optional[np.ndarray] = None,
    show_progress: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Fast voxel downsampling for large-scale integer point clouds.

    Internal coordinates are assumed to already be scaled by ×10
    and stored as integers.
    """

    points = np.asarray(points)

    if voxel_size <= 0:
        return (
            points.copy(),
            None if labels is None else np.asarray(labels).copy(),
        )

    if len(points) == 0:
        return (
            points.copy(),
            None if labels is None else np.asarray(labels).copy(),
        )

    if show_progress:
        print(
            f"  - voxel_downsample: input={len(points):,}, voxel_size={voxel_size / 10}",
            flush=True,
        )

    voxel_size_int = max(1, int(round(voxel_size)))

    # ---------------------------------------------------------
    # Step 1: voxel indexing
    # ---------------------------------------------------------
    xyz_min = points[:, :3].min(axis=0)

    voxel_indices = ((points[:, :3] - xyz_min) // voxel_size_int).astype(np.int64)

    # Encode 3D voxel index into 1D key
    vx = voxel_indices[:, 0]
    vy = voxel_indices[:, 1]
    vz = voxel_indices[:, 2]

    vy_span = int(vy.max() - vy.min() + 1)
    vz_span = int(vz.max() - vz.min() + 1)

    keys = (
        (vx - vx.min()) * vy_span * vz_span
        + (vy - vy.min()) * vz_span
        + (vz - vz.min())
    )

    # ---------------------------------------------------------
    # Step 2: sort by voxel
    # ---------------------------------------------------------
    order = np.argsort(keys, kind="mergesort")

    sorted_keys = keys[order]
    sorted_points = points[order]

    if labels is not None:
        labels = np.asarray(labels)
        sorted_labels = labels[order]

    # ---------------------------------------------------------
    # Step 3: voxel boundaries
    # ---------------------------------------------------------
    unique_keys, start_idx, counts = np.unique(
        sorted_keys,
        return_index=True,
        return_counts=True,
    )

    # ---------------------------------------------------------
    # Step 4: point aggregation (mean)
    # ---------------------------------------------------------
    down_xyz = np.vstack([
        np.add.reduceat(sorted_points[:, 0], start_idx),
        np.add.reduceat(sorted_points[:, 1], start_idx),
        np.add.reduceat(sorted_points[:, 2], start_idx),
    ]).T

    down_xyz = np.rint(down_xyz / counts[:, None]).astype(np.int32)

    # ---------------------------------------------------------
    # Step 5: label aggregation
    # ---------------------------------------------------------
    down_labels = None

    if labels is not None:
        if show_progress:
            print(
                f"  - label aggregation: {len(unique_keys):,} voxels",
                flush=True,
            )

        if sorted_labels.min() >= 0 and sorted_labels.max() <= 1:
            down_labels = _aggregate_binary_labels_fast(
                sorted_labels=sorted_labels,
                start_idx=start_idx,
                counts=counts,
            )
        else:
            raise ValueError(
                "The fast label aggregation path assumes binary labels {0, 1}."
            )
    if show_progress:
        print(
            f"  - voxel_downsample: output={len(down_xyz):,}",
            flush=True,
        )
    return down_xyz, down_labels


def statistical_outlier_removal(
    points: np.ndarray,
    nb_neighbors: int,
    std_ratio: float,
    labels: Optional[np.ndarray] = None,
    show_progress: bool = True,
    open3d_progress: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    points = np.asarray(points)
    if len(points) == 0:
        empty_idx = np.asarray([], dtype=np.int64)
        return points.copy(), None if labels is None else np.asarray(labels).copy(), empty_idx

    if show_progress:
        print(f"  - statistical_outlier_removal: input={len(points):,}, nb_neighbors={nb_neighbors}, std_ratio={std_ratio}", flush=True)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64, copy=False))
    filtered, keep_indices = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
        print_progress=open3d_progress and show_progress,
    )
    keep_indices = np.asarray(keep_indices, dtype=np.int64)
    filtered_points = points[keep_indices]

    if labels is None:
        filtered_labels = None
    else:
        labels = np.asarray(labels)
        if len(labels) != len(points):
            raise ValueError("labels and points must have the same length.")
        filtered_labels = labels[keep_indices]

    if show_progress:
        print(f"  - statistical_outlier_removal: output={len(filtered_points):,}", flush=True)
    return filtered_points, filtered_labels, keep_indices


def grid_near_ground_suppression(
    points: np.ndarray,
    grid_size: float,
    height_threshold: float,
    labels: Optional[np.ndarray] = None,
    show_progress: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Suppress near-ground points using local minimum elevation in horizontal grids.

    This implementation is optimized for integer internal coordinates.
    """
    points = np.asarray(points)
    if len(points) == 0:
        keep = np.zeros(0, dtype=bool)
        return points.copy(), None if labels is None else np.asarray(labels).copy(), keep

    if labels is not None:
        labels = np.asarray(labels)
        if len(labels) != len(points):
            raise ValueError("labels and points must have the same length.")

    if show_progress:
        print(f"  - grid_near_ground_suppression: input={len(points):,}, grid_size={grid_size}, threshold={height_threshold}", flush=True)

    grid_size_int = max(1, int(round(grid_size)))
    height_threshold_val = height_threshold

    xy_min = points[:, :2].min(axis=0)
    grid = ((points[:, :2] - xy_min) // grid_size_int).astype(np.int64)

    # Encode 2D grid index into one 1D key for faster grouping.
    gx = grid[:, 0]
    gy = grid[:, 1]
    gy_span = int(gy.max() - gy.min() + 1)
    keys = (gx - gx.min()) * gy_span + (gy - gy.min())

    order = np.argsort(keys, kind="mergesort")
    sorted_keys = keys[order]
    sorted_z = points[order, 2]

    unique_keys, start = np.unique(sorted_keys, return_index=True)
    min_z_sorted = np.minimum.reduceat(sorted_z, start)

    # Map each point to its cell minimum.
    inverse = np.searchsorted(unique_keys, keys)
    local_min_z = min_z_sorted[inverse]

    keep = (points[:, 2] - local_min_z) > height_threshold_val
    filtered_labels = None if labels is None else labels[keep]

    if show_progress:
        print(f"  - grid_near_ground_suppression: output={int(keep.sum()):,}", flush=True)
    return points[keep], filtered_labels, keep
