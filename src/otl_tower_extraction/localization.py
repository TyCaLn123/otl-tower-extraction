from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import open3d as o3d
from tqdm import tqdm

@dataclass
class CandidateTower:
    points: np.ndarray
    labels: Optional[np.ndarray]
    source_indices: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    cluster_id: int

def cluster_dbscan(points: np.ndarray, eps: float, min_points: int, show_progress: bool = True, open3d_progress: bool = True) -> np.ndarray:
    if show_progress: print(f'  - DBSCAN: input={len(points):,}, eps={eps/10}, min_points={min_points}', flush=True)
    pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64, copy=False))
    labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=show_progress and open3d_progress))
    if show_progress:
        n_cluster = int(labels.max()+1) if labels.size and labels.max() >= 0 else 0
        print(f'  - DBSCAN: clusters={n_cluster}', flush=True)
    return labels

def localize_candidate_towers(original_points: np.ndarray, dbscan_points: np.ndarray, eps: float, min_points: int, min_cluster_height: float, bbox_expand_xy: float, original_labels: Optional[np.ndarray] = None, original_indices: Optional[np.ndarray] = None, show_progress: bool = True, open3d_progress: bool = True) -> List[CandidateTower]:
    labels = cluster_dbscan(dbscan_points, eps, min_points, show_progress=show_progress, open3d_progress=open3d_progress)
    candidates: List[CandidateTower] = []
    if labels.size == 0 or labels.max() < 0: return candidates
    if original_indices is None: original_indices = np.arange(len(original_points), dtype=np.int64)
    if original_labels is not None and len(original_labels) != len(original_points): raise ValueError('original_labels and original_points must have same length.')
    z_min, z_max = original_points[:, 2].min(), original_points[:, 2].max()
    iterator = range(labels.max()+1)
    if show_progress: iterator = tqdm(iterator, desc='  candidate bbox recovery', leave=False)
    for cid in iterator:
        cluster = dbscan_points[labels == cid]
        if len(cluster) == 0: continue
        extent = cluster.max(axis=0) - cluster.min(axis=0)
        if extent[2] < min_cluster_height: continue
        bbox_min = cluster.min(axis=0); bbox_max = cluster.max(axis=0)
        bbox_min = np.array([bbox_min[0]-bbox_expand_xy, bbox_min[1]-bbox_expand_xy, z_min])
        bbox_max = np.array([bbox_max[0]+bbox_expand_xy, bbox_max[1]+bbox_expand_xy, z_max])
        mask = ((original_points[:,0]>=bbox_min[0])&(original_points[:,0]<=bbox_max[0])&(original_points[:,1]>=bbox_min[1])&(original_points[:,1]<=bbox_max[1]))
        cand_labels = None if original_labels is None else original_labels[mask]
        candidates.append(CandidateTower(original_points[mask], cand_labels, original_indices[mask], bbox_min, bbox_max, cid))
    if show_progress: print(f'  - candidate localization: candidates={len(candidates)}', flush=True)
    return candidates

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

