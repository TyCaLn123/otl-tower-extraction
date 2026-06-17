from __future__ import annotations

from pathlib import Path
import numpy as np
import open3d as o3d


def colored_cloud(points: np.ndarray, color) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color), (len(points), 1)))
    return pcd


def draw_clouds(clouds):
    o3d.visualization.draw_geometries(clouds)
