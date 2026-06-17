from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np


@dataclass
class PointCloudData:
    points: np.ndarray
    labels: Optional[np.ndarray] = None


def load_point_cloud_txt(
    path: str | Path,
    label_column: int = 3,
    integer_internal: bool = True,
    scale: int = 10,
) -> PointCloudData:
    """
    Load a labelled point cloud from a space-separated TXT file.

    Input row format:
        x y z label

    If integer_internal is true, x/y/z are multiplied by `scale`, rounded,
    and stored as np.int32. For 0.1 m quantized data, scale=10 is recommended.
    """
    path = Path(path)
    arr = np.loadtxt(path, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 4:
        raise ValueError(
            f"TXT input must contain at least four columns: x y z label. "
            f"Got shape {arr.shape} from {path}."
        )

    xyz = arr[:, :3]
    if integer_internal:
        points = np.rint(xyz * scale).astype(np.int32)
    else:
        points = xyz.astype(np.float32)

    labels = arr[:, label_column].astype(np.int8)
    return PointCloudData(points=points, labels=labels)


def save_point_cloud_txt(
    path: str | Path,
    points: np.ndarray,
    labels: Optional[np.ndarray],
    integer_internal: bool = True,
    scale: int = 10,
) -> None:
    """
    Save a labelled point cloud as a space-separated TXT file.

    Output row format:
        %.1f %.1f %.1f %d

    If integer_internal is true, points are divided by `scale` before saving.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    points = np.asarray(points)
    if labels is None:
        labels = np.ones(len(points), dtype=np.int8)
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)

    if len(points) != len(labels):
        raise ValueError("points and labels must have the same length before saving.")

    if len(points) == 0:
        path.write_text("", encoding="utf-8")
        return

    if integer_internal:
        out_points = (points[:, :3].astype(np.float32) / float(scale)).astype(np.float32)
    else:
        out_points = points[:, :3].astype(np.float32)

    arr = np.column_stack([out_points, labels.astype(np.int32)])
    np.savetxt(path, arr, fmt="%.1f %.1f %.1f %d")


def load_point_cloud(
    path: str | Path,
    label_column: int = 3,
    integer_internal: bool = True,
    scale: int = 10,
) -> PointCloudData:
    return load_point_cloud_txt(
        path,
        label_column=label_column,
        integer_internal=integer_internal,
        scale=scale,
    )


def save_point_cloud(
    path: str | Path,
    points: np.ndarray,
    labels: Optional[np.ndarray],
    integer_internal: bool = True,
    scale: int = 10,
) -> None:
    save_point_cloud_txt(
        path,
        points,
        labels,
        integer_internal=integer_internal,
        scale=scale,
    )
