from __future__ import annotations

from math import atan2, pi
from typing import Tuple
import numpy as np
from shapely.geometry import MultiPoint

try:
    from numba import njit
except Exception:  # pragma: no cover - fallback when numba is unavailable
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def decorator(func):
            return func
        return decorator


def rad_to_first_quadrant(theta: float) -> float:
    if -pi / 2 <= theta < 0:
        theta += pi / 2
    if -pi <= theta < -pi / 2:
        theta += pi
    if pi / 2 < theta <= pi:
        theta -= pi / 2
    return theta


def minimum_rotated_rectangle_xy(points_xy: np.ndarray) -> np.ndarray:
    geom = MultiPoint(points_xy).minimum_rotated_rectangle
    return np.asarray(geom.exterior.coords)[:4, :2]


def rectangle_side_lengths(rect: np.ndarray) -> Tuple[float, float]:
    return float(np.linalg.norm(rect[0] - rect[1])), float(np.linalg.norm(rect[1] - rect[2]))


def rectangle_angles_deg(rect: np.ndarray) -> np.ndarray:
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    angles = []
    for i, j in pairs:
        theta = atan2(rect[i, 1] - rect[j, 1], rect[i, 0] - rect[j, 0])
        angles.append(rad_to_first_quadrant(theta) * 180.0 / pi)
    return np.asarray(angles)


def fit_z_as_linear_of_x(x: np.ndarray, z: np.ndarray) -> Tuple[float, float]:
    a, b = np.polyfit(x, z, 1)
    return float(a), float(b)


def fit_3d_line_xz_yz(points: np.ndarray) -> Tuple[float, float, float, float]:
    z = points[:, 2]
    kx, bx = np.polyfit(z, points[:, 0], 1)
    ky, by = np.polyfit(z, points[:, 1], 1)
    return float(kx), float(bx), float(ky), float(by)


def inverse_linear(z, a: float, b: float):
    if abs(a) < 1e-12:
        raise ZeroDivisionError("Cannot invert z=a*x+b when a is close to zero.")
    return (z - b) / a


def rotate_xy(points: np.ndarray, theta: float) -> np.ndarray:
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    out = points.copy()
    out[:, :2] = out[:, :2] @ rot.T
    return out


@njit(cache=True)
def is_inside_triangle(Q, A, B, C):
    """
    Determine whether point Q lies inside triangle ABC in 3D space.
    This follows the logic of the original research script.
    """
    AB = B - A
    BC = C - B
    CA = A - C

    cross_AB_Q = np.cross(AB, Q - A)
    cross_BC_Q = np.cross(BC, Q - B)
    cross_CA_Q = np.cross(CA, Q - A)

    if (np.dot(cross_AB_Q, cross_BC_Q) >= 0.0 and
        np.dot(cross_BC_Q, cross_CA_Q) >= 0.0 and
        np.dot(cross_CA_Q, cross_AB_Q) >= 0.0):
        return True
    return False


@njit(cache=True)
def point_to_triangle_distance(P, Q, A, B, C):
    """
    Compute the distance from point P to triangle ABC.

    If the perpendicular foot Q lies inside the triangle, the point-to-plane
    distance is used. Otherwise, the minimum point-to-edge distance is used.
    This matches the original implementation.
    """
    if is_inside_triangle(Q, A, B, C):
        distance = np.linalg.norm(P - Q)
    else:
        AB = B - A
        BC = C - B
        CA = A - C
        distance_AB = np.linalg.norm(np.cross(AB, P - A)) / np.linalg.norm(AB)
        distance_BC = np.linalg.norm(np.cross(BC, P - B)) / np.linalg.norm(BC)
        distance_CA = np.linalg.norm(np.cross(CA, P - C)) / np.linalg.norm(CA)

        distance = distance_AB
        if distance_BC < distance:
            distance = distance_BC
        if distance_CA < distance:
            distance = distance_CA

    return distance


@njit(cache=True)
def point_triangular_pyramid(A, B, C, D, P):
    """
    Determine whether point P lies inside a triangular pyramid and return the
    shortest distance from P to the triangular-pyramid surface otherwise.

    Returns
    -------
    -1.0
        P is treated as inside the triangular pyramid.
    other float
        Approximate shortest distance from P to the triangular pyramid.
    """
    # Face normals.
    N1 = np.cross(B - A, C - A)
    N2 = np.cross(B - A, D - A)
    N3 = np.cross(C - A, D - A)
    N4 = np.cross(C - B, D - B)

    # Plane constants.
    D1 = -np.dot(N1, A)
    D2 = -np.dot(N2, A)
    D3 = -np.dot(N3, A)
    D4 = -np.dot(N4, B)

    # Perpendicular feet.
    denom1 = np.dot(N1, N1)
    denom2 = np.dot(N2, N2)
    denom3 = np.dot(N3, N3)
    denom4 = np.dot(N4, N4)

    # Degenerate protection.
    if denom1 == 0.0 or denom2 == 0.0 or denom3 == 0.0 or denom4 == 0.0:
        return 1.0e12

    t1 = (-D1 - np.dot(N1, P)) / denom1
    Q1 = P + t1 * N1

    t2 = (-D2 - np.dot(N2, P)) / denom2
    Q2 = P + t2 * N2

    t3 = (-D3 - np.dot(N3, P)) / denom3
    Q3 = P + t3 * N3

    t4 = (-D4 - np.dot(N4, P)) / denom4
    Q4 = P + t4 * N4

    flag1 = is_inside_triangle(Q1, A, B, C)
    flag2 = is_inside_triangle(Q2, A, B, D)
    flag3 = is_inside_triangle(Q3, A, C, D)
    flag4 = is_inside_triangle(Q4, B, C, D)

    if flag1 and flag2 and flag3 and flag4:
        return -1.0

    distance1 = point_to_triangle_distance(P, Q1, A, B, C)
    distance2 = point_to_triangle_distance(P, Q2, A, B, D)
    distance3 = point_to_triangle_distance(P, Q3, A, C, D)
    distance4 = point_to_triangle_distance(P, Q4, B, C, D)

    distance = distance1
    if distance2 < distance:
        distance = distance2
    if distance3 < distance:
        distance = distance3
    if distance4 < distance:
        distance = distance4

    return distance


# Backward-compatible alias. The previous refactored version used this name,
# but the actual logic is the original triangular-pyramid distance.
point_to_tetra_distance = point_triangular_pyramid
