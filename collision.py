# collision.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
import numpy as np
from spatialmath import SE3

Z_OFFSET_DOWN = 0.1

@dataclass(frozen=True)
class AABB:
    min_xyz: Tuple[float, float, float]
    max_xyz: Tuple[float, float, float]

    @staticmethod
    def from_center_size(center: Tuple[float, float, float],
                         size: Tuple[float, float, float]) -> "AABB":
        cx, cy, cz = center
        sx, sy, sz = size
        half = np.array([sx, sy, sz], dtype=float) * 0.5
        mn = (cx - half[0], cy - half[1], cz - half[2])
        mx = (cx + half[0], cy + half[1], cz + half[2])
        return AABB(mn, mx)

def inflate_aabb(box: AABB, margin: float) -> AABB:
    mn = np.array(box.min_xyz) - margin
    mx = np.array(box.max_xyz) + margin
    mx[2] -= Z_OFFSET_DOWN
    return AABB(tuple(mn), tuple(mx))

def _point_in_aabb(p: np.ndarray, box: AABB) -> bool:
    return np.all(p >= np.array(box.min_xyz)) and np.all(p <= np.array(box.max_xyz))

def _segment_aabb_intersect(p0: np.ndarray, p1: np.ndarray, box: AABB) -> bool:
    if _point_in_aabb(p0, box) or _point_in_aabb(p1, box):
        return True
    d = p1 - p0
    tmin, tmax = 0.0, 1.0
    for i in range(3):
        if abs(d[i]) < 1e-12:
            if p0[i] < box.min_xyz[i] or p0[i] > box.max_xyz[i]:
                return False
        else:
            inv_d = 1.0 / d[i]
            t1 = (box.min_xyz[i] - p0[i]) * inv_d
            t2 = (box.max_xyz[i] - p0[i]) * inv_d
            t_enter, t_exit = min(t1, t2), max(t1, t2)
            tmin, tmax = max(tmin, t_enter), min(tmax, t_exit)
            if tmin > tmax:
                return False
    return True

def _link_origins_world(robot, q) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = len(robot.links)
    origins = np.zeros((n + 1, 3), dtype=float)
    origins[0, :] = (robot.base if hasattr(robot, "base") else SE3()).t
    for i in range(n):
        T_i = robot.fkine(q, end=i + 1)
        origins[i + 1, :] = T_i.t
    return origins

def _robot_segments(robot, q):
    O = _link_origins_world(robot, q)
    return [(O[i, :], O[i + 1, :]) for i in range(O.shape[0] - 1)]

def config_in_collision(robot, q, obstacles: Iterable[AABB], link_radius: float = 0.03) -> bool:
    segs = _robot_segments(robot, q)
    boxes = [inflate_aabb(b, link_radius) for b in obstacles] if link_radius > 0 else list(obstacles)
    for p0, p1 in segs:
        for box in boxes:
            if _segment_aabb_intersect(p0, p1, box):
                return True
    return False

def first_collision_index(robot, q_matrix: np.ndarray, obstacles: Iterable[AABB], link_radius: float = 0.03) -> Optional[int]:
    for i, q in enumerate(np.asarray(q_matrix)):
        if config_in_collision(robot, q, obstacles, link_radius=link_radius):
            return i
    return None
