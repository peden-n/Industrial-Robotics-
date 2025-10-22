# planner.py
from __future__ import annotations
from typing import Iterable, List, Optional, Tuple
import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
from collision import AABB, first_collision_index
from controls import RunController
from pynput import keyboard
from itertools import zip_longest
donut_x_offset = -0.08
donut_y_offset = -0.08
donut_z_offset = 0.0
burnt_x_offset = -0.08
burnt_y_offset = -0.08
burnt_z_offset = 0.0

donut_offset = SE3(donut_x_offset, donut_y_offset, donut_z_offset)
burnt_offset = SE3(burnt_x_offset, burnt_y_offset, burnt_z_offset)

def solve_ik(robot, T_target,q_seed: Optional[np.ndarray] = None):
     """
     Try full pose IK first; if it fails, try position-only (ignore orientation).
     Returns a numpy array of joint angles or raises a ValueError.
     """
     q0 = np.asarray(q_seed, dtype=float) if q_seed is not None else np.asarray(robot.q, dtype=float)
     
    # print(T_target)
     # 1) full pose (position + orientation)
     try:
        sol = robot.ikine_LM(T_target, q0=q0)
        if sol.success:
            return np.array(sol.q, dtype=float)
     except Exception:
        pass

     # 2) position-only (mask: x,y,z true; roll/pitch/yaw false)
     try:
        mask = [1, 1, 1, 0, 0, 0]
        sol = robot.ikine_LM(T_target, q0=q0, mask=mask)
        if sol.success:
            return np.array(sol.q, dtype=float)
     except Exception:
        pass
     
     try: 
        mask = [1, 1, 1, 0, 0, 1]
        sol = robot.ikine_LM(T_target, q0=q0, mask=mask)
        if sol.success:
            return np.array(sol.q, dtype=float)
     except Exception:
        pass

     raise ValueError("IK did not converge for the requested target pose.")


def _stack(segments: List[np.ndarray]) -> np.ndarray:
    return np.vstack(segments) if segments else np.empty((0, 0))

def _jtraj(q_start: np.ndarray, q_goal: np.ndarray, steps: int) -> np.ndarray:
    return rtb.jtraj(q_start, q_goal, steps).q

def _top_z(obstacles: Iterable[AABB]) -> float:
    try:
        return max(b.max_xyz[2] for b in obstacles)
    except ValueError:
        return 0.0

def plan_safe_trajectory(
    robot,
    T_target: SE3,
    obstacles: List[AABB],
    *,
    link_radius: float = 0.03,
    steps_per_segment: int = 100,
    z_clear: float = 0.08,
    max_detours: int = 2,
) -> Optional[np.ndarray]:

    q_start = np.asarray(robot.q, dtype=float)
    q_goal = solve_ik(robot, T_target, q_seed=q_start)
    if q_goal is None:
        return None
    
    # Try direct
    Q0 = _jtraj(q_start, q_goal, steps_per_segment)
    if first_collision_index(robot, Q0, obstacles, link_radius) is None:
        return Q0

    # Lift over
    T_curr = robot.fkine(q_start)
    start_xy = (float(T_curr.t[0]), float(T_curr.t[1]))
    target_xy = (float(T_target.t[0]), float(T_target.t[1]))
    safe_z = max(_top_z(obstacles) + z_clear, float(T_curr.t[2]) + z_clear, float(T_target.t[2]) + z_clear)
    print(safe_z)

    T_up_start  = SE3(start_xy[0],  start_xy[1],  safe_z) * SE3.RPY(T_curr.rpy(), order="xyz")
    T_up_target = SE3(target_xy[0], target_xy[1], safe_z) * SE3.RPY(T_target.rpy(), order="xyz")

    q_up_start  = solve_ik(robot, T_up_start,  q_seed=q_start)
    q_up_target = solve_ik(robot, T_up_target, q_seed=q_up_start if q_up_start is not None else q_start)
    if q_up_start is None or q_up_target is None:
        return None

    Q1 = _jtraj(q_start,     q_up_start,  steps_per_segment)
    Q2 = _jtraj(q_up_start,  q_up_target, steps_per_segment)
    Q3 = _jtraj(q_up_target, q_goal,      steps_per_segment)
    Q = _stack([Q1, Q2, Q3])

    if first_collision_index(robot, Q, obstacles, link_radius) is None:
        return Q

    # Detours at safe_z if still blocked
    detours = [(0.25, 0.0), (-0.25, 0.0), (0.0, 0.25), (0.0, -0.25)]
    tries = 0
    for dx, dy in detours:
        if tries >= max_detours:
            break
        mid_x = 0.5 * (start_xy[0] + target_xy[0]) + dx
        mid_y = 0.5 * (start_xy[1] + target_xy[1]) + dy
        T_mid = SE3(mid_x, mid_y, safe_z) * SE3.RPY(T_target.rpy(), order="xyz")

        q_mid = solve_ik(robot, T_mid, q_seed=q_up_start)
        if q_mid is None:
            continue

        Q1 = _jtraj(q_start,    q_up_start,  steps_per_segment // 2)
        Q2 = _jtraj(q_up_start, q_mid,       steps_per_segment)
        Q3 = _jtraj(q_mid,      q_up_target, steps_per_segment)
        Q4 = _jtraj(q_up_target, q_goal,     steps_per_segment)
        Q_try = _stack([Q1, Q2, Q3, Q4])

        if first_collision_index(robot, Q_try, obstacles, link_radius) is None:
            return Q_try

        tries += 1

    return None

def on_press(key):
    if key == keyboard.Key.space:
        print('pressed space')

def move_robot_with_replanning(
    robotTargets,
    env,
    obstacles: List[AABB],
    *,
    link_radius: float = 0.03,
    steps_per_segment: int = 100,
    z_clear: float = 0.08,
    max_detours: int = 2,
    dt: float = 0.02
       ) -> bool:
    
    motions = []
    last_qs = []

    # Create the set of movements for the robot and the object that should follow it
    for r, T_target, follow_object in robotTargets:
       # q_start = np.asarray(r.q, dtype=float)
        #q_goal = solve_ik(r, T_target)
        Q = plan_safe_trajectory(
            r, T_target, obstacles,
            link_radius=link_radius,
            steps_per_segment=steps_per_segment,
            z_clear=z_clear,
            max_detours=max_detours,
               )
        if Q is None:
            print(" Could not find a collision-free plan.")
            return False
        motions.append((r, Q, follow_object))
        last_qs.append(Q[-1])

    # Execute the set of movements
    for frame in zip_longest(*[Q for (_, Q, _) in motions], fillvalue=None):
        for idx, (r, Q, follow_object) in enumerate(motions):
            q = frame[idx] if frame[idx] is not None else last_qs[idx]
            r.q = q

            # Check if there is an object to follow
            if follow_object is not None:

                T_ee = r.fkine(q)             # SE3 pose of end-effector
                T_follow_object = T_ee *donut_offset

                if hasattr(follow_object, "T"):
                    follow_object.T = T_follow_object
                elif hasattr(follow_object, "pose"):
                    follow_object.pose = T_follow_object*donut_offset
                else:
                # Fallback: try a generic attribute name
                    try:
                        setattr(follow_object, "pose", T_follow_object*donut_offset)
                    except Exception:
                        pass
        env.step(dt)
    
    return True

