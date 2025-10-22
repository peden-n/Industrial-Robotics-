
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable
import roboticstoolbox as rtb
from math import pi
from spatialmath import SE3
from spatialgeometry import Cuboid, Cylinder, Mesh
from roboticstoolbox.backends.swift import Swift
from itertools import zip_longest
import numpy as np
from roboticstoolbox import DHLink, DHRobot
from ir_support import CylindricalDHRobotPlot
from scenenatsversion import Environment
from ir_support.robots.UR3 import UR3  # package from your link
from itertools import zip_longest
from collision import AABB, first_collision_index, config_in_collision
from robot_gui import SimpleRobotGUI
from utilities.stl_manager import STLManager
import os,json 
import threading

from planner import move_robot_with_replanning
#from roboticstoolbox.models.DH import UR3


Color = Tuple[float, float, float, float]  # RGBA 0..1



#-----------------------planner ------------------------------------------
def plan_safe_trajectory(
    robot,
    T_target: SE3,
    obstacles: List[AABB],
    *,
    link_radius: float = 0.03,
    steps_per_segment: int = 80,
    z_clear_above_obstacles: float = 0.15,
    max_replans: int = 2,
 ) -> Optional[np.ndarray]:
    """
    Try direct path; if collision predicted, insert via-points to 'lift over' obstacles.
    Returns a collision-free joint path (NxDOF) or None if planning fails.
    """

    q_start = np.asarray(robot.q, dtype=float)

    # 0) Try direct path (start -> target)
    q_goal = _solve_ik(robot, T_target, q_seed=q_start)
    if q_goal is None:
        return None

    Q_direct = jtraj_q(robot, q_start, q_goal, steps_per_segment)
    idx = first_collision_index(robot, Q_direct, obstacles, link_radius=link_radius)
    if idx is None:
        return Q_direct  # ✅ direct path is clear
    
    print("Q")
    # this is the first retry (up and over) 
    T_curr = robot.fkine(q_start)
    start_xy = T_curr.t[0], T_curr.t[1]
    target_xy = T_target.t[0], T_target.t[1]

    safe_z = max(
        highest_obstacle_top(obstacles) + z_clear_above_obstacles,
        float(T_curr.t[2]) + z_clear_above_obstacles,
        float(T_target.t[2]) + z_clear_above_obstacles,
    )

    # Build via poses:
    T_up_start  = SE3(start_xy[0],  start_xy[1],  safe_z) * SE3.RPY(T_curr.rpy(), order="xyz")
    T_up_target = SE3(target_xy[0], target_xy[1], safe_z) * SE3.RPY(T_target.rpy(), order="xyz")

    # Plan segments: start -> up_start -> up_target -> target
    q_up_start  = _solve_ik(robot, T_up_start,  q_seed=q_start)
    q_up_target = _solve_ik(robot, T_up_target, q_seed=q_up_start if q_up_start is not None else q_start)
    if q_up_start is None or q_up_target is None:
        # If we can't even solve the lift, bail out early
        return None

    Q1 = jtraj_q(robot, q_start,     q_up_start,  steps_per_segment)
    Q2 = jtraj_q(robot, q_up_start,  q_up_target, steps_per_segment)
    Q3 = jtraj_q(robot, q_up_target, q_goal,      steps_per_segment)
    Q_combo = stack_traj([Q1, Q2, Q3])

    idx2 = first_collision_index(robot, Q_combo, obstacles, link_radius=link_radius)
    if idx2 is None:
        return Q_combo  #  lift-over worked

    # 2nd try ----sideways
    lateral_offsets = [(0.25, 0.0), (-0.25, 0.0), (0.0, 0.25), (0.0, -0.25), (0.35, 0.0), (-0.35, 0.0)]

    attempts = 0
    for dx, dy in lateral_offsets:
        if attempts >= max_replans:
            break

        T_mid = SE3(0, 0, 0)  # dummy init
        # Place mid waypoint halfway between start/target XY, then offset
        mid_x = 0.5 * (start_xy[0] + target_xy[0]) + dx
        mid_y = 0.5 * (start_xy[1] + target_xy[1]) + dy
        T_mid = SE3(mid_x, mid_y, safe_z) * SE3.RPY(T_target.rpy(), order="xyz")

        q_mid = solve_ik(robot, T_mid, q_seed=q_up_start)
        if q_mid is None:
            continue

        Q1 = jtraj_q(robot, q_start,    q_up_start,  steps_per_segment // 2)
        Q2 = jtraj_q(robot, q_up_start, q_mid,       steps_per_segment)
        Q3 = jtraj_q(robot, q_mid,      q_up_target, steps_per_segment)
        Q4 = jtraj_q(robot, q_up_target, q_goal,     steps_per_segment)
        Q_try = stack_traj([Q1, Q2, Q3, Q4])

        idx_try = first_collision_index(robot, Q_try, obstacles, link_radius=link_radius)
        if idx_try is None:
            return Q_try  # found a clear detour

        attempts += 1

    # If we got here, all attempts collided
    return None

# ----------------- run -----------------



#-- jtraj stuff ---------------------------------------------------------

def stack_traj(segments: List[np.ndarray]) -> np.ndarray:
    """Stack a list of (N_i x dof) arrays into one (sum N_i x dof)."""
    return np.vstack(segments) if segments else np.empty((0, 0))

def jtraj_q(robot, q_start: np.ndarray, q_goal: np.ndarray, steps: int) -> np.ndarray:
    """Joint-space quintic trajectory between two joint vectors."""
    return rtb.jtraj(q_start, q_goal, steps).q

def highest_obstacle_top(obstacles: Iterable[AABB]) -> float:
    """Max Z of all obstacles (top surface)."""
    try:
        return max(box.max_xyz[2] for box in obstacles)
    except ValueError:
        return 0.0  # no obstacles
    
def create_GP7():
    l1 = DHLink(d=0.33, a=0.4, alpha=pi/2, qlim=[-pi, pi])
    l2 = DHLink(d=0, a=0.445, alpha=0, qlim=[-pi, pi])
    l3 = DHLink(d=0, a=0.04, alpha=pi/2, qlim=[-pi, pi])
    l4 = DHLink(d=0.44, a=0, alpha=-pi/2, qlim=[-pi, pi])
    l5 = DHLink(d=0.08, a=0, alpha=pi/2, qlim=[-pi, pi])
    l6 = DHLink(d=0, a=0, alpha=0, qlim=[-pi, pi])
    r2 = DHRobot([l1, l2, l3,l4, l5, l6], name='GP7')
    cyl_viz = CylindricalDHRobotPlot(r2, cylinder_radius=0.03, color="#3478f6")
    r2 = cyl_viz.create_cylinders()
    r2.base = r2.base * SE3(-0.1,-1,0.35)
    return r2

def create_praybot():
    link1 = DHLink(d=0.445, a=0.1404, alpha=np.pi / 2, qlim=np.deg2rad([-180, 180]), offset=0)
    link2 = DHLink(d=0.17, a=0.7, alpha=0.0, qlim=np.deg2rad([-155, 95]), offset=np.pi / 2)
    link3 = DHLink(d=-0.17, a=0.115, alpha=np.pi / 2, qlim=np.deg2rad([-75, 180]))
    link4 = DHLink(d=0.8, a=0.0, alpha=np.pi / 2, qlim=np.deg2rad([-400, 400]))
    link5 = DHLink(d=0.0, a=0.0, alpha=-np.pi / 2, qlim=np.deg2rad([-120, 120]))
    link6 = DHLink(d=0.0, a=0.0, alpha=0.0, qlim=np.deg2rad([-400, 400]))

    robot = DHRobot([link1, link2, link3, link4, link5, link6], name="myRobot")
    cyl_viz = CylindricalDHRobotPlot(robot, cylinder_radius=0.03, color="#3478f6")
    robot = cyl_viz.create_cylinders()
    robot = cyl_viz.create_cylinders()
    robot.base = SE3(0.0, 1.18, 0.02)
    return robot







def add_objects(env):
    # Create STL manager for objects
    manager = STLManager(env)
    
    # Get directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")
    configs_dir = os.path.join(script_dir, "configurations")
    
    # Load all parts from their config files
    print("\n Loading objects from JSON configurations...")
    
    # Load different object types
    objects = {}
    objects['boxes'] = load_parts_from_config(manager, models_dir, configs_dir, 
                                             "box_config.json", "BoxBase.stl")
    objects['lids'] = load_parts_from_config(manager, models_dir, configs_dir,
                                            "lid_config.json", "BoxLid.stl")
    objects['donuts'] = load_parts_from_config(manager, models_dir, configs_dir,
                                              "donut_config.json", "Donut.dae")
    objects['burnt_donuts'] = load_parts_from_config(manager, models_dir, configs_dir,
                                                    "burntDonut_config.json", "Donut_burnt.dae")
    
    # Load configuration data for reference
    configs = {}
    config_files = ["box_config.json", "lid_config.json", "donut_config.json", "burntDonut_config.json"]
    
    for config_file in config_files:
        config_path = os.path.join(configs_dir, config_file)
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_name = config_file.replace('_config.json', '').replace('.json', '')
                configs[config_name] = json.load(f)

    # Print summary
    total_objects = sum(len(obj_list) for obj_list in objects.values())
    print(f"\n✓ Scene initialization complete!")
    print(f"  - Loaded objects: {total_objects} total")
    for obj_type, obj_list in objects.items():
        if obj_list:
            print(f"    • {len(obj_list)} {obj_type}")
    
    # Display initial scene for a moment
    print("\nDisplaying initial scene...")
    for i in range(30):
        env.step(0.05)
 
    return {
        'env': env,
        'manager': manager,
        'objects': objects,
        'configs': configs
    }

def load_parts_from_config(manager, models_dir, configs_dir, config_filename, model_filename):
    """
    Load multiple parts from a JSON config file at their 'start' positions.
    
    Args:
        manager: STLManager instance
        models_dir: Directory containing model files
        configs_dir: Directory containing config JSON files
        config_filename: Name of the JSON config file (e.g., 'box_config.json')
        model_filename: Name of the model file (e.g., 'BoxBase.stl')
    
    Returns:
        List of loaded part objects
    """
    config_path = os.path.join(configs_dir, config_filename)
    model_path = os.path.join(models_dir, model_filename)
    
    if not os.path.exists(config_path):
        print(f"✗ Config file not found: {config_path}")
        return []
    
    if not os.path.exists(model_path):
        print(f"✗ Model file not found: {model_path}")
        return []
    
    with open(config_path, 'r') as f:
        configs = json.load(f)
    
    parts = []
    part_type = config_filename.replace('_config.json', '').replace('.json', '')
    
    for i, config in enumerate(configs):
        name = f"{part_type}_{i+1}"
        position = config.get('start', [0, 0, 0])
        rpy = config.get('start_rpy', [0, 0, 0])
        color = config.get('color', [0.7, 0.7, 0.7, 1.0])
        
        part = manager.add(
            model_path,
            name=name,
            position=position,
            scale=0.001,
            color=tuple(color)
        )
        
        if part:
            part.rpy = rpy
            parts.append(part)
            print(f"  ✓ Loaded {name} at {position}")
    
    return parts

def describe_aabb(label, aabb):
    mn = np.array(aabb.min_xyz); mx = np.array(aabb.max_xyz)
    print(f"{label}: X[{mn[0]:.3f},{mx[0]:.3f}]  Y[{mn[1]:.3f},{mx[1]:.3f}]  Z[{mn[2]:.3f},{mx[2]:.3f}]")
    print("----------------------------------------------------")


def _dbg_obstacles(obstacles):
    print("\n[DEBUG] Obstacles:")
    for i, b in enumerate(obstacles):
        mn = tuple(round(x, 3) for x in b.min_xyz)
        mx = tuple(round(x, 3) for x in b.max_xyz)
        print(f"  [{i}] min={mn} max={mx} top_z={mx[2]:.3f}")
    print(f"  -> highest_obstacle_top = {highest_obstacle_top(obstacles):.3f}")


def run_all_moves(env, gui, r, r2, r3, scene_data):

    donuts = scene_data["objects"].get('donuts',[])
    burnt_donuts = scene_data["objects"].get('burnt_donuts',[])
    lids = scene_data['objects'].get('lids', [])
    boxes = scene_data['objects'].get('boxes', [])
    box_configs = scene_data['configs'].get('box', [])
    donut1 = donuts[6]
    dount2 = donuts[7]
    burnt_donut1 = burnt_donuts[0]
    burnt_donut2 = burnt_donuts[2]
    lid1 =lids[0]



    GRASP_FROM_TOP = SE3.Rx(pi) * SE3(0, 0, -0.002) 

    obstacles = [
        # Table top (center, size) -> width x depth x thickness
        AABB.from_center_size(center=(0.0, 0.25, 0.2), size=(1, 0.3, 0.2)),
         # Table top (center, size) -> width x depth x thickness
        AABB.from_center_size(center=(0, -0.5, 0.225), size=(1, 0.40, 0.35)),
        # other table 
        AABB.from_center_size(center=(0.65, 1.48, 0.15), size =(1, 0.40, 0.4)),
        # A no-go pillar
        AABB.from_center_size(center=(0, 0, 0.16), size=(0.2, 0.2, 0.05)),
         # A no-go pillar
        AABB.from_center_size(center=(0, -0.25, 0.16), size=(0.2, 0.2, 0.05)),
        # floor 
        AABB.from_center_size(center=(0,0,0), size=(3,3,0.01)),
        #box 
       # AABB.from_center_size(center = (0,0.5,0.29), size = (1,0.01, 0.4))
     ]

    _dbg_obstacles(obstacles)
    
    # Your original definitions (order matters for colors)
    '''
    aabbs = [
    AABB.from_center_size(center=(0.0,  0.25, 0.20), size=(1.0, 0.48, 0.10)),  # table top 1
    AABB.from_center_size(center=(0.0, -0.25, 0.16), size=(1.0, 0.48, 0.48)),  # table top 2 / block
    AABB.from_center_size(center=(0.0,  0.00, 0.16), size=(0.20, 0.20, 0.05)), # pillar 1
    AABB.from_center_size(center=(0.0, -0.25, 0.16), size=(0.20, 0.20, 0.05)), # pillar 2
    # Floor: if your ground is z=0 and you want it visible above, put center z=thickness/2.
    AABB.from_center_size(center=(0.0, 0.0, 0.005), size=(3.0, 3.0, 0.01)),    # floor
    ]
    '''
    # Then:
    
    # Movement set 1
    move_robot_with_replanning(
       [[r,SE3(-0.135, -0.28, 0.48) * GRASP_FROM_TOP, None],
        [r2,SE3(-0.495, -0.58, 0.48)* GRASP_FROM_TOP, None]],
        env, gui, 
        obstacles,
        link_radius=0.03,
        steps_per_segment=90,
        z_clear=0.08,
        max_detours=3,
        dt=0.02
    )

    
    # Movement set 2
    move_robot_with_replanning(
        [[r,SE3(0.100, 0.50, 0.48) * GRASP_FROM_TOP, donut1],
        [r2,SE3(0.045, -0.58, 0.48)* GRASP_FROM_TOP, burnt_donut1]],
        env, gui, obstacles,
        link_radius=0.03,
        steps_per_segment=90,
        z_clear=0.08, 
        max_detours=3,
        dt=0.02
    )


    # Movement set 3
    move_robot_with_replanning(
        [[r,SE3(-0.045, -0.205, 0.4) * GRASP_FROM_TOP, None],
        [r2,SE3(-0.5, -0.3,1.0) * GRASP_FROM_TOP , None]],
        env, gui, obstacles,
        link_radius=0.03,
        steps_per_segment=90,
        z_clear=0.08, 
        max_detours=3,
        dt=0.02
    )

    # Movement set 4
    move_robot_with_replanning(
        [[r,SE3(0.100, 0.530, 0.440) * GRASP_FROM_TOP, dount2 ],
        [r2,SE3(-0.6, -0.8, 0.48)* GRASP_FROM_TOP, None]],
        env, gui, obstacles,
        link_radius=0.03,
        steps_per_segment=90,
        z_clear=0.08, 
        max_detours=3,
        dt=0.02
    )

    # Movement set 5
    move_robot_with_replanning(
        [[r,SE3(-0.205, -0.205, 0.48) * GRASP_FROM_TOP, None],
        [r2,SE3(-0.5, -0.3,1.0) * GRASP_FROM_TOP , None]],
        env, gui, obstacles,
        link_radius=0.03,
        steps_per_segment=90,
        z_clear=0.08, 
        max_detours=3,
        dt=0.02
    )

    input(" Press Enter to quit...")

if __name__ == "__main__":
    #Create the main environment
    workenv = Environment()
    env = workenv.get_env()
    
    #Spawn robots
    r = UR3()
    base_pose= SE3(0,0,0.34) 
    r.base = base_pose
    r2 = create_GP7()
    r3 = create_praybot()
    r3.base = SE3(0.0, 1.18, 0.02)
    #r3 =add_robot_meshes(env,r3)
    env.add(r2)
    env.add(r3)

    q = [0, -pi/3, pi/3, 0, pi/2, 0]   
    T_ee = r2.fkine(q)
    print (T_ee)
    r.add_to_env(workenv.env)
    env.step(0.01)

    #Create rest of scene data
    scene_data = add_objects(env)

    # Create and launch GUI
    print("\n  Launching GUI control panel...")
    gui = SimpleRobotGUI()

    worker = threading.Thread(
    target=run_all_moves,
    args=(env, gui, r, r2, r3, scene_data),
    daemon=True
    )
    worker.start()
    gui.run()
 
    r2.q = [0, np.deg2rad(30), -np.deg2rad(30), 0, np.deg2rad(40), 0]