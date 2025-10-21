"""
Setup scene module - Initializes the environment and robot without blocking.
Exposes the robot and environment for external control.
"""
import os
import numpy as np
import spatialgeometry as geometry
from spatialmath import SE3
import swift
from roboticstoolbox import DHLink, DHRobot


# Robot base position
BASE_XYZ = (0.0, 1.18, 0.02)


def create_robot():
    """Create and return the robot with initial joint configuration."""
    # Define DH parameters
    link1 = DHLink(d=0.445, a=0.1404, alpha=np.pi / 2, qlim=np.deg2rad([-180, 180]), offset=0)
    link2 = DHLink(d=0.17, a=0.7, alpha=0.0, qlim=np.deg2rad([-155, 95]), offset=np.pi / 2)
    link3 = DHLink(d=-0.17, a=0.115, alpha=np.pi / 2, qlim=np.deg2rad([-75, 180]))
    link4 = DHLink(d=0.8, a=0.0, alpha=np.pi / 2, qlim=np.deg2rad([-400, 400]))
    link5 = DHLink(d=0.0, a=0.0, alpha=-np.pi / 2, qlim=np.deg2rad([-120, 120]))
    link6 = DHLink(d=0.0, a=0.0, alpha=0.0, qlim=np.deg2rad([-400, 400]))

    robot = DHRobot([link1, link2, link3, link4, link5, link6], name="myRobot")

    # Initial joint configuration
    q_init = np.deg2rad([0.0, -30.0, 45.0, 0.0, 10.0, 0.0])
    robot.q = q_init.copy()

    return robot


def add_robot_meshes(env, robot, base_xyz=BASE_XYZ):
    """
    Load and add robot link meshes to the environment.
    Returns a dictionary of mesh objects keyed by link index.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    robot_dir = os.path.join(script_dir, "robot")
    meshes = {}

    # Load Link0 through Link6 STL files from robot folder
    for k in range(0, 7):
        stl_name = f"Link{k}.stl"
        stl_path = os.path.join(robot_dir, stl_name)
        
        if os.path.exists(stl_path):
            try:
                mesh = geometry.Mesh(
                    stl_path,
                    pose=SE3(*base_xyz).A,
                    scale=(1.0, 1.0, 1.0),
                    color=(0.2, 0.2, 0.7, 1),
                )
                env.add(mesh)
                meshes[k] = mesh
                print(f"✓ Loaded {stl_name}")
            except Exception as e:
                print(f"[Warning] Failed to load {stl_name}:", e)
        else:
            print(f"[Info] {stl_name} not found; skipping.")

    # Update mesh poses based on current robot configuration
    update_robot_meshes(robot, meshes, base_xyz)
    
    return meshes


def update_robot_meshes(robot, meshes, base_xyz=BASE_XYZ):
    """Update robot mesh poses based on current joint configuration."""
    if len(meshes) == 0:
        return
    
    try:
        T_all = robot.fkine_all(robot.q)
        base_SE3 = SE3(*base_xyz)
        
        for idx, mesh in meshes.items():
            try:
                mesh.T = (base_SE3 * T_all[idx]).A
            except Exception as e:
                print(f"[Warning] Failed to update mesh {idx}:", e)
    except Exception as e:
        print(f"[Warning] Failed to compute FK:", e)


def setup_environment():
    """
    Initialize the Swift environment with all world geometry (tables, walls, etc.).
    Loads objects from scene_config.json in the configurations folder.
    
    Returns:
        env: Swift environment
        scene_objects: Dictionary of scene mesh objects {name: mesh}
    """
    import json
    
    env = swift.Swift()
    env.launch(realtime=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configurations", "scene_config.json")
    environment_dir = os.path.join(script_dir, "environment")
    
    scene_objects = {}  # Store references to scene objects
    
    # Load configuration file
    if not os.path.exists(config_path):
        print(f" Warning: Config file not found at {config_path}")
        print("  Loading scene with default hardcoded values...")
        
        # Fallback to hardcoded values
        SCALE = 0.001
        SCALE_TUPLE = (SCALE, SCALE, SCALE)
        world_objects = [
            ("Stand1.stl", SE3(0, 0, 0).A, (0.5, 0, 0, 1), SCALE_TUPLE),
            ("Stand2.stl", SE3(0.0, 0.0, 0.0).A, (0.2, 0.5, 0.2, 1), SCALE_TUPLE),
            # ("Stand3.stl", SE3(0, 0, 0).A, (0.2, 0.2, 0.7, 1), SCALE_TUPLE),
            ("TableLong.stl", SE3(0.0, 0.53, 0.4).A, (0.75, 0.75, 0.75, 1), SCALE_TUPLE),
            ("TableShort.stl", SE3(0.65, 1.48, 0.4).A, (0.75, 0.75, 0.75, 1), SCALE_TUPLE),
            ("Conveyor.stl", SE3(0, -0.5, 0.48).A, (0.3, 0.3, 0.3, 1), SCALE_TUPLE),
            ("Pallet.stl", SE3(-0.9325, 1.7625, 0.0).A, (0.545, 0.271, 0.075, 1), SCALE_TUPLE),
            ("Walls.stl", SE3(0, 0, 0).A, (1.0, 1.0, 1.0, 1), SCALE_TUPLE),
            ("TapeYellow.stl", SE3(0, 0.6, 0.1).A, (1.0, 1.0, 0.0, 1.0), SCALE_TUPLE),
            ("TapeBlack.stl", SE3(0, 0.6, 0.1).A, (0.0, 0.0, 0.0, 1.0), SCALE_TUPLE),
            ("FireExtinguisher.stl", SE3(0, -3.0, 0.0).A, (1.0, 0.0, 0.0, 1.0), SCALE_TUPLE),
            ("EStop.stl", SE3(0.05, -3.0, 0.0).A, (0.0, 1.0, 0.0, 1.0), (1.0, 1.0, 1.0)),
        ]
        
        for stl_file, pose, color, scale_tuple in world_objects:
            stl_path = os.path.join(environment_dir, stl_file)
            try:
                mesh = geometry.Mesh(stl_path, pose=pose, color=color, scale=scale_tuple)
                env.add(mesh)
                scene_objects[stl_file.replace('.stl', '')] = mesh
                print(f"✓ Loaded {stl_file}")
            except Exception as e:
                print(f"[Warning] Could not load {stl_file}:", e)
    else:
        # Load from config file
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        scale = config.get('scale', 0.001)
        scale_tuple = (scale, scale, scale)
        
        print(f"Loading scene from config: {os.path.basename(config_path)}")
        
        for obj_config in config.get('objects', []):
            stl_file = obj_config['file']
            stl_path = os.path.join(environment_dir, stl_file)
            
            if not os.path.exists(stl_path):
                print(f"   {stl_file} not found, skipping...")
                continue
            
            # Get position and rotation
            position = obj_config.get('position', [0, 0, 0])
            rotation = obj_config.get('rotation', [0, 0, 0])  # Roll, Pitch, Yaw
            color = tuple(obj_config.get('color', [0.7, 0.7, 0.7, 1.0]))
            
            # Handle individual object scale (if specified) or use global scale
            obj_scale = obj_config.get('scale', scale)
            if isinstance(obj_scale, (int, float)):
                obj_scale_tuple = (obj_scale, obj_scale, obj_scale)
            else:
                obj_scale_tuple = tuple(obj_scale)
            
            # Create SE3 pose
            pose = SE3(position[0], position[1], position[2]) * SE3.RPY(rotation)
            
            try:
                mesh = geometry.Mesh(stl_path, pose=pose.A, color=color, scale=obj_scale_tuple)
                env.add(mesh)
                scene_objects[obj_config['name']] = mesh
                print(f" Loaded {stl_file} at {position}")
            except Exception as e:
                print(f"[Warning] Could not load {stl_file}:", e)

    return env, scene_objects


def initialize_environment():
    """
    Complete initialization: create environment, robot, and load all meshes.
    
    Returns:
        env: Swift environment
        robot: DHRobot instance
        meshes: Dictionary of robot link meshes
        scene_objects: Dictionary of scene objects (tables, walls, etc.)
    
    Usage:
        env, robot, meshes, scene_objects = initialize_environment()
        
        # Move robot joints
        robot.q = np.deg2rad([10, -45, 60, 0, 20, 0])
        update_robot_meshes(robot, meshes)
        
        # Move a table
        scene_objects['TableLong'].T = SE3(0, 0.6, 0.4).A
        env.step()
    """
    print("=" * 50)
    print("Initializing environment...")
    print("=" * 50)
    
    # Setup world
    env, scene_objects = setup_environment()
    
    # Create robot
    print("\nCreating robot...")
    robot = create_robot()
    
    # Load robot meshes
    print("\nLoading robot meshes...")
    meshes = add_robot_meshes(env, robot)
    
    print("\n" + "=" * 50)
    print(" Environment initialized successfully!")
    print("=" * 50)
    print(f"Robot: {robot.name}")
    print(f"Links: {robot.n}")
    print(f"Current joint angles (deg): {np.round(np.rad2deg(robot.q), 2)}")
    print(f"Base position: {BASE_XYZ}")
    print(f"Scene objects loaded: {list(scene_objects.keys())}")
    print("=" * 50)
    
    return env, robot, meshes, scene_objects


if __name__ == '__main__':
    # Quick test
    env, robot, meshes, scene_objects = initialize_environment()
    
    print("\nTest: Moving robot to new configuration...")
    robot.q = np.deg2rad([45, -60, 90, 0, 30, 45])
    update_robot_meshes(robot, meshes)
    
    print("\nTest: Moving TableLong...")
    if 'TableLong' in scene_objects:
        scene_objects['TableLong'].T = SE3(0, 0.6, 0.4).A
    
    print("Stepping environment...")
    for _ in range(100):
        env.step(0.05)
    
    print("Test complete!")
