"""
Main Demo: Automated box and lid assembly.
Loads all parts from JSON configs, then the robot:
1. Picks up each lid from start position
2. Attaches it to the corresponding box
3. Moves the assembled box+lid to the end position (near crate) 
    ##INCOMPLETE - need to set end pos in config files##
4. Drops it off and repeats for all boxes
"""
import numpy as np
import time
import os
import json
from setup_scene import initialize_environment, update_robot_meshes
from utilities.stl_manager import STLManager
from utilities.object_array import AttachableObject
from ik_controller import solve_ik_with_limits, get_current_end_effector_pos


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


def move_to_position_smooth(robot, env, meshes, target_pos, attached_objects=None, steps=50):
    """
    Move robot to target position smoothly with interpolation.
    
    Args:
        robot: Robot instance
        env: Swift environment
        meshes: Robot meshes for visualization
        target_pos: Target [x, y, z] position
        attached_objects: List of objects to move with end-effector
        steps: Number of interpolation steps
    
    Returns:
        True if successful, False otherwise
    """
    q_target, success = solve_ik_with_limits(robot, target_pos)
    
    if not success or q_target is None:
        print(f"✗ Could not reach position {np.round(target_pos, 3)}")
        return False
    
    # Interpolate smoothly
    q_start = robot.q.copy()
    
    for step in range(steps):
        alpha = (step + 1) / steps
        robot.q = q_start * (1 - alpha) + q_target * alpha
        
        # Update visualization
        update_robot_meshes(robot, meshes)
        
        # Update attached objects
        if attached_objects:
            ee_pos = get_current_end_effector_pos(robot)
            for obj in attached_objects:
                if isinstance(obj, AttachableObject):
                    obj.update_from_parent(ee_pos)
                else:
                    obj.position = ee_pos
        
        env.step(0.02)
    
    return True


def main():
    # Initialize environment and robot
    env, robot, meshes, *_ = initialize_environment()
    
    print("\n" + "=" * 50)
    print("AUTOMATED BOX & LID ASSEMBLY")
    print("=" * 50)
    
    # Create STL manager
    manager = STLManager(env)
    
    # Get directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")
    configs_dir = os.path.join(script_dir, "configurations")
    
    # Load all parts from their config files
    print("\n📦 Loading parts from JSON configs...")
    boxes = load_parts_from_config(manager, models_dir, configs_dir, 
                                   "box_config.json", "BoxBase.stl")
    lids = load_parts_from_config(manager, models_dir, configs_dir,
                                  "lid_config.json", "BoxLid.stl")
    donuts = load_parts_from_config(manager, models_dir, configs_dir,
                                   "donut_config.json", "Donut.dae")
    burnt_donuts = load_parts_from_config(manager, models_dir, configs_dir,
                                         "burntDonut_config.json", "Donut_burnt.dae")
    
    print(f"\n✓ Scene loaded: {len(boxes)} boxes, {len(lids)} lids, {len(donuts)} donuts, {len(burnt_donuts)} burnt donuts")
    
    # Load config data for end positions
    with open(os.path.join(configs_dir, "box_config.json"), 'r') as f:
        box_configs = json.load(f)
    with open(os.path.join(configs_dir, "lid_config.json"), 'r') as f:
        lid_configs = json.load(f)
    
    # Verify we have matching numbers
    if len(boxes) != len(lids):
        print(f"⚠ Warning: {len(boxes)} boxes but {len(lids)} lids")
    
    num_pairs = min(len(boxes), len(lids))
    print(f"\n🤖 Processing {num_pairs} box-lid pairs...")
    
    # Display initial scene
    print("\nDisplaying initial scene...")
    for _ in range(30):
        env.step(0.05)
    
    # REACHABILITY TEST - just touch each object without picking up
    print("\n" + "=" * 70)
    print("🔍 REACHABILITY TEST - TOUCHING EACH OBJECT")
    print("=" * 70)
    
    approach_height = 0.15  # 15cm above each object
    
    # Test each lid
    for i, lid in enumerate(lids):
        lid_config = lid_configs[i]
        lid_start = np.array(lid_config['start'])
        touch_pos = lid_start + np.array([0, 0, approach_height])
        
        print(f"\n--- Testing Lid {i+1} ---")
        print(f"   Position: {np.round(lid_start, 3)}")
        print(f"   Touch point: {np.round(touch_pos, 3)}")
        
        success = move_to_position_smooth(robot, env, meshes, touch_pos)
        if success:
            print(f"   ✅ Successfully reached Lid {i+1}!")
        else:
            print(f"   ❌ FAILED to reach Lid {i+1}")
        
        time.sleep(0.5)
    
    # Test each box
    for i, box in enumerate(boxes):
        box_config = box_configs[i]
        box_start = np.array(box_config['start'])
        touch_pos = box_start + np.array([0, 0, approach_height])
        
        print(f"\n--- Testing Box {i+1} ---")
        print(f"   Position: {np.round(box_start, 3)}")
        print(f"   Touch point: {np.round(touch_pos, 3)}")
        
        success = move_to_position_smooth(robot, env, meshes, touch_pos)
        if success:
            print(f"   ✅ Successfully reached Box {i+1}!")
        else:
            print(f"   ❌ FAILED to reach Box {i+1}")
        
        time.sleep(0.5)
    
    # Test each end position
    for i, box_config in enumerate(box_configs):
        end_pos = np.array(box_config['end'])
        touch_pos = end_pos + np.array([0, 0, approach_height])
        
        print(f"\n--- Testing End Position {i+1} ---")
        print(f"   Position: {np.round(end_pos, 3)}")
        print(f"   Touch point: {np.round(touch_pos, 3)}")
        
        success = move_to_position_smooth(robot, env, meshes, touch_pos)
        if success:
            print(f"   ✅ Successfully reached End Position {i+1}!")
        else:
            print(f"   ❌ FAILED to reach End Position {i+1}")
        
        time.sleep(0.5)
    
    # All done!
    print("\n" + "=" * 70)
    print("✅ REACHABILITY TEST COMPLETE!")
    print("=" * 70)
    print(f"\nReview the results above to see which positions are reachable.")
    print(f"Any failed positions may need adjusting in the config files.")
    
    # Hold simulation
    print("\n💡 Simulation running. Press Ctrl+C to exit.")
    
    try:
        while True:
            env.step(0.05)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        try:
            env.close()
        except:
            pass


if __name__ == '__main__':
    main()
