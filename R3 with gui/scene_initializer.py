"""
Scene Initializer - Handles complete initialization of environment, robot, and objects.
Separates initialization logic from main control loop.
"""
import os
import json
import numpy as np
from setup_scene import initialize_environment, update_robot_meshes
from utilities.stl_manager import STLManager


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


def initialize_complete_scene():
    """
    Initialize the complete scene with environment, robot, and all objects.
    
    Returns:
        dict: Complete scene data containing:
            - env: Swift environment
            - robot: DHRobot instance
            - meshes: Robot link meshes
            - scene_objects: Scene objects (tables, walls, etc.)
            - manager: STL manager for objects
            - objects: Dictionary of loaded objects {type: [list of objects]}
            - configs: Dictionary of loaded configurations
    """
    print("\n" + "=" * 60)
    print("INITIALIZING COMPLETE SCENE")
    print("=" * 60)
    
    # Initialize environment and robot
    env, robot, meshes, scene_objects = initialize_environment()
    
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
    print(f"  - Robot: {robot.name} at base {robot.q}")
    print(f"  - Scene objects: {len(scene_objects)} (tables, walls, etc.)")
    print(f"  - Loaded objects: {total_objects} total")
    for obj_type, obj_list in objects.items():
        if obj_list:
            print(f"    • {len(obj_list)} {obj_type}")
    
    # Display initial scene for a moment
    print("\nDisplaying initial scene...")
    for i in range(30):
        env.step(0.05)
    
    print("=" * 60)
    print(" SCENE READY FOR CONTROL")
    print("=" * 60)
    
    return {
        'env': env,
        'robot': robot,
        'meshes': meshes,
        'scene_objects': scene_objects,
        'manager': manager,
        'objects': objects,
        'configs': configs
    }


if __name__ == '__main__':
    # Test the initializer
    scene_data = initialize_complete_scene()
    
    print("\nTest: Scene data keys:", list(scene_data.keys()))
    print("Environment running. Press Ctrl+C to exit.")
    
    try:
        while True:
            scene_data['env'].step(0.05)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        try:
            scene_data['env'].close()
        except:
            pass