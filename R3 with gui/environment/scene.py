"""
Scene Setup: Loads environment objects from scene_config.json
All positions, colors, and object properties are configurable via the JSON file.
"""
import swift
from spatialmath import SE3
import spatialgeometry as geometry
import spatialmath as sm
import os
import json
import numpy as np


def load_scene_from_config(config_path=None):
    """
    Load scene environment from configuration file.
    
    Args:
        config_path: Path to scene_config.json. If None, looks in ../configurations/
    
    Returns:
        env: Swift environment with all objects loaded
        objects: Dictionary of loaded mesh objects {name: mesh}
    """
    # Get config path
    if config_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(os.path.dirname(current_dir), "configurations", "scene_config.json")
    
    # Load configuration
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Scene config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Launch environment
    env = swift.Swift()
    env.launch()
    
    # Get scale
    scale = config.get('scale', 0.001)
    scale_tuple = (scale, scale, scale)
    
    # Get directory for STL files
    stl_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load all objects
    objects = {}
    
    print("Loading scene objects from config...")
    for obj_config in config.get('objects', []):
        name = obj_config['name']
        file_path = os.path.join(stl_dir, obj_config['file'])
        
        if not os.path.exists(file_path):
            print(f"  ⚠ Warning: {obj_config['file']} not found, skipping...")
            continue
        
        # Get position and rotation
        position = obj_config.get('position', [0, 0, 0])
        rotation = obj_config.get('rotation', [0, 0, 0])  # Roll, Pitch, Yaw in radians
        color = tuple(obj_config.get('color', [0.7, 0.7, 0.7, 1.0]))
        
        # Create SE3 pose from position and RPY rotation
        pose = SE3(position[0], position[1], position[2]) * SE3.RPY(rotation)
        
        # Create mesh
        mesh = geometry.Mesh(
            file_path,
            pose=pose.A,
            color=color,
            scale=scale_tuple
        )
        
        # Add to environment
        env.add(mesh)
        objects[name] = mesh
        
        print(f"  ✓ Loaded {name} at {position}")
    
    print(f"✓ Scene loaded with {len(objects)} objects")
    
    return env, objects


if __name__ == '__main__':
    # Load scene from config
    env, objects = load_scene_from_config()
    
    # Display loaded objects
    print("\nLoaded objects:")
    for name, obj in objects.items():
        print(f"  - {name}")
    
    # Hold environment
    input('\nPress Enter to step and hold the environment...')
    env.step(0.01)
    env.hold()