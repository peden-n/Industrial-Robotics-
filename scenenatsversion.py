from roboticstoolbox.backends.swift import Swift
from spatialmath import SE3
import spatialgeometry as geometry
import os, json


class Environment:
    def __init__(self):
        env = Swift()
        env.launch(realtime=True)
        env.step(0.02)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "configurations", "scene_config.json")
        environment_dir = os.path.join(script_dir, "environment")
        
        self.scene_objects = {}  # Store references to scene objects
        
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
                    self.scene_objects[stl_file.replace('.stl', '')] = mesh
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
                    self.scene_objects[obj_config['name']] = mesh
                    print(f" Loaded {stl_file} at {position}")
                except Exception as e:
                    print(f"[Warning] Could not load {stl_file}:", e)
            
            self.env = env

    def get_env(self):
        return self.env
    def get_object(self):
        return self.scene_objects
    


if __name__ == "__main__":
    Environment()
