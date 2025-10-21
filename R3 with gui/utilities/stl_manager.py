"""
Mesh Manager - Load, control, and manipulate mesh objects (STL, DAE/COLLADA).
Provides easy control over position, rotation, visibility, and lifecycle.

Supported formats:
- .stl (STereoLithography)
- .dae (COLLADA/Digital Asset Exchange)
"""
import os
import numpy as np
import spatialgeometry as geometry
from spatialmath import SE3


class STLObject:
    """
    Wrapper class for a mesh object (STL or DAE) with full control over its properties.
    
    Supported formats: .stl, .dae (COLLADA)
    
    Properties:
    -----------
    - position: [x, y, z] location in meters
    - rotation: SE3 rotation matrix or RPY angles
    - scale: uniform or [sx, sy, sz] scale factors
    - color: RGBA color tuple (0-1 range)
    - visible: whether the object is shown
    - loaded: whether the object is added to the environment
    """
    
    def __init__(self, stl_path, name=None, position=None, rotation=None, 
                 scale=1.0, color=None):
        """
        Create a mesh object.
        
        Parameters:
        -----------
        stl_path : str
            Path to the mesh file (.stl or .dae)
        name : str, optional
            Name for the object (defaults to filename)
        position : array_like (3,), optional
            Initial position [x, y, z] in meters (default: origin)
        rotation : array_like (3,) or SE3, optional
            Initial rotation as RPY angles (rad) or SE3 object (default: identity)
        scale : float or array_like (3,), optional
            Scale factor(s) (default: 1.0)
        color : tuple (4,), optional
            RGBA color (0-1 range) (default: gray)
        """
        self.stl_path = stl_path
        self.name = name if name else os.path.basename(stl_path)
        
        # Position
        self._position = np.array(position) if position is not None else np.array([0.0, 0.0, 0.0])
        
        # Rotation (stored as SE3)
        if rotation is None:
            self._rotation = SE3()
        elif isinstance(rotation, SE3):
            self._rotation = rotation
        else:
            # Assume RPY angles in radians
            self._rotation = SE3.RPY(rotation)
        
        # Scale
        if isinstance(scale, (int, float)):
            self._scale = (scale, scale, scale)
        else:
            self._scale = tuple(scale)
        
        # Color
        self._color = color if color is not None else (0.7, 0.7, 0.7, 1.0)
        
        # State
        self._mesh = None
        self._env = None
        self._loaded = False
        self._visible = True
        
        print(f"[STLObject] Created '{self.name}'")
    
    @property
    def position(self):
        """Get current position [x, y, z]."""
        return self._position.copy()
    
    @position.setter
    def position(self, value):
        """Set position [x, y, z] and update visualization."""
        self._position = np.array(value)
        self._update_pose()
        print(f"[{self.name}] Position set to {np.round(self._position, 3)}")
    
    @property
    def rotation(self):
        """Get current rotation as SE3 object."""
        return self._rotation
    
    @rotation.setter
    def rotation(self, value):
        """
        Set rotation and update visualization.
        Can be SE3 object or RPY angles [roll, pitch, yaw] in radians.
        """
        if isinstance(value, SE3):
            self._rotation = value
        else:
            self._rotation = SE3.RPY(value)
        self._update_pose()
        print(f"[{self.name}] Rotation updated")
    
    @property
    def rpy(self):
        """Get current rotation as RPY angles [roll, pitch, yaw] in radians."""
        return self._rotation.rpy()
    
    @rpy.setter
    def rpy(self, value):
        """Set rotation using RPY angles [roll, pitch, yaw] in radians."""
        self._rotation = SE3.RPY(value)
        self._update_pose()
        print(f"[{self.name}] RPY set to {np.round(np.rad2deg(value), 2)} deg")
    
    @property
    def scale(self):
        """Get current scale (sx, sy, sz)."""
        return self._scale
    
    @scale.setter
    def scale(self, value):
        """Set scale. Can be a single float or (sx, sy, sz) tuple."""
        if isinstance(value, (int, float)):
            self._scale = (value, value, value)
        else:
            self._scale = tuple(value)
        
        # Scaling requires reloading the mesh
        if self._loaded:
            print(f"[{self.name}] Reloading with new scale {self._scale}")
            self.reload()
    
    @property
    def color(self):
        """Get current RGBA color."""
        return self._color
    
    @color.setter
    def color(self, value):
        """Set RGBA color (0-1 range)."""
        old_color = self._color
        self._color = tuple(value)
        if self._mesh is not None and self._loaded:
            # Update mesh color directly
            try:
                # Try to update the internal _color attribute of the mesh
                if hasattr(self._mesh, '_color'):
                    self._mesh._color = self._color
                # Also try the public color property
                if hasattr(self._mesh, 'color'):
                    self._mesh.color = self._color
                print(f"[{self.name}] Color updated: {old_color} -> {self._color}")
            except Exception as e:
                print(f"[{self.name}] Color update failed, needs reload: {e}")
    
    @property
    def visible(self):
        """Check if object is visible."""
        return self._visible
    
    @visible.setter
    def visible(self, value):
        """Set visibility."""
        self._visible = bool(value)
        if self._mesh is not None:
            try:
                # Swift meshes don't have a direct visibility toggle,
                # so we move it far away or back to position
                if self._visible:
                    self._update_pose()
                    print(f"[{self.name}] Shown")
                else:
                    # Move far away (hack for hiding)
                    hidden_pose = SE3(10000, 10000, 10000)
                    self._mesh.T = hidden_pose.A
                    print(f"[{self.name}] Hidden")
            except Exception as e:
                print(f"[{self.name}] Visibility toggle failed: {e}")
    
    @property
    def loaded(self):
        """Check if object is loaded in the environment."""
        return self._loaded
    
    @property
    def pose(self):
        """Get current pose as SE3 object (position + rotation)."""
        return SE3.Rt(self._rotation.R, self._position)
    
    @pose.setter
    def pose(self, value):
        """Set pose using SE3 object."""
        if isinstance(value, SE3):
            self._position = value.t
            self._rotation = SE3.Rt(value.R, [0, 0, 0])
            self._update_pose()
            print(f"[{self.name}] Pose updated")
    
    def load(self, env):
        """
        Load the mesh into the Swift environment.
        
        Supports: .stl, .dae (COLLADA)
        
        Parameters:
        -----------
        env : Swift
            The Swift environment to add the mesh to
        """
        if self._loaded:
            print(f"[{self.name}] Already loaded")
            return
        
        if not os.path.exists(self.stl_path):
            print(f"[{self.name}] ERROR: Mesh file not found: {self.stl_path}")
            return
        
        try:
            # Create the mesh
            pose = self.pose
            self._mesh = geometry.Mesh(
                self.stl_path,
                pose=pose.A,
                scale=self._scale,
                color=self._color
            )
            
            # Add to environment
            env.add(self._mesh)
            self._env = env
            self._loaded = True
            
            # Apply visibility
            if not self._visible:
                self.visible = False
            
            print(f"[{self.name}] ✓ Loaded into environment")
            
        except Exception as e:
            print(f"[{self.name}] ERROR loading: {e}")
    
    def unload(self):
        """Remove the mesh from the environment."""
        if not self._loaded:
            print(f"[{self.name}] Not loaded")
            return
        
        try:
            if self._env is not None and self._mesh is not None:
                # Swift doesn't have a direct remove, but we can hide it
                self._mesh.T = SE3(10000, 10000, 10000).A
                print(f"[{self.name}] ✓ Unloaded (hidden)")
            
            self._loaded = False
            self._mesh = None
            self._env = None
            
        except Exception as e:
            print(f"[{self.name}] ERROR unloading: {e}")
    
    def reload(self):
        """Reload the mesh (useful after changing scale or color)."""
        if self._loaded and self._env is not None:
            env = self._env
            self.unload()
            self.load(env)
        else:
            print(f"[{self.name}] Not loaded, cannot reload")
    
    def show(self):
        """Make the object visible."""
        self.visible = True
    
    def hide(self):
        """Make the object invisible."""
        self.visible = False
    
    def translate(self, delta):
        """
        Translate the object by a delta vector.
        
        Parameters:
        -----------
        delta : array_like (3,)
            Translation vector [dx, dy, dz]
        """
        self.position = self._position + np.array(delta)
    
    def rotate(self, rpy_delta):
        """
        Rotate the object by RPY delta angles.
        
        Parameters:
        -----------
        rpy_delta : array_like (3,)
            Delta rotation [droll, dpitch, dyaw] in radians
        """
        delta_rot = SE3.RPY(rpy_delta)
        self._rotation = self._rotation * delta_rot
        self._update_pose()
        print(f"[{self.name}] Rotated by {np.round(np.rad2deg(rpy_delta), 2)} deg")
    
    def _update_pose(self):
        """Internal method to update the mesh transform."""
        if self._mesh is not None and self._loaded and self._visible:
            try:
                self._mesh.T = self.pose.A
            except Exception as e:
                print(f"[{self.name}] ERROR updating pose: {e}")
    
    def __repr__(self):
        status = []
        if self._loaded:
            status.append("loaded")
        if self._visible:
            status.append("visible")
        if not status:
            status.append("unloaded")
        
        return (f"STLObject('{self.name}', pos={np.round(self._position, 3)}, "
                f"status={','.join(status)})")


class STLManager:
    """
    Manager class to handle multiple STL objects.
    """
    
    def __init__(self, env=None):
        """
        Create an STL manager.
        
        Parameters:
        -----------
        env : Swift, optional
            Swift environment (can be set later)
        """
        self.env = env
        self.objects = {}
        print("[STLManager] Initialized")
    
    def add(self, stl_path, name=None, **kwargs):
        """
        Add a new STL object to the manager.
        
        Parameters:
        -----------
        stl_path : str
            Path to STL file
        name : str, optional
            Name for the object
        **kwargs : dict
            Additional arguments passed to STLObject (position, rotation, scale, color)
        
        Returns:
        --------
        obj : STLObject
            The created STL object
        """
        obj = STLObject(stl_path, name=name, **kwargs)
        obj_name = obj.name
        
        # Ensure unique name
        counter = 1
        while obj_name in self.objects:
            obj_name = f"{obj.name}_{counter}"
            counter += 1
        
        obj.name = obj_name
        self.objects[obj_name] = obj
        
        # Auto-load if environment is set
        if self.env is not None:
            obj.load(self.env)
        
        return obj
    
    def get(self, name):
        """Get an object by name."""
        return self.objects.get(name)
    
    def remove(self, name):
        """Remove an object by name."""
        if name in self.objects:
            self.objects[name].unload()
            del self.objects[name]
            print(f"[STLManager] Removed '{name}'")
    
    def load_all(self, env=None):
        """Load all objects into the environment."""
        if env is not None:
            self.env = env
        
        if self.env is None:
            print("[STLManager] ERROR: No environment set")
            return
        
        for obj in self.objects.values():
            if not obj.loaded:
                obj.load(self.env)
    
    def unload_all(self):
        """Unload all objects from the environment."""
        for obj in self.objects.values():
            obj.unload()
    
    def show_all(self):
        """Make all objects visible."""
        for obj in self.objects.values():
            obj.show()
    
    def hide_all(self):
        """Hide all objects."""
        for obj in self.objects.values():
            obj.hide()
    
    def list(self):
        """List all managed objects."""
        if not self.objects:
            print("[STLManager] No objects")
            return
        
        print(f"[STLManager] {len(self.objects)} object(s):")
        for name, obj in self.objects.items():
            print(f"  - {obj}")
    
    def __getitem__(self, name):
        """Allow dict-style access: manager['object_name']"""
        return self.get(name)


if __name__ == '__main__':
    # Demo / test
    import swift
    
    print("="*50)
    print("STL Manager Demo")
    print("="*50)
    
    # Create environment
    env = swift.Swift()
    env.launch(realtime=True)
    
    # Create manager
    manager = STLManager(env)
    
    # Example: Add some boxes (you'll need actual STL files)
    # Assuming BoxBase.stl exists in models directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(parent_dir, "models")
    
    box_path = os.path.join(models_dir, "BoxBase.stl")
    
    if os.path.exists(box_path):
        # Add first box
        box1 = manager.add(
            box_path,
            name="Box1",
            position=[0.5, 0.5, 0.5],
            color=(1, 0, 0, 1),
            scale=0.001
        )
        
        # Add second box at different location
        box2 = manager.add(
            box_path,
            name="Box2",
            position=[0.5, -0.5, 0.5],
            color=(0, 1, 0, 1),
            scale=0.001
        )
        
        # List objects
        manager.list()
        
        # Demonstrate control
        print("\n" + "="*50)
        print("Demonstrating controls...")
        print("="*50)
        
        import time
        
        # Move box1
        for _ in range(20):
            env.step(0.05)
        
        print("\nMoving Box1 up...")
        box1.translate([0, 0, 0.2])
        
        for _ in range(20):
            env.step(0.05)
        
        print("\nRotating Box1...")
        box1.rotate([0, 0, np.pi/4])
        
        for _ in range(20):
            env.step(0.05)
        
        print("\nHiding Box2...")
        box2.hide()
        
        for _ in range(20):
            env.step(0.05)
        
        print("\nShowing Box2 again...")
        box2.show()
        
        for _ in range(20):
            env.step(0.05)
        
        print("\n✓ Demo complete!")
    else:
        print(f"[Demo] BoxBase.stl not found at {box_path}")
        print("[Demo] Add your own STL files to test")
    
    print("\nPress Ctrl+C to exit")
    try:
        while True:
            env.step(0.05)
    except KeyboardInterrupt:
        print("\nExiting...")
        env.close()
