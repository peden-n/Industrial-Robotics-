"""
Object Array Manager - Create arrays of objects with attachment system.
Supports individual control of objects and parent-child attachment relationships.
"""
import numpy as np
from utilities.stl_manager import STLObject, STLManager


class AttachableObject:
    """
    Wrapper for STLObject that adds attachment capabilities.
    Objects can be attached to other objects or to robot end-effectors.
    """
    
    def __init__(self, stl_object):
        """
        Create an attachable object from an STLObject.
        
        Parameters:
        -----------
        stl_object : STLObject
            The underlying STL object
        """
        self.stl_object = stl_object
        self._parent = None
        self._children = []
        self._offset = np.array([0.0, 0.0, 0.0])  # Offset from parent
        self._rotation_offset = np.array([0.0, 0.0, 0.0])  # RPY offset from parent
        
    @property
    def name(self):
        return self.stl_object.name
    
    @property
    def position(self):
        return self.stl_object.position
    
    @position.setter
    def position(self, value):
        """Set position and update all children."""
        self.stl_object.position = value
        self._update_children()
    
    @property
    def rotation(self):
        return self.stl_object.rotation
    
    @rotation.setter
    def rotation(self, value):
        self.stl_object.rotation = value
        self._update_children()
    
    @property
    def rpy(self):
        return self.stl_object.rpy
    
    @rpy.setter
    def rpy(self, value):
        self.stl_object.rpy = value
        self._update_children()
    
    @property
    def visible(self):
        return self.stl_object.visible
    
    @visible.setter
    def visible(self, value):
        self.stl_object.visible = value
    
    @property
    def color(self):
        return self.stl_object.color
    
    @color.setter
    def color(self, value):
        self.stl_object.color = value
    
    def attach_to(self, parent, offset=None, rotation_offset=None):
        """
        Attach this object to a parent object.
        
        Parameters:
        -----------
        parent : AttachableObject or callable
            Parent object or a function that returns position (e.g., get_current_end_effector_pos)
        offset : array_like (3,), optional
            Position offset from parent [x, y, z]
        rotation_offset : array_like (3,), optional
            Rotation offset from parent [roll, pitch, yaw] in radians
        """
        # Detach from previous parent if any
        if self._parent is not None:
            self.detach()
        
        self._parent = parent
        self._offset = np.array(offset) if offset is not None else np.array([0.0, 0.0, 0.0])
        self._rotation_offset = np.array(rotation_offset) if rotation_offset is not None else np.array([0.0, 0.0, 0.0])
        
        # Add to parent's children list if parent is AttachableObject
        if isinstance(parent, AttachableObject):
            parent._children.append(self)
        
        # Update position immediately
        self.update_from_parent()
        
        print(f"[{self.name}] Attached to {parent.name if isinstance(parent, AttachableObject) else 'custom parent'}")
    
    def detach(self):
        """Detach this object from its parent."""
        if self._parent is not None:
            # Remove from parent's children list
            if isinstance(self._parent, AttachableObject):
                try:
                    self._parent._children.remove(self)
                except ValueError:
                    pass
            
            print(f"[{self.name}] Detached from parent")
            self._parent = None
            self._offset = np.array([0.0, 0.0, 0.0])
            self._rotation_offset = np.array([0.0, 0.0, 0.0])
    
    def update_from_parent(self):
        """Update position based on parent's position."""
        if self._parent is None:
            return
        
        # Get parent position
        if callable(self._parent):
            # Parent is a function (e.g., get_current_end_effector_pos)
            parent_pos = self._parent()
        elif isinstance(self._parent, AttachableObject):
            parent_pos = self._parent.position
        else:
            return
        
        # Apply offset
        new_pos = parent_pos + self._offset
        
        # Update position without triggering children update (avoid recursion)
        self.stl_object.position = new_pos
        
        # Apply rotation offset if parent has rotation
        if isinstance(self._parent, AttachableObject) and np.any(self._rotation_offset):
            parent_rpy = self._parent.rpy
            self.stl_object.rpy = parent_rpy + self._rotation_offset
        
        # Update all children recursively
        self._update_children()
    
    def _update_children(self):
        """Update all attached children."""
        for child in self._children:
            child.update_from_parent()
    
    def show(self):
        """Show object."""
        self.stl_object.show()
    
    def hide(self):
        """Hide object."""
        self.stl_object.hide()
    
    def translate(self, delta):
        """Translate object and update children."""
        self.stl_object.translate(delta)
        self._update_children()
    
    def rotate(self, rpy_delta):
        """Rotate object and update children."""
        self.stl_object.rotate(rpy_delta)
        self._update_children()
    
    def __repr__(self):
        parent_info = ""
        if self._parent is not None:
            if isinstance(self._parent, AttachableObject):
                parent_info = f", attached_to={self._parent.name}"
            else:
                parent_info = ", attached=True"
        
        children_info = ""
        if self._children:
            children_info = f", children={len(self._children)}"
        
        return f"AttachableObject({self.name}{parent_info}{children_info})"


class ObjectArray:
    """
    Manager for creating and controlling arrays of objects.
    Supports grid patterns, individual control, and attachment.
    """
    
    def __init__(self, stl_manager):
        """
        Create an object array manager.
        
        Parameters:
        -----------
        stl_manager : STLManager
            The STL manager to use for creating objects
        """
        self.stl_manager = stl_manager
        self.objects = {}
        self.arrays = {}
        print("[ObjectArray] Initialized")
    
    def create_grid(self, stl_path, name_prefix, rows, cols, spacing, 
                    start_pos, scale=0.001, color=None, **kwargs):
        """
        Create a grid array of objects.
        
        Parameters:
        -----------
        stl_path : str
            Path to STL/DAE file
        name_prefix : str
            Prefix for object names (will append _row_col)
        rows : int
            Number of rows
        cols : int
            Number of columns
        spacing : float or tuple
            Spacing between objects (uniform or (row_spacing, col_spacing))
        start_pos : array_like (3,)
            Position of first object [x, y, z]
        scale : float or tuple
            Scale factor(s)
        color : tuple, optional
            RGBA color for all objects
        **kwargs : dict
            Additional arguments for STLObject
            
        Returns:
        --------
        array_objects : list of AttachableObject
            List of created objects
        """
        if isinstance(spacing, (int, float)):
            row_spacing = col_spacing = spacing
        else:
            row_spacing, col_spacing = spacing
        
        start_pos = np.array(start_pos)
        array_objects = []
        
        print(f"\n[ObjectArray] Creating {rows}x{cols} grid of {name_prefix}")
        print(f"  Spacing: {row_spacing} x {col_spacing}")
        print(f"  Start position: {start_pos}")
        
        for row in range(rows):
            for col in range(cols):
                # Calculate position
                pos = start_pos + np.array([
                    0,
                    col * col_spacing,
                    row * row_spacing
                ])
                
                # Create object name
                obj_name = f"{name_prefix}_{row}_{col}"
                
                # Create STL object
                stl_obj = self.stl_manager.add(
                    stl_path,
                    name=obj_name,
                    position=pos,
                    scale=scale,
                    color=color,
                    **kwargs
                )
                
                # Wrap in AttachableObject
                attachable = AttachableObject(stl_obj)
                array_objects.append(attachable)
                self.objects[obj_name] = attachable
        
        # Store array reference
        array_name = f"{name_prefix}_grid"
        self.arrays[array_name] = array_objects
        
        print(f"✓ Created {len(array_objects)} objects")
        return array_objects
    
    def create_line(self, stl_path, name_prefix, count, spacing, 
                    start_pos, direction='x', scale=0.001, color=None, **kwargs):
        """
        Create a line array of objects.
        
        Parameters:
        -----------
        stl_path : str
            Path to STL/DAE file
        name_prefix : str
            Prefix for object names
        count : int
            Number of objects
        spacing : float
            Spacing between objects
        start_pos : array_like (3,)
            Starting position
        direction : str
            Direction of line ('x', 'y', or 'z')
        scale : float or tuple
            Scale factor(s)
        color : tuple, optional
            RGBA color
            
        Returns:
        --------
        array_objects : list of AttachableObject
            List of created objects
        """
        start_pos = np.array(start_pos)
        direction_map = {'x': 0, 'y': 1, 'z': 2}
        dir_idx = direction_map.get(direction.lower(), 0)
        
        array_objects = []
        
        print(f"\n[ObjectArray] Creating line of {count} {name_prefix}")
        print(f"  Direction: {direction}, Spacing: {spacing}")
        
        for i in range(count):
            # Calculate position
            offset = np.zeros(3)
            offset[dir_idx] = i * spacing
            pos = start_pos + offset
            
            # Create object
            obj_name = f"{name_prefix}_{i}"
            stl_obj = self.stl_manager.add(
                stl_path,
                name=obj_name,
                position=pos,
                scale=scale,
                color=color,
                **kwargs
            )
            
            attachable = AttachableObject(stl_obj)
            array_objects.append(attachable)
            self.objects[obj_name] = attachable
        
        # Store array
        array_name = f"{name_prefix}_line"
        self.arrays[array_name] = array_objects
        
        print(f"✓ Created {len(array_objects)} objects")
        return array_objects
    
    def create_circle(self, stl_path, name_prefix, count, radius, 
                      center_pos, plane='xy', scale=0.001, color=None, **kwargs):
        """
        Create a circular array of objects.
        
        Parameters:
        -----------
        stl_path : str
            Path to STL/DAE file
        name_prefix : str
            Prefix for object names
        count : int
            Number of objects
        radius : float
            Radius of circle
        center_pos : array_like (3,)
            Center position
        plane : str
            Plane of circle ('xy', 'xz', or 'yz')
        scale : float or tuple
            Scale factor(s)
        color : tuple, optional
            RGBA color
            
        Returns:
        --------
        array_objects : list of AttachableObject
            List of created objects
        """
        center_pos = np.array(center_pos)
        array_objects = []
        
        print(f"\n[ObjectArray] Creating circle of {count} {name_prefix}")
        print(f"  Radius: {radius}, Plane: {plane}")
        
        for i in range(count):
            angle = 2 * np.pi * i / count
            
            # Calculate position based on plane
            if plane == 'xy':
                offset = np.array([radius * np.cos(angle), radius * np.sin(angle), 0])
            elif plane == 'xz':
                offset = np.array([radius * np.cos(angle), 0, radius * np.sin(angle)])
            elif plane == 'yz':
                offset = np.array([0, radius * np.cos(angle), radius * np.sin(angle)])
            else:
                offset = np.array([radius * np.cos(angle), radius * np.sin(angle), 0])
            
            pos = center_pos + offset
            
            # Create object
            obj_name = f"{name_prefix}_{i}"
            stl_obj = self.stl_manager.add(
                stl_path,
                name=obj_name,
                position=pos,
                scale=scale,
                color=color,
                **kwargs
            )
            
            attachable = AttachableObject(stl_obj)
            array_objects.append(attachable)
            self.objects[obj_name] = attachable
        
        # Store array
        array_name = f"{name_prefix}_circle"
        self.arrays[array_name] = array_objects
        
        print(f"✓ Created {len(array_objects)} objects")
        return array_objects
    
    def get(self, name):
        """Get object by name."""
        return self.objects.get(name)
    
    def get_array(self, array_name):
        """Get array by name."""
        return self.arrays.get(array_name)
    
    def attach_array_to_parent(self, array_name, parent, maintain_spacing=True):
        """
        Attach entire array to a parent object.
        
        Parameters:
        -----------
        array_name : str
            Name of array to attach
        parent : AttachableObject or callable
            Parent object
        maintain_spacing : bool
            If True, maintain relative positions between objects
        """
        array = self.arrays.get(array_name)
        if not array:
            print(f"[ObjectArray] Array '{array_name}' not found")
            return
        
        if maintain_spacing and len(array) > 0:
            # Get first object's position as reference
            first_pos = array[0].position.copy()
            
            # Attach first object
            array[0].attach_to(parent)
            
            # Attach rest with relative offsets
            for i, obj in enumerate(array[1:], 1):
                offset = obj.position - first_pos
                obj.attach_to(parent, offset=offset)
        else:
            # Attach all to same position
            for obj in array:
                obj.attach_to(parent)
        
        print(f"[ObjectArray] Attached {len(array)} objects from '{array_name}'")
    
    def update_all_attachments(self):
        """Update all objects that have parents."""
        for obj in self.objects.values():
            if obj._parent is not None:
                obj.update_from_parent()
    
    def show_array(self, array_name):
        """Show all objects in an array."""
        array = self.arrays.get(array_name)
        if array:
            for obj in array:
                obj.show()
    
    def hide_array(self, array_name):
        """Hide all objects in an array."""
        array = self.arrays.get(array_name)
        if array:
            for obj in array:
                obj.hide()
    
    def list(self):
        """List all objects and arrays."""
        print(f"\n[ObjectArray] {len(self.objects)} objects, {len(self.arrays)} arrays:")
        
        for array_name, array_objs in self.arrays.items():
            print(f"  Array '{array_name}': {len(array_objs)} objects")
        
        print("\n  Individual objects:")
        for name, obj in self.objects.items():
            print(f"    - {obj}")


if __name__ == '__main__':
    # Demo
    import os
    import swift
    from utilities.stl_manager import STLManager
    
    print("="*50)
    print("Object Array Manager Demo")
    print("="*50)
    
    # Create environment
    env = swift.Swift()
    env.launch(realtime=True)
    
    # Create managers
    stl_manager = STLManager(env)
    array_manager = ObjectArray(stl_manager)
    
    # Get models directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(parent_dir, "models")
    box_path = os.path.join(models_dir, "BoxBase.stl")
    
    if os.path.exists(box_path):
        # Create a 3x3 grid of boxes
        grid = array_manager.create_grid(
            box_path,
            name_prefix="Box",
            rows=3,
            cols=3,
            spacing=0.15,
            start_pos=[0.5, -0.3, 0.8],
            scale=0.001,
            color=(0.2, 0.6, 1.0, 0.7)
        )
        
        # Create a line of boxes
        line = array_manager.create_line(
            box_path,
            name_prefix="LineBox",
            count=5,
            spacing=0.12,
            start_pos=[1.0, -0.3, 0.8],
            direction='y',
            scale=0.001,
            color=(1.0, 0.3, 0.3, 0.7)
        )
        
        # List all objects
        array_manager.list()
        
        # Wait
        for _ in range(50):
            env.step(0.05)
        
        # Demonstrate individual control
        print("\nDemonstrating individual control...")
        box_1_1 = array_manager.get("Box_1_1")  # Center box
        if box_1_1:
            print(f"Moving {box_1_1.name} up...")
            box_1_1.translate([0, 0, 0.2])
        
        for _ in range(30):
            env.step(0.05)
        
        # Demonstrate attachment
        print("\nDemonstrating attachment...")
        box_0_0 = array_manager.get("Box_0_0")
        box_0_1 = array_manager.get("Box_0_1")
        
        if box_0_0 and box_0_1:
            print(f"Attaching {box_0_1.name} to {box_0_0.name}")
            box_0_1.attach_to(box_0_0, offset=[0, 0, 0.1])
            
            print(f"Moving {box_0_0.name} - attached object should follow")
            for i in range(60):
                angle = i * 2 * np.pi / 60
                offset = np.array([0.1 * np.cos(angle), 0.1 * np.sin(angle), 0])
                box_0_0.position = [0.5, -0.3, 0.8] + offset
                env.step(0.03)
        
        print("\n✓ Demo complete!")
    else:
        print(f"Box file not found: {box_path}")
    
    print("\nPress Ctrl+C to exit")
    try:
        while True:
            env.step(0.05)
    except KeyboardInterrupt:
        print("\nExiting...")
        env.close()
