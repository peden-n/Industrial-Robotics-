"""
Interactive Scene Setup Tool
Allows real-time adjustment of table positions with sliders and testing robot reachability.
Displays current position values to copy into config files.
"""
import numpy as np
import time
import os
import json
import tkinter as tk
from tkinter import ttk
import threading
from setup_scene import initialize_environment, update_robot_meshes
from utilities.stl_manager import STLManager
from utilities.object_array import AttachableObject
from ik_controller import solve_ik_with_limits, get_current_end_effector_pos


class InteractiveSceneGUI:
    def __init__(self):
        self.env = None
        self.robot = None
        self.meshes = None
        self.manager = None
        self.boxes = []
        self.lids = []
        self.donuts = []
        self.burnt_donuts = []
        self.box_configs = []
        self.lid_configs = []
        
        # Table references (from environment)
        self.table_long = None
        self.table_short = None
        self.conveyor = None
        self.pallet = None
        
        # Current positions
        self.table_long_pos = [0.0, 0.53, 0.4]
        self.table_short_pos = [0.65, 1.48, 0.4]
        self.conveyor_pos = [0.0, -0.5, 0.48]
        self.pallet_pos = [-0.9325, 1.7625, 0.0]
        self.robot_base_pos = [0.0, 1.18, 0.02]
        
        # Store initial positions for calculating offsets
        self.initial_table_long_pos = self.table_long_pos.copy()
        self.initial_table_short_pos = self.table_short_pos.copy()
        self.initial_box_positions = []
        self.initial_lid_positions = []
        
        # Lock for thread safety
        self.update_lock = threading.Lock()
        self.running = True
        
        # Create GUI
        self.create_gui()
        
    def create_gui(self):
        """Create the interactive GUI window."""
        self.root = tk.Tk()
        self.root.title("Interactive Scene Setup Tool")
        self.root.geometry("800x900")
        
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 1: Table Long
        self.create_table_tab(notebook, "Table Long", self.table_long_pos, self.update_table_long)
        
        # Tab 2: Table Short
        self.create_table_tab(notebook, "Table Short", self.table_short_pos, self.update_table_short)
        
        # Tab 3: Conveyor
        self.create_table_tab(notebook, "Conveyor", self.conveyor_pos, self.update_conveyor)
        
        # Tab 4: Pallet
        self.create_table_tab(notebook, "Pallet", self.pallet_pos, self.update_pallet)
        
        # Tab 5: Robot Base
        self.create_table_tab(notebook, "Robot Base", self.robot_base_pos, self.update_robot_base)
        
        # Tab 6: Robot Testing
        self.create_test_tab(notebook)
        
        # Bottom panel: Status and controls
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(fill='x', padx=10, pady=10)
        
        self.status_label = tk.Label(bottom_frame, text="Status: Initializing...", font=('Arial', 10))
        self.status_label.pack()
        
        # Quit button
        quit_btn = tk.Button(bottom_frame, text="Quit", command=self.quit_app, 
                            bg='red', fg='white', font=('Arial', 12, 'bold'))
        quit_btn.pack(pady=10)
        
    def create_table_tab(self, notebook, name, pos_list, update_callback):
        """Create a tab for adjusting a table's position."""
        tab = tk.Frame(notebook)
        notebook.add(tab, text=name)
        
        # Title
        title = tk.Label(tab, text=f"{name} Position", font=('Arial', 16, 'bold'))
        title.pack(pady=10)
        
        # Current position display
        pos_frame = tk.LabelFrame(tab, text="Current Position", font=('Arial', 12))
        pos_frame.pack(fill='x', padx=20, pady=10)
        
        pos_text = tk.Text(pos_frame, height=3, font=('Courier', 10))
        pos_text.pack(padx=10, pady=10)
        pos_text.insert('1.0', f"X: {pos_list[0]:.4f}\nY: {pos_list[1]:.4f}\nZ: {pos_list[2]:.4f}")
        pos_text.config(state='disabled')
        
        # Store reference for updating
        setattr(self, f'{name.lower().replace(" ", "_")}_pos_text', pos_text)
        
        # Sliders
        slider_frame = tk.LabelFrame(tab, text="Adjust Position (meters)", font=('Arial', 12))
        slider_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # X slider
        x_frame = tk.Frame(slider_frame)
        x_frame.pack(fill='x', padx=10, pady=5)
        tk.Label(x_frame, text="X:", width=3, font=('Arial', 11)).pack(side='left')
        x_scale = tk.Scale(x_frame, from_=-2.0, to=2.0, resolution=0.01, orient='horizontal',
                          command=lambda v: self.on_slider_change(0, float(v), pos_list, update_callback, pos_text))
        x_scale.set(pos_list[0])
        x_scale.pack(side='left', fill='x', expand=True)
        
        # Y slider
        y_frame = tk.Frame(slider_frame)
        y_frame.pack(fill='x', padx=10, pady=5)
        tk.Label(y_frame, text="Y:", width=3, font=('Arial', 11)).pack(side='left')
        y_scale = tk.Scale(y_frame, from_=-2.0, to=2.0, resolution=0.01, orient='horizontal',
                          command=lambda v: self.on_slider_change(1, float(v), pos_list, update_callback, pos_text))
        y_scale.set(pos_list[1])
        y_scale.pack(side='left', fill='x', expand=True)
        
        # Z slider
        z_frame = tk.Frame(slider_frame)
        z_frame.pack(fill='x', padx=10, pady=5)
        tk.Label(z_frame, text="Z:", width=3, font=('Arial', 11)).pack(side='left')
        z_scale = tk.Scale(z_frame, from_=0.0, to=2.0, resolution=0.01, orient='horizontal',
                          command=lambda v: self.on_slider_change(2, float(v), pos_list, update_callback, pos_text))
        z_scale.set(pos_list[2])
        z_scale.pack(side='left', fill='x', expand=True)
        
        # Copy to clipboard button
        copy_btn = tk.Button(tab, text="Copy Position to Clipboard", 
                            command=lambda: self.copy_position(pos_list),
                            font=('Arial', 11))
        copy_btn.pack(pady=10)
        
    def create_test_tab(self, notebook):
        """Create tab for robot reachability testing."""
        tab = tk.Frame(notebook)
        notebook.add(tab, text="Robot Test")
        
        # Title
        title = tk.Label(tab, text="Robot Reachability Test", font=('Arial', 16, 'bold'))
        title.pack(pady=10)
        
        # Instructions
        instructions = tk.Label(tab, text="Test if robot can reach all box, lid, and crate positions.\n" +
                                         "Green = Reachable, Red = Unreachable",
                               font=('Arial', 10), justify='left')
        instructions.pack(pady=10)
        
        # Test all button
        test_all_btn = tk.Button(tab, text="Test All Positions (Math Only)", 
                                command=self.test_all_positions,
                                bg='blue', fg='white', font=('Arial', 14, 'bold'),
                                width=25, height=2)
        test_all_btn.pack(pady=10)
        
        # Visualize reachable button
        visualize_btn = tk.Button(tab, text="Show Reachable Positions (Visual)", 
                                 command=self.visualize_reachable,
                                 bg='green', fg='white', font=('Arial', 14, 'bold'),
                                 width=25, height=2)
        visualize_btn.pack(pady=10)
        
        # Individual test buttons frame
        individual_frame = tk.LabelFrame(tab, text="Test Individual Positions (with Robot Movement)", font=('Arial', 12))
        individual_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Boxes
        box_frame = tk.Frame(individual_frame)
        box_frame.pack(fill='x', pady=5)
        tk.Label(box_frame, text="Boxes:", font=('Arial', 11, 'bold')).pack(side='left', padx=10)
        for i in range(6):
            btn = tk.Button(box_frame, text=f"Box {i+1}", 
                           command=lambda idx=i: self.test_single_position('box', idx),
                           width=8)
            btn.pack(side='left', padx=2)
        
        # Lids
        lid_frame = tk.Frame(individual_frame)
        lid_frame.pack(fill='x', pady=5)
        tk.Label(lid_frame, text="Lids:", font=('Arial', 11, 'bold')).pack(side='left', padx=10)
        for i in range(6):
            btn = tk.Button(lid_frame, text=f"Lid {i+1}", 
                           command=lambda idx=i: self.test_single_position('lid', idx),
                           width=8)
            btn.pack(side='left', padx=2)
        
        # End positions (crate)
        end_frame = tk.Frame(individual_frame)
        end_frame.pack(fill='x', pady=5)
        tk.Label(end_frame, text="Crate:", font=('Arial', 11, 'bold')).pack(side='left', padx=10)
        for i in range(6):
            btn = tk.Button(end_frame, text=f"Pos {i+1}", 
                           command=lambda idx=i: self.test_single_position('end', idx),
                           width=8)
            btn.pack(side='left', padx=2)
        
        # Results display
        results_frame = tk.LabelFrame(tab, text="Test Results", font=('Arial', 12))
        results_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.results_text = tk.Text(results_frame, height=10, font=('Courier', 9))
        self.results_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Add scrollbar
        scrollbar = tk.Scrollbar(self.results_text)
        scrollbar.pack(side='right', fill='y')
        self.results_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.results_text.yview)
        
    def on_slider_change(self, axis, value, pos_list, update_callback, text_widget):
        """Handle slider value changes."""
        pos_list[axis] = value
        
        # Update text display
        text_widget.config(state='normal')
        text_widget.delete('1.0', 'end')
        text_widget.insert('1.0', f"X: {pos_list[0]:.4f}\nY: {pos_list[1]:.4f}\nZ: {pos_list[2]:.4f}")
        text_widget.config(state='disabled')
        
        # Update scene (call the update callback)
        if self.env is not None and update_callback is not None:
            try:
                update_callback()
                # Force a step to show the update
                self.env.step(0.01)
            except Exception as e:
                print(f"Error updating scene: {e}")
            
    def copy_position(self, pos_list):
        """Copy position to clipboard."""
        pos_str = f'"position": [{pos_list[0]:.4f}, {pos_list[1]:.4f}, {pos_list[2]:.4f}]'
        self.root.clipboard_clear()
        self.root.clipboard_append(pos_str)
        self.update_status(f"Copied: {pos_str}")
        
    def update_table_long(self):
        """Update TableLong position and all boxes on it."""
        if self.table_long is not None:
            from spatialmath import SE3
            self.table_long.T = SE3(*self.table_long_pos).A
            print(f"Table Long moved to: {self.table_long_pos}")
            
        # Calculate offset from initial position
        offset = np.array(self.table_long_pos) - np.array(self.initial_table_long_pos)
        
        # Move all boxes with the table
        for i, box in enumerate(self.boxes):
            if i < len(self.initial_box_positions):
                new_pos = self.initial_box_positions[i] + offset
                box.position = new_pos
                print(f"  Box {i+1} moved to: {new_pos}")
        
    def update_table_short(self):
        """Update TableShort position and all lids on it."""
        if self.table_short is not None:
            from spatialmath import SE3
            self.table_short.T = SE3(*self.table_short_pos).A
            print(f"Table Short moved to: {self.table_short_pos}")
            
        # Calculate offset from initial position
        offset = np.array(self.table_short_pos) - np.array(self.initial_table_short_pos)
        
        # Move all lids with the table
        for i, lid in enumerate(self.lids):
            if i < len(self.initial_lid_positions):
                new_pos = self.initial_lid_positions[i] + offset
                lid.position = new_pos
                print(f"  Lid {i+1} moved to: {new_pos}")
        
    def update_conveyor(self):
        """Update Conveyor position."""
        if self.conveyor is not None:
            from spatialmath import SE3
            self.conveyor.T = SE3(*self.conveyor_pos).A
            print(f"Conveyor moved to: {self.conveyor_pos}")
            
    def update_pallet(self):
        """Update Pallet position and objects on it."""
        if self.pallet is not None:
            from spatialmath import SE3
            self.pallet.T = SE3(*self.pallet_pos).A
            print(f"Pallet moved to: {self.pallet_pos}")
    
    def update_robot_base(self):
        """Update robot base position and all robot meshes."""
        if self.robot is not None and self.meshes is not None:
            print(f"Robot base moved to: {self.robot_base_pos}")
            # Update all robot link meshes to new base position
            update_robot_meshes(self.robot, self.meshes, base_xyz=tuple(self.robot_base_pos))
            
            # Also need to update the robot's actual forward kinematics
            # This is handled by update_robot_meshes which uses the base_xyz parameter
        
    def test_all_positions(self):
        """Test robot reachability for all positions with color coding (math only, fast)."""
        if self.robot is None:
            self.log_result("Robot not initialized!")
            return
            
        self.log_result("\n" + "="*70)
        self.log_result("TESTING ALL POSITIONS (MATH ONLY - FAST)")
        self.log_result("="*70 + "\n")
        
        approach_height = 0.15
        total = 0
        reachable = 0
        
        # Get current table offsets
        table_long_offset = np.array(self.table_long_pos) - np.array(self.initial_table_long_pos)
        table_short_offset = np.array(self.table_short_pos) - np.array(self.initial_table_short_pos)
        
        self.log_result(f"Current Configuration:")
        self.log_result(f"  Table Long offset: {np.round(table_long_offset, 3)}")
        self.log_result(f"  Table Short offset: {np.round(table_short_offset, 3)}")
        self.log_result(f"  Robot base: {np.round(self.robot_base_pos, 3)}\n")
        
        reachability_results = []
        
        # Test lids (on short table)
        self.log_result("Testing Lids:")
        for i, lid_config in enumerate(self.lid_configs):
            lid_start = np.array(lid_config['start']) + table_short_offset
            touch_pos = lid_start + np.array([0, 0, approach_height])
            
            success = self.test_reach(touch_pos, visualize=False)
            reachability_results.append(('lid', i, success, touch_pos))
            
            # Color code the lid if it exists
            if i < len(self.lids):
                try:
                    if success:
                        self.lids[i].color = (0.0, 1.0, 0.0, 1.0)  # Green
                    else:
                        self.lids[i].color = (1.0, 0.0, 0.0, 1.0)  # Red
                except Exception as e:
                    self.log_result(f"  ⚠ Could not color lid {i+1}: {e}")
            
            status = "REACHABLE" if success else "UNREACHABLE"
            self.log_result(f"  Lid {i+1} at {np.round(lid_start, 3)}: {status}")
            
            total += 1
            if success:
                reachable += 1
                
        # Test boxes (on long table)
        self.log_result("\nTesting Boxes:")
        for i, box_config in enumerate(self.box_configs):
            box_start = np.array(box_config['start']) + table_long_offset
            touch_pos = box_start + np.array([0, 0, approach_height])
            
            success = self.test_reach(touch_pos, visualize=False)
            reachability_results.append(('box', i, success, touch_pos))
            
            # Color code the box if it exists
            if i < len(self.boxes):
                try:
                    if success:
                        self.boxes[i].color = (0.0, 1.0, 0.0, 1.0)  # Green
                    else:
                        self.boxes[i].color = (1.0, 0.0, 0.0, 1.0)  # Red
                except Exception as e:
                    self.log_result(f"  ⚠ Could not color box {i+1}: {e}")
            
            status = "REACHABLE" if success else "UNREACHABLE"
            self.log_result(f"  Box {i+1} at {np.round(box_start, 3)}: {status}")
            
            total += 1
            if success:
                reachable += 1
        
        # Force scene refresh after all color updates
        if self.env:
            try:
                for _ in range(10):
                    self.env.step(0.02)
            except:
                pass
                
        # Test end positions (near pallet - not affected by table movement)
        self.log_result("\nTesting End Positions (Crate):")
        for i, box_config in enumerate(self.box_configs):
            end_pos = np.array(box_config['end'])
            touch_pos = end_pos + np.array([0, 0, approach_height])
            
            success = self.test_reach(touch_pos, visualize=False)
            reachability_results.append(('end', i, success, touch_pos))
            
            status = "REACHABLE" if success else "UNREACHABLE"
            self.log_result(f"  End Pos {i+1} at {np.round(end_pos, 3)}: {status}")
            
            total += 1
            if success:
                reachable += 1
                
        self.log_result("\n" + "="*70)
        self.log_result(f"RESULTS: {reachable}/{total} positions reachable ({100*reachable/total:.1f}%)")
        self.log_result("="*70)
        self.log_result("\n Green objects in 3D = Reachable")
        self.log_result(" Red objects in 3D = Unreachable")
        self.log_result("\nUse 'Show Reachable Positions' to visualize robot movement\n")
        
        # Store results for visualization
        self.reachability_results = reachability_results
    
    def visualize_reachable(self):
        """Visualize robot moving to each reachable position."""
        if not hasattr(self, 'reachability_results'):
            self.log_result("Run 'Test All Positions' first!")
            return
            
        if self.robot is None:
            self.log_result("Robot not initialized!")
            return
        
        if self.env is None:
            self.log_result("Environment not initialized!")
            return
            
        self.log_result("\n" + "="*70)
        self.log_result("VISUALIZING REACHABLE POSITIONS")
        self.log_result("="*70 + "\n")
        
        reachable_count = sum(1 for _, _, success, _ in self.reachability_results if success)
        self.log_result(f"Found {reachable_count} reachable positions to visualize\n")
        
        approach_height = 0.15
        count = 0
        
        try:
            for obj_type, index, success, touch_pos in self.reachability_results:
                if not success:
                    continue  # Skip unreachable positions
                    
                if obj_type == 'lid':
                    name = f"Lid {index+1}"
                elif obj_type == 'box':
                    name = f"Box {index+1}"
                else:
                    name = f"End Pos {index+1}"
                    
                count += 1
                self.log_result(f"{count}/{reachable_count}: Moving to {name}...")
                
                try:
                    self.test_reach(touch_pos, visualize=True)
                    self.log_result(f"  ✓ Reached {name}")
                except Exception as e:
                    self.log_result(f"  ✗ Error reaching {name}: {e}")
                    
                time.sleep(0.5)  # Pause between movements
                
            self.log_result(f"\n Visualization complete! Showed {count} positions.\n")
        except Exception as e:
            self.log_result(f"\n Visualization error: {e}\n")
        
    def test_single_position(self, obj_type, index):
        """Test a single position with visual movement and color coding."""
        if self.robot is None:
            self.log_result(" Robot not initialized!")
            return
            
        approach_height = 0.15
        obj = None
        
        try:
            if obj_type == 'box':
                if index >= len(self.box_configs):
                    self.log_result(f" Box {index+1} config not found!")
                    return
                if index >= len(self.boxes):
                    self.log_result(f" Box {index+1} object not loaded!")
                    return
                pos = np.array(self.box_configs[index]['start'])
                # Apply current table offset
                offset = np.array(self.table_long_pos) - np.array(self.initial_table_long_pos)
                pos = pos + offset
                obj = self.boxes[index]
                name = f"Box {index+1}"
            elif obj_type == 'lid':
                if index >= len(self.lid_configs):
                    self.log_result(f" Lid {index+1} config not found!")
                    return
                if index >= len(self.lids):
                    self.log_result(f" Lid {index+1} object not loaded!")
                    return
                pos = np.array(self.lid_configs[index]['start'])
                # Apply current table offset
                offset = np.array(self.table_short_pos) - np.array(self.initial_table_short_pos)
                pos = pos + offset
                obj = self.lids[index]
                name = f"Lid {index+1}"
            elif obj_type == 'end':
                if index >= len(self.box_configs):
                    self.log_result(f" End position {index+1} config not found!")
                    return
                pos = np.array(self.box_configs[index]['end'])
                name = f"End Position {index+1}"
            else:
                return
                
            touch_pos = pos + np.array([0, 0, approach_height])
            
            self.log_result(f"\nTesting {name}...")
            self.log_result(f"  Object position: {np.round(pos, 3)}")
            self.log_result(f"  Touch position (above): {np.round(touch_pos, 3)}")
            
            # Get offset info for debugging
            if obj_type == 'box':
                offset = np.array(self.table_long_pos) - np.array(self.initial_table_long_pos)
                self.log_result(f"  Table Long offset: {np.round(offset, 3)}")
            elif obj_type == 'lid':
                offset = np.array(self.table_short_pos) - np.array(self.initial_table_short_pos)
                self.log_result(f"  Table Short offset: {np.round(offset, 3)}")
            
            # Test with visualization
            success = self.test_reach(touch_pos, visualize=True)
            
            # Color code the object
            if obj is not None:
                try:
                    if success:
                        obj.color = (0.0, 1.0, 0.0, 1.0)  # Green = reachable
                        # Force scene update
                        if self.env:
                            self.env.step(0.01)
                        status = "✅ REACHABLE (colored GREEN)"
                    else:
                        obj.color = (1.0, 0.0, 0.0, 1.0)  # Red = unreachable
                        # Force scene update
                        if self.env:
                            self.env.step(0.01)
                        status = " UNREACHABLE (colored RED)"
                except Exception as e:
                    status = " REACHABLE" if success else " UNREACHABLE"
                    self.log_result(f"  ⚠ Could not update color: {e}")
            else:
                status = " REACHABLE" if success else " UNREACHABLE"
            
            self.log_result(f"{name}: {status}\n")
            
        except Exception as e:
            self.log_result(f" Error testing position: {e}\n")
        
    def test_reach(self, target_pos, visualize=False):
        """Test if robot can reach a position using IK (math only by default)."""
        try:
            q_target, success = solve_ik_with_limits(self.robot, target_pos)
            
            if visualize and success and q_target is not None:
                # Smoothly move robot to show the position
                q_start = self.robot.q.copy()
                steps = 50
                
                for i in range(steps):
                    alpha = (i + 1) / steps
                    self.robot.q = q_start * (1 - alpha) + q_target * alpha
                    update_robot_meshes(self.robot, self.meshes, base_xyz=tuple(self.robot_base_pos))
                    if self.env is not None:
                        self.env.step(0.02)
                        time.sleep(0.02)
            elif visualize and not success:
                self.log_result("   IK failed, cannot visualize position")
                    
            return success
        except Exception as e:
            self.log_result(f"   Error testing position: {e}")
            return False
        
    def log_result(self, message):
        """Add message to results text widget."""
        if hasattr(self, 'results_text'):
            self.results_text.insert('end', message + '\n')
            self.results_text.see('end')
            self.root.update()
            
    def update_status(self, message):
        """Update status label."""
        self.status_label.config(text=f"Status: {message}")
        self.root.update()
        
    def initialize_scene(self):
        """Initialize the 3D scene and load all objects."""
        try:
            self.update_status("Initializing environment...")
            print("Initializing environment...")
            
            # Initialize environment
            self.env, self.robot, self.meshes, scene_objects = initialize_environment()
            
            # Store table references
            self.table_long = scene_objects.get('TableLong')
            self.table_short = scene_objects.get('TableShort')
            self.conveyor = scene_objects.get('Conveyor')
            self.pallet = scene_objects.get('Pallet')
            
            print(f"Scene objects loaded: {list(scene_objects.keys())}")
            
            # Create STL manager
            self.manager = STLManager(self.env)
            
            # Get directories
            script_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(script_dir, "models")
            configs_dir = os.path.join(script_dir, "configurations")
            
            # Load configs
            with open(os.path.join(configs_dir, "box_config.json"), 'r') as f:
                self.box_configs = json.load(f)
            with open(os.path.join(configs_dir, "lid_config.json"), 'r') as f:
                self.lid_configs = json.load(f)
                
            # Load parts
            self.update_status("Loading parts...")
            from main import load_parts_from_config
            
            self.boxes = load_parts_from_config(self.manager, models_dir, configs_dir, 
                                               "box_config.json", "BoxBase.stl")
            self.lids = load_parts_from_config(self.manager, models_dir, configs_dir,
                                              "lid_config.json", "BoxLid.stl")
            
            # Store initial positions for offset calculations
            for box in self.boxes:
                self.initial_box_positions.append(np.array(box.position))
            for lid in self.lids:
                self.initial_lid_positions.append(np.array(lid.position))
            
            print(f"Loaded {len(self.boxes)} boxes and {len(self.lids)} lids")
            print(f"Initial box positions: {self.initial_box_positions}")
            print(f"Initial lid positions: {self.initial_lid_positions}")
            
            self.donuts = load_parts_from_config(self.manager, models_dir, configs_dir,
                                               "donut_config.json", "Donut.dae")
            self.burnt_donuts = load_parts_from_config(self.manager, models_dir, configs_dir,
                                                       "burntDonut_config.json", "Donut_burnt.dae")
            
            # Get references to tables (would need to modify setup_scene to return these)
            # For now, we'll store their positions and update via config
            
            self.update_status("Ready!")
            print("Initialization complete!")
            
        except Exception as e:
            print(f"Initialization error: {e}")
            import traceback
            traceback.print_exc()
            self.update_status(f"Error: {e}")
        
    def run_simulation(self):
        """Run the Swift simulation loop in background thread."""
        print("Starting simulation loop...")
        while self.running:
            if self.env is not None:
                try:
                    self.env.step(0.05)
                except Exception as e:
                    print(f"Simulation step error: {e}")
            time.sleep(0.05)
            
    def quit_app(self):
        """Clean shutdown."""
        self.running = False
        try:
            if self.env is not None:
                self.env.close()
        except:
            pass
        self.root.quit()
        self.root.destroy()
        
    def run(self):
        """Start the application."""
        # Initialize scene in background
        init_thread = threading.Thread(target=self.initialize_scene, daemon=True)
        init_thread.start()
        
        # Start simulation loop
        sim_thread = threading.Thread(target=self.run_simulation, daemon=True)
        sim_thread.start()
        
        # Run GUI
        self.root.mainloop()


if __name__ == '__main__':
    app = InteractiveSceneGUI()
    app.run()
