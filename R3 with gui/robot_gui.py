"""
Robot Control GUI - Emergency controls and system management interface.
Provides emergency stop, reset controls, and environment management.
"""
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import numpy as np


class RobotControlGUI:
    """GUI for robot emergency controls and system management."""
    
    def __init__(self, controller=None, scene_data=None, assembly_function=None):
        """
        Initialize the robot control GUI.
        
        Args:
            controller: RobotController instance
            scene_data: Scene data dictionary from initialize_complete_scene()
            assembly_function: Function to call when continuing assembly (optional)
        """
        self.controller = controller
        self.scene_data = scene_data
        self.assembly_function = assembly_function
        
        # Control states
        self.emergency_stopped = False
        self.system_paused = False
        self.assembly_running = False
        
        # Create the GUI window
        self.create_gui()
        
    def create_gui(self):
        """Create the main GUI window and controls."""
        self.root = tk.Tk()
        self.root.title("Robot Control Panel")
        self.root.geometry("400x500")
        self.root.configure(bg='#2b2b2b')
        
        # Configure style for better appearance
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main title
        title_label = tk.Label(
            self.root, 
            text="🤖 ROBOT CONTROL PANEL", 
            font=('Arial', 16, 'bold'),
            bg='#2b2b2b', 
            fg='white'
        )
        title_label.pack(pady=10)
        
        # Status display
        self.create_status_section()
        
        # Emergency controls
        self.create_emergency_section()
        
        # Reset controls
        self.create_reset_section()
        
        # Environment controls
        self.create_environment_section()
        
        # System controls
        self.create_system_section()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("System Ready")
        status_bar = tk.Label(
            self.root, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN, 
            anchor=tk.W,
            bg='#404040',
            fg='white'
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Make window stay on top
        self.root.attributes('-topmost', True)
        
    def create_status_section(self):
        """Create the status display section."""
        status_frame = tk.LabelFrame(
            self.root, 
            text="System Status", 
            font=('Arial', 10, 'bold'),
            bg='#2b2b2b',
            fg='white'
        )
        status_frame.pack(pady=5, padx=10, fill='x')
        
        self.status_labels = {}
        
        # Emergency status
        self.status_labels['emergency'] = tk.Label(
            status_frame, 
            text=" System Normal", 
            font=('Arial', 10),
            bg='#2b2b2b',
            fg='green'
        )
        self.status_labels['emergency'].pack(pady=2)
        
        # Assembly status
        self.status_labels['assembly'] = tk.Label(
            status_frame, 
            text=" Assembly Stopped", 
            font=('Arial', 10),
            bg='#2b2b2b',
            fg='orange'
        )
        self.status_labels['assembly'].pack(pady=2)
        
        # Robot position status
        self.status_labels['position'] = tk.Label(
            status_frame, 
            text="📍 Position: Unknown", 
            font=('Arial', 10),
            bg='#2b2b2b',
            fg='white'
        )
        self.status_labels['position'].pack(pady=2)
        
    def create_emergency_section(self):
        """Create the emergency control section."""
        emergency_frame = tk.LabelFrame(
            self.root, 
            text="Emergency Controls", 
            font=('Arial', 10, 'bold'),
            bg='#2b2b2b',
            fg='white'
        )
        emergency_frame.pack(pady=5, padx=10, fill='x')
        
        # Emergency Stop button
        self.emergency_button = tk.Button(
            emergency_frame,
            text=" EMERGENCY STOP",
            font=('Arial', 12, 'bold'),
            bg='red',
            fg='white',
            activebackground='darkred',
            command=self.emergency_stop,
            height=2
        )
        self.emergency_button.pack(pady=5, fill='x')
        
        # Reset Emergency Stop button
        self.reset_emergency_button = tk.Button(
            emergency_frame,
            text=" Reset Emergency Stop",
            font=('Arial', 10),
            bg='green',
            fg='white',
            activebackground='darkgreen',
            command=self.reset_emergency_stop,
            state='disabled'
        )
        self.reset_emergency_button.pack(pady=2, fill='x')
        
        # Continue button
        self.continue_button = tk.Button(
            emergency_frame,
            text=" Continue Assembly",
            font=('Arial', 10),
            bg='blue',
            fg='white',
            activebackground='darkblue',
            command=self.continue_assembly,
            state='disabled'
        )
        self.continue_button.pack(pady=2, fill='x')
        
    def create_reset_section(self):
        """Create the reset control section."""
        reset_frame = tk.LabelFrame(
            self.root, 
            text="Robot Reset Controls", 
            font=('Arial', 10, 'bold'),
            bg='#2b2b2b',
            fg='white'
        )
        reset_frame.pack(pady=5, padx=10, fill='x')
        
        # Full Reset button
        self.full_reset_button = tk.Button(
            reset_frame,
            text=" Full Reset (Home Position)",
            font=('Arial', 10),
            bg='orange',
            fg='white',
            activebackground='darkorange',
            command=self.full_reset
        )
        self.full_reset_button.pack(pady=2, fill='x')
        
        # Hard Reset button
        self.hard_reset_button = tk.Button(
            reset_frame,
            text=" Hard Reset (Full System)",
            font=('Arial', 10),
            bg='purple',
            fg='white',
            activebackground='darkviolet',
            command=self.hard_reset
        )
        self.hard_reset_button.pack(pady=2, fill='x')
        
    def create_environment_section(self):
        """Create the environment control section."""
        env_frame = tk.LabelFrame(
            self.root, 
            text="Environment Controls", 
            font=('Arial', 10, 'bold'),
            bg='#2b2b2b',
            fg='white'
        )
        env_frame.pack(pady=5, padx=10, fill='x')
        
        # Environment Reset button
        self.env_reset_button = tk.Button(
            env_frame,
            text=" Reset Environment (Objects)",
            font=('Arial', 10),
            bg='teal',
            fg='white',
            activebackground='darkcyan',
            command=self.reset_environment
        )
        self.env_reset_button.pack(pady=2, fill='x')
        
    def create_system_section(self):
        """Create the system control section."""
        system_frame = tk.LabelFrame(
            self.root, 
            text="System Controls", 
            font=('Arial', 10, 'bold'),
            bg='#2b2b2b',
            fg='white'
        )
        system_frame.pack(pady=5, padx=10, fill='x')
        
        # Start Assembly button
        self.start_assembly_button = tk.Button(
            system_frame,
            text=" Start Assembly",
            font=('Arial', 10),
            bg='darkgreen',
            fg='white',
            activebackground='green',
            command=self.start_assembly
        )
        self.start_assembly_button.pack(pady=2, fill='x')
        
        # Pause Assembly button
        self.pause_assembly_button = tk.Button(
            system_frame,
            text=" Pause Assembly",
            font=('Arial', 10),
            bg='goldenrod',
            fg='white',
            activebackground='orange',
            command=self.pause_assembly,
            state='disabled'
        )
        self.pause_assembly_button.pack(pady=2, fill='x')
        
    def emergency_stop(self):
        """Execute emergency stop - halt all robot motion immediately."""
        self.emergency_stopped = True
        self.assembly_running = False
        self.system_paused = True
        
        print(" EMERGENCY STOP ACTIVATED!")
        self.update_status(" EMERGENCY STOP - All motion halted")
        
        # Update GUI
        self.status_labels['emergency'].config(text=" EMERGENCY STOP", fg='red')
        self.status_labels['assembly'].config(text=" Assembly Halted", fg='red')
        
        self.emergency_button.config(state='disabled')
        self.reset_emergency_button.config(state='normal')
        self.start_assembly_button.config(state='disabled')
        self.pause_assembly_button.config(state='disabled')
        
        # Stop robot immediately if controller available
        if self.controller:
            try:
                self.controller.emergency_stop()
                print("Controller emergency stop activated - all motion will halt")
            except Exception as e:
                print(f"Warning: Could not execute emergency stop on controller: {e}")
        
        messagebox.showwarning(
            "Emergency Stop", 
            "EMERGENCY STOP ACTIVATED!\n\nAll robot motion has been halted.\nPress 'Reset Emergency Stop' to resume control."
        )
    
    def reset_emergency_stop(self):
        """Reset the emergency stop condition."""
        self.emergency_stopped = False
        
        # Reset controller's emergency stop flag
        if self.controller:
            try:
                self.controller.reset_emergency_stop()
            except Exception as e:
                print(f"Warning: Could not reset controller emergency stop: {e}")
        
        print(" Emergency stop reset")
        self.update_status(" Emergency stop reset - System ready")
        
        # Update GUI
        self.status_labels['emergency'].config(text=" System Normal", fg='green')
        self.status_labels['assembly'].config(text="⏸ Assembly Stopped", fg='orange')
        
        self.emergency_button.config(state='normal')
        self.reset_emergency_button.config(state='disabled')
        self.continue_button.config(state='normal')
        self.start_assembly_button.config(state='normal')
        
        messagebox.showinfo("Emergency Stop Reset", "Emergency stop has been reset.\nSystem is ready for operation.")
    
    def continue_assembly(self):
        """Continue assembly after emergency stop or reset."""
        if self.emergency_stopped:
            messagebox.showwarning("Cannot Continue", "Please reset emergency stop first.")
            return
        
        self.system_paused = False
        
        print(" Continuing assembly...")
        self.update_status(" Continuing assembly process")
        
        # Update GUI
        self.status_labels['assembly'].config(text=" Assembly Running", fg='green')
        self.continue_button.config(state='disabled')
        self.start_assembly_button.config(state='disabled')
        self.pause_assembly_button.config(state='normal')
        
        # Continue assembly if function provided
        if self.assembly_function:
            try:
                threading.Thread(target=self.assembly_function, daemon=True).start()
            except Exception as e:
                print(f"Error continuing assembly: {e}")
                messagebox.showerror("Assembly Error", f"Could not continue assembly: {e}")
    
    def full_reset(self):
        """Reset robot to home position."""
        if self.emergency_stopped:
            messagebox.showwarning("Cannot Reset", "Please reset emergency stop first.")
            return
        
        print(" Executing full reset - moving to home position...")
        self.update_status(" Moving robot to home position...")
        
        # Update GUI
        self.system_paused = True
        self.assembly_running = False
        self.status_labels['assembly'].config(text=" Resetting to Home", fg='blue')
        
        # Execute reset in separate thread to avoid blocking GUI
        def reset_thread():
            try:
                if self.controller:
                    # Clear any attached objects
                    self.controller.attached_objects.clear()
                    if hasattr(self.controller, 'object_offsets'):
                        self.controller.object_offsets.clear()
                    
                    # Move to home position
                    success = self.controller.move_to_home_position()
                    
                    if success:
                        self.update_status(" Robot reset to home position")
                        self.status_labels['assembly'].config(text=" Assembly Stopped", fg='orange')
                        self.continue_button.config(state='normal')
                        messagebox.showinfo("Reset Complete", "Robot has been reset to home position.")
                    else:
                        self.update_status(" Failed to reset robot position")
                        messagebox.showerror("Reset Failed", "Could not move robot to home position.")
                else:
                    messagebox.showwarning("No Controller", "No robot controller available for reset.")
            except Exception as e:
                print(f"Error during full reset: {e}")
                self.update_status(" Reset failed")
                messagebox.showerror("Reset Error", f"Reset failed: {e}")
        
        threading.Thread(target=reset_thread, daemon=True).start()
    
    def hard_reset(self):
        """Perform hard reset of entire robot system."""
        if self.emergency_stopped:
            messagebox.showwarning("Cannot Reset", "Please reset emergency stop first.")
            return
        
        result = messagebox.askyesno(
            "Hard Reset Confirmation", 
            "This will completely reset the robot system.\nAll current progress will be lost.\n\nAre you sure?"
        )
        
        if not result:
            return
        
        print(" Executing hard reset - full system reset...")
        self.update_status(" Performing hard system reset...")
        
        # Update GUI
        self.system_paused = True
        self.assembly_running = False
        self.status_labels['assembly'].config(text=" Hard Resetting", fg='purple')
        
        def hard_reset_thread():
            try:
                if self.controller:
                    # Clear all attachments and states
                    self.controller.attached_objects.clear()
                    if hasattr(self.controller, 'object_offsets'):
                        self.controller.object_offsets.clear()
                    
                    # Reset robot to initial configuration
                    home_angles = [0.0, -30.0, 45.0, 0.0, 10.0, 0.0]
                    success = self.controller.move_joints(home_angles, steps=60, degrees=True)
                    
                    if success:
                        self.update_status(" Hard reset complete")
                        self.status_labels['assembly'].config(text=" Assembly Stopped", fg='orange')
                        self.continue_button.config(state='normal')
                        messagebox.showinfo("Hard Reset Complete", "System has been completely reset.")
                    else:
                        self.update_status(" Hard reset failed")
                        messagebox.showerror("Hard Reset Failed", "Could not complete hard reset.")
                else:
                    messagebox.showwarning("No Controller", "No robot controller available for reset.")
            except Exception as e:
                print(f"Error during hard reset: {e}")
                self.update_status(" Hard reset failed")
                messagebox.showerror("Hard Reset Error", f"Hard reset failed: {e}")
        
        threading.Thread(target=hard_reset_thread, daemon=True).start()
    
    def reset_environment(self):
        """Reset environment objects (boxes and lids) to starting positions."""
        result = messagebox.askyesno(
            "Environment Reset", 
            "This will reset all boxes and lids to their starting positions.\n\nContinue?"
        )
        
        if not result:
            return
        
        print(" Resetting environment objects...")
        self.update_status(" Resetting boxes and lids to start positions...")
        
        try:
            if self.scene_data and 'objects' in self.scene_data and 'configs' in self.scene_data:
                # Reset boxes to start positions
                if 'boxes' in self.scene_data['objects'] and 'box' in self.scene_data['configs']:
                    boxes = self.scene_data['objects']['boxes']
                    box_configs = self.scene_data['configs']['box']
                    
                    for i, (box, config) in enumerate(zip(boxes, box_configs)):
                        if 'start' in config:
                            box.position = config['start']
                            print(f"Reset box_{i+1} to {config['start']}")
                
                # Reset lids to start positions
                if 'lids' in self.scene_data['objects'] and 'lid' in self.scene_data['configs']:
                    lids = self.scene_data['objects']['lids']
                    lid_configs = self.scene_data['configs']['lid']
                    
                    for i, (lid, config) in enumerate(zip(lids, lid_configs)):
                        if 'start' in config:
                            lid.position = config['start']
                            print(f"Reset lid_{i+1} to {config['start']}")
                
                # Clear any attached objects from controller
                if self.controller:
                    self.controller.attached_objects.clear()
                    if hasattr(self.controller, 'object_offsets'):
                        self.controller.object_offsets.clear()
                
                self.update_status(" Environment reset complete")
                messagebox.showinfo("Environment Reset", "All objects have been reset to starting positions.")
                
            else:
                messagebox.showwarning("No Scene Data", "No scene data available for environment reset.")
                
        except Exception as e:
            print(f"Error during environment reset: {e}")
            self.update_status(" Environment reset failed")
            messagebox.showerror("Environment Reset Error", f"Environment reset failed: {e}")
    
    def start_assembly(self):
        """Start the assembly process."""
        if self.emergency_stopped:
            messagebox.showwarning("Cannot Start", "Please reset emergency stop first.")
            return
        
        self.assembly_running = True
        self.system_paused = False
        
        print(" Starting assembly process...")
        self.update_status(" Starting assembly process...")
        
        # Update GUI
        self.status_labels['assembly'].config(text=" Assembly Running", fg='green')
        self.start_assembly_button.config(state='disabled')
        self.pause_assembly_button.config(state='normal')
        self.continue_button.config(state='disabled')
        
        # Start assembly if function provided
        if self.assembly_function:
            try:
                threading.Thread(target=self.assembly_function, daemon=True).start()
            except Exception as e:
                print(f"Error starting assembly: {e}")
                messagebox.showerror("Assembly Error", f"Could not start assembly: {e}")
        else:
            messagebox.showinfo("No Assembly Function", "No assembly function has been configured.")
    
    def pause_assembly(self):
        """Pause the assembly process."""
        self.system_paused = True
        self.assembly_running = False
        
        print(" Pausing assembly...")
        self.update_status(" Assembly paused")
        
        # Update GUI
        self.status_labels['assembly'].config(text=" Assembly Paused", fg='orange')
        self.start_assembly_button.config(state='normal')
        self.pause_assembly_button.config(state='disabled')
        self.continue_button.config(state='normal')
    
    def update_status(self, message):
        """Update the status bar message."""
        self.status_var.set(message)
        
        # Update position if controller available
        if self.controller:
            try:
                pos = self.controller.get_current_position()
                pos_str = f" Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]"
                self.status_labels['position'].config(text=pos_str)
            except Exception:
                pass
    
    def is_emergency_stopped(self):
        """Check if system is in emergency stop state."""
        return self.emergency_stopped
    
    def is_paused(self):
        """Check if system is paused."""
        return self.system_paused
    
    def is_running(self):
        """Check if assembly is currently running."""
        return self.assembly_running and not self.system_paused and not self.emergency_stopped
    
    def run(self):
        """Start the GUI main loop."""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\nGUI interrupted by user")
            self.close()
    
    def close(self):
        """Close the GUI window."""
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass
    
    def update_loop(self):
        """Update loop for real-time status updates (call this periodically)."""
        try:
            # Update position display
            if self.controller and not self.emergency_stopped:
                pos = self.controller.get_current_position()
                pos_str = f" Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]"
                self.status_labels['position'].config(text=pos_str)
        except Exception:
            pass
        
        # Schedule next update
        if hasattr(self, 'root'):
            self.root.after(1000, self.update_loop)  # Update every second


# Convenience function to create and run GUI
def create_robot_gui(controller=None, scene_data=None, assembly_function=None):
    """
    Create and return a robot control GUI.
    
    Args:
        controller: RobotController instance
        scene_data: Scene data dictionary
        assembly_function: Function to call for assembly operations
    
    Returns:
        RobotControlGUI instance
    """
    gui = RobotControlGUI(controller, scene_data, assembly_function)
    return gui


if __name__ == '__main__':
    # Test the GUI standalone
    print("Testing Robot Control GUI...")
    
    # Create a test GUI
    gui = create_robot_gui()
    
    # Start update loop
    gui.update_loop()
    
    # Run GUI
    gui.run()