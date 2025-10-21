"""
Robot Box Lid Assembly System - Pick up lids and boxes, and place them in the pallet.
This script:
1. Initializes the complete scene (environment, robot, boxes, lids, etc.)
2. Sets up three specified positions for joint 1 rotation:
   - "Box" position: -90°
   - "Lid" position: 0°
   - "Pallet" position: 170°
3. Loops through all available boxes and lids to:
   - Go to Lid position
   - Pick up a lid
   - Move up 0.2m
   - Rotate to Box position
   - Pick up corresponding box
   - Move up 0.2m above default box height
   - Rotate to Pallet position
   - Place both lid and box at box end position
   - Move up 0.4m above pallet
   - Return to Lid position for next iteration
4. Provides GUI controls for emergency stop, reset, and system management
"""
import numpy as np
import time
import threading
from scene_initializer import initialize_complete_scene
from robot_controller import RobotController
from robot_gui import create_robot_gui


# Define the three specified positions for joint 1
SPECIFIED_POSITIONS = {
    "Box": -90.0,      # Joint 1 angle for box position
    "Lid": 0.0,        # Joint 1 angle for lid position  
    "Pallet": 170.0    # Joint 1 angle for pallet position
}


def rotate_joint1_to_position(controller, position_name):
    """
    Rotate joint 1 to one of the specified positions.
    
    Args:
        controller: RobotController instance
        position_name: One of "Box", "Lid", or "Pallet"
    
    Returns:
        bool: True if successful
    """
    if position_name not in SPECIFIED_POSITIONS:
        print(f" Unknown position: {position_name}")
        return False
    
    target_angle = SPECIFIED_POSITIONS[position_name]
    print(f" Rotating joint 1 to {position_name} position ({target_angle}°)")
    
    # Get current joint configuration
    current_joints = controller.get_current_joints(degrees=True)
    
    # Set joint 1 to target angle
    target_joints = current_joints.copy()
    target_joints[0] = target_angle
    
    return controller.move_joints(target_joints, steps=40, degrees=True)


def move_up_by_height(controller, height):
    """
    Move the end-effector up by a specified height from current position.
    
    Args:
        controller: RobotController instance
        height: Height to move up in meters
    
    Returns:
        bool: True if successful
    """
    current_pos = controller.get_current_position()
    target_pos = current_pos + np.array([0, 0, height])
    
    print(f"  Moving up by {height}m to {np.round(target_pos, 3)}")
    return controller.move_to_position_smooth(target_pos, steps=30)


def pick_up_object_at_position(controller, obj, object_name, offset=0.15):
    """
    Pick up an object at its current position with specified offset.
    
    Args:
        controller: RobotController instance
        obj: Object to pick up
        object_name: Name for logging purposes
        offset: Distance to maintain from end-effector (meters)
    
    Returns:
        bool: True if successful
    """
    print(f"\n Picking up {object_name} with {offset}m offset")
    
    # Use the controller's pick_object method with offset
    success = controller.pick_object(obj, approach_from_above=True, offset_distance=offset)
    
    if success:
        print(f" Successfully picked up {object_name} with offset!")
    else:
        print(f" Failed to pick up {object_name}")
    
    return success


def place_objects_at_end_position(controller, end_pos, box_name, lid_name, placement_height_adjustment=1.0):
    """
    Place both lid and box at the same specified end position in one motion.
    Accounts for gripper offset to ensure objects land at the correct target position.
    
    Args:
        controller: RobotController instance
        end_pos: Target end position [x, y, z]
        box_name: Box name for logging
        lid_name: Lid name for logging
        placement_height_adjustment: Additional height adjustment for placement (negative = lower, positive = higher)
    
    Returns:
        bool: True if successful
    """
    print(f"\n Placing {lid_name} and {box_name} together at end position {np.round(end_pos, 3)}")
    
    attached_objects = controller.get_attached_objects().copy()
    if not attached_objects:
        print(" No objects attached to place")
        return False
    
    end_pos = np.array(end_pos)
    
    # Calculate gripper position needed to place objects at target position
    # Account for the stored offsets between gripper and objects
    if attached_objects and controller.object_offsets:
        # Find the average offset for proper gripper positioning
        total_offset = np.array([0.0, 0.0, 0.0])
        offset_count = 0
        
        for obj in attached_objects:
            if id(obj) in controller.object_offsets:
                offset = controller.object_offsets[id(obj)]
                total_offset += offset
                offset_count += 1
                print(f"  → Object offset: {np.round(offset, 3)}")
        
        if offset_count > 0:
            avg_offset = total_offset / offset_count
            # To place objects at end_pos, gripper should be at end_pos - avg_offset
            # Apply additional height adjustment if specified
            adjusted_end_pos = end_pos - avg_offset + np.array([0, 0, placement_height_adjustment])
            print(f"  → Average object offset: {np.round(avg_offset, 3)}")
            print(f"  → Height adjustment: {placement_height_adjustment}m")
            print(f"  → Adjusted gripper target: {np.round(adjusted_end_pos, 3)}")
        else:
            adjusted_end_pos = end_pos
            print(f"  → No offsets found, using direct position")
    else:
        adjusted_end_pos = end_pos
        print(f"  → No attached objects or offsets, using direct position")
    
    # Step 1: Approach position (above adjusted target)
    approach_pos = adjusted_end_pos + np.array([0, 0, controller.approach_height])
    print(f"  → Moving to approach position: {np.round(approach_pos, 3)}")
    if not controller.move_to_position_smooth(approach_pos):
        return False
    time.sleep(0.5)
    
    # Step 2: Move to place position (just above adjusted target)
    place_pos = adjusted_end_pos + np.array([0, 0, controller.pick_height])
    print(f"  → Moving to place position: {np.round(place_pos, 3)}")
    if not controller.move_to_position_smooth(place_pos):
        return False
    time.sleep(0.5)
    
    # Step 3: "Release" all objects simultaneously (simulate gripper opening)
    print(f"  → Opening gripper and releasing both {box_name} and {lid_name}")
    released_objects = []
    
    # Get current gripper position for accurate object placement
    current_gripper_pos = controller.get_current_position()
    
    while controller.attached_objects:
        obj = controller.attached_objects.pop()  # Remove from attached list
        
        # Calculate where the object should be placed based on current gripper position and stored offset
        if id(obj) in controller.object_offsets:
            stored_offset = controller.object_offsets[id(obj)]
            # Object position = current gripper position + stored offset
            calculated_obj_pos = current_gripper_pos + stored_offset
            obj.position = calculated_obj_pos
            print(f"  → Released object with offset compensation at {np.round(calculated_obj_pos, 3)}")
            print(f"  → Expected final position: {np.round(end_pos, 3)}")
            
            # Clean up the stored offset
            print(f"  → Cleared stored offset: {np.round(stored_offset, 3)}")
            del controller.object_offsets[id(obj)]
        else:
            # Fallback: place at target position if no offset stored
            obj.position = end_pos
            print(f"  → Released object (no offset data) at target {np.round(end_pos, 3)}")
        
        released_objects.append(obj)
    
    print(f" Placed both {box_name} and {lid_name} together at {np.round(end_pos, 3)}")
    
    # Debug: Check object positions before retraction
    for i, obj in enumerate(released_objects):
        print(f"  Debug: Object {i+1} position before retraction: {np.round(obj.position, 3)}")
    
    # Give a moment for the object positions to be properly committed
    time.sleep(0.2)
    
    # Step 4: Retract gripper (move up from place position)
    retract_pos = place_pos + np.array([0, 0, 0.05])  # Move up 5cm
    print(f"  → Retracting to: {np.round(retract_pos, 3)}")
    if not controller.move_to_position_smooth(retract_pos):
        return False
    
    # Debug: Check object positions after retraction
    for i, obj in enumerate(released_objects):
        print(f"  Debug: Object {i+1} position after retraction: {np.round(obj.position, 3)}")
    
    return True


def assembly_cycle(controller, lid_obj, box_obj, box_config, cycle_num, total_cycles):
    """
    Perform one complete assembly cycle: pick lid, pick box, place both.
    
    Args:
        controller: RobotController instance
        lid_obj: Lid object to pick up
        box_obj: Box object to pick up
        box_config: Box configuration containing end position
        cycle_num: Current cycle number (1-based)
        total_cycles: Total number of cycles
    
    Returns:
        bool: True if cycle completed successfully
    """
    print(f"\n{'='*70}")
    print(f" ASSEMBLY CYCLE {cycle_num}/{total_cycles}")
    print(f"{'='*70}")
    
    lid_name = f"lid_{cycle_num}"
    box_name = f"box_{cycle_num}"
    
    # Helper function to check for emergency stop
    def check_emergency_stop():
        if controller.is_emergency_stopped():
            print(" Assembly cycle halted due to emergency stop")
            return True
        return False
    
    # Step 1: Go to Lid position
    print(f"Step 1: Moving to Lid position")
    if check_emergency_stop():
        return False
    if not rotate_joint1_to_position(controller, "Lid"):
        print(f" Failed to rotate to Lid position")
        return False
    time.sleep(0.5)
    
    # Step 2: Pick up lid
    print(f"Step 2: Picking up {lid_name}")
    if check_emergency_stop():
        return False
    if not pick_up_object_at_position(controller, lid_obj, lid_name):
        print(f" Failed to pick up {lid_name}")
        return False
    time.sleep(0.5)
    
    # Step 3: Move up by 0.2m
    print(f"Step 3: Moving up by 0.2m after picking lid")
    if check_emergency_stop():
        return False
    if not move_up_by_height(controller, 0.2):
        print(f" Failed to move up after picking lid")
        return False
    time.sleep(0.5)
    
    # Step 4: Rotate joint 1 to Box position
    print(f"Step 4: Rotating to Box position")
    if check_emergency_stop():
        return False
    if not rotate_joint1_to_position(controller, "Box"):
        print(f" Failed to rotate to Box position")
        return False
    time.sleep(0.5)
    
    # Step 5: Pick up corresponding box
    print(f"Step 5: Picking up {box_name}")
    if check_emergency_stop():
        return False
    if not pick_up_object_at_position(controller, box_obj, box_name):
        print(f" Failed to pick up {box_name}")
        return False
    time.sleep(0.5)
    
    # Step 6: Move up 0.2m above default box height
    print(f"Step 6: Moving up 0.2m above default box height")
    if check_emergency_stop():
        return False
    if not move_up_by_height(controller, 0.2):
        print(f" Failed to move up after picking box")
        return False
    time.sleep(0.5)
    
    # Step 7: Rotate joint 1 to Pallet position
    print(f"Step 7: Rotating to Pallet position")
    if check_emergency_stop():
        return False
    if not rotate_joint1_to_position(controller, "Pallet"):
        print(f" Failed to rotate to Pallet position")
        return False
    time.sleep(0.5)
    
    # Step 8: Place both objects at box end position
    print(f"Step 8: Placing objects at end position")
    if check_emergency_stop():
        return False
    if 'end' not in box_config:
        print(f" No end position defined for {box_name}")
        return False
    
    end_pos = np.array(box_config['end'])
    if not place_objects_at_end_position(controller, end_pos, box_name, lid_name, placement_height_adjustment=-0.03):
        print(f" Failed to place objects at end position")
        return False
    time.sleep(0.5)
    
    # Step 9: Move up 0.4m above the pallet
    print(f"Step 9: Moving up 0.4m above the pallet")
    if check_emergency_stop():
        return False
    if not move_up_by_height(controller, 0.4):
        print(f" Failed to move up above pallet")
        return False
    time.sleep(0.5)
    
    print(f" Assembly cycle {cycle_num} completed successfully!")
    return True


def run_assembly_process(controller, scene_data, gui):
    """
    Run the complete assembly process with GUI control integration.
    
    Args:
        controller: RobotController instance
        scene_data: Scene data dictionary
        gui: RobotControlGUI instance
    """
    # Get objects and configurations
    lids = scene_data['objects'].get('lids', [])
    boxes = scene_data['objects'].get('boxes', [])
    box_configs = scene_data['configs'].get('box', [])
    
    # Validate we have the required objects
    if not lids or not boxes or not box_configs:
        print(" Missing required objects for assembly!")
        return
    
    # Determine number of assembly cycles
    num_cycles = min(len(lids), len(boxes), len(box_configs))
    
    print(f"\n Starting assembly process:")
    print(f"  • Planned assembly cycles: {num_cycles}")
    
    successful_cycles = 0
    
    for i in range(num_cycles):
        # Check for emergency stop or pause
        if gui.is_emergency_stopped() or (controller and controller.is_emergency_stopped()):
            print(" Assembly stopped due to emergency stop")
            break
        
        if gui.is_paused():
            print("⏸ Assembly paused")
            # Wait until unpaused or emergency stopped
            while gui.is_paused() and not gui.is_emergency_stopped() and not (controller and controller.is_emergency_stopped()):
                time.sleep(0.1)
            
            if gui.is_emergency_stopped() or (controller and controller.is_emergency_stopped()):
                print(" Assembly stopped due to emergency stop")
                break
        
        lid_obj = lids[i]
        box_obj = boxes[i]
        box_config = box_configs[i]
        
        print(f"\n Starting assembly cycle {i + 1}/{num_cycles}")
        
        # Perform assembly cycle with error handling
        try:
            cycle_success = assembly_cycle(
                controller, 
                lid_obj, 
                box_obj, 
                box_config, 
                i + 1, 
                num_cycles
            )
            
            if cycle_success:
                successful_cycles += 1
                print(f" Assembly cycle {i + 1} completed successfully!")
            else:
                print(f" Assembly cycle {i + 1} failed!")
        
        except Exception as e:
            print(f" Assembly cycle {i + 1} failed with error: {e}")
        
        # Brief pause between cycles (check for stop during pause)
        for pause_step in range(10):  # 1 second total pause
            if gui.is_emergency_stopped() or gui.is_paused():
                break
            time.sleep(0.1)
    
    # Final positioning if not stopped
    if not gui.is_emergency_stopped():
        print(f"\n Assembly process complete! Returning to Lid position...")
        try:
            rotate_joint1_to_position(controller, "Lid")
        except Exception as e:
            print(f"Warning: Could not return to Lid position: {e}")
    
    # Print final summary
    print(f"\n{'='*70}")
    print(" ASSEMBLY SUMMARY")
    print(f"{'='*70}")
    print(f"Total assembly cycles planned: {num_cycles}")
    print(f"Successful cycles: {successful_cycles}")
    print(f"Failed cycles: {num_cycles - successful_cycles}")
    success_rate = (successful_cycles / num_cycles * 100) if num_cycles > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    
    if successful_cycles == num_cycles:
        print(" ALL ASSEMBLIES COMPLETED SUCCESSFULLY!")
    else:
        print("  Some assemblies failed - check robot configurations and object positions")
    
    print(f"{'='*70}")


def main():
    """Main function - initialize scene and start GUI-controlled assembly system."""
    print(f"\n{'='*70}")
    print(" ROBOT BOX LID ASSEMBLY SYSTEM WITH GUI CONTROL")
    print(f"{'='*70}")
    print("This program will:")
    print("  1. Initialize the complete scene (robot, environment, objects)")
    print("  2. Set up three specified positions for joint 1:")
    print(f"     • Box position: {SPECIFIED_POSITIONS['Box']}°")
    print(f"     • Lid position: {SPECIFIED_POSITIONS['Lid']}°")
    print(f"     • Pallet position: {SPECIFIED_POSITIONS['Pallet']}°")
    print("  3. Launch GUI control panel for:")
    print("     • Emergency stop and reset controls")
    print("     • Robot position and system resets")
    print("     • Environment reset (objects to start positions)")
    print("     • Assembly start/pause/continue controls")
    print("  4. Perform controlled assembly cycles with GUI oversight")
    print(f"{'='*70}")
    
    # Initialize the complete scene
    print("\n  Initializing scene...")
    scene_data = initialize_complete_scene()
    
    # Create robot controller
    print("\n Creating robot controller...")
    controller = RobotController(
        scene_data['robot'],
        scene_data['env'], 
        scene_data['meshes']
    )
    
    # Show initial robot status
    controller.print_status()
    
    # Move to home position first
    print("\n Moving to home position...")
    controller.move_to_home_position()
    
    # Validate we have the required objects
    lids = scene_data['objects'].get('lids', [])
    boxes = scene_data['objects'].get('boxes', [])
    box_configs = scene_data['configs'].get('box', [])
    
    if not lids:
        print(" No lids found in scene!")
        return
    
    if not boxes:
        print(" No boxes found in scene!")
        return
    
    if not box_configs:
        print(" No box configurations found!")
        return
    
    # Determine number of assembly cycles
    num_cycles = min(len(lids), len(boxes), len(box_configs))
    
    print(f"\n Assembly Planning:")
    print(f"  • Available lids: {len(lids)}")
    print(f"  • Available boxes: {len(boxes)}")
    print(f"  • Available box end positions: {len(box_configs)}")
    print(f"  • Planned assembly cycles: {num_cycles}")
    
    if num_cycles == 0:
        print(" Cannot perform assembly - no matching lid/box/config sets!")
        return
    
    # Create assembly function for GUI
    def assembly_function():
        run_assembly_process(controller, scene_data, gui)
    
    # Create and launch GUI
    print("\n  Launching GUI control panel...")
    gui = create_robot_gui(controller, scene_data, assembly_function)
    
    # Start GUI update loop
    gui.update_loop()
    
    print(" GUI launched! Use the control panel to manage the robot system.")
    print("   • Click 'Start Assembly' to begin the assembly process")
    print("   • Use 'Emergency Stop' for immediate halt")
    print("   • Use reset buttons to recover from errors")
    print("   • Close the GUI window to exit the program")
    
    # Run GUI (this will block until GUI is closed)
    try:
        gui.run()
    except KeyboardInterrupt:
        print("\n Program interrupted by user")
    
    # Cleanup when GUI closes
    print(f"\n Cleaning up...")
    try:
        if hasattr(gui, 'root'):
            gui.close()
        scene_data['env'].close()
    except Exception as e:
        print(f"Cleanup warning: {e}")
    
    print(" Program complete!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Interrupted by user")
        print("Shutting down...")
    except Exception as e:
        print(f"\n Error occurred: {e}")
        print("Check your configuration files and robot setup.")