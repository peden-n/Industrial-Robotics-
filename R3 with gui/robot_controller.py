"""
Robot Control Interface - Provides high-level control functions for robot manipulation.
Handles picking, placing, and movement operations with proper error handling.
"""
import numpy as np
import time
from ik_controller import solve_ik_with_limits, get_current_end_effector_pos
from setup_scene import update_robot_meshes
from utilities.object_array import AttachableObject


class RobotController:
    """High-level robot control interface."""
    
    def __init__(self, robot, env, meshes):
        """
        Initialize robot controller.
        
        Args:
            robot: DHRobot instance
            env: Swift environment
            meshes: Dictionary of robot link meshes
        """
        self.robot = robot
        self.env = env
        self.meshes = meshes
        self.attached_objects = []  # Objects currently attached to end-effector
        self.object_offsets = {}  # Store relative offsets for each attached object
        self.approach_height = 0.10  # Default approach height (15cm)
        self.pick_height = 0.05  # Height above object for picking (reduced for better placement)
        self.emergency_stop_flag = False  # Flag to immediately halt all motion
        
    def move_to_position_smooth(self, target_pos, steps=50, check_reachability=True):
        """
        Move robot to target position smoothly with interpolation.
        
        Args:
            target_pos: Target [x, y, z] position
            steps: Number of interpolation steps
            check_reachability: Whether to check if position is reachable first
        
        Returns:
            bool: True if successful, False otherwise
        """
        if check_reachability:
            q_target, success = solve_ik_with_limits(self.robot, target_pos)
            if not success or q_target is None:
                print(f"✗ Cannot reach position {np.round(target_pos, 3)}")
                return False
        else:
            q_target, success = solve_ik_with_limits(self.robot, target_pos)
            if not success:
                return False
        
        # Interpolate smoothly
        q_start = self.robot.q.copy()
        
        for step in range(steps):
            # Check for emergency stop before each movement step
            if self.emergency_stop_flag:
                print("EMERGENCY STOP - Movement halted immediately!")
                return False
            
            alpha = (step + 1) / steps
            self.robot.q = q_start * (1 - alpha) + q_target * alpha
            
            # Update visualization
            update_robot_meshes(self.robot, self.meshes)
            
            # Update attached objects with their relative offsets
            if self.attached_objects:
                ee_pos = get_current_end_effector_pos(self.robot)
                for obj in self.attached_objects:
                    if isinstance(obj, AttachableObject):
                        obj.update_from_parent(ee_pos)
                    else:
                        # Maintain relative offset from when object was picked up
                        if id(obj) in self.object_offsets:
                            obj.position = ee_pos + self.object_offsets[id(obj)]
                        else:
                            obj.position = ee_pos
            
            self.env.step(0.02)
        
        print(f" Moved to {np.round(target_pos, 3)}")
        return True
    
    def move_joints(self, joint_angles, steps=50, degrees=True):
        """
        Move robot to specific joint configuration.
        
        Args:
            joint_angles: Target joint angles (6 values)
            steps: Number of interpolation steps
            degrees: Whether joint_angles are in degrees (True) or radians (False)
        
        Returns:
            bool: True if successful
        """
        if degrees:
            target_q = np.deg2rad(joint_angles)
        else:
            target_q = np.array(joint_angles)
        
        # Check joint limits
        for i, (angle, link) in enumerate(zip(target_q, self.robot.links)):
            if not (link.qlim[0] <= angle <= link.qlim[1]):
                print(f"Joint {i+1} angle {np.rad2deg(angle):.1f}° exceeds limits "
                      f"[{np.rad2deg(link.qlim[0]):.1f}°, {np.rad2deg(link.qlim[1]):.1f}°]")
                return False
        
        # Interpolate smoothly
        q_start = self.robot.q.copy()
        
        for step in range(steps):
            # Check for emergency stop before each movement step
            if self.emergency_stop_flag:
                print(" EMERGENCY STOP - Joint movement halted immediately!")
                return False
            
            alpha = (step + 1) / steps
            self.robot.q = q_start * (1 - alpha) + target_q * alpha
            
            # Update visualization
            update_robot_meshes(self.robot, self.meshes)
            
            # Update attached objects with their relative offsets
            if self.attached_objects:
                ee_pos = get_current_end_effector_pos(self.robot)
                for obj in self.attached_objects:
                    if isinstance(obj, AttachableObject):
                        obj.update_from_parent(ee_pos)
                    else:
                        # Maintain relative offset from when object was picked up
                        if id(obj) in self.object_offsets:
                            obj.position = ee_pos + self.object_offsets[id(obj)]
                        else:
                            obj.position = ee_pos
            
            self.env.step(0.02)
        
        if degrees:
            print(f" Moved to joint angles: {np.round(np.rad2deg(target_q), 1)}°")
        else:
            print(f" Moved to joint angles: {np.round(target_q, 3)} rad")
        return True
    
    def pick_object(self, obj, approach_from_above=True, offset_distance=0.2):
        """
        Pick up an object by moving to it and attaching it with a specified offset.
        
        Args:
            obj: Object to pick up (must have .position attribute)
            approach_from_above: Whether to approach from above (True) or directly (False)
            offset_distance: Distance to maintain from end-effector when picking (meters)
        
        Returns:
            bool: True if successful
        """
        if not hasattr(obj, 'position'):
            print(" Object does not have position attribute")
            return False
        
        obj_pos = np.array(obj.position)
        print(f"\n Picking up object at {np.round(obj_pos, 3)} with {offset_distance}m offset")
        
        # Calculate pick position with offset (approach from above the object)
        pick_pos = obj_pos + np.array([0, 0, offset_distance])
        
        # Step 1: Approach position (higher above object)
        if approach_from_above:
            approach_pos = obj_pos + np.array([0, 0, self.approach_height + offset_distance])
            print(f"  → Moving to approach position: {np.round(approach_pos, 3)}")
            if not self.move_to_position_smooth(approach_pos):
                return False
            time.sleep(0.5)
        
        # Step 2: Move to pick position (with offset above object)
        print(f"  → Moving to pick position with offset: {np.round(pick_pos, 3)}")
        if not self.move_to_position_smooth(pick_pos):
            return False
        time.sleep(0.5)
        
        # Step 3: "Attach" object (simulate gripper closing)
        print("  → Closing gripper and attaching object")
        
        # Calculate and store the relative offset between end-effector and object
        current_ee_pos = self.get_current_position()
        offset = obj_pos - current_ee_pos
        self.object_offsets[id(obj)] = offset
        
        self.attached_objects.append(obj)
        print(f"  → Stored relative offset: {np.round(offset, 3)}")
        
        # Step 4: Lift object (maintaining the offset)
        lift_pos = pick_pos + np.array([0, 0, 0.05])  # Lift 5cm
        print(f"  → Lifting object to: {np.round(lift_pos, 3)}")
        if not self.move_to_position_smooth(lift_pos):
            return False
        
        print(f" Successfully picked up object with {offset_distance}m offset!")
        return True
    
    def place_object(self, target_pos, approach_from_above=True):
        """
        Place the currently held object at target position.
        Accounts for gripper offset to ensure object lands at correct target position.
        
        Args:
            target_pos: Target [x, y, z] position to place object
            approach_from_above: Whether to approach from above
        
        Returns:
            bool: True if successful
        """
        if not self.attached_objects:
            print(" No object attached to place")
            return False
        
        target_pos = np.array(target_pos)
        print(f"\n Placing object at {np.round(target_pos, 3)}")
        
        obj = self.attached_objects[0]  # Get the object we're about to place
        
        # Calculate gripper position needed to place object at target position
        if id(obj) in self.object_offsets:
            stored_offset = self.object_offsets[id(obj)]
            # To place object at target_pos, gripper should be at target_pos - stored_offset
            adjusted_target_pos = target_pos - stored_offset
            print(f"  → Object offset: {np.round(stored_offset, 3)}")
            print(f"  → Adjusted gripper target: {np.round(adjusted_target_pos, 3)}")
        else:
            adjusted_target_pos = target_pos
            print(f"  → No offset stored, using direct positioning")
        
        # Step 1: Approach position (above adjusted target)
        if approach_from_above:
            approach_pos = adjusted_target_pos + np.array([0, 0, self.approach_height])
            print(f"  → Moving to approach position: {np.round(approach_pos, 3)}")
            if not self.move_to_position_smooth(approach_pos):
                return False
            time.sleep(0.5)
        
        # Step 2: Move to place position (just above adjusted target)
        place_pos = adjusted_target_pos + np.array([0, 0, self.pick_height])
        print(f"  → Moving to place position: {np.round(place_pos, 3)}")
        if not self.move_to_position_smooth(place_pos):
            return False
        time.sleep(0.5)
        
        # Step 3: "Detach" object (simulate gripper opening)
        print("  → Opening gripper and releasing object")
        obj = self.attached_objects.pop()  # Remove from attached list
        
        # Calculate where the object should be placed based on current gripper position and stored offset
        current_gripper_pos = self.get_current_position()
        if id(obj) in self.object_offsets:
            stored_offset = self.object_offsets[id(obj)]
            # Object position = current gripper position + stored offset
            calculated_obj_pos = current_gripper_pos + stored_offset
            obj.position = calculated_obj_pos
            print(f"  → Placed object with offset compensation at {np.round(calculated_obj_pos, 3)}")
            print(f"  → Expected final position: {np.round(target_pos, 3)}")
            
            # Clean up the stored offset
            print(f"  → Cleared stored offset: {np.round(stored_offset, 3)}")
            del self.object_offsets[id(obj)]
        else:
            # Fallback: place at target position if no offset stored
            obj.position = target_pos
            print(f"  → Placed object (no offset data) at target {np.round(target_pos, 3)}")
        
        # Step 4: Retract gripper
        retract_pos = place_pos + np.array([0, 0, 0.05])  # Move up 5cm
        print(f"  → Retracting to: {np.round(retract_pos, 3)}")
        if not self.move_to_position_smooth(retract_pos):
            return False
        
        print(f" Successfully placed object!")
        return True
    
    def move_to_home_position(self):
        """Move robot to a safe home position."""
        home_angles = [0.0, -30.0, 45.0, 0.0, 10.0, 0.0]  # degrees
        print("\n Moving to home position")
        return self.move_joints(home_angles)
    
    def get_current_position(self):
        """Get current end-effector position."""
        return get_current_end_effector_pos(self.robot)
    
    def get_current_joints(self, degrees=True):
        """Get current joint angles."""
        if degrees:
            return np.rad2deg(self.robot.q)
        return self.robot.q.copy()
    
    def check_reachability(self, target_pos):
        """
        Check if a position is reachable without moving the robot.
        
        Args:
            target_pos: Target [x, y, z] position
        
        Returns:
            bool: True if reachable
        """
        q_target, success = solve_ik_with_limits(self.robot, target_pos)
        return success and q_target is not None
    
    def trace_path(self, waypoints, steps_per_segment=30):
        """
        Move robot through a series of waypoints.
        
        Args:
            waypoints: List of [x, y, z] positions
            steps_per_segment: Steps between each waypoint
        
        Returns:
            bool: True if all waypoints reached successfully
        """
        print(f"\n Tracing path with {len(waypoints)} waypoints")
        
        for i, waypoint in enumerate(waypoints):
            print(f"  → Waypoint {i+1}/{len(waypoints)}: {np.round(waypoint, 3)}")
            if not self.move_to_position_smooth(waypoint, steps=steps_per_segment):
                print(f"✗ Failed to reach waypoint {i+1}")
                return False
            time.sleep(0.2)
        
        print(" Path complete!")
        return True
    
    def emergency_stop(self):
        """Stop all robot motion immediately."""
        self.emergency_stop_flag = True
        print(" EMERGENCY STOP - Robot motion halted immediately!")
        print("   All ongoing movements will be interrupted")
        return True
    
    def reset_emergency_stop(self):
        """Reset the emergency stop flag to allow movement again."""
        self.emergency_stop_flag = False
        print(" Emergency stop reset - Robot motion enabled")
        return True
    
    def is_emergency_stopped(self):
        """Check if robot is in emergency stop state."""
        return self.emergency_stop_flag
    
    def get_attached_objects(self):
        """Get list of currently attached objects."""
        return self.attached_objects.copy()
    
    def is_holding_object(self):
        """Check if robot is currently holding an object."""
        return len(self.attached_objects) > 0
    
    def test_all_positions(self, scene_data, approach_height=0.15):
        """
        Test going to all object positions and end positions from scene data.
        
        Args:
            scene_data: Scene data dictionary from initialize_complete_scene()
            approach_height: Height above each position to test (default 15cm)
        
        Returns:
            dict: Results of reachability test {position_name: success_bool}
        """
        print("\n" + "=" * 70)
        print(" COMPREHENSIVE POSITION TEST")
        print("=" * 70)
        
        results = {}
        
        # Test all loaded objects
        objects = scene_data.get('objects', {})
        configs = scene_data.get('configs', {})
        
        for obj_type, obj_list in objects.items():
            if not obj_list:
                continue
            
            print(f"\n--- Testing {obj_type.upper()} positions ---")
            
            for i, obj in enumerate(obj_list):
                obj_pos = np.array(obj.position)
                test_pos = obj_pos + np.array([0, 0, approach_height])
                
                print(f"\n{obj_type[:-1].capitalize()} {i+1}:")
                print(f"   Position: {np.round(obj_pos, 3)}")
                print(f"   Test point: {np.round(test_pos, 3)}")
                
                success = self.move_to_position_smooth(test_pos, steps=30)
                results[f"{obj_type}_{i+1}_start"] = success
                
                if success:
                    print(f"    Successfully reached {obj_type[:-1]} {i+1}!")
                else:
                    print(f"    FAILED to reach {obj_type[:-1]} {i+1}")
                
                time.sleep(0.5)
        
        # Test end positions from configs
        print(f"\n--- Testing END POSITIONS ---")
        
        for config_type, config_data in configs.items():
            if not isinstance(config_data, list):
                continue
                
            for i, config in enumerate(config_data):
                if 'end' not in config:
                    continue
                    
                end_pos = np.array(config['end'])
                test_pos = end_pos + np.array([0, 0, approach_height])
                
                print(f"\n{config_type.capitalize()} {i+1} End Position:")
                print(f"   Position: {np.round(end_pos, 3)}")
                print(f"   Test point: {np.round(test_pos, 3)}")
                
                success = self.move_to_position_smooth(test_pos, steps=30)
                results[f"{config_type}_{i+1}_end"] = success
                
                if success:
                    print(f"    Successfully reached {config_type} {i+1} end position!")
                else:
                    print(f"    FAILED to reach {config_type} {i+1} end position")
                
                time.sleep(0.5)
        
        # Print summary
        print("\n" + "=" * 70)
        print("📊 POSITION TEST SUMMARY")
        print("=" * 70)
        
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        success_rate = (successful / total * 100) if total > 0 else 0
        
        print(f"Total positions tested: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {total - successful}")
        print(f"Success rate: {success_rate:.1f}%")
        
        print("\nDetailed results:")
        for position_name, success in results.items():
            status = " PASS" if success else " FAIL"
            print(f"  {position_name}: {status}")
        
        print("=" * 70)
        
        if total - successful > 0:
            print("  Some positions failed - consider adjusting configurations")
        else:
            print(" All positions are reachable!")
        
        return results

    def print_status(self):
        """Print current robot status."""
        current_pos = self.get_current_position()
        current_joints = self.get_current_joints()
        
        print(f"\n ROBOT STATUS")
        print(f"  End-effector position: {np.round(current_pos, 3)}")
        print(f"  Joint angles (deg): {np.round(current_joints, 1)}")
        print(f"  Attached objects: {len(self.attached_objects)}")
        if self.attached_objects:
            for i, obj in enumerate(self.attached_objects):
                print(f"    • Object {i+1}: {getattr(obj, 'name', 'Unknown')}")
        print()


if __name__ == '__main__':
    # Test the robot controller
    from scene_initializer import initialize_complete_scene
    
    # Initialize scene
    scene_data = initialize_complete_scene()
    
    # Create controller
    controller = RobotController(
        scene_data['robot'],
        scene_data['env'],
        scene_data['meshes']
    )
    
    print("\n" + "="*50)
    print("TESTING ROBOT CONTROLLER")
    print("="*50)
    
    # Test basic movement
    controller.print_status()
    
    # Test joint movement
    print("\nTest 1: Moving joints...")
    controller.move_joints([30, -45, 60, 0, 20, 30])
    time.sleep(1)
    
    # Test position movement
    print("\nTest 2: Moving to position...")
    target = [0.5, 1.3, 0.8]
    controller.move_to_position_smooth(target)
    
    # Test path tracing
    print("\nTest 3: Tracing path...")
    path = [
        [0.4, 1.2, 0.7],
        [0.6, 1.2, 0.7],
        [0.6, 1.4, 0.7],
        [0.4, 1.4, 0.7]
    ]
    controller.trace_path(path)
    
    # Return to home
    controller.move_to_home_position()
    controller.print_status()
    
    print("\nController test complete! Press Ctrl+C to exit.")
    try:
        while True:
            scene_data['env'].step(0.05)
    except KeyboardInterrupt:
        print("\nShutting down...")
        scene_data['env'].close()