"""
Inverse Kinematics Controller
Converts end-effector positions (and optionally orientations) to joint angles.
Accounts for robot base offset from world origin.
"""
import numpy as np
from spatialmath import SE3
from roboticstoolbox.backends.swift import Swift

# Robot base position in world coordinates (must match setup_scene.py)
ROBOT_BASE_XYZ = np.array([0.0, 1.18, 0.02])


def solve_ik(robot, target_pos, target_orient=None, q0=None, method='LM', max_iterations=100, tolerance=1e-4, base_offset=ROBOT_BASE_XYZ):
    """
    Solve inverse kinematics for the robot.
    
    Parameters:
    -----------
    robot : DHRobot
        The robot object
    target_pos : array_like (3,)
        Target position [x, y, z] in meters (in world coordinates)
    target_orient : array_like (3,3) or SE3 or None, optional
        Target orientation as rotation matrix or SE3 object.
        If None, only position is constrained.
    q0 : array_like (n,), optional
        Initial joint configuration for IK solver.
        If None, uses current robot.q
    method : str, optional
        IK method: 'LM' (Levenberg-Marquardt), 'NR' (Newton-Raphson), 'GN' (Gauss-Newton)
    max_iterations : int, optional
        Maximum iterations for IK solver
    tolerance : float, optional
        Convergence tolerance
    base_offset : array_like (3,), optional
        Robot base position offset from world origin
        
    Returns:
    --------
    q : ndarray
        Joint angles that achieve the target pose, or None if failed
    success : bool
        Whether IK converged successfully
    """
    # Use current joint angles as initial guess if not provided
    if q0 is None:
        q0 = robot.q.copy()
    
    # Convert world coordinates to robot base coordinates
    target_pos_robot = np.array(target_pos) - base_offset
    
    # Create target SE3 transform (relative to robot base)
    if target_orient is None:
        # Position-only IK: use current orientation
        current_pose = robot.fkine(q0)
        target_pose = SE3.Rt(current_pose.R, target_pos_robot)
    elif isinstance(target_orient, SE3):
        target_pose = target_orient
        target_pose.t = target_pos_robot  # Update position to be relative to base
    else:
        # Assume target_orient is a rotation matrix
        target_pose = SE3.Rt(target_orient, target_pos_robot)
    
    try:
        # Use robotics toolbox IK solver
        solution = robot.ikine_LM(
            target_pose,
            q0=q0,
            mask=[1, 1, 1, 0, 0, 0] if target_orient is None else [1, 1, 1, 1, 1, 1],
            ilimit=max_iterations,
            tol=tolerance
        )
        
        if solution.success:
            return solution.q, True
        else:
            print(f"[Warning] IK did not converge. Error: {solution.reason}")
            return solution.q, False
            
    except Exception as e:
        print(f"[Error] IK failed: {e}")
        return None, False


def solve_ik_with_limits(robot, target_pos, target_orient=None, q0=None, num_attempts=5, base_offset=ROBOT_BASE_XYZ):
    """
    Solve IK with multiple attempts using different initial configurations.
    Useful when IK fails due to local minima.
    
    Parameters:
    -----------
    robot : DHRobot
        The robot object
    target_pos : array_like (3,)
        Target position [x, y, z] (in world coordinates)
    target_orient : array_like (3,3) or SE3 or None
        Target orientation
    q0 : array_like (n,), optional
        Preferred initial configuration
    num_attempts : int
        Number of random initial configurations to try if first attempt fails
    base_offset : array_like (3,), optional
        Robot base position offset from world origin
        
    Returns:
    --------
    q : ndarray
        Joint angles, or None if all attempts failed
    success : bool
        Whether any attempt succeeded
    """
    # Try with provided initial guess first
    q, success = solve_ik(robot, target_pos, target_orient, q0, base_offset=base_offset)
    
    if success:
        return q, True
    
    # Try with random initial configurations
    print(f"[Info] Trying {num_attempts} alternative initial configurations...")
    for i in range(num_attempts):
        # Generate random configuration within joint limits
        q_rand = np.array([
            np.random.uniform(link.qlim[0], link.qlim[1])
            for link in robot.links
        ])
        
        q, success = solve_ik(robot, target_pos, target_orient, q_rand, base_offset=base_offset)
        if success:
            print(f"[Info] IK converged on attempt {i+1}")
            return q, True
    
    print("[Error] IK failed after all attempts")
    return None, False


def move_to_position(robot, meshes, env, target_pos, target_orient=None, update_func=None, steps=50):
    """
    Move robot to target end-effector position with smooth interpolation.
    
    Parameters:
    -----------
    robot : DHRobot
        The robot object
    meshes : dict
        Dictionary of robot link meshes
    env : Swift
        Swift environment
    target_pos : array_like (3,)
        Target position [x, y, z]
    target_orient : array_like or SE3 or None
        Target orientation (optional)
    update_func : callable, optional
        Function to update robot visualization: update_func(robot, meshes)
    steps : int
        Number of interpolation steps for smooth motion
        
    Returns:
    --------
    success : bool
        Whether the move was successful
    """
    # Solve IK
    q_target, success = solve_ik_with_limits(robot, target_pos, target_orient)
    
    if not success or q_target is None:
        print(f"[Error] Could not reach position {target_pos}")
        return False
    
    # Interpolate from current to target configuration
    q_start = robot.q.copy()
    
    for i in range(steps):
        alpha = (i + 1) / steps
        robot.q = q_start * (1 - alpha) + q_target * alpha
        
        if update_func is not None:
            update_func(robot, meshes)
        
        env.step(0.02)
    
    # Verify final position (convert robot frame to world frame)
    final_pose = robot.fkine(robot.q)
    final_pos_world = final_pose.t + ROBOT_BASE_XYZ
    error = np.linalg.norm(final_pos_world - target_pos)
    
    if error > 0.01:  # 1cm tolerance
        print(f"[Warning] Position error: {error*1000:.2f}mm")
        print(f"  Target (world): {np.round(target_pos, 3)}")
        print(f"  Actual (world): {np.round(final_pos_world, 3)}")
    else:
        print(f"✓ Reached target position (error: {error*1000:.2f}mm)")
    
    return True


def trace_path(robot, meshes, env, waypoints, target_orient=None, update_func=None, steps_per_segment=30):
    """
    Move robot through a series of waypoints.
    
    Parameters:
    -----------
    robot : DHRobot
        The robot object
    meshes : dict
        Dictionary of robot link meshes
    env : Swift
        Swift environment
    waypoints : list of array_like (3,)
        List of target positions to visit in sequence
    target_orient : array_like or SE3 or None
        Constant orientation for all waypoints (optional)
    update_func : callable
        Function to update visualization
    steps_per_segment : int
        Number of steps between each waypoint
        
    Returns:
    --------
    success : bool
        Whether all waypoints were reached
    """
    print(f"\n{'='*50}")
    print(f"Tracing path with {len(waypoints)} waypoints")
    print(f"{'='*50}")
    
    for i, target_pos in enumerate(waypoints):
        print(f"\nWaypoint {i+1}/{len(waypoints)}: {np.round(target_pos, 3)}")
        
        success = move_to_position(
            robot, meshes, env, target_pos, 
            target_orient=target_orient,
            update_func=update_func,
            steps=steps_per_segment
        )
        
        if not success:
            print(f"[Error] Failed to reach waypoint {i+1}")
            return False
    
    print(f"\n{'='*50}")
    print("✓ Path complete!")
    print(f"{'='*50}\n")
    return True


def get_current_end_effector_pos(robot, base_offset=ROBOT_BASE_XYZ):
    """
    Get the current end-effector position in world coordinates.
    
    Parameters:
    -----------
    robot : DHRobot
        The robot object
    base_offset : array_like (3,), optional
        Robot base position offset from world origin
    
    Returns:
    --------
    pos : ndarray (3,)
        Current [x, y, z] position in world coordinates
    """
    T = robot.fkine(robot.q)
    # Add base offset to get world coordinates
    return T.t + base_offset


def get_current_end_effector_pose(robot):
    """
    Get the current end-effector pose (position + orientation).
    
    Returns:
    --------
    pose : SE3
        Current end-effector pose
    """
    return robot.fkine(robot.q)


if __name__ == '__main__':
    # Quick test
    from setup_scene import initialize_environment, update_robot_meshes
    
    env, robot, meshes, *_ = initialize_environment()
    
    print("\n" + "="*50)
    print("Testing IK Controller")
    print("="*50)
    
    # Get current position
    current_pos = get_current_end_effector_pos(robot)
    print(f"\nCurrent end-effector position: {np.round(current_pos, 3)}")
    
    # Test 1: Move to a new position
    print("\nTest 1: Moving to offset position...")
    target = current_pos + np.array([0.1, 0.1, 0.0])
    success = move_to_position(robot, meshes, env, target, update_func=update_robot_meshes)
    
    # Test 2: Move in a small square
    print("\nTest 2: Tracing a square...")
    center = get_current_end_effector_pos(robot)
    size = 0.15
    square_waypoints = [
        center + np.array([size/2, size/2, 0]),
        center + np.array([size/2, -size/2, 0]),
        center + np.array([-size/2, -size/2, 0]),
        center + np.array([-size/2, size/2, 0]),
        center + np.array([size/2, size/2, 0]),  # Close the square
    ]
    
    trace_path(robot, meshes, env, square_waypoints, update_func=update_robot_meshes, steps_per_segment=20)
    
    print("\nTest complete! Press Ctrl+C to exit.")
    try:
        while True:
            env.step(0.05)
    except KeyboardInterrupt:
        print("\nShutting down...")
        env.close()
