import numpy as np
import matplotlib.pyplot as plt
import time
import os
import spatialgeometry as geometry
from spatialmath import SE3
import keyboard
import swift
from roboticstoolbox import DHLink, DHRobot
from ir_support import (
    CylindricalDHRobotPlot,
)

"""
Question 4 Derive the DH parameters for the simple 3 link manipulator provided.
Use these to generate a model of the manipulator using the Robot Toolbox in MATLAB
"""


def main():
    plt.close()

    link1 = DHLink(d=0.445, a=0.1404, alpha=np.pi / 2, qlim=np.deg2rad([-180, 180]), offset=0)
    link2 = DHLink(d=0.17, a=0.7, alpha=0.0, qlim=np.deg2rad([-155, 95]), offset=np.pi / 2)
    link3 = DHLink(d=-0.17, a=0.115, alpha=np.pi / 2, qlim=np.deg2rad([-75, 180]))
    link4 = DHLink(d=0.8, a=0.0, alpha=np.pi / 2, qlim=np.deg2rad([-400, 400]))
    link5 = DHLink(d=0.0, a=0.0, alpha=-np.pi / 2, qlim=np.deg2rad([-120, 120]))
    link6 = DHLink(d=0.0, a=0.0, alpha=0.0, qlim=np.deg2rad([-400, 400]))

    robot = DHRobot([link1, link2, link3, link4, link5, link6], name="myRobot")
    workspace = [-3, 3, -3, 3, -3, 3]
    # q =  np.zeros([1,3]) # Initial joint angles = 0
    # Initial joint configuration (radians). 
    q_init = np.deg2rad([0.0, -30.0, 45.0, 0.0, 10.0, 0.0])
    q = q_init.copy()  # six joints

    robot.q = q_init.copy()

    # compute script directory once for potential STL meshes
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Continuously toggle between teach GUI and Swift until Ctrl+C
    mode = "teach"
    final_q = robot.q.copy()
    print("Entering interactive loop. Use Enter to switch modes; use Ctrl+C to exit.")
    try:
        while True:
            if mode == "teach":
                print("Starting teach GUI. Move the joints to your desired pose.")
                plt.close()  # Close any existing figure because teach will create a new one
                fig = robot.teach(robot.q, limits=workspace, block=False, backend="pyplot")

                # Wait until user presses Enter to finish adjusting the pose
                while True:
                    try:
                        if keyboard.is_pressed("enter"):
                            break
                    except Exception:
                        # fallback to blocking input if keyboard module can't detect keys
                        input("Press Enter to continue to simulation...")
                        break
                    print("q = ", robot.q)
                    try:
                        fig.step(0.05)
                    except Exception:
                        time.sleep(0.05)

                # Capture the pose the user set in the teach GUI
                final_q = robot.q.copy()
                print("\nFinal joint state (radians):", np.round(final_q, 4))
                print("Final joint state (degrees):", np.round(np.rad2deg(final_q), 2))
                plt.close("all")
                mode = "swift"

            elif mode == "swift":
                # Now launch Swift and apply the final pose to the simulation
                try:
                    # Launch a Swift environment and add the robot
                    env = swift.Swift()
                    env.launch(realtime=True)

                    # compute initial link transforms (fkine_all returns base + link transforms)
                    robot.q = final_q
                    try:
                        T_all = robot.fkine_all(robot.q)
                    except Exception:
                        T_all = None

                    # Attempt to add Link0..Link6 meshes (base + links) if present in the same folder
                    meshes = {}
                    for k in range(0, 7):
                        stl_name = f"Link{k}.stl"
                        stl_path = os.path.join(script_dir, stl_name)
                        if os.path.exists(stl_path):
                            try:
                                mesh = geometry.Mesh(
                                    stl_path,
                                    pose=SE3(),
                                    scale=(1.0, 1.0, 1.0),
                                    color=(0.2, 0.2, 0.7, 1),
                                )
                                env.add(mesh)
                                # if we have fkine results, set the initial pose to match the robot link
                                try:
                                    if T_all is not None:
                                        mesh.T = T_all[k].A
                                except Exception:
                                    # ignore per-mesh pose assignment errors
                                    pass
                                meshes[k] = mesh
                                print(f"Added mesh {stl_name} from: {stl_path}")
                            except Exception as mesh_e:
                                print(f"[Warning] Could not load {stl_name}:", mesh_e)
                        else:
                            print(f"STL not found: {stl_path}; skipping.")

                    # one step to ensure the scene updates
                    try:
                        env.step(0.05)
                    except Exception:
                        pass
                    print("Launched Swift and added available STL meshes at final pose.")

                    # Run Swift until user presses Enter, then return to teach
                    while True:
                        try:
                            if keyboard.is_pressed("enter"):
                                break
                        except Exception:
                            # fallback to blocking input if keyboard module can't detect keys
                            input("Press Enter to return to teach GUI...")
                            break

                        # Update mesh poses to match robot forward kinematics 
                        try:
                            if 'meshes' in locals() and len(meshes) > 0:
                                try:
                                    T_all = robot.fkine_all(robot.q)
                                    for (idx, mesh) in meshes.items():
                                        try:
                                            mesh.T = T_all[idx].A
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                        except Exception:
                            pass

                        try:
                            env.step(0.05)
                        except Exception:
                            time.sleep(0.05)

                    # Close/cleanup the Swift environment before switching back
                    try:
                        if hasattr(env, "close"):
                            env.close()
                    except Exception:
                        pass

                except Exception as e:
                    # Fall back to matplotlib if Swift is not available or fails
                    print("[Warning] Could not open Swift or add robot:", str(e))
                    print("Falling back to matplotlib view.")
                    robot.q = final_q
                    view = robot.plot(q=final_q, backend="pyplot", limits=workspace)

                    # Wait here until Enter to return to teach
                    while True:
                        try:
                            if keyboard.is_pressed("enter"):
                                break
                        except Exception:
                            input("Press Enter to return to teach GUI...")
                            break
                        try:
                            view.step(0.05)
                        except Exception:
                            time.sleep(0.05)
                    plt.close("all")

                mode = "teach"

    except KeyboardInterrupt:
        # Allow Ctrl+C to exit the interactive loop
        print("\nKeyboard interrupt received, exiting interactive loop.")
        try:
            plt.close("all")
        except Exception:
            pass
        try:
            if 'env' in locals() and hasattr(env, "close"):
                env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
