# demo_option_b.py
import numpy as np
from spatialmath import SE3
#from roboticstoolbox.models.DH import UR3
from spatialgeometry import Cuboid, Cylinder, Mesh
from roboticstoolbox.backends.swift import Swift
from ir_support.robots.UR3 import UR3  # package from your link
from collision import AABB
from planner import move_robot_with_replanning
from math import pi

#Define offests to fix donut positions
donut_x_offset = -0.08
donut_y_offset = -0.08
donut_z_offset = 0.0
burnt_x_offset = -0.08
burnt_y_offset = -0.08
burnt_z_offset = 0.0

donut_offset = SE3(donut_x_offset, donut_y_offset, donut_z_offset)
burnt_offset = SE3(burnt_x_offset, burnt_y_offset, burnt_z_offset)


ounut1 = Mesh(
    filename="/Users/nataliebusch/Documents/Industal Robotics/Assignment 2/Industrial-Robotics-/Donut2.dae",
    pose=(SE3(-0.305, -0.405, 0.48)+ donut_offset),  # place + rotate (degrees)
    scale= (0.001, 0.001, 0.001)                                          # adjust if needed
  )

class PleaseAppear:
    def __init__(self):
         print('hi')


    def coltest(self):
            

            # 1) Swift environment
        swift = Swift(); 
        swift.launch(realtime=True)

        # 2) UR3 at your base pose
        r = UR3()
        r.base = SE3(-0.1, -0.1, 0.0)
        #swift.add(r)
        r.add_to_env(swift)
        swift.step(0.05)
        #swift.hold()
        
        # 3) Two keep-out boxes in the way
        obstacles = [
            AABB.from_center_size(center=(0.55, 0.8, 0.05), size=(1.5, 1.3, 0.4)),  # tabletop slab
           # AABB.from_center_size(center=(0.55, 0.60, 1.00), size=(0.20, 0.20, 1.00)),  # pillar/post
        ]

        # (Optional) visualize the boxes
        try:
            from spatialgeometry import Cuboid
            def add_box(aabb, color=[1, 0, 0, 0.25]):
                cx = 0.5 * (aabb.min_xyz[0] + aabb.max_xyz[0])
                cy = 0.5 * (aabb.min_xyz[1] + aabb.max_xyz[1])
                cz = 0.5 * (aabb.min_xyz[2] + aabb.max_xyz[2])
                sx = (aabb.max_xyz[0] - aabb.min_xyz[0])
                sy = (aabb.max_xyz[1] - aabb.min_xyz[1])
                sz = (aabb.max_xyz[2] - aabb.min_xyz[2])
                swift.add(Cuboid([sx, sy, sz], pose=SE3(cx, cy, cz), color=color))
                
            for b in obstacles:
                add_box(b, color=[1,0,0,0.25])
                add_box(b, color=[0,0,1,0.15], inflate=0.03)
        except Exception:
            pass

        # 4) Pick a simple goal: move +0.35 m in X, keep current orientation
        T_now = r.fkine(r.q)
        T_goal = SE3(-0.3,-0.2,0.42) * SE3.Rx(pi) 

       # rc = RunController(make_gui=True)    # opens a tiny window with Pause/STOP

        ok = move_robot_with_replanning(
            robot=r,
            T_target=T_goal,
            env=swift,
            obstacles=obstacles,
                                # pass the controller in
        )



        # 5) Move with Option B (auto-replan)
        ok = move_robot_with_replanning(
            robot=r,
            T_target=T_goal,
            env=swift,
            obstacles=obstacles,
            link_radius=0.03,        # pretend links are ~3 cm thick
            steps_per_segment=90,    # smoothness
            z_clear=0.02,            # fly 18 cm above the tallest obstacle
            max_detours=3,           # try up to 3 side detours
            dt=0.02,
        )

        #T_goal = SE3(0,0.21,0.2) 
        #T_goal= SE3(0.2,0.2, 0.9)

        T_goal = SE3(0,0.21,0.27) * SE3.Rx(pi)
        #cr = RunController
        ok = move_robot_with_replanning(
            robot=r,
            T_target=T_goal,
            env=swift,
            obstacles=obstacles,
            link_radius=0.03,        # pretend links are ~3 cm thick
            steps_per_segment=90,    # smoothness
            z_clear=0.02,            # fly 18 cm above the tallest obstacle
            max_detours=3,           # try up to 3 side detours
            dt=0.02
        )
        print("Executed:", ok)


if __name__ == "__main__":
    go = PleaseAppear()
    PleaseAppear.coltest(go)