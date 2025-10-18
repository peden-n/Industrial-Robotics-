
from dataclasses import dataclass
from typing import List, Tuple, Optional
import roboticstoolbox as rtb
from math import pi
from spatialmath import SE3
from spatialgeometry import Cuboid, Cylinder, Mesh
from roboticstoolbox.backends.swift import Swift
from itertools import zip_longest
import numpy as np
from roboticstoolbox import DHLink, DHRobot
from ir_support import CylindricalDHRobotPlot
from scenenatsversion import Environment
from ir_support.robots.UR3 import UR3  # package from your link

#from roboticstoolbox.models.DH import UR3


Color = Tuple[float, float, float, float]  # RGBA 0..1

#Define offests to fix donut positions
donut_x_offset = -0.08
donut_y_offset = -0.08
donut_z_offset = 0.0
burnt_x_offset = -0.08
burnt_y_offset = -0.08
burnt_z_offset = 0.0

donut_offset = SE3(donut_x_offset, donut_y_offset, donut_z_offset)
burnt_offset = SE3(burnt_x_offset, burnt_y_offset, burnt_z_offset)

tool_offset = SE3(0, 0, 0)  # 40 mm above the donut when grasping from top
#----------------- meshes -----------------
dounut1 = Mesh(
    filename="/Users/nataliebusch/Documents/Industal Robotics/Assignment 2/Industrial-Robotics-/Donut2.dae",
    pose=(SE3(-0.305, -0.405, 0.48)+ donut_offset),  # place + rotate (degrees)
    scale= (0.001, 0.001, 0.001)                                          # adjust if needed
  )


dounut2 = Mesh(
    filename="/Users/nataliebusch/Documents/Industal Robotics/Assignment 2/Industrial-Robotics-/Donut2.dae",
    pose=(SE3(-0.495, -0.405, 0.48)+ donut_offset),  # place + rotate (degrees)
    scale= (0.001, 0.001, 0.001)                                          # adjust if needed
  )

burnt = Mesh(
    filename="/Users/nataliebusch/Documents/Industal Robotics/Assignment 2/Industrial-Robotics-/Donut_burnt.dae",
    pose=(SE3(-0.495, -0.58, 0.48)+ burnt_offset),  # place + rotate (degrees)
    scale= (0.001, 0.001, 0.001)                                          # adjust if needed
 )

import numpy as np
from spatialmath import SE3

# --- Geometry helpers ---



# ----------------- run -----------------
def move_ur3(env,robotTargets=[]):
        
        motions = []
        last_qs = []

        for ur3, T_target, donut in robotTargets:
            q_start = np.asarray(ur3.q, dtype=float)
            q_goal = solve_ik(ur3, T_target)
            #T_donut = None
            #if donut is not None:
            #    T_ee = ur3.fkine(ur3.q)
            #    T_donut = T_ee * donut_offset

            Q = rtb.jtraj(q_start, q_goal, 150).q
            motions.append((ur3, Q, donut))
            last_qs.append(Q[-1])


        for frame in zip_longest(*[Q for (_, Q, _) in motions], fillvalue=None):
        # frame is a tuple: (q1 or None, q2 or None, ...)
            for idx, (ur3, Q, donut) in enumerate(motions):
                q = frame[idx] if frame[idx] is not None else last_qs[idx]
                ur3.q = q
                if donut is not None:

                    T_ee = ur3.fkine(q)             # SE3 pose of end-effector
                    T_donut = T_ee * donut_offset

                    if hasattr(donut, "T"):
                        donut.T = T_donut
                    elif hasattr(donut, "pose"):
                        donut.pose = T_donut
                    else:
                    # Fallback: try a generic attribute name
                        try:
                            setattr(donut, "pose", T_donut)
                        except Exception:
                            pass
            env.step(0.02)
            
            



        #for q in traj:
        #    ur3.q = q
        #    if donut is not None:
        #      T_ee = ur3.fkine(q)             # SE3 pose of end-effector
        #      T_donut = T_ee * donut_offset     # apply any offset so it doesn’t intersect the gripper

               # Support both common spatialgeometry attributes
        #      if hasattr(donut, "T"):
        #        donut.T = T_donut
        #      elif hasattr(donut, "pose"):
        #        donut.pose = T_donut
        #      else:
                # Fallback: try a generic attribute name
        #        try:
        #            setattr(donut, "pose", T_donut)
        #        except Exception:
        #            pass
        #    self._env.step(0.02)


def solve_ik(robot, T_target):
     """
     Try full pose IK first; if it fails, try position-only (ignore orientation).
     Returns a numpy array of joint angles or raises a ValueError.
     """
     q0 = np.array(robot.q if robot.q is not None else np.zeros(robot.n), dtype=float)
    # print(T_target)
     # 1) full pose (position + orientation)
     try:
        sol = robot.ikine_LM(T_target, q0=q0)
        if sol.success:
            return np.array(sol.q, dtype=float)
     except Exception:
        pass

     # 2) position-only (mask: x,y,z true; roll/pitch/yaw false)
     try:
        mask = [1, 1, 1, 0, 0, 0]
        sol = robot.ikine_LM(T_target, q0=q0, mask=mask)
        if sol.success:
            return np.array(sol.q, dtype=float)
     except Exception:
        pass

     raise ValueError("IK did not converge for the requested target pose.")


if __name__ == "__main__":
    # platform along the Y axis in front of the table, included in the enclosure
    workenv = Environment()
    print("A---------------")
    
    base_pose= SE3(0,0,0.34) 
    
    env = workenv.get_env()
    print("A")
    r = UR3()
    r2= UR3()
    r.base = base_pose 
    r2.base = base_pose * SE3(-0.4, 0.0, 0.0) 
   
    
    q = [0, -pi/3, pi/3, 0, pi/2, 0]   # nice pose
    T_ee = r2.fkine(q)
    print("A")
    print (T_ee)
    r.add_to_env(workenv.env)
    print("A")
    env.step(0.01)

    env.add(dounut1)
    env.add(dounut2)

    env.add(burnt)
    l1 = DHLink(d=0.33, a=0.4, alpha=pi/2, qlim=[-pi, pi])
    l2 = DHLink(d=0, a=0.445, alpha=0, qlim=[-pi, pi])
    l3 = DHLink(d=0, a=0.04, alpha=pi/2, qlim=[-pi, pi])
    l4 = DHLink(d=0.44, a=0, alpha=-pi/2, qlim=[-pi, pi])
    l5 = DHLink(d=0.08, a=0, alpha=pi/2, qlim=[-pi, pi])
    l6 = DHLink(d=0, a=0, alpha=0, qlim=[-pi, pi])



    r2 = DHRobot([l1, l2, l3,l4, l5, l6], name='my_robot')
    #robot = LinearUR3()
    #Give the robot a cylinder mesh (links) to display in Swift environment
    cyl_viz = CylindricalDHRobotPlot(r2, cylinder_radius=0.03, color="#3478f6")
    r2 = cyl_viz.create_cylinders()
    r2.q = [0, np.deg2rad(30), -np.deg2rad(30), 0, np.deg2rad(40), 0]
    #print(burnt.T [0], burnt.T [1], burnt.T [2])
    #get_nut = SE3(dounut1.T[0], dounut1.T[1], dounut1.T[2])  # 10 cm above 
    #move_ur3(cell, r, T_above)
    
    r2.base = r2.base * SE3(-0.1,-1,0.35)
    env.add(r2)
    GRASP_FROM_TOP = SE3.Rx(pi) * SE3(0, 0, -0.08) 
    #print(get_nut)
    get_nut = SE3(0.1, 0.2, 0.95)
   

    
    

    
    




    move_ur3(env,robotTargets=[
        [r,SE3(-0.305, -0.405, 0.48) * GRASP_FROM_TOP, None],
        [r2,SE3(-0.495, -0.58, 0.48)* GRASP_FROM_TOP, None]
        ]
        )
       # -0.405, -0.405, 0.48
    
    move_ur3(env,robotTargets=[
        [r,SE3(-0.40, 0.3, 0.48) * GRASP_FROM_TOP, dounut1],
         [r2,SE3(-0.6, -0.8, 0.48)* GRASP_FROM_TOP, burnt]]
        )
    #-0.40, 0.135, 0.40
    move_ur3(env,robotTargets=[
        [r,SE3(-0.405, -0.405, 0.48) * GRASP_FROM_TOP, None],
        [r2,SE3(-0.5, -0.3,1.0) * GRASP_FROM_TOP , None]
        ]
        )         
    
    move_ur3(env,robotTargets=
        [[r,SE3(-0.40, 0.0, 0.40)* GRASP_FROM_TOP, dounut2]])

    move_ur3(env,robotTargets=
        [[r,SE3(-0.4, 0.4,1.1) * GRASP_FROM_TOP, None]])
    
    #cell.move_ur3(cell._env.robots[0],[0, -pi/3, pi/3, 0, pi/2, 0])
    #move_ur3(cell,r,SE3(0.2, 0.9, 0.90) * GRASP_FROM_TOP, donut=dounut1)
   # x, y, z = dounut1.T[:3, -1]
    #cell._env.add(dounut2)
    #move_ur3(cell,r,get_nut * GRASP_FROM_TOP)
    #move_ur3(cell,r,SE3(0.2, 0.9, 0.93) * GRASP_FROM_TOP, donut=dounut2)
  #  get_nut = (SE3(x, y, z)- donut_offset) # 10 cm above
    #move_ur3(cell,r,get_nut * GRASP_FROM_TOP)
   # cell._env.add(dounut2)

    input("Scene ready (platform on Y axis, enclosure includes table+platform, 3 UR3s). Press Enter to quit...")

