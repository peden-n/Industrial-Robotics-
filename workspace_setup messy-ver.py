# workcell_platform_y_axis.py
# Swift scene: table + legs + bowl + adaptive enclosure + platform on X or Y axis (with 3 UR3s facing table).

from dataclasses import dataclass
from typing import List, Tuple, Optional
import roboticstoolbox as rtb
from math import pi
from spatialmath import SE3
from spatialgeometry import Cuboid, Cylinder, Mesh
from roboticstoolbox.backends.swift import Swift
from itertools import zip_longest
import numpy as np
from ir_support.robots.UR3 import UR3  # package from your link
from ir_support.robots.DHRobot3D import DHRobot3D
from ir_support import CylindricalDHRobotPlot
from roboticstoolbox import DHLink, DHRobot, jtraj
import numpy as np

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
    pose=(SE3(0.15, 0.26, 0.9)+ donut_offset),  # place + rotate (degrees)
    scale= (0.001, 0.001, 0.001)                                          # adjust if needed
  )

dounut2 = Mesh(
    filename="/Users/nataliebusch/Documents/Industal Robotics/Assignment 2/Industrial-Robotics-/Donut2.dae",
    pose=(SE3(0.0, 0.2, 0.9)+ donut_offset),  # place + rotate (degrees)
    scale= (0.001, 0.001, 0.001)                                          # adjust if needed
  )

burnt = Mesh(
    filename="/Users/nataliebusch/Documents/Industal Robotics/Assignment 2/Industrial-Robotics-/Donut_burnt.dae",
    pose=(SE3(0.3, 0.1, 0.9)+ burnt_offset),  # place + rotate (degrees)
    scale= (0.001, 0.001, 0.001)                                          # adjust if needed
 )

# ----------------- fixtures -----------------

@dataclass
class Table:
    length: float = 2.5
    width: float = 0.8
    thickness: float = 0.06
    top_z: float = 0.90
    color: Color = (0.80, 0.80, 0.80, 1.0)

    def build(self) -> Cuboid:
        return Cuboid([self.length, self.width, self.thickness],
                      pose=SE3(0, 0, self.top_z - self.thickness/2),
                      color=self.color)

    @property
    def underside_z(self) -> float:
        return self.top_z - self.thickness


@dataclass
class TableLegs:
    table: Table
    leg_size: float = 0.07
    inset: float = 0.08
    color: Color = (0.35, 0.35, 0.38, 1.0)

    def build(self) -> List[Cuboid]:
        leg_h = self.table.underside_z
        zc = leg_h / 2.0
        ox = self.table.length/2 - self.inset - self.leg_size/2
        oy = self.table.width/2  - self.inset - self.leg_size/2
        dims = [self.leg_size, self.leg_size, leg_h]
        return [Cuboid(dims, pose=SE3(sx*ox, sy*oy, zc), color=self.color)
                for sx in (+1, -1) for sy in (+1, -1)]


@dataclass
class Platform:
    """Box next to the table for mounting robots."""
    length: float = 0.8   # X dimension
    width: float  = 1.0   # Y dimension
    height: float = 0.85  # Z top surface
    gap_to_table: float = 0.52
    axis: str = "y"       # "x" (left/right of table) or "y" (front/back of table)
    side: str = "front"   # if axis="y": "front"(+Y) or "back"(-Y); if axis="x": "left"(-X) or "right"(+X)
    color: Color = (0.55, 0.55, 0.58, 0.60)

    def center_pose(self, table: Table) -> SE3:
        if self.axis == "y":
            y_sign = +1 if self.side == "front" else -1
            x = 0.0
            y = y_sign * (table.width/2 + self.gap_to_table + self.width/2)
        else:
            x_sign = -1 if self.side == "left" else +1
            x = x_sign * (table.length/2 + self.gap_to_table + self.length/2)
            y = 0.0
        z = self.height/2
        return SE3(x, y, z)

    def build(self, table: Table) -> Cuboid:
        return Cuboid([self.length, self.width, self.height],
                      pose=self.center_pose(table),
                      color=self.color)

   


@dataclass
class Enclosure:
    """Auto-sized clear walls to include both table and platform."""
    wall_thickness: float = 0.01
    wall_height: float = 1.8
    clearance: float = 0.7
    color: Color = (0.80, 0.90, 1.00, 0.20)

    def walls_from_bounds(self, minx: float, maxx: float, miny: float, maxy: float) -> List[Cuboid]:
        # expand bounds by clearance
        minx -= self.clearance
        maxx += self.clearance
        miny -= self.clearance
        maxy += self.clearance

        inner_x = maxx - minx
        inner_y = maxy - miny
        cx = (maxx + minx) / 2.0
        cy = (maxy + miny) / 2.0
        zc = self.wall_height / 2.0

        w_px = Cuboid([self.wall_thickness, inner_y, self.wall_height], pose=SE3(cx + inner_x/2, cy, zc), color=self.color)
        w_nx = Cuboid([self.wall_thickness, inner_y, self.wall_height], pose=SE3(cx - inner_x/2, cy, zc), color=self.color)
        w_py = Cuboid([inner_x, self.wall_thickness, self.wall_height], pose=SE3(cx, cy + inner_y/2, zc), color=self.color)
        w_ny = Cuboid([inner_x, self.wall_thickness, self.wall_height], pose=SE3(cx, cy - inner_y/2, zc), color=self.color)
        return [w_px, w_nx, w_py, w_ny]


# ----------------- workcell -----------------

class Workcell:
    def __init__(
        self,
        table: Optional[Table] = None,
        legs: Optional[TableLegs] = None,
        platform: Optional[Platform] = None,
        enclosure: Optional[Enclosure] = None,
    ) -> None:
        self.table = table or Table()
        self.legs = legs or TableLegs(table=self.table)
        self.platform = platform or Platform(axis="y", side="front")  # default per your ask
        self.enclosure = enclosure or Enclosure()
        self._env: Optional[Swift] = None

    def launch(self) -> Swift:
        env = Swift()
        env.launch(realtime=True)
        self._env = env
        return env

    def populate(self) -> None:
        if self._env is None:
            raise RuntimeError("Call launch() before populate()")
        # build main geometry
        tbl = self.table.build()
        self._env.add(tbl)
        for leg in self.legs.build(): self._env.add(leg)

        plat = self.platform.build(self.table)
        self._env.add(plat)

        # mount UR3s on platform, facing table
       # self.add_ur3(base_pose= SE3(0,0.75,0.85) )
         
         #elf.platform.center_pose(self.table) * SE3(-0.3, -0.2, 0)
        # compute union bounds (table + platform) for enclosure
        minx, maxx, miny, maxy = self._plan_bounds_for_enclosure()
        for w in self.enclosure.walls_from_bounds(minx, maxx, miny, maxy):

            self._env.add(w)

   

    def add_ur3(self, base_pose: SE3):
        """Place ONE UR3 at the given base pose (world frame)."""

        r = UR3()
        r.base = base_pose
        r.add_to_env(self._env)
        self._env.add(r)
        self._env.step(0)  # force a render after adding robots
        return r

    

    def _plan_bounds_for_enclosure(self) -> Tuple[float, float, float, float]:

        # table bounds (center at 0,0)
        t_minx = -self.table.length/2
        t_maxx =  self.table.length/2
        t_miny = -self.table.width/2
        t_maxy =  self.table.width/2

        # platform bounds
        P = self.platform
        C = P.center_pose(self.table)
        p_minx = C.t[0] - P.length/2
        p_maxx = C.t[0] + P.length/2
        p_miny = C.t[1] - P.width/2
        p_maxy = C.t[1] + P.width/2

        # union
        minx = min(t_minx, p_minx)
        maxx = max(t_maxx, p_maxx)
        miny = min(t_miny, p_miny)
        maxy = max(t_maxy, p_maxy)
        return minx, maxx, miny, maxy
     # ---- helpers ----
    def solve_ik(robot, T_target):
     """
     Try full pose IK first; if it fails, try position-only (ignore orientation).
     Returns a numpy array of joint angles or raises a ValueError.
     """
     q0 = np.array(robot.q if robot.q is not None else np.zeros(robot.n), dtype=float)

     # 1) full pose (position + orientation)
     try:
        mask = [1, 1, 1, 1, 1, 1]
        sol = robot.ikine_LM(T_target, q0=q0, mask=mask)
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
    
    

# ----------------- run -----------------
def move_ur3(self,robotTargets=[]):
        
        motions = []
        last_qs = []

        for robot, T_target, donut in robotTargets:
            q_start = np.asarray(robot.q, dtype=float)
            q_goal = solve_ik(robot, T_target)
            #T_donut = None
            #if donut is not None:
            #    T_ee = ur3.fkine(ur3.q)
            #    T_donut = T_ee * donut_offset

            Q = rtb.jtraj(q_start, q_goal, 150).q
            motions.append((robot, Q, donut))
            last_qs.append(Q[-1])


        for frame in zip_longest(*[Q for (_, Q, _) in motions], fillvalue=None):
        # frame is a tuple: (q1 or None, q2 or None, ...)
            for idx, (robot, Q, donut) in enumerate(motions):
                q = frame[idx] if frame[idx] is not None else last_qs[idx]
                robot.q = q
                if donut is not None:

                    T_ee = robot.fkine(q)             # SE3 pose of end-effector
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
            self._env.step(0.02)
            
            



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

def solve_ik2(robot, T_target):
     """
     Try full pose IK first; if it fails, try position-only (ignore orientation).
     Returns a numpy array of joint angles or raises a ValueError.
     """
     q0 = np.array(robot.q if robot.q is not None else np.zeros(robot.n), dtype=float)
     #print(T_target)
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



def move_bot(self,DHRobot, T_target: SE3):
        
        q_start = np.array(DHRobot.q, dtype=float)
        q_goal = solve_ik2(DHRobot, T_target)

           
        traj = rtb.jtraj(q_start, q_goal, 150).q
        for q in traj:
            DHRobot.q = q
            self._env.step(0.02)
# ---------------------------------------------------------------------------------------#



if __name__ == "__main__":
    # platform along the Y axis in front of the table, included in the enclosure
    cell = Workcell(
        platform=Platform(axis="y", side="front", length=1.4, width=1.0, height=0.85, gap_to_table=0.02)
    )
    base_pose= SE3(0,0.60,0.85) 
    
    env = cell.launch()
    cell.populate()
    r = UR3()
    #r2= UR3()
    r.base = base_pose 
    #r2.base = base_pose * SE3(0.4, 0.0, 0.0) 
    r.add_to_env(cell._env)
    #r2.add_to_env(cell._env)
    q = [0, -pi/3, pi/3, 0, pi/2, 0]   # nice pose
    #T_ee = r2.fkine(q)

   # print (T_ee)
    cell._env.add(r)
    cell._env.step(0) 
    cell._env.add(dounut1)
    cell._env.add(dounut2)

    cell._env.add(burnt)



    l1 = DHLink(d=0.33, a=0.4, alpha=pi/2, qlim=[-pi, pi])
    l2 = DHLink(d=0, a=0.445, alpha=0, qlim=[-pi, pi])
    l3 = DHLink(d=0, a=0.04, alpha=pi/2, qlim=[-pi, pi])
    l4 = DHLink(d=0.44, a=0, alpha=-pi/2, qlim=[-pi, pi])
    l5 = DHLink(d=0.08, a=0, alpha=pi/2, qlim=[-pi, pi])
    l6 = DHLink(d=0, a=0, alpha=0, qlim=[-pi, pi])



    robot = DHRobot([l1, l2, l3,l4, l5, l6], name='my_robot')
    #robot = LinearUR3()
    #Give the robot a cylinder mesh (links) to display in Swift environment
    cyl_viz = CylindricalDHRobotPlot(robot, cylinder_radius=0.03, color="#3478f6")
    robot = cyl_viz.create_cylinders()
    robot.q = [0, np.deg2rad(30), -np.deg2rad(30), 0, np.deg2rad(40), 0]


    #Call 3 robot models
    #r1 = LinearUR3()

    robot.base = robot.base * SE3(0.4,0.70,0.85) # * SE3.Rz(-pi/2) # adjust the robot's base postion 




    #print(burnt.T [0], burnt.T [1], burnt.T [2])
    #get_nut = SE3(dounut1.T[0], dounut1.T[1], dounut1.T[2])  # 10 cm above 
    #move_ur3(cell, r, T_above)
    x, y, z = dounut1.T[:3, -1]
    #print(x, y, z)
   # get_nut = (SE3(x, y, z)- donut_offset) # 10 cm above
    GRASP_FROM_TOP = SE3.Rx(pi) * SE3(0, 0, -0.08) 
    #print(get_nut)
    get_nut = SE3(0.1, 0.2, 0.95)
   
    cell._env.add(robot)
    cell._env.step (5)
    move_ur3(cell,robotTargets=[
        [r,SE3(0.0, 0.2, 0.9) * GRASP_FROM_TOP, None],
      #  [r2,SE3(0.3, 0.1,0.9)* GRASP_FROM_TOP, None],
        [robot,SE3(0.3, 0.1,0.9) * GRASP_FROM_TOP, None]]
        )
    
    pls_move = SE3(-0.05,0.05,1.5) 
   # move_bot(cell,robot, pls_move)
   

    move_ur3(cell,robotTargets=[
        [r,SE3(-0.25, 0.3,0.9) * GRASP_FROM_TOP, dounut2],
        [robot,SE3(0.7, 0.5,0.9)* GRASP_FROM_TOP, burnt]]
        )
    
    move_ur3(cell,robotTargets=[
        [r,SE3(0.15, 0.25, 0.9) * GRASP_FROM_TOP, None],
        [robot,SE3(0.5, 0.3,1.0) * GRASP_FROM_TOP , None]]
        )         
    print (robot.q)
    move_ur3(cell,robotTargets=
        [[r,SE3(-0.25, 0.3, 0.94)* GRASP_FROM_TOP, dounut1]])

    move_ur3(cell,robotTargets=
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

