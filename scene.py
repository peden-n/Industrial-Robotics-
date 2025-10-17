import swift
from spatialmath import SE3
import spatialgeometry as geometry
import os

# Uniform scale (set to 0.001 if your STL is in mm)
SCALE = 0.001
SCALE_TUPLE = (SCALE, SCALE, SCALE)

# launch the environment
env = swift.Swift()
env.launch()

# read objects from stl files and put them at global (0,0,0)
current_dir = os.path.dirname(os.path.abspath(__file__))

Stand1 = os.path.join(current_dir, "Stand1.stl")
Stand2 = os.path.join(current_dir, "Stand2.stl")
Stand3 = os.path.join(current_dir, "Stand3.stl")
TableLong = os.path.join(current_dir, "TableLong.stl")
TableShort = os.path.join(current_dir, "TableShort.stl")
Conveyor = os.path.join(current_dir, "Conveyor.stl")
Pallet = os.path.join(current_dir, "Pallet.stl")
Walls = os.path.join(current_dir, "Walls.stl")


# Create meshes at world origin and apply uniform scale. Adjust colors as needed.
Stand1 = geometry.Mesh(Stand1, color=(0.5, 0, 0, 1), scale=SCALE_TUPLE)
Stand2 = geometry.Mesh(Stand2, pose=SE3(0.0, 0.0, 0.0).A, color=(0.2, 0.5, 0.2, 1), scale=SCALE_TUPLE)
Stand3 = geometry.Mesh(Stand3, color=(0.2, 0.2, 0.7, 1), scale=SCALE_TUPLE)
TableLong = geometry.Mesh(TableLong, pose=SE3(0.0, 0.53, 0.4).A, color=(0.75, 0.75, 0.75, 1), scale=SCALE_TUPLE)
TableShort = geometry.Mesh(TableShort, pose=SE3(0.65, 1.48, 0.4).A, color=(0.75, 0.75, 0.75, 1), scale=SCALE_TUPLE)
Conveyor = geometry.Mesh(Conveyor, pose=SE3(0, -0.5, 0.48).A, color=(0.3, 0.3, 0.3, 1), scale=SCALE_TUPLE)
Pallet = geometry.Mesh(Pallet, pose=SE3(-0.9325, 1.7625, 0.0).A, color=(0.545, 0.271, 0.075, 1), scale=SCALE_TUPLE)
Walls = geometry.Mesh(Walls, color=(1.0, 1.0, 1.0, 1), scale=SCALE_TUPLE)


env.add(Stand1)
env.add(Stand2)
env.add(Stand3)
env.add(TableLong)
env.add(TableShort)
env.add(Conveyor)
env.add(Pallet)
env.add(Walls)


# If you want them not to overlap, set poses individually, e.g.:
# TableLong.T = SE3(0.0, 0.53, 0).A
# Stand3.T = SE3(-0.5, 0, 0).A
# Stand4.T = SE3(0, 0.5, 0).A

# update the scene to see change
input('Press Enter to step and hold the environment...')
env.step(0.01)
env.hold()