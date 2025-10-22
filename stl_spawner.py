# stl_spawner.py - Tiny utility to spawn, place, and update STL meshes in a Swift scene.

from __future__ import annotations
import os
from typing import Optional, Tuple, Union
from spatialmath import SE3
import spatialgeometry as geometry

Vector3 = Tuple[float, float, float]
RGBA = Tuple[float, float, float, float]


class STLSpawner:
    """Utility class for importing and manipulating STL meshes in a Swift scene."""

    # Sets up default empty environment with colour and scale.
    def __init__(self, env, default_scale: float = 0.001, default_color: RGBA = (0.8, 0.8, 0.8, 1.0)):
        """Initialise an STL spawner.

        Args:
            env (swift.Swift): Swift simulation environment.
            default_scale (float, optional): Default scaling factor for meshes.
                Defaults to 0.001.
            default_color (tuple[float, float, float, float], optional): Default RGBA
                colour for meshes. Defaults to (0.8, 0.8, 0.8, 1.0).
        """
        self.env = env
        self.default_scale = default_scale
        self.default_color = default_color

    # Public API
    # Function to import the stl with a pre attached usable config
    def import_stl(
        self,
        stl_filename: str,
        folder: Optional[str] = None,
        translation: Vector3 = (0.0, 0.0, 0.0),
        rpy_deg: Optional[Vector3] = None,                              
        quat_xyzw: Optional[Tuple[float, float, float, float]] = None,  
        matrix_A: Optional["list[list[float]]"] = None,                 
        scale: Union[float, Tuple[float, float, float]] = None,
        color: Optional[RGBA] = None,
        show_axes: bool = False,
        axes_len: float = 0.05,
    ):
        """Import an STL file into the Swift environment.

        Args:
            stl_filename (str): Name of the STL file.
            folder (str, optional): Folder containing the STL file. Defaults to None.
            translation (tuple[float, float, float], optional): Position (x,y,z) in meters.
                Defaults to (0.0, 0.0, 0.0).
            rpy_deg (tuple[float, float, float], optional): Orientation (roll, pitch, yaw)
                in degrees. Defaults to None.
            quat_xyzw (tuple[float, float, float, float], optional): Quaternion (x,y,z,w).
                Defaults to None.
            matrix_A (list[list[float]], optional): 4x4 homogeneous transform matrix.
                Defaults to None.
            scale (float or tuple[float, float, float], optional): Mesh scaling factor.
                Defaults to self.default_scale.
            color (tuple[float, float, float, float], optional): Mesh RGBA colour.
                Defaults to self.default_color.
            show_axes (bool, optional): Whether to add a frame axes for the mesh.
                Defaults to False.
            axes_len (float, optional): Length of the axes lines. Defaults to 0.05.

        Returns:
            geometry.Mesh or tuple: Mesh object, or (mesh, axes) if show_axes is True.
        """
        # Import an STL with an initial transform into the Swift environment.
        stl_path = self._resolve_path(stl_filename, folder)
        pose = self._make_pose(translation, rpy_deg, quat_xyzw, matrix_A)
        scale_tuple = self._scale_tuple(scale if scale is not None else self.default_scale)
        rgba = color if color is not None else self.default_color

        mesh = geometry.Mesh(stl_path, pose=pose, scale=scale_tuple, color=rgba)
        self.env.add(mesh)

        axes = None 
        if show_axes:
            # Axes object visualises the mesh's frame
            axes = geometry.Axes(length=axes_len, pose=mesh.T)
            self.env.add(axes)

        self.env.step(0.01)
        return (mesh, axes) if show_axes else mesh

    def add_axes(self, pose: Union[SE3, "list[list[float]]"], length: float = 0.05):
        """Adds a coordinate axes object to the environment.

        Args:
            pose (SE3 or list[list[float]]): Pose of the axes.
            length (float, optional): Length of the axes lines. Defaults to 0.05.

        Returns:
            geometry.Axes: The created axes object.
        """
        axes = geometry.Axes(length=length, pose=pose.A if hasattr(pose, "A") else pose)
        self.env.add(axes)
        self.env.step(0.01)
        return axes

    def set_pose(
        self,
        mesh_or_axes: geometry.Geometry,
        translation: Optional[Vector3] = None,
        rpy_deg: Optional[Vector3] = None,
        quat_xyzw: Optional[Tuple[float, float, float, float]] = None,
        matrix_A: Optional["list[list[float]]"] = None,
    ):
        """Set an absolute pose on a mesh or axes object.

        Args:
            mesh_or_axes (geometry.Geometry): The mesh or axes to update.
            translation (tuple[float, float, float], optional): Position (x,y,z) in meters.
                Defaults to (0,0,0).
            rpy_deg (tuple[float, float, float], optional): Orientation (roll, pitch, yaw)
                in degrees. Defaults to None.
            quat_xyzw (tuple[float, float, float, float], optional): Quaternion (x,y,z,w).
                Defaults to None.
            matrix_A (list[list[float]], optional): 4x4 homogeneous transform matrix.
                Defaults to None.
        """
        # Convenience to set an absolute pose on a mesh/axes.
        T = self._make_pose(translation or (0, 0, 0), rpy_deg, quat_xyzw, matrix_A)
        mesh_or_axes.T = T.A if hasattr(T, "A") else T
        self.env.step(0.01)

    def apply_delta(self, mesh_or_axes: geometry.Geometry, dSE3: SE3):
        """Apply an incrementtal transform to a mesh or axes.

        Args:
            mesh_or_axes (geometry.Geometry): The object to update.
            dSE3 (SE3): Delta transformation to apply.
        """
        # Moving from gloobal to local coordinate frames.
        mesh_or_axes.T = (dSE3.A @ mesh_or_axes.T)
        self.env.step(0.01)

    # Resolve file path - HELPER to figure out the path of the stl file
    @staticmethod
    def _resolve_path(stl_filename: str, folder: Optional[str]) -> str:
        """Resolve the absolute path of an STL file.

        Args:
            stl_filename (str): Name of the STL file.
            folder (str, optional): Folder path. Defaults to script's directory.

        Returns:
            str: Absolute path to the STL file.
        """
        if os.path.isabs(stl_filename):
            return stl_filename
        base = folder if folder is not None else os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base, stl_filename)

    # Build pose - HELPER to create a full SE3 transform
    @staticmethod
    def _make_pose(
        translation: Vector3,
        rpy_deg: Optional[Vector3],
        quat_xyzw: Optional[Tuple[float, float, float, float]],
        matrix_A: Optional["list[list[float]]"],
    ) -> SE3:
        """Create an SE3 transform from multiple possible input formats.

        Args:
            translation (tuple[float, float, float]): Translation vector (x, y, z).
            rpy_deg (tuple[float, float, float], optional): Roll, pitch, yaw in degrees.
                Defaults to None.
            quat_xyzw (tuple[float, float, float, float], optional): Quaternion (x,y,z,w).
                Defaults to None.
            matrix_A (list[list[float]], optional): 4x4 homogeneous transform matrix.
                Defaults to None.

        Returns:
            SE3: Constructed pose as an SE3 object.
        """
        T = SE3()
        if matrix_A is not None:
            T = SE3(matrix_A)
        elif quat_xyzw is not None:
            x, y, z, w = quat_xyzw
            T = SE3.Rquat([w, x, y, z])
        elif rpy_deg is not None:
            rx, ry, rz = rpy_deg
            T = SE3.Rx(rx, unit="deg") * SE3.Ry(ry, unit="deg") * SE3.Rz(rz, unit="deg")
        tx, ty, tz = translation
        return SE3(tx, ty, tz) * T

    # Scale conversion - HELPER to to ensure mesh is always scaling in (x, y, z) format.
    @staticmethod
    def _scale_tuple(scale: Union[float, Tuple[float, float, float]]):
        """Ensure scaling is in (x,y,z) tuple format.

        Args:
            scale (float or tuple[float, float, float]): Uniform scale factor
                or explicit per-axis scale.

        Returns:
            tuple[float, float, float]: Scale factors in (x,y,z) format.
        """
        return (scale, scale, scale) if isinstance(scale, (int, float)) else scale
