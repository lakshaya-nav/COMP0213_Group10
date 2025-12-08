import pybullet as p
from abc import ABC, abstractmethod
import numpy as np


# Abstract class for Grippers
class Gripper(ABC):
    """
    Abstract base class for all gripper types used in the grasp planning system.

    This class defines the fundamental interface and common functionality for
    urdf_files simulated in PyBullet. Concrete subclasses must implement
    the abstract methods for opening, closing, orienting, and executing 
    grasp-and-lift actions.

    Parameters:
    urdf_path : str
        Path to the URDF file describing the gripper model.
    base_position : sequence of float (len 3)
        Initial (x, y, z) position where the gripper should be spawned.
    base_orientation : sequence of float (len 3)
        Euler angles (roll, pitch, yaw) used to compute the initial orientation.

    Attributes:
    _urdf_path : str
        Stored URDF path for internal use.
    base_position : list[float]
        Mutable list storing the gripper’s base world position.
    base_orientation : tuple[float]
        Quaternion orientation derived from the Euler input.
    id : int or None
        The PyBullet body ID returned after loading.
    constraint_id : int or None
        ID of a fixed constraint if the gripper is locked to the world.
    grasp_moving : bool
        Whether the gripper is currently performing a grasp movement.
    num_joints : int
        Number of joints in the loaded URDF model.
    """

    def __init__(self, urdf_path: str, base_position: tuple[float, float, float],
                  base_orientation: tuple[float, float, float]) -> None:
        self._urdf_path = urdf_path
        self.base_position = list(base_position)
        self.base_orientation = p.getQuaternionFromEuler(base_orientation)
        self.id = None
        self.constraint_id = None
        self.grasp_moving = False
        self.num_joints = 0

    def load(self) -> int:
        """
        Load gripper URDF into the PyBullet simulation

        Returns:
        int
            The PyBullet body unique ID of the loaded gripper
        """

        self.id = p.loadURDF(self._urdf_path, basePosition=self.base_position, 
                             baseOrientation=self.base_orientation)

        # Storing num of joints
        self.num_joints = p.getNumJoints(self.id)
        return self.id

    def attach_fixed(self) -> None:
        """
        Attaching gripper to the pybullet world using a fixed constraint

        Prevents gripper from moving freely; is useful for 
        stationary grasping experiments or calibration setups.
        """

        self.constraint_id = p.createConstraint(
            parentBodyUniqueId=self.id,
            parentLinkIndex=-1, # -1 indicates the base link
            childBodyUniqueId=-1, # -1 binds to the world
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=self.base_position,
            childFrameOrientation=self.base_orientation
        )

    @staticmethod

    def create_points_sphere(obj_pos: tuple[float, float, float], radius: float, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate n random points on a spherical surface around the object

        This method samples points uniformly on a spherical surface restricted to a polar angle range 
        (0° to 80°), with norm/gaussian noise
        The sampled points can be used as the approach or grasp points

        Parameters:
        obj_pos : tuple[float, float, float]
            Center of the sphere (typically the object's position)
        radius : float
            Radius of the spherical sampling region
        n : int
            Number of sample points

        Returns:
        tuple of np.ndarray
            Arrays (x, y, z) each of length n, representing sampled points
        """
        (cx, cy, cz) = obj_pos

        # Restricting phi to form a spherical region above object from (0° to 80°)
        phi_min = np.deg2rad(0)
        phi_max = np.deg2rad(80)

        # Random sampling for spherical coordinates
        u = np.random.rand(n)
        v = np.random.uniform(np.cos(phi_max), np.cos(phi_min), n)

        theta = 2 * np.pi * u
        phi = np.arccos(v)

        # Standard deviation of applied Gaussian noise
        std_dev = 0.02 * radius

        # Converting spherical to Cartesian coordinates
        x = radius * np.sin(phi) * np.cos(theta) + cx
        y = radius * np.sin(phi) * np.sin(theta) + cy
        z = radius * np.cos(phi) + cz

        # Adding Gaussian noise
        x += np.random.normal(0, std_dev, n)
        y += np.random.normal(0, std_dev, n)
        z += np.random.normal(0, std_dev, n)
        return x, y, z

    @abstractmethod
    def open_gripper(self) -> None:
        """
        Open the gripper fingers/jaws

        Subclasses to implement according to their joints
        """
        pass

    @abstractmethod
    def close_gripper(self) -> None:
        """
        Close the gripper fingers/jaws

        Subclasses to implement according to their joints
        """
        pass

    @abstractmethod
    def find_orientation(self, obj_pos: tuple[float, float, float]) -> tuple:
        """
        Compute desired gripper orientation relative to object/target position

        Parameters:
        obj_pos : tuple[float, float, float]
            The target object's position.

        Returns:
        tuple or list
            Orientation values (Euler or quaternion)
        """
        pass

    @abstractmethod
    def grasp_lift(self, obj: object, target_pos: tuple[float, float, float], lift_height:float =0.4, lift_steps:int =150) -> bool:
        """
        Perform a grasp and lift on an object

        Parameters:
        obj : Object
            The object instance to grasp
        target_pos : tuple[float, float, float]
            Coordinates to move toward when initiating the grasp
        lift_height : float, optional
            Height above the grasp point to lift the object
        lift_steps : int, optional
            Number of simulation steps used during the lifting motion

        Returns:
        bool
            True if the lift  is successful, False otherwise
        """
        pass

