from grippers.BaseGripper import Gripper
from objects.BaseObject import BaseObject
import pybullet as p
import time
import math
import numpy as np


class TwoFingerGripper(Gripper):
    """
    Simple two-finger gripper implementation.

    This class provides specific functionality for:
    - opening and closing two finger joints,
    - computing a grasp orientation based on an object's position,
    - performing upward lifting motions using a fixed world constraint

    It is the subclasses of Gripper, which defines the core interface for all
    gripper types used in the grasp planning pipeline.
    """
    def __init__(self) -> None:
        """
        Initialise a two-finger gripper using the PR2 gripper URDF
        The gripper is spawned at pos: (0, 0, 0) with zero orientation
        """
        super().__init__('pr2_gripper.urdf', (0, 0, 0), (0, 0, 0))

    # Opening the gripper 
    def open_gripper(self) -> None:
        """
        Fully opens both fingers of the gripper 

        ! Note:
        The PR2 gripper has four joints, where joints 0 and 2 are linked
        to the finger motion. Joints 1 and 3 are fixed.
        """

        # Joint positions for an open position of the PR2 gripper
        initial_positions = [0.550569, 0.0, 0.549657, 0.0]
        for i, pos in enumerate(initial_positions):
            p.resetJointState(self.id, i, pos)

    # Closing the gripper 
    def close_gripper(self) -> None:
        """
        Closing the two finger joints(0 and 2) using position control

        ! Note:
        Only joints 0 and 2 must be commanded, because they control
        the two symmetric moving fingers
        """
        for joint in [0, 2]:
            p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL,
                                    targetPosition=0.1, maxVelocity=10, force=100)

    # Finding orientation for grasping
    def find_orientation(self, obj_position: tuple[float, float, float]) -> tuple[float, float, float]:
        """
        Computes a noisy approach orientation toward the object

        The orientation is estimated by creating a vector from the
        gripper base to the object. Euler angles are computed for pitch,
        roll and yaw. Gaussian noise is added to avoid repeated identical
        grasp attempts in stochastic grasp planning.

        Parameters: 
        obj_position : tuple[float, float, float]
            Position of the target object.

        Returns:
        tuple[float, float, float]
            (pitch, roll, yaw) Euler angles with Gaussian noise.
        """

        # Vector pointing from object to gripper
        vector = (self.base_position[0] - obj_position[0],
                  self.base_position[1] - obj_position[1],
                  self.base_position[2] - obj_position[2])
        
        # Normalising the directional vector
        length = math.sqrt(sum(v ** 2 for v in vector))
        self.direction = [v / length for v in vector]

        # Compute noise using standard deviation
        std_dev = 0.05

        # Compute roll and yaw (tilt toward the object and rotation around vertical axis)
        yaw = ((3 / 2) * math.pi) - math.atan2(vector[0], vector[1])
        roll = math.atan2(vector[2], math.sqrt(vector[0] ** 2 + vector[1] ** 2))

        # Adding Gaussian noise to each angle
        roll_noisy = roll + np.random.normal(0, std_dev)
        yaw_noisy = yaw + np.random.normal(0, std_dev)
        pitch_noisy = 0 + np.random.normal(0, std_dev)

        return pitch_noisy, roll_noisy, yaw_noisy

    # Moving up the gripper
    def move_up(self, z: float, yaw:float =0.0) -> None:
        """
        Moves the gripper vertically upward, but also maintains current orientation

        Parameters: 
        z : float
            Desired Z-height for the gripper base.
        yaw_angle : float, optional
            Desired yaw angle for the gripper orientation.

        Raises: 
        ValueError
            If the gripper has not been attached to the world with a fixed
            constraint before attempting to move it.
        """

        # Ensure the gripper is fixed with a constraint
        if self.constraint_id is None:
            raise ValueError("Gripper must be fixed before moving.")

        # Get current position and orientation
        current_pos, current_ori = p.getBasePositionAndOrientation(self.id)

        # Update only the Z-coordinate
        new_pos = [current_pos[0], current_pos[1], z]

        # Update the fixed constraint to the new position
        p.changeConstraint(
            self.constraint_id,
            jointChildPivot=new_pos,
            jointChildFrameOrientation=current_ori,  # Use current orientation
            maxForce=50  # Increase maxForce to ensure object follows
        )

    # Grasp and lift method
    def grasp_lift(self, obj: BaseObject, target_pos, lift_height:float =0.4, lift_steps:int =150) -> None:
        """
        Perform a grasping lift operation after making contact with an object.

        Parameters:
        obj : object
            The object being grasped. Must provide grasp_height attribute
        lift_height : float, optional
            Final height to lift the object to.
        lift_steps : int, optional
            Number of simulation steps to perform during the lifting motion.

        Notes:
        - The method assumes the gripper is already placed at the correct
          grasp position above the object from before
        - Finger forces are applied during the lift to maintain secure contact.
        """
        yaw_angle = 0.0
        grasp_height = obj.grasp_height

        # Set friction on contact surfaces 
        for _ in range(10):  # allow contact to form
            p.stepSimulation()
            time.sleep(1. / 240.)

        # Initialise lifting parameters
        z_current = grasp_height
        z_target = lift_height
        z_step = (z_target - z_current) / lift_steps

        # Perform the lifting motion in steps
        for _ in range(lift_steps):
            z_current += z_step
            self.move_up(z_current, yaw_angle)

            for joint in [0, 2]:
                p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL,
                                        targetPosition=0.12, force=400, maxVelocity=2)
            p.stepSimulation()
            time.sleep(1. / 240.)




