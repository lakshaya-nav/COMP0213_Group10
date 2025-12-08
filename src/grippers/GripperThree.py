from grippers.BaseGripper import Gripper
import pybullet as p
import time
import math
import numpy as np

class ThreeFingerGripper(Gripper):
    """
    Three-finger adaptive gripper implementation

    This gripper provides functionality for:
    - opening and closing the three fingers in the gripper,
    - computing a grasp orientation based on an object's position,
    - performing upward lifting motions using a fixed world constraint.

    It is a subclass of Gripper, which defines the core interface for all
    gripper types used in the grasp planning pipeline.
    """

    def __init__(self) -> None:
        """
        Initialises the three-finger gripper using the SDH URDF.
        It is spawned at position (0, 0, 0) with zero Euler orientation.

        Notes: 
        The SDH hand uses 9 total joints:
        - Each finger has 3 joints: proximal, middle and distal.
        """

        super().__init__("grippers/urdf_files/threeFingers/sdh/sdh.urdf", (0, 0, 0), (0, 0, 0))

        # Set the number of joints and initial states
        self.num_joints = 9
        self.open = False
        self.close = True

    # Utilities for joint control
    def apply_joint_command(self, joint: int, target: float) -> None:
        """
        Applies a simple position control command to one joint.

        Parameters: 
        joint : int
            Joint index to be commanded.
        target : float
            Target joint angle in radians.
        """
        p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL,
                                targetPosition=target, maxVelocity=2, force=5)

    def get_joint_positions(self) -> list[float]:
        """
        Returns current joint angles for all joints in the gripper.

        Return: 
        list[float]
            A list of joint positions of length `self.num_joints`.

        ! Note:  
        If gripper ID = invalid (for example not loaded yet), a default list
        of 0 values is returned.
        """

        if self.id is None:
            print("Warning: Gripper ID is None, cannot get joint positions.")
            return [0.0] * self.num_joints
        try:
            return [p.getJointState(self.id, i)[0] for i in range(self.num_joints)]
        except Exception as e:
            print(f"Error getting joint states for gripper {self.id}: {e}")
            return [0.0] * self.num_joints

    # Opening the gripper
    def open_gripper(self) -> None:
        """
        Opens the three-finger gripper.

        Notes: 
        For this gripper model:
        - Joints 1, 4, and 7 represent the main flexion joints
          for the fingers.
        - Opening is performed by gradually reducing their angles until
          the minimum safe extension value.
        """
        closed = True
        Gripper.open = False
        iteration = 0
        while closed and not Gripper.open:
            joints = self.get_joint_positions()
            closed = False
            for k in range(self.num_joints):
                if k in [1, 4, 7] and joints[k] <= 0.9:

                    # Move finger joints toward an open angle
                    self.apply_joint_command(k, joints[k] - 0.05)

                    closed = True
            iteration += 1
            if iteration > 500:
                break
            p.stepSimulation()

    # Computing grasp orientation
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
        
        # Normalise the vector
        length = math.sqrt(sum(v ** 2 for v in vector))
        self.direction = [v / length for v in vector]

        # Compute noise using standard deviation
        std_dev = 0.05

        # Compute yaw and pitch
        yaw = (2 * math.pi) - math.atan2(vector[0], vector[1])
        pitch = math.atan2(vector[2], math.sqrt(vector[0] ** 2 + vector[1] ** 2)) + math.pi / 2

        # Adding Gaussian noise to each angle
        roll_noisy = 0 + np.random.normal(0, std_dev)
        yaw_noisy = yaw + np.random.normal(0, std_dev)
        pitch_noisy = pitch + np.random.normal(0, std_dev)

        return pitch_noisy, roll_noisy, yaw_noisy

    # Closing the gripper
    def close_gripper(self) -> None:
        """
        Closes all three fingers by applying target angles to all joints.

        Notes:
        Each finger has 3 joints out of the 9 total:
        - Proximal (0, 3, 6)
        - Middle   (1, 4, 7)
        - Distal   (2, 5, 8)
        """
        target_positions = {0: 0.6, 1: 0.5, 2: 0.8, 3: 0.6, 4: 0.5, 5: 0.8, 6: 0.6, 7: 0.5, 8: 0.8}

        for k, pos in target_positions.items():
            self.apply_joint_command(k, pos)

        # Allow motion to settle
        for _ in range(200):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

    # Performing a grasp and lift
    def grasp_lift(self, lift_height:float =0.4, lift_steps: int =150) -> None:
        """
        Performs a grasping lift operation after contact has formed.

        Parameters:
        lift_height : float, optional
            Height to lift the object by.
        lift_steps : int, optional
            Number of simulation steps used for the lifting motion.

        Notes:
        During the lift:
        - The three main flexion joints (1, 4, 7) apply an increased grip.
        - A fixed constraint must already be created before lifting begins.
        """
        current_pos = p.getBasePositionAndOrientation(self.id)[0]

        start_pos = np.array([current_pos[0], current_pos[1], current_pos[2]])
        end_pos = np.array([current_pos[0], current_pos[1], current_pos[2] + lift_height])
        
        # Interpolate between start and end positions
        for interp_pos in np.linspace(start_pos, end_pos, lift_steps):
            for j in [1, 4]:
                p.setJointMotorControl2(self.id, j, p.POSITION_CONTROL,
                                        targetPosition=0.05, maxVelocity=1, force=400)
            p.setJointMotorControl2(self.id, 7, p.POSITION_CONTROL,
                                    targetPosition=0.05, maxVelocity=1, force=400)

            p.changeConstraint(self.constraint_id, interp_pos,
                               jointChildFrameOrientation=self.base_orientation, maxForce=50)
            p.stepSimulation()
            time.sleep(0.01)

