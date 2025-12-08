import numpy as np
import csv
import pybullet as p
from objects.BaseObject import BaseObject
from grippers.BaseGripper import Gripper
from grippers.GripperTwo import TwoFingerGripper
from grippers.GripperThree import ThreeFingerGripper

class DataStorage:
    def __init__(self, radii: list[float], obj: BaseObject, gripper_type: int) -> None:
        """
            Class to manage the generation, storage, and export of gripper-object pair data.

            Attributes:
                obj (BaseObject): Object in the simulation for which data is being collected.
                obj_pos (tuple[float, float, float]): Position of the object.
                data (dict): Stores positions (x, y, z), orientations (pitch, roll, yaw), output states, radii +object position.
                base_position (dict): Stores computed base positions for gripper.
                data_points (list[int]): Number of points to generate for each radius of semisphere around object.
                gripper_type (int): Type of gripper being used (2 for TwoFinger, 3 for ThreeFinger).
                gripper (Gripper): Instance of selected gripper.
        """
        self.obj = obj
        self.obj_pos = obj.position
        self.data = {'x': [], 'y': [], 'z': [], 'pitch': [], 'roll': [], 'yaw': [], 'outputs': [],
                     'radii': radii, 'obj pos': self.obj_pos}
        self.base_position = {'x': [], 'y': [], 'z': []}
        self.data_points = [300, 250, 450]
        self.gripper_type = gripper_type
        if self.gripper_type == 2:
            self.gripper = TwoFingerGripper()
        elif self.gripper_type == 3:
            self.gripper = ThreeFingerGripper()

    def add_position(self) -> None:
        """
            Generates sample points on spheres of different radii around object.
            Stores the points in keys: x, y, z in data dictionary.
        """
        for i in range(len(self.data_points)):
            # Create points around object using gripper's static method
            x, y, z = Gripper.create_points_sphere(self.data['obj pos'], self.data['radii'][i], self.data_points[i])
            self.data['x'].extend(x)
            self.data['y'].extend(y)
            self.data['z'].extend(z)

    def find_base_position(self) -> None:
        """
            Calculates gripper's base positions for each sampled point around object.
            Approach pose (base positions) are extended be a factor of 2 from grasp pose (sampled points) along the vector from object to point.
        """
        scalar = 2
        # append the calculated base positions to the base position dictionary
        for i in range(len(self.data['x'])):
            self.base_position['x'].append(((self.data['x'][i] - self.obj_pos[0]) * scalar) + self.data['x'][i])
            self.base_position['y'].append(((self.data['y'][i] - self.obj_pos[1]) * scalar) + self.data['y'][i])
            self.base_position['z'].append(((self.data['z'][i] - self.obj_pos[2]) * scalar) + self.data['z'][i])

    def add_orientation(self) -> None:
        """
            Calculates orientation (pitch, roll, yaw) of gripper for each base position.
            Stores the results in keys: pitch, roll and yaw in data dictionary.
        """
        for i in range(len(self.data['x'])):
            # Set gripper's base position for current point
            self.gripper.base_position = (self.base_position['x'][i],
                                          self.base_position['y'][i],
                                          self.base_position['z'][i])
            # Compute orientation of gripper relative to object and append to dictionary
            pitch, roll, yaw = self.gripper.find_orientation(self.obj_pos)
            self.data['pitch'].append(pitch)
            self.data['roll'].append(roll)
            self.data['yaw'].append(yaw)

    def lifted_or_not(self, terminate=0) -> bool :
        """
            Determines if object has been lifted successfully based on its z-position.

            Args:
                terminate (int): If set to 1, indicates trial is terminated and stores value as NaN.

            Returns:
                bool: True if terminated, False otherwise.
        """
        state = 1
        if terminate == 1:
            self.data['outputs'].append(np.nan)
            return True

        # Check object's z position to determine if lifted
        if p.getBasePositionAndOrientation(self.obj._id)[0][2] < 0.25:
            state = 0
        # Append outputs to the data dictionary
        self.data['outputs'].append(state)
        return False

    def dict_to_csv(self, filename: str) -> None:
        """
            Exports collected gripper + object data saved in a dictionary to a CSV file.

            Args:
                filename (str): Name of CSV file to create.
        """
        keys = list(self.data.keys())[:7]
        values = zip(self.data['x'], self.data['y'], self.data['z'], self.data['pitch'], self.data['roll'],
                     self.data['yaw'], self.data['outputs'])

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(keys)
            writer.writerows(values)

    def input_data(self) -> None:
        """
            Generates all input data for gripper-object pair.

            1. Generates sample points around object at specified radii.
            2. Computes gripper base positions for each sampled point.
            3. Calculates gripper's orientation for each base position.
        """
        self.add_position()
        self.find_base_position()
        self.add_orientation()

    def output_data(self, filename: str) -> None:
        """
           Exports collected input + output data to a CSV file.

           Args:
               filename (str): The name of CSV file where data will be saved.
        """
        self.dict_to_csv(filename)