from objects.BaseObject import BaseObject

class Cylinder(BaseObject):
    """
        Represents a cylinder in the simulation.

        Inherits from BaseObject and specifically defines the urdf path of the cylinder

        Attributes:
            grasp_height (float): The default height above the cylinder at which it can be grasped.
    """
    def __init__(self, position, orientation=(0, 0, 0), scale=1.0):
        super().__init__('objects/urdf_files/cylinder.urdf', position, orientation, scale)
        self.grasp_height = 0.1