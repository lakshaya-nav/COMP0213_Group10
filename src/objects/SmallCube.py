from objects.BaseObject import BaseObject

class SmallCube(BaseObject):
    """
        Represents a small cube object in the simulation.

        Inherits from BaseObject and specifically defines the urdf path of the small cube

        Attributes:
            grasp_height (float): The default height above the cube at which it can be grasped.
    """
    def __init__(self, position, orientation=(0, 0, 0), scale=1.0) -> None:
        super().__init__('objects/urdf_files/cube_small.urdf', position, orientation, scale)
        self.grasp_height = 0.03