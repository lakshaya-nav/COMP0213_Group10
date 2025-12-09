import pybullet as p

class BaseObject:
    """
        A parent class representing a general object in a PyBullet simulation scene.

        Attributes:
            __urdf_file (str): Path to the URDF file representing the object's model.
            position (tuple[float, float, float]): The (x, y, z) coordinates for the object's initial position.
            orientation (tuple[float, float, float]): The Euler angles (roll, pitch, yaw) for the object's initial orientation.
            scale (float): Scaling factor applied to the object when loaded in the simulation.
            _id (int or None): The unique PyBullet ID assigned to the object after loading. Initialized as None.
            _name (str or None): The name of the object, typically derived from its class and ID. Initialized as None.
        """
    def __init__(self, urdf_file: str, position: tuple[float, float, float], orientation: tuple[float, float, float], scale: float) -> None:
        self.__urdf_file = urdf_file
        self.position = position
        self.orientation = p.getQuaternionFromEuler(orientation)
        self.scale = scale
        self._id = None
        self._name = None

    def load(self) -> int:
        """
            Loads the object into the PyBullet simulation.

            Returns:
                _id (int): The unique PyBullet ID of the loaded object.
        """
        self._id = p.loadURDF(self.__urdf_file, basePosition=self.position, baseOrientation=self.orientation,
                              globalScaling=self.scale)
        return self._id

    def update_name(self, _id) -> None:
        """
            Updates the object's name based on its class and PyBullet ID.

            Args:
                _id (int): The PyBullet ID of the object including its name.
        """
        self._name = f'{self.__class__.__name__}{_id}'