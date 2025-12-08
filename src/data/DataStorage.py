from src.grippers.GripperTwo import TwoFingerGripper
from src.grippers.GripperThree import ThreeFingerGripper

class DataStorage:
    def __init__(self, radii, obj, gripper_type):
        self.obj = obj
        self.obj_pos = obj.position
        self.data = {'x': [], 'y': [], 'z': [], 'pitch': [], 'roll': [], 'yaw': [], 'outputs': [],
                     'radii': radii, 'obj pos': self.obj_pos}
        self.base_position = {'x': [], 'y': [], 'z': []}
        # 300,240,180
        self.data_points = [300, 250, 450]
        self.gripper_type = gripper_type
        if self.gripper_type == 2:
            self.gripper = TwoFingerGripper()
        elif self.gripper_type == 3:
            self.gripper = ThreeFingerGripper()

    def add_position(self):
        for i in range(len(self.data_points)):
            x, y, z = Gripper.create_points_sphere(self.data['obj pos'], self.data['radii'][i], self.data_points[i])
            self.data['x'].extend(x)
            self.data['y'].extend(y)
            self.data['z'].extend(z)

    def find_base_position(self):
        scalar = 2
        for i in range(len(self.data['x'])):
            self.base_position['x'].append(((self.data['x'][i] - self.obj_pos[0]) * scalar) + self.data['x'][i])
            self.base_position['y'].append(((self.data['y'][i] - self.obj_pos[1]) * scalar) + self.data['y'][i])
            self.base_position['z'].append(((self.data['z'][i] - self.obj_pos[2]) * scalar) + self.data['z'][i])

    def add_orientation(self):
        for i in range(len(self.data['x'])):
            self.gripper.base_position = (self.base_position['x'][i], self.base_position['y'][i],
                                          self.base_position['z'][i])
            pitch, roll, yaw = self.gripper.find_orientation(self.obj_pos)
            self.data['pitch'].append(pitch)
            self.data['roll'].append(roll)
            self.data['yaw'].append(yaw)

    def lifted_or_not(self, terminate=0):
        state = 1
        if terminate == 1:
            self.data['outputs'].append(np.nan)
            return True
        if p.getBasePositionAndOrientation(self.obj._id)[0][2] < 0.25:
            state = 0
        self.data['outputs'].append(state)

    def dict_to_csv(self, filename):
        keys = list(self.data.keys())[:7]
        values = zip(self.data['x'], self.data['y'], self.data['z'], self.data['pitch'], self.data['roll'],
                     self.data['yaw'], self.data['outputs'])

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(keys)
            writer.writerows(values)

    def input_data(self):
        self.add_position()
        self.find_base_position()
        self.add_orientation()

    def output_data(self, filename):
        self.dict_to_csv(filename)