import pybullet as p

def gripper_orientation_position_update(gripper, env, i):
    """
        Updates gripper's base position + orientation for i-th data point.

        Args:
            gripper (Gripper): An instance of a gripper (TwoFingerGripper / ThreeFingerGripper).
            env (DataStorage): Environment containing positions + orientations of gripper.
            i (int): Index of current data point to use from environment.
    """
    # Update gripper's base position
    gripper.base_position = [env.base_position['x'][i], env.base_position['y'][i], env.base_position['z'][i]]
    # Convert Euler angles (pitch, roll, yaw) to quaternion
    gripper.base_orientation = p.getQuaternionFromEuler((env.data['pitch'][i], env.data['roll'][i], env.data['yaw'][i]))
