import pybullet as p
import time

# Import object classes
from objects.SmallCube import SmallCube
from objects.Cylinder import Cylinder

# Import data storage and gripper classes
from data.DataStorage import DataStorage
from grippers.GripperThree import ThreeFingerGripper
from grippers.GripperTwo import TwoFingerGripper

# Import simulation logic and utility functions
from simulation.Simulation import Simulation
from simulation.utils import gripper_orientation_position_update

# Import classifier class
from classifier.ClassifierModel import run_models


def run_simulation():
    """
        Initialises objects, grippers, environments + runs full grasping simulation.

        Performs following steps:
            1. Create objects (cubes & cylinders)
            2. Create DataStorage environments with sampling radii
            3. Generate all grasping positions & orientations
            4. Create two-finger and three-finger grippers
            5. Initialise simulation class
            6. Run simulation for each grasp pose:
                - Spawn objects
                - Update gripper pose
                - Spawn grippers
                - Attempt grasping
                - Lift object or terminate
                - Save grasp outcome
                - Remove objects + continue
            7. Save results to CSV files
            8. Disconnect PyBullet
    """

    # Two-finger gripper + cube
    obj_g2_cube = SmallCube((0, 0, 0.03))
    env_g2_cube = DataStorage([0.3, 0.33, 0.37], obj_g2_cube, 2)
    env_g2_cube.input_data()

    # Two-finger gripper + cylinder
    obj_g2_cylinder = Cylinder((0, 2, 0.06))
    env_g2_cylinder = DataStorage([0.3, 0.32, 0.38], obj_g2_cylinder, 2)
    env_g2_cylinder.input_data()

    # Three-finger gripper + scaled cube
    obj_g3_cube = SmallCube((2, 0, 0.05), scale=2.0)
    env_g3_cube = DataStorage([0.2, 0.23, 0.28], obj_g3_cube, 3)
    env_g3_cube.input_data()

    # Three-finger gripper + scaled cylinder
    obj_g3_cylinder = Cylinder((2, 2, 0.1), scale=1.5)
    env_g3_cylinder = DataStorage([0.21, 0.25, 0.30], obj_g3_cylinder, 3)
    env_g3_cylinder.input_data()

    # Create all grippers
    g2_cube = TwoFingerGripper()
    g2_cylinder = TwoFingerGripper()
    g3_cube = ThreeFingerGripper()
    g3_cylinder = ThreeFingerGripper()

    # Initialise simulation engine
    sim = Simulation([obj_g2_cube, obj_g2_cylinder, obj_g3_cube, obj_g3_cylinder],
                     [g2_cube, g2_cylinder, g3_cube, g3_cylinder],
                     [env_g2_cube, env_g2_cylinder, env_g3_cube, env_g3_cylinder])

    # Prepare PyBullet environment
    sim.setup_environment_visual()

    # Run simulation for all training data
    for i in range(sum(env_g2_cube.data_points)):
        print(f'Simulation no.{i}')

        # Spawn all objects into environment
        sim.spawn_objects()

        # Update pose of each gripper before spawning them
        gripper_orientation_position_update(g2_cube, env_g2_cube, i)
        gripper_orientation_position_update(g2_cylinder, env_g2_cylinder, i)
        gripper_orientation_position_update(g3_cube, env_g3_cube, i)
        gripper_orientation_position_update(g3_cylinder, env_g3_cylinder, i)

        # Spawn all grippers
        sim.spawn_grippers()

        # Move all grippers simultaneously towards object and close
        terminate = sim.move_towards_objs_and_close(i)

        # If the terminate condition is True then remove all objects from environment and set output to NaN
        if terminate == 1:
            sim.remove_objs()
            env_g2_cube.lifted_or_not(terminate)
            env_g2_cylinder.lifted_or_not(terminate)
            env_g3_cube.lifted_or_not(terminate)
            env_g3_cylinder.lifted_or_not(terminate)
            continue

        # Lift objects after closing grippers
        sim.lift_objects()

        # Check which objects were successfully lifted after holding them for three seconds
        env_g2_cube.lifted_or_not()
        env_g2_cylinder.lifted_or_not()
        sim.hold_3_seconds()
        env_g3_cube.lifted_or_not()
        env_g3_cylinder.lifted_or_not()

        # Remove objects and reset the environment
        sim.remove_objs()
        time.sleep(0.5)

        # Update data to separate csv files after each iteration
        env_g2_cube.output_data('output_g2_cube.csv')
        env_g2_cylinder.output_data('output_g2_cylinder.csv')
        env_g3_cube.output_data('output_g3_cube.csv')
        env_g3_cylinder.output_data('output_g3_cylinder.csv')

    # Disconnect PyBullet after simulating all grasp poses
    p.disconnect()

if __name__ == "__main__":
    # Run simulation and generate training data
    run_simulation()
    # Run model classification on training data + output model accuracy
    run_models()