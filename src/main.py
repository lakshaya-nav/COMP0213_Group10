import pybullet as p
import time
from objects.SmallCube import SmallCube
from objects.Cylinder import Cylinder
from data.DataStorage import DataStorage
from grippers.GripperThree import ThreeFingerGripper
from grippers.GripperTwo import TwoFingerGripper
from simulation.Simulation import Simulation
from simulation.utils import gripper_orientation_position_update


def run_simulation():
    obj_g2_cube = SmallCube((0, 0, 0.03))
    env_g2_cube = DataStorage([0.3, 0.33, 0.37], obj_g2_cube, 2)
    env_g2_cube.input_data()

    obj_g2_cylinder = Cylinder((0, 2, 0.06))
    env_g2_cylinder = DataStorage([0.3, 0.32, 0.38], obj_g2_cylinder, 2)
    env_g2_cylinder.input_data()

    obj_g3_cube = SmallCube((2, 0, 0.05), scale=2.0)
    env_g3_cube = DataStorage([0.2, 0.23, 0.28], obj_g3_cube, 3)
    env_g3_cube.input_data()

    obj_g3_cylinder = Cylinder((2, 2, 0.1), scale=1.5)
    env_g3_cylinder = DataStorage([0.21, 0.25, 0.30], obj_g3_cylinder, 3)
    env_g3_cylinder.input_data()

    g2_cube = TwoFingerGripper()
    g2_cylinder = TwoFingerGripper()
    g3_cube = ThreeFingerGripper()
    g3_cylinder = ThreeFingerGripper()

    sim = Simulation([obj_g2_cube, obj_g2_cylinder, obj_g3_cube, obj_g3_cylinder],
                     [g2_cube, g2_cylinder, g3_cube, g3_cylinder],
                     [env_g2_cube, env_g2_cylinder, env_g3_cube, env_g3_cylinder])

    sim.setup_environment_visual()
    for i in range(sum(env_g2_cube.data_points)):
        print(f'Simulation no.{i}')

        sim.spawn_objects()
        gripper_orientation_position_update(g2_cube, env_g2_cube, i)
        gripper_orientation_position_update(g2_cylinder, env_g2_cylinder, i)
        gripper_orientation_position_update(g3_cube, env_g3_cube, i)
        gripper_orientation_position_update(g3_cylinder, env_g3_cylinder, i)
        sim.spawn_grippers()

        print("Gripper1 ID:", g2_cube.id)
        print("Number of joints:", p.getNumJoints(g2_cube.id))
        print("Gripper2 ID:", g2_cylinder.id)
        print("Number of joints:", p.getNumJoints(g2_cylinder.id))
        print("Gripper3 ID:", g3_cube.id)
        print("Number of joints:", p.getNumJoints(g3_cube.id))
        print("Gripper4 ID:", g3_cylinder.id)
        print("Number of joints:", p.getNumJoints(g3_cylinder.id))

        terminate = sim.move_towards_objs_and_close(i)
        if terminate == 1:
            sim.remove_objs()
            env_g2_cube.lifted_or_not(terminate)
            env_g2_cylinder.lifted_or_not(terminate)
            env_g3_cube.lifted_or_not(terminate)
            env_g3_cylinder.lifted_or_not(terminate)
            continue

        sim.lift_objects()
        env_g2_cube.lifted_or_not()
        env_g2_cylinder.lifted_or_not()
        sim.hold_3_seconds()
        env_g3_cube.lifted_or_not()
        env_g3_cylinder.lifted_or_not()
        sim.remove_objs()
        time.sleep(0.5)

        env_g2_cube.output_data('output_g2_cube.csv')
        env_g2_cylinder.output_data('output_g2_cylinder.csv')
        env_g3_cube.output_data('output_g3_cube.csv')
        env_g3_cylinder.output_data('output_g3_cylinder.csv')
    p.disconnect()


if __name__ == "__main__":
    run_simulation()