import pybullet as p
import pybullet_data
import time
import numpy as np


class Simulation:
    """
    Simulation manager for object–gripper interactions within PyBullet.

    This class provides functionality for:
    - setting up the physics environment (visual and non-visual),
    - spawning objects and urdf_files into the scene,
    - commanding urdf_files to approach target objects,
    - executing grasp attempts and lifting sequences,
    - maintaining friction, constraints, and simulation timing.

    It acts as the high-level controller for multi-gripper, multi-object
    experiments used in grasp planning and evaluation.
    """

    def __init__(self, objects: list, grippers: list, envs: list):
        """
        Initialises the simulation controller.

        Parameters:
        objects : list
            A list of object instances to be spawned.
        urdf_files : list
            A list of gripper instances (TwoFingerGripper, ThreeFingerGripper, etc.).
        envs : list
            A list of environment data structures storing sampling positions,
            orientation targets, and associated metadata.
        """
                
        self.objects = objects
        self.grippers = grippers
        self.envs = envs

        # Number of objects and urdf_files
        self.num_objects = len(self.objects)
        self.num_grippers = len(self.grippers)

    # VISUAL ENVIRONMENT SETUP
    @staticmethod
    def setup_environment_visual():
        """
        Sets up a PyBullet visual simulation environment, with GUI.

        Returns:
        int  
            The body ID of the loaded ground plane.

        """
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -10) # SetS gravity in negative Z direction, at 10 m/s^2

        p.setRealTimeSimulation(0)
        return p.loadURDF("plane.urdf")

    # NON-VISUAL ENVIRONMENT SETUP
    @staticmethod
    def setup_environment_non_visual():
        """
        Sets up a PyBullet non-visual simulation environment, without GUI.

        Returns:
        int  
            The body ID of the loaded ground plane.

        Notes:
        - This environment runs faster because no rendering is required.
        """
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -10) # SetS gravity in negative Z direction, at 10 m/s^2

        p.setRealTimeSimulation(0)
        return p.loadURDF("plane.urdf")

    def spawn_objects(self):
        """
        Loads all objects into the simulation using their internal load() method.
        """
        for i in range(self.num_objects):
            self.objects[i].load()

    def spawn_grippers(self):
        """
        Loads all urdf_files into the simulation and prepares them for interaction.

        Notes:
        - A fixed constraint is created to anchor each gripper.
        - Grippers are opened immediately after spawning.
        - If a gripper fails to load, a RuntimeError is raised.
        """
        for i in range(self.num_grippers):
            self.grippers[i].load()
            g = self.grippers[i]

            # Validating the gripper spawn/raising error if it fails
            if g.id < 0 or p.getNumJoints(g.id) == 0:
                raise RuntimeError(f"Failed to load gripper {g.id} from {g._urdf_path}")
            else:
                print(f"Successfully loaded gripper {g.id}")

            # Fix and open gripper
            self.grippers[i].attach_fixed()
            self.grippers[i].open_gripper()

    def move_towards_objs_and_close(self, i: int):
        """
        Moves each gripper toward its target object position and closes the gripper
        once all urdf_files have reached their respective approach points.

        Parameters:
        i : int  
            Index of the sampled grasp point in the environment data.

        Returns:
        int  
            1 if movement terminated early due to exceeding iteration limits,
            0 otherwise. (Deals with potential infinite loops.)

        Notes:
        - All urdf_files are moved simultaneously in a loop until they reach their targets.
        - Grippers move along the direction vector computed by find_orientation().
        - A threshold distance of 0.03 m is used to consider a gripper as "reached".
        """
                
        step_size = 0.05

        # Ensure all urdf_files are properly anchored
        for g in self.grippers:
            if g.constraint_id is None:
                raise ValueError("Gripper must be fixed before moving.")

        reached = [0,0,0,0]
        count = 0
        terminate = 0

        # Motion loop until all urdf_files reach targets
        while reached != [1,1,1,1]:
            if count > 100:
                terminate = 1
                return terminate

            for idx, (g, env)  in enumerate(zip(self.grippers, self.envs)):
                if reached[idx] == 1:
                    continue
                
                # Getting current and target positions
                curr_pos = np.array(p.getBasePositionAndOrientation(g.id)[0])
                target_pos = np.array((env.data['x'][i], env.data['y'][i], env.data['z'][i]))
                dist = np.linalg.norm(np.array(curr_pos) - np.array(target_pos))

                # Mark as reached if within threshold
                if dist < 0.03:
                    reached[idx] = 1
                    continue

                # Compute orientation and step direction
                g.find_orientation(env.obj_pos)
                new_pos = [curr_pos[0] - (g.direction[0] * step_size),
                           curr_pos[1] - (g.direction[1] * step_size),
                           curr_pos[2] - (g.direction[2] * step_size)]

                # Move gripper through the constraint update
                p.changeConstraint(
                    g.constraint_id,
                    jointChildPivot=new_pos,
                    jointChildFrameOrientation=g.base_orientation,
                    maxForce=50
                    )
            count += 1
            p.stepSimulation()
            time.sleep(1. / 240.)

        # Apply friction and initiate finger closure
        for g, obj in zip(self.grippers, self.objects):
            p.changeDynamics(obj._id, -1, lateralFriction=2.0, rollingFriction=0.1, spinningFriction=0.1)
            p.changeDynamics(g.id, -1, lateralFriction=2.0, rollingFriction=0.1, spinningFriction=0.1)
            g.close_gripper()

        return terminate

    def lift_objects(self, lift_height: float =0.4):
        """
        Lifts all grasped objects by raising the urdf_files vertically over a given height.

        Parameters:
        lift_height : float, optional
            The total vertical distance applied during the lifting action.

        Notes:
        - Two-finger and three-finger urdf_files use different joint actuation schemes.
        - The motion is carried out incrementally for simulation stability.
        """

        grippers_2 = []
        grippers_3 = []
        start_pos_2 = []
        start_pos_3 = []
        final_pos = []
        step_size = 0.01

        # Categorise urdf_files by type
        for g in self.grippers:
            position = p.getBasePositionAndOrientation(g.id)[0]
            position_list = list(position)

            # Handling 2 Finger Grippers
            if type(g).__name__ == "TwoFingerGripper":
                grippers_2.append(g)
                start_pos_2.append(position_list)

            # Handling 3 Finger Grippers
            elif type(g).__name__ == "ThreeFingerGripper":
                grippers_3.append(g)
                start_pos_3.append(position_list)

            # Final z-target for each gripper
            final_pos.append([position_list[0], 
                              position_list[1], 
                              position_list[2] + lift_height])

        current_pos_2 = start_pos_2
        current_pos_3 = start_pos_3

        # Lift sequence for two-finger urdf_files
        while current_pos_2[0][2] <= final_pos[0][2] and current_pos_2[1][2] <= final_pos[1][2]:
            # Move two-finger urdf_files up
            for g, pos in zip(grippers_2, current_pos_2):
                pos[2] += step_size
                g.move_up(pos[2], 0.0)

                # maintain grip force during lift
                for joint in [0, 2]:
                    p.setJointMotorControl2(g.id, joint, p.POSITION_CONTROL,
                                            targetPosition=0.12, force=400, maxVelocity=2)
                p.stepSimulation()
                time.sleep(0.01)
                
        # Lift sequence for three-finger urdf_files
        while current_pos_3[0][2] <= final_pos[2][2] and current_pos_3[1][2] <= final_pos[3][2]:
            for g, pos in zip(grippers_3, current_pos_3):

                # Actuate main finger joints (1, 4, 7) to hold the object during lift
                for j in [1, 4]:
                    p.setJointMotorControl2(g.id, j, p.POSITION_CONTROL,
                                            targetPosition=0.05, maxVelocity=1, force=400)
                p.setJointMotorControl2(g.id, 7, p.POSITION_CONTROL,
                                        targetPosition=0.05, maxVelocity=1, force=400)
                
                # Update constraint position
                p.changeConstraint(g.constraint_id, pos,
                                   jointChildFrameOrientation=g.base_orientation, maxForce=50)

                pos[2] += step_size
                p.stepSimulation()
                time.sleep(0.01)

    def hold_3_seconds(self):
        """
        Maintains finger closure for ≈3 seconds to stabilise the grasp.

        Notes:
        - Applies sustained joint torques for three-finger urdf_files.
        - Ensures the object does not slip after lifting.
        """

        # Duration and timing setup
        hold_time = 3.0
        time_step = 1. / 240.
        num_steps = int(hold_time / time_step)

        # Holding loop
        for _ in range(num_steps):
            for g in [self.grippers[2], self.grippers[3]]:
                for j in [1, 4]:
                    p.setJointMotorControl2(g.id, j, p.POSITION_CONTROL,
                                            targetPosition=0.05, maxVelocity=1, force=400)
                p.setJointMotorControl2(g.id, 7, p.POSITION_CONTROL,
                                        targetPosition=0.05, maxVelocity=1, force=400)

            p.stepSimulation()
            time.sleep(time_step)

    def remove_objs(self):
        """
        Removes all urdf_files and objects from the simulation environment.

        Notes:
        - removes constraints for safety, then removes bodies.
        - Ensures no IDs remain in the scene.
        """

        for g in self.grippers:

            # Remove constraints if they exist
            if g.constraint_id is not None:
                try:
                    p.removeConstraint(g.constraint_id)
                except Exception as e:
                    print(f"Failed to remove constraint {g.constraint_id}: {e}")
                g.constraint_id = None

            # Remove gripper bodies from simulation
            if g.id is not None:
                try:
                    p.removeBody(g.id)
                except Exception as e:
                    print(f"Failed to remove body {g.id}: {e}")
                g.id = None

        # Remove objects from simulation
        for obj in self.objects:
            p.removeBody(obj._id)
        time.sleep(0.5)

