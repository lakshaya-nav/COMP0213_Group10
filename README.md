# COMP0213_Group10

## Robotic Grasping Simulation & Classification
This project uses PyBullet to create a physics-based simulation of grasp planning, an important characteristic of modern robotics. The code is designed to generate data from the simulation and use a machine learning classifier to predict the success rate of future grasps at a given position and orientation.

The project utilises 2 different grippers (2 and 3 finger gripper), as well as 2 different types of geometric objects (cube and cylinder) to collect data. These different grippers and objects were used to evaluate the success rates of each gripper-object pair, as all the data collected from the 4 possible combinations were used to individually train a Logistic Regression model that classifies grasp configurations (using the position and orientation approach) as either successful or failed. 

## Features
* __Physics Simulation__: Uses PyBullet for realistic object interactions, collisions and real-world forces like gravity.
* __Multi-Gripper Support__: Includes implementations for a Two-Finger Gripper and a Three-Finger Gripper.
* __Data Generation__: Automates the collection of labelled grasp data (6-DOF pose + Success/Fail label): ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'output'\]
* __Machine Learning__: Integrated a Logistic Regression pipeline (using scikit-learn) to train and evaluate grasp quality.
* __Visualisation__: Generates Confusion Matrices and evaulates the models using metrics like accuracy and f1 score.

## Features
This project is based in Python. Ensure that Python 3.x is installed. Additionally, the project makes use of the following dependencies:
* PyBullet
* Numpy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn

## Installation
1. Clone or download the repository
2. Navigate to the project root directory from the terminal
3. Install the required Python packages from the requirements.txt file by using the following command:
   ```
   pip install -r requirements.txt
   ```

## Usage
### Running the entire pipeline
The entire pipeline is designed to be carried out by the ``` main.py ``` script.
1. Navigate to the source directory: ``` cd src ```
2. Run the main script:  ``` python main.py ```

__What you should see when you run it:__
1. The simulation should start, with all 4 gripper-object pairs loading in.
2. The gripper will start attempting hundreds of grasps on different objects. The grippers should spawn, move towards the object, and attempt to lift it. NOTE: This step will take the longest
3. The results of the simulation will automatically save in dedicated CSV files
4. After the simulation, the script triggers model training on the acquired data.
5. The terminal will output accuracy metrics and plots for the Confusion Matrices and ROC will be generated
6. The trained model is saved as a ``` .pkl ``` file.

### Running using argparse
While the entire pipeline can be run using the ``` main.py ``` script, generating and testing new samples can also be done using ``` argparse ```. 

__To generate new data__:
```
python main.py --mode generate_data --num_samples 150      # 150 can be changed to number of desired samples
```

__To train classifier on existing data__:
```
python main.py --mode train_classifier
```

__To test the generated data on existing model__:
```
python main.py --mode test_planner
```

## Project Structure

```
COMP0213_Group10/
├── requirements.txt         # List of python dependencies
├── README.md                # Project documentation
└── src/
    ├── main.py              # Entry point for simulation and training
    ├── simulation/          # Physics engine wrapper
    │   ├── Simulation.py    # Main simulation class (setup, loops)
    │   └── utils.py         # Helper functions
    ├── objects/             # Object definitions (URDFs, logic)
    │   ├── BaseObject.py
    │   ├── SmallCube.py
    │   └── Cylinder.py
    ├── grippers/            # Gripper definitions (Kinematics, Control)
    │   ├── BaseGripper.py
    │   ├── GripperTwo.py
    │   └── GripperThree.py
    ├── data/                # Data storage logic and CSV datasets
    │   └── DataStorage.py
    └── classifier/          # Machine Learning logic
        └── ClassifierModel.py
```
## Troubleshooting
* Path Issues: If you encounter ```FileNotFoundError``` related to CSVs or models, ensure you are running the script from inside the src folder, as the relative paths (e.g., ../data/) depend on the execution context. Ensure that the src is the source root folder in the local IDE configuration. Example, for VS Code:
   1. Open settings
   2. In the search bar, type: "extraPaths"
   3. Click 'Add item' for Python › Analysis: Extra Paths, and add the path to the src folder. ``` ${workspaceFolder}/src```
   4. Click  'Edit in Settings' for Python › Auto Complete: Extra Paths, and add the path to the src folder. ``` ${workspaceFolder}/src```

* PyBullet GUI: The simulation runs in ```setup_environment_non_visual()``` mode by default. If you wish to visualise the simulation, you may need to change ```setup_environment_non_visual()``` to ```setup_environment_visual()``` by replacing the ```run_simulation()``` function in ```main.py``` with the following:
```
def run_simulation(num_samples=None):
    """
        Initialises objects, grippers, environments + runs full grasping simulation.

        Args:
            num_samples: Number of samples to generate for each gripper-object combination.
                        If None, uses the default sampling radii to generate data points.

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
    if num_samples:
        env_g2_cube.data_points = [num_samples]
    env_g2_cube.input_data()

    # Two-finger gripper + cylinder
    obj_g2_cylinder = Cylinder((0, 2, 0.06))
    env_g2_cylinder = DataStorage([0.3, 0.32, 0.38], obj_g2_cylinder, 2)
    if num_samples:
        env_g2_cylinder.data_points = [num_samples]
    env_g2_cylinder.input_data()

    # Three-finger gripper + scaled cube
    obj_g3_cube = SmallCube((2, 0, 0.05), scale=2.0)
    env_g3_cube = DataStorage([0.2, 0.23, 0.28], obj_g3_cube, 3)
    if num_samples:
        env_g3_cube.data_points = [num_samples]
    env_g3_cube.input_data()

    # Three-finger gripper + scaled cylinder
    obj_g3_cylinder = Cylinder((2, 2, 0.1), scale=1.5)
    env_g3_cylinder = DataStorage([0.21, 0.25, 0.30], obj_g3_cylinder, 3)
    if num_samples:
        env_g3_cylinder.data_points = [num_samples]
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


```

  
