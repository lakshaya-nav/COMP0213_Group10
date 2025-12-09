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
3. Install the required Python packages from the requirements.txt file by using the following command: ``` pip install -r requirements.txt ```

## Usage
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
* Path Issues: If you encounter ```FileNotFoundError``` related to CSVs or models, ensure you are running the script from inside the src folder, as the relative paths (e.g., ../data/) depend on the execution context.

* PyBullet GUI: The simulation runs in ```p.GUI``` mode by default. If you are running this on a headless server, you may need to change ```p.connect(p.GUI)``` to ```p.connect(p.DIRECT)``` in ```src/simulation/Simulation.py.```

  
