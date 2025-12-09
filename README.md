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
  pip install -r requirements.txt

## Usage

   




  
