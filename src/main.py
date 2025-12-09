import pybullet as p
import time
import argparse

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


def test_planner():
    """
        Test the planner functionality by evaluating newly generated data
        on pre-trained logistic regression models.
        
        This function:
        1. Loads pre-trained models from trained_models folder
        2. Loads newly generated output data (output_g2_cube.csv, etc.)
        3. Evaluates the data on the models without retraining
        4. Displays performance metrics
    """
    import pickle
    import pandas as pd
    from pathlib import Path
    from sklearn import metrics
    
    print("Testing planner with pre-trained models...")
    print("=" * 60)
    
    # Define file mappings: (output_file, model_file, display_name)
    test_configs = [
        ("output_g2_cube.csv", "trained_model_g2_cube_final.pkl", "G2 Cube"),
        ("output_g2_cylinder.csv", "trained_model_g2_cylinder_final.pkl", "G2 Cylinder"),
        ("output_g3_cube.csv", "trained_model_g3_cube_final.pkl", "G3 Cube"),
        ("output_g3_cylinder.csv", "trained_model_g3_cylinder_final.pkl", "G3 Cylinder")
    ]
    
    # Get paths
    script_dir = Path(__file__).parent
    model_dir = script_dir / "classifier" / "trained_models"
    
    for output_file, model_file, name in test_configs:
        print(f"\n---------- Testing {name} ----------")
        
        # Load the output data
        output_path = script_dir / output_file
        model_path = model_dir / model_file
        
        # Check if files exist
        if not output_path.exists():
            print(f"Output file not found: {output_file}")
            print(f"Please run 'python main.py --mode generate_data' first.")
            continue
            
        if not model_path.exists():
            print(f"Model file not found: {model_file}")
            print(f"Please run 'python main.py --mode train_classifier' first.")
            continue
        
        # Load the model
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        # Load and prepare the data
        data = pd.read_csv(output_path)
        data = data.dropna(subset=['outputs'])
        
        if len(data) == 0:
            print(f"No valid data in {output_file}")
            continue
        
        # Separate features and target
        feature_cols = ['x', 'y', 'z', 'pitch', 'roll', 'yaw']
        X = data[feature_cols]
        y = data['outputs'].astype(int)
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        accuracy = metrics.accuracy_score(y, y_pred)
        precision = metrics.precision_score(y, y_pred, zero_division=0)
        recall = metrics.recall_score(y, y_pred, zero_division=0)
        f1 = metrics.f1_score(y, y_pred, zero_division=0)
        
        # Display results
        print(f"Data samples: {len(data)}")
        print(f"Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision:    {precision:.4f}")
        print(f"Recall:       {recall:.4f}")
        print(f"F1-Score:     {f1:.4f}")
        
        # Show confusion matrix with labels to handle edge cases
        cm = metrics.confusion_matrix(y, y_pred, labels=[0, 1])
        print(f"\nConfusion Matrix:")
        print(f"  Predicted:     Fail    Success")
        print(f"  Actual Fail:   {cm[0][0]:4d}    {cm[0][1]:4d}")
        print(f"  Actual Success: {cm[1][0]:4d}    {cm[1][1]:4d}")
    
    print("\n" + "=" * 60)
    print("Planner testing completed!")


def parse_arguments():
    """
        Parse command line arguments for different modes of operation.
        
        Returns:
            argparse.Namespace: Parsed arguments containing mode and num_samples
    """
    parser = argparse.ArgumentParser(
        description='Robot Grasping Simulation and Classification System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode generate_data --num_samples 150
  python main.py --mode train_classifier
  python main.py --mode test_planner
  python main.py  (runs default - data generation and model training)
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['generate_data', 'train_classifier', 'test_planner'],
        default=None,
        help='Operation mode: generate_data (run simulation), train_classifier (train models), or test_planner (test planner)'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Number of samples to generate per gripper-object combination (only used with generate_data mode)'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # If no mode specified, run default behavior (generate data + train classifier)
    if args.mode is None:
        print("Running default mode: generating data and training classifier...")
        run_simulation()
        run_models()
    
    # Generate data mode
    elif args.mode == 'generate_data':
        print(f"Generating training data{f' with {args.num_samples} samples per combination' if args.num_samples else ''}...")
        run_simulation(num_samples=args.num_samples)
        print("Data generation completed")
    
    # Train classifier mode
    elif args.mode == 'train_classifier':
        print("Training classifier models...")
        run_models()
        print("Model training completed")
    
    # Test planner mode
    elif args.mode == 'test_planner':
        test_planner()