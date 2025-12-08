# Imports for Classification
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

class LogReg:
    """
    A class to perform Logistic Regression on 6-DOF sensor data.

    This class handles data loading, preprocessing, model training,
    evaluation, and visualization of results using a logistic regression model.

    Attributes:
        col_names (list): The expected column names for the input CSV:
                          ['x', 'y', 'z', 'pitch', 'roll', 'yaw', 'output'].
        file_name (str): The path to the input CSV file.
        X (pd.DataFrame): The feature dataset (sensor readings).
        y (pd.Series): The target variable (output class).
        y_test (np.array): True labels from the test set.
        y_pred (np.array): Predicted labels from the model.
    """
    col_names = ['x', 'y', 'z', 'pitch', 'roll', 'yaw', 'output']

    def __init__(self, file_name: str):
        """
        Initializes the LogReg class with the target file.

        Args:
            file_name (str): Path to the CSV file containing sensor data.
        """
        self.file_name = file_name

    def read_file(self):
        """
        Reads and preprocesses the CSV data.

        The method read_file loads the data, assigns column names, removes the first row
        as it is a header from data generation, removes rows with missing
        target values if simulation fails, and separates the features from targets get X and y.

        Returns:
            tuple: A tuple containing:
                - self.X (pd.DataFrame): Features ['x', 'y', 'z', 'pitch', 'roll', 'yaw'].
                - self.y (pd.Series): Target variable ['output'] cast to integers.
        """
        # Reading the csv file and generating a pandas dataframe
        data = pd.read_csv(self.file_name, header=None, names=LogReg.col_names)
        # Dropping row 1 of headers
        data = data.drop(0)
        # Dropping NaNs from the dataset
        data = data.dropna(subset=['output'])
        feature_cols = ['x', 'y', 'z', 'pitch', 'roll', 'yaw']
        self.X = data[feature_cols]  # Features
        self.y = data.output.astype(int)  # Target variable
        return self.X, self.y

    def conduct_regression(self):
        """
        Trains the Logistic Regression model and generates the predictions with y_pred.

        The method splits the data (70% train, 30% test), scales the features
        using StandardScaler, and fits a balanced Logistic Regression model.

        Args:
            X1_range, X2_range, X3_range:

        Returns:
            tuple: A tuple containing:
                - y_test (np.array): The actual values of the test set.
                - y_pred (np.array): The predicted values of the test set.
        """
        self.X, self.y = self.read_file()

        # Splitting the data into test and training sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=20)

        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Conducting Logistic Regression
        self.logreg = LogisticRegression(max_iter=5000, random_state=16, class_weight='balanced')
        # Fitting the data
        self.logreg.fit(X_train_scaled, y_train)
        y_pred = self.logreg.predict(X_test_scaled)
        self.y_test = y_test
        self.y_pred = y_pred
        return y_test, y_pred

    def accuracy_precision_recall_f1(self):
        """
        Calculates and prints standard classification performance metrics.

        This method triggers the regression pipeline and evaluates the model
        against the test set using:
        - Accuracy: Ratio of correct predictions to total predictions.
        - Precision: Ratio of correctly predicted positive observations to total predicted positives.
        - Recall: Ratio of correctly predicted positive observations to all observations in actual class.
        - F1 Score: The weighted average of Precision and Recall.
        """
        self.y_test, self.y_pred = self.conduct_regression()
        # Accuracy, Precision, Recall, F1 Score
        print("Logistic Regression Accuracy:", metrics.accuracy_score(self.y_test, self.y_pred))
        print("Logistic Regression Precision:", metrics.precision_score(self.y_test, self.y_pred))
        print("Logistic Regression Recall:", metrics.recall_score(self.y_test, self.y_pred))
        print("Logistic Regression F1 Score:", metrics.f1_score(self.y_test, self.y_pred))

    def confusion_matrix(self):
        """
        Generates and displays a Confusion Matrix heatmap.

        Visualizes the performance of the classification algorithm by comparing
        predicted values against actual values.

        """
        self.y_test, self.y_pred = self.conduct_regression()

        # Plotting a Confusion Matrix
        cnf_matrix = metrics.confusion_matrix(self.y_test, self.y_pred)
        class_names = [0, 1]  # name  of classes
        fig, ax = plt.subplots()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)

        # Create heatmap to visually depict the confusion matrix
        sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
        ax.xaxis.set_label_position("top")
        plt.tight_layout()
        plt.title('Confusion matrix', y=1.1)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.show(block=False)

    def roc_curve(self):
        """
        Plots the Receiver Operating Characteristic (ROC) curve.

        Calculates the Area Under the Curve (AUC) and plots the True Positive Rate (TPR)
        against the False Positive Rate (FPR) at various threshold settings.

        Note:
            self.logreg is an accessible attribute of the class.
        """
        self.y_test, self.y_pred = self.conduct_regression()

        # Calculating True Positive Rate (TPR), False Positive Rate (FPR) and the AUC
        y_pred_proba = self.logreg.predict_proba(self.X)[:, 1]
        fpr, tpr, _ = metrics.roc_curve(self.y, y_pred_proba)
        auc = metrics.roc_auc_score(self.y, y_pred_proba)

        # Plotting the ROC Curve
        plt.figure(2)
        plt.plot(fpr, tpr, label="AUC=" + str(auc))
        plt.legend(loc=4)
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()

    def matthews_cc(self):
        """
        Calculates and prints the Matthews Correlation Coefficient (MCC).

        MCC is used as a measure of the quality of binary classifications,
        particularly useful when classes are of very different sizes.
        Returns a value between -1 and +1.
        """
        self.y_test, self.y_pred = self.conduct_regression()
        print("Matthew's correlation coefficient:", metrics.matthews_corrcoef(self.y_test, self.y_pred))

    def full_operation(self):
        """
        Orchestrates the complete data analysis pipeline.

        Sequence of operations:
        1. Reads the dataset.
        2. Trains the model (conduct_regression).
        3. Prints file metadata.
        4. Calculates and displays all statistical metrics and plots.
        """

        self.read_file()
        self.conduct_regression()
        print(f"---------- For {self.file_name} ----------")
        self.accuracy_precision_recall_f1()
        self.matthews_cc()
        self.confusion_matrix()
        self.roc_curve()


def run_models():
    """
    Instantiates and executes the LogReg pipeline for multiple dataset configurations.

    This driver function iterates through a predefined list of CSV files
    (Cube and Cylinder shapes for groups G2 and G3), creates a LogReg
    instance for each, and triggers the full regression and evaluation workflow.

    Files processed:
        - output_g2_cube.csv
        - output_g2_cylinder.csv
        - output_g3_cube.csv
        - output_g3_cylinder.csv

    Raises:
        FileNotFoundError: If any of the hardcoded CSV files are missing from the working directory.
    """
    file1 = LogReg("output_g2_cube.csv")
    file1.full_operation()

    file2 = LogReg("output_g2_cylinder.csv")
    file2.full_operation()

    file3 = LogReg("output_g3_cube.csv")
    file3.full_operation()

    file4 = LogReg("output_g3_cylinder.csv")
    file4.full_operation()