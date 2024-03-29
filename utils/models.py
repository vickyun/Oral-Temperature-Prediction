import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import os
from utils.constant import OUTPUT
import mlflow

class GradientDescentRegressor:
    """ Batch Gradient Descent for Linear Regression """
    def __init__(self, iterations: int, learning_rate: float):

        self.learning_rate = learning_rate
        self.iterations = iterations

        self.history = {}

        self.model = None
        self.std = None
        self.mean = None
        self.schema = None

    def preprocessing(self, X, y):
        """
        Preprocess dataset,
        make one hot encoding for numerical features

        :param df: input X, y
        :return: transformed X, y
        """

        dummy_cols = []

        # Fill na Distance feature
        if len(X.columns) > 1:
            # One hot encode
            # categorical features
            dummy_cols = X.columns.to_list()[:3]

            if "Distance" in X.columns:
                X.loc[:, 'Distance'] = X['Distance'].fillna(X['Distance'].median())

        # Feature Scaling
        for i in X.columns:
            if i not in dummy_cols:
                X.loc[:, i] = (X[i] - X[i].mean()) / X[i].std()

        X = pd.get_dummies(X, dtype=float, columns=dummy_cols)

        if self.schema is None:
            self.schema = X.columns
        else:
            missing_cols = list(set(X.columns).symmetric_difference(self.schema))
            if missing_cols:
                for c in missing_cols:
                    X.loc[:, c] = 0

                X = X[self.schema]

        X = np.array(X)
        # Add column that corresponds to x0 = 1
        # for each training example
        X = np.insert(X, 0, 1, axis=1)

        if self.std is None:
            self.std = y.std()
            self.mean = y.mean()

            # Target Scaling
            y = (y - self.mean) / self.std

        y = np.array(y).reshape(y.shape[0], 1)

        return X, y

    def fit(self, X, y):
        """ Fit model to data """

        X, y = self.preprocessing(X, y)

        # Initialize weights
        params = initialize_parameters(X.shape[1])

        history = {'iteration': [], 'loss': []}

        nb_iter = self.iterations

        while self.iterations > 0:
            # Repeat until convergence
            loss = calculate_loss(X, y, params)

            mlflow.log_metric("loss", loss)
            mlflow.log_metric("iteration", nb_iter - self.iterations)

            # Update parameters
            params = update_parameters(X, y, params, self.learning_rate)

            history['iteration'].append(nb_iter - self.iterations)
            history['loss'].append(loss[0])

            if self.iterations % 100 == 0:
                print(f"Iteration - {history['iteration'][nb_iter - self.iterations]},", "\t",
                      f"Loss: {history['loss'][nb_iter - self.iterations]}")

            self.iterations -= 1

        self.model = params
        self.history = history

        if not os.path.exists(OUTPUT.split("/")[0]):
            os.mkdir(OUTPUT.split("/")[0])

        os.mkdir(OUTPUT)

        return history

    def predict(self, X_test):
        """ Predict """

        res = np.sum(X_test*self.model, axis=1)*self.std + self.mean
        return res

    def plot_learning_curve(self, OUTPUT):
        """ Plot learning curve """
        plt.title("Learning curve")

        plt.plot(self.history['iteration'], self.history['loss'])

        plt.xlabel("# iteration")
        plt.ylabel("Loss")

        plt.savefig(OUTPUT + "learning_curve.png")
        plt.close('all')

        mlflow.log_artifact(OUTPUT + "learning_curve.png")

def RMSE(y_pred: np.array, y_target: Series) -> float:
    """ Calculate MSE between predicted value and target value """
    m = y_target.shape[0]
    return np.sqrt(1/m*np.sum((y_pred.reshape(m, 1) - y_target.reshape(m, 1))**2))

def MAE(y_pred: np.array, y_target: Series) -> float:
    """ Calculate MAE between predicted value and target value """
    m = y_target.shape[0]
    return 1 / m * np.sum(np.abs(y_pred.reshape(m, 1) - y_target.reshape(m, 1)))

def initialize_parameters(num_features):
    """ Initialize teta array """
    teta = []

    for i in range(num_features):
        if i == 0:
            teta.append(random.uniform(0, 1))
        else:
            teta.append(random.uniform(0, 1))

    return np.array(teta)

def calculate_loss(X, y, teta):
    """ Calculate loss """
    m = X.shape[0]
    loss = 1/(2*m)*np.sum((np.sum(X*teta, axis=1).reshape(m, 1) - y)**2, axis=0)

    return loss

def update_parameters(X, y, teta, learning_rate):
    """ Update parameters simultaneously """
    m = X.shape[0]
    # Update parameters teta
    updated_teta = teta - learning_rate/m*get_grads(X, y, teta)

    return updated_teta

def get_grads(X, y, teta):
    """ Calculate array of gradients """
    m = X.shape[0]
    grads = np.sum((np.sum(X*teta, axis=1).reshape(m, 1) - y)*X, axis=0)

    return grads
