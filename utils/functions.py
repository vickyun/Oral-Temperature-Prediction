import pandas as pd
import matplotlib.pyplot as plt
from utils.models import RMSE, MAE
import random
import mlflow
import seaborn as sns

def train_test_split(X, y, SEED, frac=None):
    """ Split data into train, validation and test set"""

    # Shuffle data before splitting
    X = X.sample(frac=1, random_state=SEED)
    y = y.sample(frac=1, random_state=SEED)

    # Train, Validation and Test sets
    if frac is None:
        frac = [0.6, 0.2, 0.2]

    size_train = int(X.shape[0] * frac[0])
    size_val = int(X.shape[0] * frac[1])
    size_test = int(X.shape[0] * frac[2])

    X_train = X[:size_train]
    y_train = y[:size_train]

    X_val = X[size_train:size_train + size_val]
    y_val = y[size_train:size_train + size_val]

    X_test = X[size_train + size_val:]
    y_test = y[size_train + size_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def estimate_performance(model_folder, model, X, y, dataset="test"):
    """ Estimate model performance """
    # Estimate model performance on training set
    # evaluate model
    X, y = model.preprocessing(X, y)

    y_pred = model.predict(X)

    rmse = RMSE(y_pred, y)
    mlflow.log_metric(f"{dataset}_set RMSE", round(rmse, 3))

    mae = MAE(y_pred, y)
    mlflow.log_metric(f"{dataset}_set MAE", round(mae, 3))

    print(f"\n{dataset} set\n"
          f"RMSE:   {round(rmse, 3)}\n"
          f"MAE:   {round(mae, 3)}\n")

    plt.scatter(y_pred, y)
    plt.xlabel("y_pred")
    plt.ylabel("y_test")

    plt.savefig(model_folder + f"/{dataset}.png")

    plt.close()

    # Plot Residuals
    residuals = y.reshape(y.shape[0]) - y_pred

    sns.residplot(x=y_pred, y=residuals, scatter_kws={"color": "green"})

    # Set plot labels and title
    plt.xlabel('Predicted Oral Temperature (°C)')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot - {dataset} set')

    plt.savefig(model_folder + f"/{dataset}_set_residuals.png")

    plt.close()

    mlflow.log_artifact(model_folder + f"/{dataset}.png")
    mlflow.log_artifact(model_folder + f"/{dataset}_set_residuals.png")


def save_to_csv(dictionary, path):
    """Save to csv"""
    df = pd.DataFrame(dictionary)

    df.to_csv(path)


def log_params(FRAC, NB_ITER, ALPHA, SEED, X):
    """
    Log parameters to MlFlow
    :param FRAC: Train/Test split fraction
    :param NB_ITER: Number of iterations
    :param ALPHA: Learning Rate
    :param SEED: Random seed
    :param N Features: Number of features
    :param Features: Set of features
    :return: None
    """
    mlflow.log_params({"N iterations": NB_ITER,
                       "Learning Rate": ALPHA,
                       "Train/Test split": FRAC,
                       "Seed": SEED,
                       "N Features": len(X.columns),
                       "Features": list(X.columns)
                       }
                      )
