import glob
import pandas as pd
from utils.functions import train_test_split, estimate_performance, save_to_csv
from utils.constant import OUTPUT, FRAC, SEED
import pickle
from ucimlrepo import fetch_ucirepo


def load_test_data(INPUT, FRAC):
    """Load test dataset"""

    df = pd.read_csv(INPUT, low_memory=False)

    pd.options.mode.copy_on_write = True

    X_columns = ["type_local", "surface_reelle_bati", "nombre_pieces_principales",
                 "surface_terrain", "commune_principale", "longitude", "latitude"]

    y_col = "valeur_fonciere"

    # Separate features and target
    X = df[X_columns]
    y = df[y_col]

    # Train Test Split
    X_train, y_train, X_val, y_val, X_test, y_test = train_test_split(X, y, SEED, frac=FRAC)

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":

    # get most recent data folder
    model_folder = sorted(glob.glob("out/*"))[-1]
    model_path = model_folder + "/model.pkl"

    # load model from pickle file
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # fetch dataset
    infrared_thermography_temperature = fetch_ucirepo(id=925)

    # data (as pandas dataframes)
    X = infrared_thermography_temperature.data.features
    y = infrared_thermography_temperature.data.targets.aveOralM

    # Train Test Split
    X_train, y_train, X_val, y_val, X_test, y_test = train_test_split(X, y, SEED, frac=FRAC)

    # evaluate model
    estimate_performance(model_folder, model, X_test, y_test, "train")

    # evaluate model
    estimate_performance(model_folder, model, X_train, y_train, "test")
