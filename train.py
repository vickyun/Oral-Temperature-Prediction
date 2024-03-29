from utils.models import GradientDescentRegressor
from utils.functions import train_test_split, estimate_performance, save_to_csv, log_params
from utils.constant import OUTPUT, FRAC, NB_ITER, ALPHA, SEED, RUN_NAME, FEATURES
import pickle
from ucimlrepo import fetch_ucirepo
import mlflow


if __name__ == '__main__':

    # fetch dataset
    infrared_thermography_temperature = fetch_ucirepo(id=925)

    # data (as pandas dataframes)
    X = infrared_thermography_temperature.data.features.loc[:, FEATURES]
    y = infrared_thermography_temperature.data.targets.aveOralM

    # Train Test Split
    X_train, y_train, X_val, y_val, X_test, y_test = train_test_split(X, y, SEED, frac=FRAC)

    # Compile the model
    model = GradientDescentRegressor(iterations=NB_ITER, learning_rate=ALPHA)

    with mlflow.start_run(run_name=RUN_NAME) as run:

        log_params(FRAC, NB_ITER, ALPHA, SEED, X)

        # Fit to the training set
        history = model.fit(X_train, y_train)

        save_to_csv(history, OUTPUT + "history.csv")

        model.plot_learning_curve(OUTPUT)

        # Save model in .pkl file
        model_pkl_file = OUTPUT + "model.pkl"

        with open(model_pkl_file, 'wb') as file:
            pickle.dump(model, file)

        mlflow.log_artifact(model_pkl_file)

        estimate_performance(OUTPUT, model, X_train, y_train, "train")

        estimate_performance(OUTPUT, model, X_test, y_test, "test")

