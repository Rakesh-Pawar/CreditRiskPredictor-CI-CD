import pandas as pd
from sklearn.model_selection import train_test_split

from src import utils
from src.tune_model import tune_model


def train_model(data_path, model_save_path, features_save_path, k):
    """
        Train a machine learning model on the provided data and save the trained model and selected features.

        Args:
            data_path (str): Path to the CSV file containing the input data.
            model_save_path (str): Path to save the trained model.
            features_save_path (str): Path to save the selected feature names.
            k (int): Number of top features to select.

        Returns:
            None

        This function performs the following steps:

        1. Load the input data from the specified `data_path` using pandas.
        2. Call the `tune_model` function (which should be defined elsewhere) with the loaded data and the specified
           `k` value to obtain the best model pipeline.
        3. Separate the features (`X`) and target (`y`) from the loaded data.
        4. Split the data into training and testing sets using `train_test_split` from scikit-learn.
        5. Fit the `best_model` pipeline on the training data using `best_model.fit(X_train, y_train)`.
        6. Save the trained `best_model` pipeline to the specified `model_save_path` using `utils.save_model`.
        7. Get the names of the selected features after applying the preprocessing and feature selection steps
           using `best_model.named_steps['feature_selection'].get_feature_names_out()`.
        8. Save the selected feature names to the specified `features_save_path` using `utils.save_selected_features`.

        Note:
        This function assumes the following:
        - The `tune_model` function is defined elsewhere and returns the best model pipeline.
        - The `utils` module contains the `save_model` and `save_selected_features` functions for saving
          the model and feature names, respectively.
        - The input data has columns named "applicant_id" and "high_risk_applicant", which are dropped
          from the feature matrix `X`.
        """
    data = pd.read_csv(data_path)
    best_model, selected_feature_names = tune_model(data_path, k)
    X = data.drop(["applicant_id", "high_risk_applicant"], axis=1)
    y = data['high_risk_applicant']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)
    model = best_model.fit(X_train, y_train)
    utils.save_model(model, model_save_path)

    utils.save_selected_features(selected_feature_names, features_save_path)


train_model("../input/train.csv",
            "../models/final_model.pkl",
            "../models/selected_features.json",
            15)
