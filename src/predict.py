import json
import pickle
import pandas as pd


def predict_new_data(model_path, feature_path, new_data_path):
    """
    Predict the labels for new data using the trained model.

    Args:
        model_path (str): Path to the saved trained model.
        feature_path (str): Path to the saved selected feature names.
        new_data_path (str): Path to the new data CSV file.

    Returns:
        pd.DataFrame: DataFrame with predictions and corresponding applicant IDs.
    """
    # Load the trained model
    with open(model_path, 'rb') as model_file:
        best_model = pickle.load(model_file)

    # Load the selected feature names
    with open(feature_path, 'r') as f:
        selected_feature_names = json.load(f)

    # Load new data
    new_data = pd.read_csv(new_data_path)
    applicant_ids = new_data['applicant_id']
    X_new = new_data.drop(['applicant_id'], axis=1)

    # Apply the same preprocessing as during training
    preprocessor = best_model.named_steps['preprocessor']
    X_new_preprocessed = preprocessor.transform(X_new)

    # Get feature names after preprocessing
    numerical_features = X_new.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_new.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(
        categorical_features)
    preprocessed_feature_names = list(numerical_features) + list(categorical_feature_names)

    # Ensure all selected features are present in the preprocessed features
    missing_features = [name for name in selected_feature_names if name not in preprocessed_feature_names]
    if missing_features:
        print(f"Missing features from preprocessed data: {missing_features}")
        raise ValueError("Some selected features are not present in the preprocessed data.")

    # Find indices of the selected features in the preprocessed feature names
    selected_feature_indices = [preprocessed_feature_names.index(name) for name in selected_feature_names]

    # Debugging: Print selected feature indices
    print("Selected feature indices:", selected_feature_indices)
    print("Shape of X_new_preprocessed:", X_new_preprocessed.shape)

    # Ensure the new data contains only the selected features
    X_new_selected = X_new_preprocessed[:, selected_feature_indices]

    # Make predictions
    predictions = best_model.named_steps['model'].predict(X_new_selected)

    # Return a DataFrame with applicant IDs and predictions
    results = pd.DataFrame({'applicant_id': applicant_ids, 'prediction': predictions})
    return results


# Example usage:
predictions = predict_new_data("../models/final_model.pkl",
                               "../models/selected_features.json",
                               "../input/test.csv")
predictions.to_csv("../output/predictions.csv", index=False)
