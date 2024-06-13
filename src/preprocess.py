import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_and_select_features(X, y, k):
    """
    Preprocess the input data by applying standard scaling to numerical features,
    one-hot encoding to categorical features, and SelectKBest for feature selection.

    Parameters:
    - X: pd.DataFrame, feature matrix
    - y: pd.Series or pd.DataFrame, target variable
    - k: int, number of top features to select (default: 10)

    Returns:
    - X_preprocessed: np.ndarray, preprocessed and feature-selected feature matrix
    - y: np.ndarray, target variable (unchanged)
    - preprocessor: ColumnTransformer, fitted preprocessor for future use
    """
    # Ensure X is a DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=['number']).columns.tolist()

    # Preprocessing pipelines for both numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
        ],
        remainder='passthrough'
    )

    # Fit the preprocessor on the data
    X_preprocessed = preprocessor.fit_transform(X)

    # Select top k features
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X_preprocessed, y)

    # Get feature names after preprocessing
    numerical_feature_names = numerical_features
    categorical_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(
        categorical_features)
    preprocessed_feature_names = list(numerical_feature_names) + list(categorical_feature_names)
    selected_feature_indices = selector.get_support(indices=True)
    selected_feature_names = [preprocessed_feature_names[i] for i in selected_feature_indices]

    return X_selected, y, preprocessor, selector, selected_feature_names
