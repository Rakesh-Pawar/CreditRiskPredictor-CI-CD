import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.preprocess import preprocess_and_select_features


def tune_model(data_path, k):
    """
    Tune various machine learning models using grid search cross-validation and select the best model.

    Args:
        data (pd.DataFrame): Input data containing features and target variable.
        n_top_features (int): Number of top features to select.

    Returns:
        Pipeline: Best model pipeline with preprocessing, feature selection, and the best model.

    Models and Hyperparameters:
        1. Logistic Regression:
            - C (float): Inverse of regularization strength. Smaller values specify stronger regularization.
                Values to try: [0.1, 1.0, 10.0]

        2. Support Vector Machine (SVM):
            - C (float): Penalty parameter for mis-classification. Smaller values specify stronger regularization.
                Values to try: [0.1, 1.0, 10.0]
            - kernel (str): Kernel type to be used in the algorithm. Options: 'linear', 'poly', 'rbf'.

        3. Decision Tree Classifier:
            - max_depth (int or None): Maximum depth of the tree. If None, the tree is expanded until all leaves are pure.
                Values to try: [None, 10, 20, 30]

        4. Gradient Boosting Classifier:
            - n_estimators (int): Number of boosting stages to perform.
                Values to try: [50, 100, 200]
            - learning_rate (float): Learning rate shrinks the contribution of each tree.
                Values to try: [0.05, 0.1, 0.2]

        5. Random Forest Classifier:
            - n_estimators (int): Number of trees in the forest.
                Values to try: [50, 100, 200]
            - max_depth (int or None): Maximum depth of the tree. If None, the tree is expanded until all leaves are pure.
                Values to try: [None, 10, 20, 30]
            - max_features (int, float, str or None): Number of features to consider when looking for the best split.
                If int, then consider `max_features` features at each split.
                If floated, then `max_features` is a fraction and `int(max_features * n_features)` features are considered at each split.
                If "sqrt", then `max_features=sqrt(n_features)`.
                If "log2", then `max_features=log2(n_features)`.
                Values to try: ['sqrt', 'log2', None]
                :param k:
                :param data_path:
    """
    data = pd.read_csv(data_path)
    y = data['high_risk_applicant']
    X = data.drop(["applicant_id", "high_risk_applicant"], axis=1)

    # Preprocess the data and select top k features
    X_preprocessed, y, preprocessor, selector, selected_feature_names = preprocess_and_select_features(X, y, k=k)

    # Define the models with their respective hyperparameters
    models = [
        ('LogisticRegression', LogisticRegression(max_iter=100), {'model__C': [0.1, 1.0, 10.0]}),
        ('SVM', SVC(), {'model__C': [0.1, 1.0, 10.0], 'model__kernel': ['linear', 'poly', 'rbf']}),
        ('DecisionTree', DecisionTreeClassifier(), {'model__max_depth': [None, 10, 20, 30]}),
        ('GradientBoosting', GradientBoostingClassifier(),
         {'model__n_estimators': [50, 100, 200], 'model__learning_rate': [0.05, 0.1, 0.2]}),
        ('RandomForest', RandomForestClassifier(),
         {'model__n_estimators': [50, 100, 200], 'model__max_depth': [None, 10, 20, 30],
          'model__max_features': ['sqrt', 'log2', None]})
    ]

    best_model = None
    best_score = 0

    # Iterate over models and perform grid search
    for name, model, params in models:
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('feature_selection', selector),
            ('model', model)
        ])
        grid = GridSearchCV(estimator=pipeline, param_grid=params, cv=5, scoring='accuracy')
        grid.fit(X, y)
        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_

    return best_model, selected_feature_names


# best_model = tune_model("../input/train.csv", k=7)
