# src/utils.py

import json
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_data(filepath):
    """Loads the dataset from the specified filepath."""
    return pd.read_csv(filepath)


def save_selected_features(features, filepath):
    with open(filepath, 'w') as f:
        json.dump(features, f)


def load_selected_features(file_path):
    with open(file_path, 'r') as f:
        selected_features = json.load(f)
    return selected_features


def save_model(model, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_json(data, filepath):
    """Saves the data to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f)


def load_json(filepath):
    """Loads data from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def evaluate_model(y_true, y_pred):
    """Evaluates the model and returns evaluation metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
