from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def select_model():
    models = {
        'RandomForest': RandomForestClassifier(),
        'LogisticRegression': LogisticRegression(),
        'SVC': SVC()
    }
    return models
