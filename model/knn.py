"""Model 3: k-Nearest Neighbours."""
from sklearn.neighbors import KNeighborsClassifier

NAME = "kNN"
FILE = "knn.joblib"


def make_estimator():
    return KNeighborsClassifier()


PARAM_GRID = {
    "clf__n_neighbors": [3, 5, 7, 11, 15, 25],
    "clf__weights": ["uniform", "distance"],
}
