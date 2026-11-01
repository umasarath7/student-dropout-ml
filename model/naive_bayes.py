"""Model 4: Gaussian Naive Bayes."""
from sklearn.naive_bayes import GaussianNB

NAME = "Naive Bayes"
FILE = "naive_bayes.joblib"


def make_estimator():
    return GaussianNB()


# var_smoothing adds a little variance to every feature so that
# near-constant one-hot columns don't dominate the probabilities.
PARAM_GRID = {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-2, 1e-1, 0.3, 1.0]}
