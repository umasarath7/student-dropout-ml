"""Model 1: Logistic Regression (multinomial, one equation per class)."""
from sklearn.linear_model import LogisticRegression

NAME = "Logistic Regression"
FILE = "logistic_regression.joblib"


def make_estimator():
    return LogisticRegression(max_iter=3000)


# C controls regularisation: small C = simpler model.
PARAM_GRID = {"clf__C": [0.05, 0.1, 0.5, 1.0, 5.0, 20.0]}
