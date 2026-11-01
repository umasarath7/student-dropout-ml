"""Model 5: Support Vector Machine (RBF kernel)."""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

NAME = "Support Vector Machine"
FILE = "svm.joblib"


def make_estimator():
    # A plain SVM only gives a decision score. Calibration turns that score
    # into class probabilities, which we need for the AUC metric.
    return CalibratedClassifierCV(SVC(kernel="rbf", random_state=26), ensemble=False)


PARAM_GRID = {
    "clf__estimator__C": [0.5, 1.0, 3.0, 10.0],
    "clf__estimator__gamma": ["scale", 0.01],
}
