"""Model 2: Decision Tree."""
from sklearn.tree import DecisionTreeClassifier

NAME = "Decision Tree"
FILE = "decision_tree.joblib"


def make_estimator():
    return DecisionTreeClassifier(random_state=26)


# Limiting depth and leaf size stops the tree from memorising the training set.
PARAM_GRID = {
    "clf__max_depth": [4, 6, 8, 10],
    "clf__min_samples_leaf": [5, 15, 30],
    "clf__criterion": ["gini", "entropy"],
}
