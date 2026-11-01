"""The six evaluation metrics asked for in the assignment."""
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# The three classes are not balanced (Enrolled is the smallest), so we use
# macro averaging: every class counts equally, no matter how big it is.
AVERAGE = "macro"


def score_all(y_true, y_pred, y_proba, classes):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba, multi_class="ovr", average=AVERAGE,
                             labels=classes),
        "Precision": precision_score(y_true, y_pred, average=AVERAGE, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=AVERAGE, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=AVERAGE, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
