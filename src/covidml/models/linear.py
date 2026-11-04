"""Logistic regression with the original experiment's defaults."""

from sklearn.linear_model import LogisticRegression


def make_logistic_regression(**parameters) -> LogisticRegression:
    """
    Build a balanced, L2-regularized classifier with overridable defaults.
    """
    defaults = {
        "l1_ratio": 0,
        "class_weight": "balanced",
        "max_iter": 1000,
        "random_state": 42,
    }
    return LogisticRegression(**(defaults | parameters))
