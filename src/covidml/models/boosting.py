"""XGBoost constructors shared by the comparison and final pipeline."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def make_xgboost_classifier(**parameters) -> XGBClassifier:
    """
    Build an XGBoost classifier with reproducible experiment defaults.
    """
    defaults = {"eval_metric": "logloss", "random_state": 42}
    return XGBClassifier(**(defaults | parameters))


def make_xgboost_pipeline(**parameters) -> Pipeline:
    """
    Build the standardized XGBoost pipeline used for tuning and ablation.

    Parameters are forwarded to the classifier. Search spaces can address
    classifier parameters with the usual ``classifier__`` prefix.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", make_xgboost_classifier(**parameters)),
        ]
    )
