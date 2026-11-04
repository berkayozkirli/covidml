"""Classification metrics shared by all model experiments."""

from dataclasses import dataclass
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ModelResults:
    model_name: str
    model: Any
    hyperparameters: dict
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float

    @classmethod
    def evaluate(cls, model_name: str, model, hyperparameters: dict, y_true, y_pred, y_prob):
        return cls(
            model_name=model_name,
            model=model,
            hyperparameters=hyperparameters,
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, zero_division=0),
            recall=recall_score(y_true, y_pred, zero_division=0),
            f1=f1_score(y_true, y_pred, zero_division=0),
            roc_auc=roc_auc_score(y_true, y_prob),
            pr_auc=average_precision_score(y_true, y_prob),
        )

    def __str__(self):
        hp_str = ""
        if self.hyperparameters:
            hp_formatted = "\n".join([f"   - {k}: {v}" for k, v in self.hyperparameters.items()])
            hp_str = f"Hyperparameters:\n{hp_formatted}\n--------------------------------------------------\n"

        return (
            f"   {self.model_name} EVALUATION\n{hp_str}"
            f"Accuracy:  {self.accuracy:.4f}\nPrecision: {self.precision:.4f}\n"
            f"Recall:    {self.recall:.4f}\nF1-Score:  {self.f1:.4f}\n"
            f"ROC-AUC:   {self.roc_auc:.4f}\nPR-AUC:    {self.pr_auc:.4f}\n"
        )
