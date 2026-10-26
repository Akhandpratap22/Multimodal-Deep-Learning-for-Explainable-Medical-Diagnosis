import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    auc,
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computes all clinical diagnostic metrics detailed in Section IV-E:
    - Accuracy
    - Precision (Positive Predictive Value)
    - Recall / Sensitivity
    - Specificity (True Negative Rate)
    - F1-Score
    - ROC-AUC
    - PR-AUC
    - Brier Score (Probability Calibration)
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred_probs = np.asarray(y_pred_probs).ravel().astype(np.float64)
    y_pred_binary = (y_pred_probs >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred_binary))
    prec = float(precision_score(y_true, y_pred_binary, zero_division=0))
    rec = float(recall_score(y_true, y_pred_binary, zero_division=0))
    f1 = float(f1_score(y_true, y_pred_binary, zero_division=0))
    
    try:
        roc_auc = float(roc_auc_score(y_true, y_pred_probs))
    except ValueError:
        roc_auc = 0.5

    # Specificity = TN / (TN + FP)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # PR-AUC
    p_prec, p_rec, _ = precision_recall_curve(y_true, y_pred_probs)
    pr_auc = float(auc(p_rec, p_prec))

    # Calibration Brier score (lower is better calibrated)
    brier = float(brier_score_loss(y_true, y_pred_probs))

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "threshold": round(threshold, 4),
    }


def find_optimal_threshold(y_true: np.ndarray, y_pred_probs: np.ndarray) -> float:
    """
    Finds the optimal F1 threshold on validation set (Section IV-E).
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred_probs = np.asarray(y_pred_probs).ravel()

    best_thresh = 0.5
    best_f1 = -1.0

    thresholds = np.linspace(0.1, 0.9, 81)
    for t in thresholds:
        b = (y_pred_probs >= t).astype(int)
        score = f1_score(y_true, b, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(t)

    return best_thresh
