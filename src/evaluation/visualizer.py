from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve

from src.config import FIGURES_DIR


def plot_figure_3a_masking_auc(
    masking_rates: List[float],
    multimodal_aucs: List[float],
    missing_aware_aucs: List[float],
    save_path: Path = None
):
    """
    Reproduces Figure 3(a) from the paper:
    ROC-AUC against masking rate (%) for the multimodal and missing-data-aware models.
    """
    if save_path is None:
        save_path = FIGURES_DIR / "fig3a_masking_rate_vs_roc_auc.png"

    rates_pct = [r * 100 for r in masking_rates]

    plt.figure(figsize=(7, 6), dpi=300)
    plt.plot(rates_pct, multimodal_aucs, marker='o', color='#2563eb', linewidth=2.2, label='Multimodal (Standard)')
    plt.plot(rates_pct, missing_aware_aucs, marker='s', color='#dc2626', linewidth=2.2, linestyle='--', label='Missing-data-aware')

    plt.title("Fig. 3(a): ROC-AUC vs Clinical Masking Rate", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Masking rate (%)", fontsize=11)
    plt.ylabel("ROC-AUC", fontsize=11)
    plt.xticks([0, 10, 30, 50])
    plt.ylim([0.55, 1.02])
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, loc='lower left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_figure_3b_roc_curves(
    roc_data: Dict[str, Dict[str, np.ndarray]],
    save_path: Path = None
):
    """
    Reproduces Figure 3(b) from the paper:
    ROC curves (Sensitivity vs 1 - Specificity) with Chance level diagonal.
    """
    if save_path is None:
        save_path = FIGURES_DIR / "fig3b_roc_curves.png"

    plt.figure(figsize=(7, 6), dpi=300)

    colors = {
        "Image-only": "#8b5cf6",
        "Clinical-only": "#f59e0b",
        "Multimodal": "#2563eb",
        "Missing-data-aware": "#10b981",
    }

    for model_name, data in roc_data.items():
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_pred_probs"])
        score = auc(fpr, tpr)
        c = colors.get(model_name, "#475569")
        plt.plot(fpr, tpr, color=c, lw=2.2, label=f"{model_name} (AUC = {score:.3f})")

    # Chance diagonal
    plt.plot([0, 1], [0, 1], color='#94a3b8', linestyle='--', lw=1.5, label='Chance level')

    plt.title("Fig. 3(b): Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("1 − Specificity (False Positive Rate)", fontsize=11)
    plt.ylabel("Sensitivity / Recall (True Positive Rate)", fontsize=11)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_figure_3c_calibration_curves(
    calibration_data: Dict[str, Dict[str, np.ndarray]],
    save_path: Path = None
):
    """
    Reproduces Figure 3(c) from the paper:
    Calibration curves (Observed fraction vs Mean predicted probability) with Perfect calibration diagonal.
    """
    if save_path is None:
        save_path = FIGURES_DIR / "fig3c_calibration_curves.png"

    plt.figure(figsize=(7, 6), dpi=300)

    colors = {
        "Image-only": "#8b5cf6",
        "Clinical-only": "#f59e0b",
        "Multimodal": "#2563eb",
        "Missing-data-aware": "#10b981",
    }

    # Perfect calibration reference
    plt.plot([0, 1], [0, 1], color='#94a3b8', linestyle='--', lw=1.5, label='Perfect calibration')

    for model_name, data in calibration_data.items():
        prob_true, prob_pred = calibration_curve(data["y_true"], data["y_pred_probs"], n_bins=8, strategy='uniform')
        c = colors.get(model_name, "#475569")
        plt.plot(prob_pred, prob_true, marker='o', color=c, lw=2.0, label=f"{model_name}")

    plt.title("Fig. 3(c): Diagnostic Probability Calibration Curves", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Mean predicted probability", fontsize=11)
    plt.ylabel("Observed fraction of positives", fontsize=11)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_clinical_attribution_waterfall(
    attributions: Dict[str, float],
    patient_id: str,
    save_path: Path = None
):
    """Plots clinical feature attribution bar chart."""
    if save_path is None:
        save_path = FIGURES_DIR / f"attribution_{patient_id}.png"

    sorted_feats = sorted(attributions.items(), key=lambda x: x[1])
    names = [x[0].replace("_", " ").title() for x in sorted_feats]
    values = [x[1] for x in sorted_feats]
    colors = ['#ef4444' if v > 0 else '#3b82f6' for v in values]

    plt.figure(figsize=(8, 5), dpi=300)
    bars = plt.barh(names, values, color=colors, height=0.6)
    plt.axvline(0, color='#64748b', linestyle='-', linewidth=0.9)
    plt.title(f"Clinical Feature Attribution (Integrated Gradients) - {patient_id}", fontsize=12, fontweight='bold', pad=10)
    plt.xlabel("Contribution to Pathology Prediction (Δ Logit/Score)", fontsize=10)
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
