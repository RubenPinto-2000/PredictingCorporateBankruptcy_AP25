"""
Model evaluation and visualization (non statitisque).

This module evaluates the performance of models from models.py using standard metrics.
It also displays a model's ability to correctly predict bankruptcy, as well as the distribution
between false positives, true positives, etc.
"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import numpy as np
import os


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model and print comprehensive results.
    Analyzes the overall performance of a model after training.
    """
    # Generate predictions (binary and probabilities)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate evaluation metrics as recommended by the assistant teacher
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print("=" * 80)
    print(f"{model_name} - Evaluation Results")
    print("=" * 80)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print()
    
    # Compare the model's predictions with actual values to see how well the model performs after training
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print()
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Bankruptcy", "Bankruptcy"], zero_division=0))
    print()
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba
    }
    
    return metrics


def plot_roc_curve(y_test, y_pred_proba, model_name, save_path=None):
    """
    Plot ROC curve to display the relationship between true positive rate and false positive rate.
    """
    # Calculate ROC curve points and AUC score
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Save plot if path is provided, and create the directory if it doesn't exist
    if (save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.close()


def plot_confusion_matrix(confusion_matrix, model_name, save_path=None):
    """
    Plot confusion matrix to visualize the distribution of True Negatives, False Positives,
    False Negatives, and True Positives for bankruptcy or non-bankruptcy predictions.
    """
    # Confusion matrix structure:
    #                Predicted
    #              No Bankruptcy  Bankruptcy
    # Actual  No Bankruptcy    TN          FP
    #         Bankruptcy       FN          TP
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["No Bankruptcy", "Bankruptcy"],
                yticklabels=["No Bankruptcy", "Bankruptcy"])
    plt.ylabel("True class")
    plt.xlabel("Predicted class")
    plt.title(f"Confusion Matrix - {model_name}")
    
    # Save plot if path is provided
    # os.makedirs ensures the directory exists before saving (creates it if needed)
    if (save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.close()

