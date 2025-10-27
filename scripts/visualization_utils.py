"""
Visualization Utilities

Common utility functions for data visualization and plotting.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Tuple
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


def plot_training_history(history, metrics: List[str] = ['loss', 'accuracy'], 
                          figsize: Tuple[int, int] = (12, 4)) -> None:
    """
    Plot training history for deep learning models.
    
    Args:
        history: Training history object (e.g., from Keras)
        metrics: List of metrics to plot
        figsize: Figure size as (width, height)
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        axes[idx].plot(history.history[metric], label=f'Train {metric}')
        if f'val_{metric}' in history.history:
            axes[idx].plot(history.history[f'val_{metric}'], label=f'Val {metric}')
        axes[idx].set_title(f'{metric.capitalize()} vs. Epoch')
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel(metric.capitalize())
        axes[idx].legend()
        axes[idx].grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=False,
                         figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names for display
        normalize: Whether to normalize the confusion matrix
        figsize: Figure size as (width, height)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        figsize: Figure size as (width, height)
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_precision_recall_curve(y_true, y_pred_proba, 
                                figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        figsize: Figure size as (width, height)
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(feature_names: List[str], importances: np.ndarray,
                           top_n: int = 20, figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot feature importance.
    
    Args:
        feature_names: List of feature names
        importances: Array of feature importances
        top_n: Number of top features to display
        figsize: Figure size as (width, height)
    """
    # Sort features by importance
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=figsize)
    plt.title(f'Top {top_n} Feature Importances')
    plt.barh(range(top_n), importances[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_class_distribution(y, labels=None, figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Plot class distribution.
    
    Args:
        y: Target labels
        labels: Class names for display
        figsize: Figure size as (width, height)
    """
    import pandas as pd
    
    class_counts = pd.Series(y).value_counts().sort_index()
    
    plt.figure(figsize=figsize)
    class_counts.plot(kind='bar')
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    if labels:
        plt.xticks(range(len(labels)), labels, rotation=45)
    plt.tight_layout()
    plt.show()


def plot_learning_curve(train_sizes, train_scores, val_scores,
                       figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot learning curve to diagnose bias/variance.
    
    Args:
        train_sizes: Training set sizes
        train_scores: Training scores
        val_scores: Validation scores
        figsize: Figure size as (width, height)
    """
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=figsize)
    plt.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    plt.plot(train_sizes, val_mean, label='Validation score', color='red', marker='s')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                     alpha=0.1, color='red')
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
