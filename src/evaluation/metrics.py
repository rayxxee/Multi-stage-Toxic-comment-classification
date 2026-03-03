import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

def compute_metrics(y_true, y_pred, y_proba, label_dict):
    """
    Computes classification metrics given true labels, predictions, and probabilities.
    label_dict: e.g. {0: 'not_safe', 1: 'safe'} mapping integers to names
    """
    if len(y_true) == 0:
        return {}

    acc = accuracy_score(y_true, y_pred)
    # Using 'weighted' to handle imbalanced datasets smoothly
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    unique_labels = sorted(list(label_dict.keys()))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    auc = None
    try:
        if y_proba is not None and len(y_proba) > 0:
            if len(unique_labels) == 2:
                # Binary class
                # y_proba shape is typically (n_samples, 2), we want class 1
                if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                    auc = roc_auc_score(y_true, y_proba[:, 1])
                elif y_proba.ndim == 1:
                    auc = roc_auc_score(y_true, y_proba)
            else:
                # Multiclass (OvR)
                auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted', labels=unique_labels)
    except Exception as e:
        # e.g. only one class present in y_true
        pass

    return {
        'accuracy': acc,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'precision': precision,
        'recall': recall,
        'roc_auc': auc,
        'confusion_matrix': cm.tolist(),
        'label_order': [label_dict[i] for i in unique_labels]
    }
