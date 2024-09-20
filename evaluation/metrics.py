from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, f1_score, accuracy_score, confusion_matrix
import numpy as np

def calculate_auc(y_true, y_scores):
    """Calculate the Area Under the ROC Curve (AUC)."""
    return roc_auc_score(y_true, y_scores)

def calculate_average_precision(y_true, y_scores):
    """Calculate the Average Precision Score."""
    return average_precision_score(y_true, y_scores)

def calculate_precision_recall_curve(y_true, y_scores):
    """Calculate precision and recall for various thresholds."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    return precision, recall, thresholds

def calculate_f1_score(y_true, y_pred):
    """Calculate the F1 Score."""
    return f1_score(y_true, y_pred)

def calculate_accuracy(y_true, y_pred):
    """Calculate accuracy."""
    return accuracy_score(y_true, y_pred)

def calculate_confusion_matrix(y_true, y_pred):
    """Calculate the confusion matrix."""
    return confusion_matrix(y_true, y_pred)

def evaluate_model(y_true, y_pred, y_scores):
    """Evaluate the model and return various metrics."""
    auc = calculate_auc(y_true, y_scores)
    average_precision = calculate_average_precision(y_true, y_scores)
    f1 = calculate_f1_score(y_true, y_pred)
    accuracy = calculate_accuracy(y_true, y_pred)
    cm = calculate_confusion_matrix(y_true, y_pred)
    
    return {
        "AUC": auc,
        "Average Precision": average_precision,
        "F1 Score": f1,
        "Accuracy": accuracy,
        "Confusion Matrix": cm
    }
