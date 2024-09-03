"""
Module that allows to receive true and predicted values (or trained model and data to compute 
predictions) and computes multiple metrics and returns pandas DataFrame with the results. 

METHODS:
--------
 - get_metric_evaluations_from_yscore
 - get_metric_evaluations_from_ypred
 - get_metric_evaluations

"""
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,accuracy_score
import numpy as np

def get_metric_evaluations_from_yscore(y: np.ndarray, y_score: np.ndarray, description:str=None):
    """
    Evaluates classification metrics given true labels and predicted scores.

    Args:
        y (np.ndarray): True binary labels (0 or 1) for the classification problem.
        y_score (np.ndarray): Predicted scores (probabilities) for the positive class.
        description (str, optional): An optional description for the evaluation. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the evaluation metrics including Accuracy, Precision, Recall, 
                      F1-Score, AUC, and counts of True Negatives (TN), True Positives (TP), False Negatives (FN),
                      and False Positives (FP). Each metric is represented as a single-element list.
    """
    y_pred = y_score>0.5
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    results = {'Description': '' if description is None else description,
               'Accuracy': accuracy_score(y, y_pred),
               'Precision': precision_score(y, y_pred),
               'Recal': recall_score(y, y_pred),
               'F1-Score': f1_score(y, y_pred),
               'AUC': roc_auc_score(y_true=y, y_score=y_score),
               'TN': tn,
               'TP': tp,
               'FN': fn,
               'FP': fp,
               }
    results = {key: [value] for key, value in results.items()}
    return pd.DataFrame(results)

def get_metric_evaluations_from_ypred(y_true:np.ndarray,
                                      y_pred:np.ndarray,
                                      y_score:np.ndarray=None,
                                      description:str=None):
    """
    Evaluates classification metrics given true labels, predicted labels, and optional predicted scores.

    Args:
        y_true (np.ndarray): True binary labels (0 or 1) for the classification problem.
        y_pred (np.ndarray): Predicted binary labels (0 or 1) for the classification problem.
        y_score (np.ndarray, optional): Predicted scores (probabilities) for the positive class. Defaults to None.
        description (str, optional): An optional description for the evaluation. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the evaluation metrics including Accuracy, Precision, Recall, 
                      F1-Score, and counts of True Negatives (TN), True Positives (TP), False Negatives (FN),
                      and False Positives (FP). If `y_score` is provided, AUC is also included; otherwise, 'N/A' is returned for AUC.
                      Each metric is represented as a single-element list.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    results = {'Description': '' if description is None else description,
                        'Accuracy': accuracy_score(y_true, y_pred),
                        'Precision': precision_score(y_true, y_pred),
                        'Recal': recall_score(y_true, y_pred),
                        'F1-Score': f1_score(y_true, y_pred),
                        'TN': tn,
                        'TP': tp,
                        'FN': fn,
                        'FP': fp
                        }
    if not y_score is None:
        results['AUC'] = roc_auc_score(y_true=y_true, y_score=y_score)
    else:
        results['AUC']='N/A'

    results = {key: [value] for key, value in results.items()}
    return pd.DataFrame(results)

def get_metric_evaluations(trained_model,
                            X:np.ndarray,
                            y:np.ndarray,
                            model_config_name:str, 
                            experiment_config_name:str, 
                            description='', ):
    """
    Evaluates classification metrics using a trained model, input features, and true labels.

    Args:
        trained_model: A trained classification model with `predict` and `predict_proba` methods.
        X (np.ndarray): Input features for prediction.
        y (np.ndarray): True binary labels (0 or 1) for the classification problem.
        model_config_name (str): The name or identifier of the model configuration used.
        experiment_config_name (str): The name or identifier of the experiment configuration.
        description (str, optional): An optional description for the evaluation. Defaults to ''.

    Returns:
        pd.DataFrame: A DataFrame containing the evaluation metrics including Accuracy, Precision, Recall,
                      F1-Score, AUC, and counts of True Negatives (TN), True Positives (TP), False Negatives (FN),
                      and False Positives (FP). It also includes the experiment and model configuration names.
                      Each metric is represented as a single-element list.
    """
    y_pred = trained_model.predict(X)
    y_score = trained_model.predict_proba(X)[:,1]

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    results = {'Description': description,
                        'Accuracy': accuracy_score(y, y_pred),
                        'Precision': precision_score(y, y_pred),
                        'Recal': recall_score(y, y_pred),
                        'F1-Score': f1_score(y, y_pred),
                        'AUC': roc_auc_score(y_true=y, y_score=y_score),
                        'TN': tn,
                        'TP': tp,
                        'FN': fn,
                        'FP': fp,
                        'Experiment config':experiment_config_name,
                        'Model config': model_config_name
                        }
    results = {key: [value] for key, value in results.items()}
    return pd.DataFrame(results)
