import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,accuracy_score


def get_metric_evaluations_from_yscore(y, y_score, description=None):
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

# def get_metric_evaluations_from_yhat_and_ypred(y_true, y_pred, y_score=None, description=None):
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#     results = {'Description': '' if description is None else description,
#                         'Accuracy': accuracy_score(y_true, y_pred),
#                         'Precision': precision_score(y_true, y_pred),
#                         'Recal': recall_score(y_true, y_pred),
#                         'F1-Score': f1_score(y_true, y_pred),
#                         'TN': tn,
#                         'TP': tp,
#                         'FN': fn,
#                         'FP': fp
#                         }
#     if not y_score is None:
#         results['AUC'] = roc_auc_score(y_true=y_true, y_score=y_score)

#     results = {key: [value] for key, value in results.items()}
#     return pd.DataFrame(results)

def get_metric_evaluations(trained_model,
                            X,
                            y,
                            model_config_name, 
                            experiment_config_name, 
                            description='', ):
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
