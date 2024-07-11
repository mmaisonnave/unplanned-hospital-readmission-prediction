import json
import pandas as pd
import numpy as np

import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,accuracy_score

import sys
sys.path.append('..')
from utilities import configuration
from utilities import health_data
from utilities import io
from utilities import metrics


def _logit_pvalue(model, x):
    """ Calculate z-scores for scikit-learn LogisticRegression.
     parameters:
     model: fitted sklearn.linear_model.LogisticRegression with intercept and large C
        x:     matrix on which the model was fit
            This function uses asymtptics for maximum likelihood estimates.
    """
    p = model.predict_proba(x)
    n = len(p)
    m = len(model.coef_[0]) + 1
    coefs = np.concatenate([model.intercept_, model.coef_[0]])
    x_full = np.matrix(np.insert(np.array(x), 0, 1, axis = 1))
    ans = np.zeros((m, m))
    for i in range(n):
        ans = ans + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p[i,1] * p[i, 0]
    vcov = np.linalg.inv(np.matrix(ans))
    se = np.sqrt(np.diag(vcov))
    t =  coefs/se
    p = (1 - scipy.stats.norm.cdf(abs(t))) * 2
    return p


if __name__ == '__main__':
    EXPERIMENT_CONFIGURATION_NAME='configuration_93' # All binary features, min_df=2, with feature selection 7500
    MODEL_CONFIGURATION_NAME = 'model_8' # (N)+(C)+(I)+ Combined D (CD) + class balanced weights

    io.debug(f'Using EXPERIMENT_CONFIGURATION_NAME={EXPERIMENT_CONFIGURATION_NAME}')
    io.debug(f'Using MODEL_CONFIGURATION_NAME=     {MODEL_CONFIGURATION_NAME}')

    config = configuration.get_config()

    experiment_configurations = json.load(open(config['experiments_config'], encoding='utf-8'))
    X_train, y_train, X_test, y_test, features_names = health_data.Admission.get_train_test_matrices(experiment_configurations[EXPERIMENT_CONFIGURATION_NAME])

    io.debug(f'X_train.shape={X_train.shape}')
    io.debug(f'y_train.shape={y_train.shape}')

    io.debug(f'X_test.shape= {X_test.shape}')
    io.debug(f'y_test.shape= {y_test.shape}')



    io.debug('Creating model ...')
    logreg = configuration.model_from_configuration_name(MODEL_CONFIGURATION_NAME)

    io.debug('Standarizing data ...')
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.toarray())
    X_test = scaler.transform(X_test.toarray())

    io.debug('Training data ...')
    logreg.fit(X_train, y_train)

    def _get_metric_evaluations(evaluated_model, X, y_true, model_config_name, experiment_config_name, description=''):
        y_pred = evaluated_model.predict(X)
        y_score = evaluated_model.predict_proba(X)[:,1]

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        results = {'Description': description,
                            'Accuracy': accuracy_score(y_true, y_pred),
                            'Precision': precision_score(y_true, y_pred),
                            'Recal': recall_score(y_true, y_pred),
                            'F1-Score': f1_score(y_true, y_pred),
                            'AUC': roc_auc_score(y_true=y_true, y_score=y_score),
                            'TN': tn,
                            'TP': tp,
                            'FN': fn,
                            'FP': fp,
                            'Model config': model_config_name,
                            'Experiment config': experiment_config_name
                            }
        results = {key: [results[key]] for key in results}
        return pd.DataFrame(results)

    io.debug('Computing model results ...')
    df = pd.concat([metrics.get_metric_evaluations(logreg, 
                                                   X_train, 
                                                   y_train, 
                                                   MODEL_CONFIGURATION_NAME, 
                                                   EXPERIMENT_CONFIGURATION_NAME, 
                                                    description='LogReg train'),
                    metrics.get_metric_evaluations(logreg,
                                                   X_test, 
                                                   y_test, 
                                                   MODEL_CONFIGURATION_NAME, 
                                                   EXPERIMENT_CONFIGURATION_NAME, 
                                                   description='LogReg test')])


    df.to_csv(config['explainable_lr_metrics'], index=False)

    io.debug('Formating and storing resuts ...')
    diagnosis_mapping = health_data.Admission.get_diagnoses_mapping()
    intervention_mapping = health_data.Admission.get_intervention_mapping()

    def code2description(code):
        if code.upper() in diagnosis_mapping or code in diagnosis_mapping:
            assert not (code.upper() in intervention_mapping or code in intervention_mapping)
            return "DIAG: "+diagnosis_mapping[code.upper()]
        
        if code.upper() in intervention_mapping or code in intervention_mapping:
            assert not (code.upper() in diagnosis_mapping or code in diagnosis_mapping)
            return "INT: "+intervention_mapping[code.upper()]
        return "N/A"


    io.debug('Finished computing coefficients (normalized features), formating and saving ...')
    scored_feature_names = list(zip(list(logreg.coef_[0,:]),
                                    features_names))

    # scored_feature_names = sorted(scored_feature_names, 
    #                             key=lambda x:np.abs(x[0]), reverse=True)

    coefficients_df = pd.DataFrame(scored_feature_names,
                                columns=['Score', 'Feature Name'])

    coefficients_df = coefficients_df[['Feature Name', 'Score']]
    coefficients_df['Code Description'] = list(map(code2description, coefficients_df['Feature Name']))


    # I could never run this process, it dies, 7 days is not enough time to run this part. Pvalues are not computed
    # coefficients_df['pvalues'] = _logit_pvalue(logreg, X_train)

    coefficients_df.to_csv(config['explainable_lr_coefficients'], index=False)
    io.debug('DONE')
