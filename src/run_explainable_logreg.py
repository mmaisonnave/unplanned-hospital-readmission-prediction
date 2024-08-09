"""
This module trains a Logistic Regression (LR) model for the unplanned hospital 
readmission task, utilizing the configuration that achieved the best performance 
(`configuration_93`).

Key functionalities:
1. Loads the data and model configuration specified by `configuration_93`.
2. Standardizes the input data to ensure the LR model coefficients are comparable.
3. Trains the LR model on the standardized training data.
4. Evaluates the model's performance on both the training and test sets.
5. Stores the model's performance metrics and the LR coefficients, which can be 
   used to explain the model's decisions.
6. Maps the feature names to human-readable descriptions, including whether they 
   are diagnosis or intervention codes.
7. Saves the coefficients along with their descriptions to a CSV file for 
   interpretability.

This module allows for the analysis of feature importance through the model's 
coefficients.
"""
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


    coefficients_df.to_csv(config['explainable_lr_coefficients'], index=False)
    io.debug('DONE')
