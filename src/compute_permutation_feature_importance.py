"""
This script computes the permutation feature importance for a 
BalancedRandomForestClassifier (BRF) model, focusing on a subset of features 
that includes only categorical and numerical variables, excluding diagnosis 
and intervention codes.

Key functionalities:
1. Loads the data and model configuration specified by `configuration_93`.
2. Filters out diagnosis and intervention codes from the dataset, retaining only 
   categorical and numerical variables.
3. Trains a new BRF model using the filtered dataset.
4. Evaluates the performance of the trained BRF model on both the training and 
   test sets.
5. Computes the permutation feature importance for the BRF model, providing 
   insights into the importance of each feature.
6. Saves the performance metrics of the BRF model and the permutation feature 
   importance results to CSV files for further analysis.

The results are stored in:
- `permutation_feature_importance_only_num_and_cat.csv`: Contains the sorted 
  features along with their associated importance scores and standard deviations.
- `brf_with_cat_and_num.csv`: Contains the performance metrics for the BRF 
  model trained without diagnosis and intervention features.
"""
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,accuracy_score

import sys

sys.path.append('..')

from utilities import configuration
from utilities import health_data
from utilities import metrics
from utilities import io



if __name__ == '__main__':
    SEED = 1593085724
    EXPERIMENT_CONFIGURATION_NAME = 'configuration_93' # (N)+(C)+(I)+ Combined D (CD)
    MODEL_CONFIGURATION_NAME='model_1' # BRF, balanced
    PERMUTATION_REPETITION_COUNT=10

    config = configuration.get_config()

    experiment_configurations = json.load(open(config['experiments_config'], encoding='utf-8'))
    X_train, y_train, X_test, y_test, feature_names = health_data.Admission.get_train_test_matrices(experiment_configurations[EXPERIMENT_CONFIGURATION_NAME])
    io.debug(f'X_train.shape={X_train.shape}')
    io.debug(f'y_train.shape={y_train.shape}')

    io.debug(f'X_test.shape= {X_test.shape}')
    io.debug(f'y_test.shape= {y_test.shape}')


    io.debug('Filtering intervention and diagnosis codes from X_train and X_test')
    diagnosis_mapping = health_data.Admission.get_diagnoses_mapping()
    intervention_mapping = health_data.Admission.get_intervention_mapping()
    codes = set(map(str.lower, 
                    intervention_mapping.keys())).union(map(str.lower,
                                                            set(diagnosis_mapping.keys())))

    cat_and_num_variable = list(filter(lambda feature: not feature.lower() in codes, feature_names))

    X_train = X_train[:,:len(cat_and_num_variable)]
    X_test = X_test[:,:len(cat_and_num_variable)]
    feature_names=feature_names[:len(cat_and_num_variable)]
    io.debug(f'X_train.shape={X_train.shape}')
    io.debug(f'y_train.shape={y_train.shape}')

    io.debug(f'X_test.shape= {X_test.shape}')
    io.debug(f'y_test.shape= {y_test.shape}')

    io.debug('Training new BRF model with new X_train and X_test')
    brf = configuration.model_from_configuration_name(MODEL_CONFIGURATION_NAME)
    io.debug(f'Training model MODEL_CONFIGURATION_NAME={MODEL_CONFIGURATION_NAME} ...')
    brf.fit(X_train, y_train)
    io.debug('Storing performance of new BRF model')

    df = pd.concat([metrics.get_metric_evaluations(brf,
                                        X_train,
                                        y_train,
                                        MODEL_CONFIGURATION_NAME,
                                        experiment_config_name=EXPERIMENT_CONFIGURATION_NAME,
                                        description='Main BRF only cat and num (train)'
                                        ),
                    metrics.get_metric_evaluations(brf,
                                        X_test,
                                        y_test,
                                        MODEL_CONFIGURATION_NAME,
                                        experiment_config_name=EXPERIMENT_CONFIGURATION_NAME,
                                        description='Main BRF only cat and num (test)')])
    
    io.debug(df[['Precision', 'Recal', 'F1-Score', 'AUC']])

    df.to_csv(config['pfi_performance'],
            index=True)

    io.debug('Computing permutation feature importance ...')
    r = permutation_importance(brf,
                            X_test.toarray(),
                            y_test,
                            n_repeats=PERMUTATION_REPETITION_COUNT,
                            scoring='roc_auc',
                            random_state=np.random.RandomState(seed=SEED))
    io.debug('Permutation feature importance computation finished, formating results. ')

    del r['importances']
    results = pd.DataFrame(r)
    results['variable'] = feature_names

    columns = results.columns
    columns = list(columns[-1:]) + list(columns[:-1])
    results = results[columns]

    results = results.sort_values(by='importances_mean',
                                  ascending=False)

    output_filename = config['pfi_results']

    results.to_csv(output_filename,
                   index=None)
    io.debug(f'Results stored to disk (output_filename={output_filename}).')
    io.debug('Done')