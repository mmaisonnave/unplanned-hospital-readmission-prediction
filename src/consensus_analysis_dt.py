"""
This script runs four DT with four different group of features and one ensemble.
"""

import pandas as pd
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score

import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from utilities import configuration
from utilities import health_data
from utilities import metrics
import json
from collections import Counter

def _capitalize_feature_name(feature_name:str)->str:
    if feature_name=='cmg':
        return 'CMG'
    elif feature_name=='case_weight':
        return 'RIW'
    else:
        aux = feature_name.replace('_', ' ').replace('-', ' ').strip()
        if aux.split(' ')=='':
            print('Error')
            print(aux)
        return ' '.join([word[0].upper()+word[1:] for word in aux.split(' ')])


def main():
    EXPERIMENT_CONFIG_NAME='configuration_93' 
    MODEL_CONFIG_NAME= 'model_6_depth_3'

    NC_FEATURE_LIST_GROUP_I=['acute_days', 
                             'cmg', 
                             'New Acute Patient',
                             'Unplanned Readmit', 
                             'urgent admission']
    
    NC_FEATURE_LIST_GROUP_II=['alc_days', 
                              'Day Surgery Entry', 
                              'Emergency Entry',
                              'General Surgery', 
                              'level 1 comorbidity', 
                              'transfusion given']
    
    NC_FEATURE_LIST_GROUP_III=['age',
                               'case_weight',
                               'Direct Entry',
                               'elective admission',
                               'Family Practice',
                               'female',
                               'General Medicine',
                               'is alc',
                               'is central zone',
                              'level 4 comorbidity', 
                               'male',
                               'OBS Delivered',
                               'Oral Surgery',
                               'Orthopaedic Surgery',
                               'Palliative Care',
                               'Panned Readmit',
                               'Psychiatry',
                               'Urology'
                               ]
    DI_FEATURE_LIST_GROUP_I=['j441',
                             'i500',
                             'z515',
                             'Z515',
                             'z38000',
                             '5md50aa',
                             ]

    NC_FEATURE_LIST_GROUP_I_TO_III_DI_GROUP_I = NC_FEATURE_LIST_GROUP_I + \
                                                NC_FEATURE_LIST_GROUP_II + \
                                                NC_FEATURE_LIST_GROUP_III + \
                                                DI_FEATURE_LIST_GROUP_I

    config = configuration.get_config()

    PARAMS = configuration.configuration_from_configuration_name(EXPERIMENT_CONFIG_NAME)
    print(f"use_idf={PARAMS['use_idf']}")
    X_train, y_train, X_test, y_test, feature_names = health_data.Admission.get_train_test_matrices(PARAMS)

    print(f'X_train.shape={X_train.shape}')
    print(f'y_train.shape={y_train.shape}')
    print(f'X_test.shape= {X_test.shape}')
    print(f'y_test.shape= {y_test.shape}')
    print()

    ensemble=[]
    results_df=None

    for FEATURE_LIST,DESCRIPTION in zip([NC_FEATURE_LIST_GROUP_I, 
                                        NC_FEATURE_LIST_GROUP_II, 
                                        NC_FEATURE_LIST_GROUP_III, 
                                        DI_FEATURE_LIST_GROUP_I,
                                        NC_FEATURE_LIST_GROUP_I_TO_III_DI_GROUP_I,
                                        ],
                                        ['NC_group_I',
                                         'NC_group_II',
                                         'NC_group_III',
                                         'DI_group_I',
                                         'NC_group_I_TO_III_AND_DI_group_I'
                                         ]
                                        ):
        DESCRIPTION = f'{DESCRIPTION}_{MODEL_CONFIG_NAME}_{EXPERIMENT_CONFIG_NAME}'
        print(f'Filtering columns (len(FEATURE_LIST)={len(FEATURE_LIST)})')
        print(f'FEATURE_LIST={FEATURE_LIST}')
        selected_columns_ix = [ix for ix,feature_name in enumerate(feature_names) if feature_name in FEATURE_LIST]

        temp_X_train = X_train[:,selected_columns_ix]
        temp_X_test = X_test[:,selected_columns_ix]
        temp_feature_names = feature_names[selected_columns_ix]


        print(f'temp_X_train.shape={temp_X_train.shape}')
        print(f'y_train.shape={y_train.shape}')
        print(f'temp_X_test.shape= {temp_X_test.shape}')
        print(f'y_test.shape= {y_test.shape}')
        print()

        print(f'temp_feature_names={list(temp_feature_names)}')
        dt_model = configuration.model_from_configuration_name(MODEL_CONFIG_NAME)

        dt_model.fit(temp_X_train, y_train)
        ensemble.append(dt_model.predict(temp_X_test))

        print('Computing model results ...')
        new_results = pd.concat([metrics.get_metric_evaluations(dt_model, temp_X_train, y_train, MODEL_CONFIG_NAME, EXPERIMENT_CONFIG_NAME, description=f'DT train({DESCRIPTION})'),
                        metrics.get_metric_evaluations(dt_model, temp_X_test, y_test, MODEL_CONFIG_NAME, EXPERIMENT_CONFIG_NAME, description=f'DT test({DESCRIPTION})')])

        if results_df is None:
            results_df = new_results
        else:
            results_df = pd.concat([results_df, new_results])


        fig, ax = plt.subplots(figsize=(22,10))
        tree.plot_tree(dt_model,
                    feature_names=list(map(_capitalize_feature_name, temp_feature_names)),
                    class_names=['NR', 'R'],
                    fontsize=11,
                    impurity=False,
                    label='none',
                    filled=True,
                    node_ids=False,
                    )

        output_file = config['consensus_dt_figures'].replace('.jpg', f'_{DESCRIPTION}.jpg')

        fig.savefig(output_file, bbox_inches='tight')
    

    # Ensemble performance
    y_pred = (np.sum(ensemble,axis=0)==len(ensemble)).astype('int')
    y_true = y_test

    new_results = metrics.get_metric_evaluations_from_ypred(y_true, 
                                                           y_pred, 
                                                           description=f'test ensemble_{MODEL_CONFIG_NAME}_{EXPERIMENT_CONFIG_NAME}')

    results_df = pd.concat([results_df, new_results])

    metrics_csv_name = config['consensus_dt_metrics']

    results_df.to_csv(metrics_csv_name, index=False)

if __name__ == "__main__":
    main()