"""
This module contains *no* functions to import. It only contains a single main function that runs all 
Scikit-learn experiments (all models obtained from Scikit-learn that use the fit, predict, and 
predict_proba methods.). For computing the metrics it uses **cross_validation**. 

You can pass a single argument (--simulation) that computes how many experiments are pending, but 
runs none and makes no change in disk (dry run).

The main function takes as INPUT:
(1). 'experiment_configuration.json' file and
(2). 'models.json' file
(the path for both files is obtained from paths.yaml)

In (1), I have the configuration on how to compute the training and testing matrices (in the 
config file I describe the pre-processing steps for the matrices).

In (2), I describe the parameters of the models (the kernel for SVM, number of estimators for 
random forests, etc.). 

Both files (1 and 2) have multiple configurations that describe ALL experiments to be run.

For example, the first model (model_0) in `models.json` is a SVC model that contains 
the following parameters:
    "model_0": {
        "model_name": "SVC",
        "C": 1.0,
        "kernel": "rbf",
        "degree": 3,
        "gamma": "scale",
        "coef0": 0.0,
        "shrinking": true,
        "probability": true,
        "tol": 0.001,
        "cache_size": 200,
        "class_weight": null,
        "verbose": false,
        "max_iter": 5000,
        "decision_function_shape": "ovr",
        "break_ties": false,
        "random_state": null,
        "configuration_ids": ["configuration_0"]
    }

Model_0 uses "configuration_0" to set up the matrices (see last line of model config). 
Configuration_0 looks like this:
    "configuration_0": {
        "fix_skew": false,
        "normalize": false,
        "fix_missing_in_testing": true,
        "numerical_features": true,
        "categorical_features": true,
        "diagnosis_features": true,
        "intervention_features": true,
        "use_idf": false,
        "class_balanced": false,
        "remove_outliers": true,
        "under_sample_majority_class": false,
        "over_sample_minority_class": false,
        "smote_and_undersampling":false
    }

In total, more than a hundred experiments are described in the config files, when this methods run, 
it computes all experiments unless they are already computed and we already have the results stored.


The OUTPUT of the experiments is stored in: 'experiments_results.csv' (path obtained from paths.yaml)


On a high level, the main functions does the following:
1. Recovers all experiment and model configurations from config files
2. Computes from all the described experiments how many are already ran.
3. for each experiment_config found in 'experiment_configuration.json':
4.     X_train, y_train, X_test, y_test <= get_matrices_from config(experiment_config)
5.     for each model_config in 'models.json':
6.         model <= get_model_from_config(model_config)
7.         model.fit(X_train,y_train)
8.         compute_training_metrics(model, X_train, y_train)
8.         compute_testing_metrics(model, X_test, y_test)
9.         append new results to existing result file ('experiments_results.csv')

** From step(2), we know which experiments were already ran, so in (3) and (4) we skip model_config and
   experiment_config that were already run.


"""
# INPUT: 
    # model_configurations = json.load(open(config['models_config'], encoding='utf-8'))
    # experiment_configurations = json.load(open(config['experiments_config'], encoding='utf-8'))

# INPUT AND OUTPUT
    # new_df.to_csv(config['experiment_results'], index=False, sep=';')

from scipy import sparse
import pandas as pd
import numpy as np
import json
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

import os
import argparse

import sys
sys.path.append('..')

from utilities import configuration
from utilities import health_data
from utilities import io

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

if __name__ == '__main__':
    N_SPLITS=3
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- #
    # Simulation parameter used to compute pending experiments without running them           #
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- #
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulation',
                        dest='simulation',
                        required=True,
                        help='Wether to save to disk or not',
                        type=str,
                        choices=['True', 'False']
                        )
    parser.add_argument('--experiment-configuration',
                        dest='experiment_configuration',
                        required=True,
                        help="Choose one particular configuration to run (configuration_name:str) or all ('all')",
                        type=str,
                        )
    args = parser.parse_args()
    simulation = args.simulation=='True'


    # ---------- ---------- ---------- ---------- ---------- ---------- #
    # Retrieving configuration (paths.yaml)                             #
    # ---------- ---------- ---------- ---------- ---------- ---------- #
    config = configuration.get_config()

    if simulation:
        io.debug('Running only a SIMULATION run.')
    io.debug('Starting all experiments ...')

    if args.experiment_configuration == 'all':
        csv_output_file = config['experiment_results_cv']
    else:
        csv_output_file = config['custom_conf_results_cv'][:-4]+f'_{args.experiment_configuration}.csv'
    print(f'STORING results in: {csv_output_file}')

    # ---------- ---------- ---------- ---------- ---------- ---------- #
    # Retrieving model (models.json) and experiment configurations      #
    # (experiment_configuration.json)                                   #
    # ---------- ---------- ---------- ---------- ---------- ---------- #
    model_configurations = json.load(open(config['models_config'], encoding='utf-8'))
    experiment_configurations = json.load(open(config['experiments_config'], encoding='utf-8'))
    io.debug(f'Using {len(model_configurations):4} different models.')
    io.debug(f'Using {len(experiment_configurations):4} different configurations.')

    # ---------- ---------- ---------- #
    # COMPUTING TO-DO EXPERIMENTS      #
    # ---------- ---------- ---------- #
    to_do = []
    for model_id, model_config in model_configurations.items():
        if not ('skipping' in model_config and model_config['skipping']):
            configurations_to_run = [config_id for config_id in model_config['configuration_ids']]
            to_do += [(config_id, model_id) for config_id in configurations_to_run]
    to_do = set(to_do)
    io.debug(f'Number of experiments found (in total): {len(to_do)}')
    
    already_ran = {}
    if os.path.isfile(csv_output_file):
        already_run_df = pd.read_csv(csv_output_file, sep=';')

        already_ran = [(config_id, model_id) for config_id, model_id in zip(already_run_df['config_id'], already_run_df['model_id'])]
        already_ran = set(already_ran)
    io.debug(f'Number of experiments already ran found (in total): {len(already_ran)}')

    pending = to_do.difference(already_ran)
    pending_conf = set([config_id for config_id, model_id in pending])


    io.debug(f'Pending experiments ({len(pending)})={pending}')
    io.debug(f'Number of configuration founds: {len(experiment_configurations)} ({experiment_configurations.keys()})')

    # Filtering configurations not needed for this run:
    experiment_configurations = {configuration_id:configuration_dict 
                                 for configuration_id, configuration_dict in experiment_configurations.items()
                                 if configuration_id in pending_conf
                                 }
    
    if args.experiment_configuration!='all':
        io.debug(f'Filtering using custom configuration={args.experiment_configuration}')
        experiment_configurations = {configuration_id:configuration_dict
                                 for configuration_id, configuration_dict in experiment_configurations.items()
                                 if configuration_id==args.experiment_configuration
                                 }
        io.debug(f'Number of configuration founds after filtering: {len(experiment_configurations)} ({experiment_configurations.keys()})')

        pending = [(config_id, model_id) 
                    for config_id, model_id in pending
                    if config_id==args.experiment_configuration
                    ]
        io.debug(f'Pending experiments after selecting custom configuration ({args.experiment_configuration})({len(pending)})={pending}')

    io.debug(f'Number of config pending to-do: {len(experiment_configurations)} ({experiment_configurations.keys()})')

    if simulation:
        io.debug('Ending simulation without running any experiment ...')
        sys.exit(0)



    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    # RUNNING ALL CONFIGURATIONS
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    for configuration_id, configuration_dict in experiment_configurations.items():
        io.debug(f'Running on configuration ID: {configuration_id}')
        params = configuration_dict
         
        # Computing training and testing matrices.
        # X, y, columns = health_data.Admission.get_train_test_matrices(params)
        X, y, feature_names = health_data.Admission.get_both_train_test_matrices(params)
        print(f'X.shape = {X.shape}')
        print(f'y.shape = {y.shape}')
        print(f'feature_names.shape = {feature_names.shape}')



        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
        # RUNNING ALL MODELS WITH CALCULATED MATRICES
        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
        for model_id, model_dict in model_configurations.items():
            io.debug(f'Working on model ID= {model_id}')

            # Skipping if already ran.
            if os.path.isfile(csv_output_file):
                auxdf = pd.read_csv(csv_output_file, sep=';')
                model_ids = ([model_id_ for model_id_ in auxdf['model_id']])
                configuration_ids = ([config_id_ for config_id_ in auxdf['config_id']])

                already_run_pairs = set([(model_id_, config_id_) 
                                     for model_id_,config_id_ in zip(model_ids,configuration_ids)])
                
                if (model_id, configuration_id) in already_run_pairs:
                    io.debug(f'SKIPPING, configuration ({configuration_id}) and model ({model_id}) already found ...')
                    continue
                io.debug('Results not found, running experiments ...')

            # Skipping model if "skipping"==True
            if 'skipping' in model_dict and model_dict['skipping']:
                io.debug(f'SKIPPING model ({model_id}), as requested by configuration ...')
                continue
            
            # Skipping model if not assigned to current configuration
            if not configuration_id in model_dict['configuration_ids']:
                io.debug(f'SKIPPING model {model_id} because it is not in the configuration_ids list.')
                continue
            else:
                io.debug('Configuration found in configuration_ids list, preparing to run ...')

            # Creating model and fitting
            MODEL_SEED = 1270833263
            model_random_state=np.random.RandomState(MODEL_SEED)
            model = configuration.model_from_configuration(model_dict, random_state=model_random_state)
            model_name = model_dict['model_name']

            io.debug(f'Training model {str(model)}')


            combined_X = X
            combined_y = y

            print(f'combined_X.shape= {combined_X.shape}')
            print()

            print(f'combined_y.shape= {combined_y.shape}')
            print()
            
            
            kf = KFold(n_splits=N_SPLITS, random_state=np.random.RandomState(seed=1149035622), shuffle=True)
            all_results=[]
            for i, (train_index, test_index) in enumerate(kf.split(combined_X)):
                fold_X_train = combined_X[train_index,:]
                fold_y_train = combined_y[train_index]

                fold_X_test = combined_X[test_index,:]
                fold_y_test = combined_y[test_index]

                fold_feature_names = feature_names.copy()


                print(f'fold_X_train.shape={fold_X_train.shape=}')
                print(f'fold_y_train.shape={fold_y_train.shape=}')
                print(f'fold_feature_names.shape={fold_feature_names.shape=}')
                print()

                print(f'fold_X_test.shape={fold_X_test.shape=}')
                print(f'fold_y_test.shape={fold_y_test.shape=}')
                print()

                # -----------------------------------------------
                # ----------------- RE-SAMPLING -----------------
                # -----------------------------------------------
                SAMPLING_SEED = 1270833263
                sampling_random_state = np.random.RandomState(SAMPLING_SEED)

                if configuration_dict['under_sample_majority_class']:
                    assert not configuration_dict['over_sample_minority_class']
                    assert not configuration_dict['smote_and_undersampling']

                    under_sampler = RandomUnderSampler(sampling_strategy=1, random_state=sampling_random_state)
                    print('Under Sampling training set before calling fit ....')

                    # Under sampling:
                    fold_X_train, fold_y_train = under_sampler.fit_resample(fold_X_train, 
                                                                            fold_y_train)
                    print(f'[INSIDE running_all_experiments_csv] resampled(fold_X_train).shape = {fold_X_train.shape}')
                    print(f'[INSIDE running_all_experiments_csv] resampled(fold_y_train).shape = {fold_y_train.shape}')

                elif configuration_dict['over_sample_minority_class']:
                    assert not configuration_dict['under_sample_majority_class']
                    assert not configuration_dict['smote_and_undersampling']

                    over_sample = SMOTE(sampling_strategy=1, random_state=sampling_random_state)
                    print('[INSIDE running_all_experiments_csv] Over Sampling training set before calling fit ....')

                    fold_X_train, fold_y_train = over_sample.fit_resample(fold_X_train, 
                                                                          fold_y_train)
                    print(f'[INSIDE running_all_experiments_csv] resampled(fold_X_train).shape = {fold_X_train.shape}')
                    print(f'[INSIDE running_all_experiments_csv] resampled(fold_y_train).shape = {fold_y_train.shape}')

                elif configuration_dict['smote_and_undersampling']:
                    assert not configuration_dict['under_sample_majority_class']
                    assert not configuration_dict['over_sample_minority_class']

                    over = SMOTE(sampling_strategy=params['over_sampling_ration'],
                                 random_state=sampling_random_state
                                 )
                    under = RandomUnderSampler(sampling_strategy=params['under_sampling_ration'], 
                                               random_state=sampling_random_state
                                               )
                    
                    steps = [('o', over),
                             ('u', under)]
                    
                    pipeline = Pipeline(steps=steps)
                    print('[INSIDE running_all_experiments_csv] Applying both under and over sampling ....')

                    fold_X_train, fold_y_train = pipeline.fit_resample(fold_X_train, 
                                                                       fold_y_train)
                    print(f'[INSIDE running_all_experiments_csv] resampled(fold_X_train).shape = {fold_X_train.shape}')
                    print(f'[INSIDE running_all_experiments_csv] resampled(y_train).shape = {fold_y_train.shape}')

                else:
                    print('[INSIDE running_all_experiments_csv] Using X_train, y_train, no samplig strategy ...')
                # ------------------------------------------------------
                # ----------------- END of RE-SAMPLING -----------------
                # ------------------------------------------------------

                # 
                # AFTER DOING RESAMPLING, some columns now might be constant (only instances with  a 
                # certain value=v were selcted)
                # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
                # REMOVING CONSTANT VARIABLES (CHANGING NUMBER OF COLUMNS, need to update all matrices)
                # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
                io.debug('Looking for constant variables ...')
                fold_feature_names = np.array(fold_feature_names) 


                io.debug('Using memory efficient solution')
                constant_variables = np.array(list(
                    map(lambda ix: True if np.var(fold_X_train[:,ix].toarray())==0 else False, range(fold_X_train.shape[1]))
                ))


                if np.sum(constant_variables)>0:
                    print(f'Removing {np.sum(constant_variables)} constant variables ...')
                    # X = X[:,~constant_variables]
                    fold_X_train = fold_X_train[:,~constant_variables]
                    fold_X_test = fold_X_test[:,~constant_variables]
                    fold_feature_names = fold_feature_names[~constant_variables]
                    print(f'Removed {np.sum(constant_variables)} columns')
                else:
                    print('Not constant variables found ...')
                # ---------- ---------- ---------- ---------- ---------- ---------- #
                #           END OF REMOVING CONSTANT VARIABLES                      #
                # ---------- ---------- ---------- ---------- ---------- ---------- #

                # --------------------------------------------------------------
                # ----------------- BEGIN OF FEATURE SELECTION -----------------
                # --------------------------------------------------------------
                if 'feature_selection' in configuration_dict and configuration_dict['feature_selection']:
                    print('Applying feature selection')
                    clf = SelectKBest(f_classif, 
                                      k=configuration_dict['k_best_features'],).fit(fold_X_train,
                                                                                    fold_y_train)
                    print()
                    assert len(fold_feature_names) == fold_X_train.shape[1], f'{len(fold_feature_names)} != {fold_X_train.shape[1]}'
                    fold_X_train = clf.transform(fold_X_train)
                    fold_X_test = clf.transform(fold_X_test)               
                    fold_feature_names = clf.transform(fold_feature_names.reshape(1,-1))[0,:]     
                    assert len(fold_feature_names) == fold_X_train.shape[1], f'{len(fold_feature_names)} != {fold_X_train.shape[1]}'

                    print(f'fold_X_train.shape={fold_X_train.shape}')
                    print(f'fold_X_test.shape= {fold_X_test.shape}')
                    print(f'fold_feature_names.shape=     {fold_feature_names.shape}')

                # --------------------------------------------------------------
                # -----------------  END OF FEATURE SELECTION  -----------------
                # --------------------------------------------------------------                
                print('Fitting model')
                model.fit(fold_X_train, fold_y_train)

                # EVALUATION ON TESTING
                y_true = fold_y_test
                y_pred = model.predict(fold_X_test)
                y_score= model.predict_proba(fold_X_test)[:,1] #model.predict_proba(X_train)[:,1]

                results = np.array([
                    accuracy_score(y_true, y_pred,),
                    precision_score(y_true, y_pred,),
                    recall_score(y_true, y_pred,),
                    f1_score(y_true, y_pred,),
                    roc_auc_score(y_true=y_true, y_score=y_score),
                ])
                all_results.append(results)
                
            METRIC_COUNT = 5 # Acc, Prec, Recall, F1-score, ROC AUC
            averages = np.average(np.vstack([all_results]), axis=0)
            print(f'averages.shape={averages.shape}')
            assert averages.shape[0] == METRIC_COUNT, f'{averages.shape[0]}'

            stds = np.std(np.vstack([all_results]), axis=0)
            assert stds.shape[0] == METRIC_COUNT, f'{stds.shape[0]}'
            print(f'stds.shape={stds.shape}')

            columns = ['accuracy_avg',
                       'precision_avg',
                       'recall_avg', 
                       'f1_avg', 
                       'roc_auc_avg',
                       'accuracy_std',
                       'precision_std',
                       'recall_std',
                       'f1_std',
                       'roc_auc_std',
                       ]


            results_df = pd.DataFrame(np.hstack([averages, stds]).reshape(1,-1), columns = columns)



            config_df = pd.DataFrame({'Model': [model_name],
                                      'experiment_params': str(configuration_dict) , 
                                      'model_params':  str(model_dict), 
                                      'config_id': configuration_id, 
                                      'model_id': model_id,
                                      })

            new_df = results_df.join(config_df)

            # SAVING
            if os.path.isfile(csv_output_file):
                old_df = pd.read_csv(csv_output_file, sep=';')
                io.debug(f'Previous experiments found: {old_df.shape[0]} entries found')
                new_df = pd.concat([old_df,new_df])

            new_df.to_csv(csv_output_file, index=False, sep=';')
            io.debug(f"Saving results in {csv_output_file.split('/')[-1]}")
            io.debug(f'Saving {new_df.shape[0]} entries.')
