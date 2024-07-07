"""
Module to handle the configuration yaml configuration file.
"""
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB, BernoulliNB
from imblearn.ensemble import BalancedRandomForestClassifier
import yaml
import os
import numpy as np
import json

def get_config() -> dict:
    """
    Function to get the dictionary with all the configuration parameters.
    """
    with open('../config/paths.yaml', 'r') as file:
        config = yaml.safe_load(file)

    for key, value in config.items():
        if isinstance(value, list):
            for ix, elem in enumerate(value):
                if elem.startswith('$') and elem[1:] in config:
                    value[ix] = config[elem[1:]]
            config[key] = os.path.join(*value)

    return config



def model_from_configuration(params, random_state):
    if 'class_weight' in params and isinstance(params['class_weight'],dict):
        params['class_weight'] = {float(key):value for key, value in params['class_weight'].items()}
        print(params['class_weight'])
    if params['model_name'].startswith('SVC'):
        return SVC(C=params['C'],
                   kernel=params['kernel'],
                   degree=params['degree'],
                   gamma=params['gamma'],
                   coef0=params['coef0'],
                   shrinking=params['shrinking'],
                   probability=params['probability'],
                   tol=params['tol'],
                   cache_size=params['cache_size'],
                   class_weight=params['class_weight'],
                   verbose=params['verbose'],
                   max_iter=params['max_iter'],
                   decision_function_shape=params['decision_function_shape'],
                   break_ties=params['break_ties'],
                   random_state=random_state,
                   )
    if params['model_name'].startswith('DecisionTreeClassifier'):
        return DecisionTreeClassifier(criterion=params['criterion'],
                                      splitter=params['splitter'],
                                      max_depth=params['max_depth'],
                                      min_samples_split=params['min_samples_split'],
                                      min_samples_leaf=params['min_samples_leaf'],
                                      min_weight_fraction_leaf=params['min_weight_fraction_leaf'],
                                      max_features=params['max_features'],
                                      random_state=random_state,
                                      max_leaf_nodes=params['max_leaf_nodes'],
                                      min_impurity_decrease=params['min_impurity_decrease'],
                                      class_weight=params['class_weight'],
                                      ccp_alpha=params['ccp_alpha'],
                                     )
    if params['model_name'].startswith('LogisticRegression'):
        return LogisticRegression(penalty=params['penalty'],
                                  dual=params['dual'],
                                  tol=params['tol'],
                                  C=params['C'],
                                  fit_intercept=params['fit_intercept'],
                                  intercept_scaling=params['intercept_scaling'],
                                  class_weight=params['class_weight'],
                                  random_state=random_state,
                                  solver=params['solver'],
                                  max_iter=params['max_iter'],
                                  multi_class=params['multi_class'],
                                  verbose=params['verbose'],
                                  warm_start=params['warm_start'],
                                  n_jobs=params['n_jobs'],
                                  l1_ratio=params['l1_ratio'],
                                  )
    if params['model_name'].startswith('RandomForestClassifier'):
        return RandomForestClassifier(n_estimators=params['n_estimators'],
                                      criterion=params['criterion'],
                                      max_depth=params['max_depth'],
                                      min_samples_split=params['min_samples_split'],
                                      min_samples_leaf=params['min_samples_leaf'],
                                      min_weight_fraction_leaf=params['min_weight_fraction_leaf'],
                                      max_features=params['max_features'],
                                      max_leaf_nodes=params['max_leaf_nodes'],
                                      min_impurity_decrease=params['min_impurity_decrease'],
                                      bootstrap=params['bootstrap'],
                                      oob_score=params['oob_score'],
                                      n_jobs=params['n_jobs'],
                                      random_state=random_state,
                                      verbose=params['verbose'],
                                      warm_start=params['warm_start'],
                                      class_weight=params['class_weight'],
                                      ccp_alpha=params['ccp_alpha'],
                                      max_samples=params['max_samples'],
                                     )
    if params['model_name'].startswith('MLPClassifier'):
        return MLPClassifier(hidden_layer_sizes=params['hidden_layer_sizes'],
                             activation=params['activation'],
                             solver=params['solver'],
                             alpha=params['alpha'],
                             batch_size=params['batch_size'],
                             learning_rate=params['learning_rate'],
                             learning_rate_init=params['learning_rate_init'],
                             power_t=params['power_t'],
                             max_iter=params['max_iter'],
                             shuffle=params['shuffle'],
                             random_state=random_state,
                             tol=params['tol'],
                             verbose=params['verbose'],
                             warm_start=params['warm_start'],
                             momentum=params['momentum'],
                             nesterovs_momentum=params['nesterovs_momentum'],
                             early_stopping=params['early_stopping'],
                             validation_fraction=params['validation_fraction'],
                             beta_1=params['beta_1'],
                             beta_2=params['beta_2'],
                             epsilon=params['epsilon'],
                             n_iter_no_change=params['n_iter_no_change'],
                             max_fun=params['max_fun'],
                             )
    if params['model_name'].startswith('GaussianNB'):
        return GaussianNB(priors=params['priors'],
                          var_smoothing=params['var_smoothing'],
                          )
    if params['model_name'].startswith('ComplementNB'):
        return ComplementNB(alpha=params['alpha'],
                            force_alpha=params['force_alpha'],
                            fit_prior=params['fit_prior'],
                            class_prior=params['class_prior'],
                            norm=params['norm'],
                            )
    if params['model_name'].startswith('BernoulliNB'):
        return BernoulliNB(alpha=params['alpha'],
                           force_alpha=params['force_alpha'],
                           binarize=params['binarize'],
                           fit_prior=params['fit_prior'],
                           class_prior=params['class_prior'],
                          )
    if params['model_name'].startswith('BalancedRandomForestClassifier'):
        return BalancedRandomForestClassifier(n_estimators=params['n_estimators'],
                                              criterion=params['criterion'],
                                              max_depth=params['max_depth'],
                                              min_samples_split=params['min_samples_split'],
                                              min_samples_leaf=params['min_samples_leaf'],
                                              min_weight_fraction_leaf=params['min_weight_fraction_leaf'],
                                              max_features=params['max_features'],
                                              max_leaf_nodes=params['max_leaf_nodes'],
                                              min_impurity_decrease=params['min_impurity_decrease'],
                                              bootstrap=params['bootstrap'],
                                              oob_score=params['oob_score'],
                                              sampling_strategy=params['sampling_strategy'],
                                              replacement=params['replacement'],
                                              n_jobs=params['n_jobs'],
                                              verbose=params['verbose'],
                                              warm_start=params['warm_start'],
                                              class_weight=params['class_weight'],
                                              ccp_alpha=params['ccp_alpha'],
                                              max_samples=params['max_samples'],
                                              random_state=random_state,
                                              )

def model_from_configuration_name(configuration_name):
    MODEL_SEED = 1270833263
    config = get_config()
    with open(config['models_config'], encoding='utf-8') as reader:
        model_configurations = json.load(reader)

    model_dict = model_configurations[configuration_name]
    model_random_state = np.random.RandomState(MODEL_SEED)
    return model_from_configuration(model_dict, random_state=model_random_state)

def configuration_from_configuration_name(configuration_name):
    config = get_config()
    experiment_configurations = json.load(open(config['experiments_config'], encoding='utf-8'))
    return experiment_configurations[configuration_name]
