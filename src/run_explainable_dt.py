import pandas as pd

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from sklearn import tree
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from utilities import configuration
from utilities import health_data
from utilities import metrics
from utilities import io

import json

def _capitalize_feature_name(feature_name:str)->str:
    if feature_name=='cmg':
        return 'CMG'
    elif feature_name=='case_weight':
        return 'RIW'
    else:
        aux = feature_name.replace('_', ' ').replace('-', ' ').strip()
        if aux.split(' ')=='':
            io.debug('Error')
            io.debug(aux)
        return ' '.join([word[0].upper()+word[1:] for word in aux.split(' ')])



if __name__ == '__main__':
    EXPERIMENT_CONFIGURATION_NAMES = [('configuration_27', '(N)'),      #  + U(1.0) + O(0.1)
                                      ('configuration_28', '(C)'),      # + U(1.0) + O(0.1)
                                      ('configuration_87','(N)_(C)'), # + U(1.0) + O(0.1)
                                      ('configuration_30', '(I)'),      #  + U(1.0) + O(0.1)
                                      ('configuration_29_combined', 'Combined D (CD)'), #  +Combined D (CD)
                                      ('configuration_93', '(N)_(C)_(D)_(I)') #(N)+(C)+(I)+ Combined D (CD)
                                      ]
    config = configuration.get_config()
    dfs=[]
    for experiment_configuration_name, experiment_configuration_description in EXPERIMENT_CONFIGURATION_NAMES:
        io.debug(f'EXPERIMENT_CONFIGURATION_NAME={experiment_configuration_name}')
        experiment_configurations = json.load(open(config['experiments_config'], encoding='utf-8'))
        X_train, y_train, X_test, y_test, features_names = health_data.Admission.get_train_test_matrices(experiment_configurations[experiment_configuration_name])

        io.debug(f'X_train.shape={X_train.shape}')
        io.debug(f'y_train.shape={X_train.shape}')

        io.debug(f'X_test.shape= {X_test.shape}')
        io.debug(f'y_test.shape= {y_test.shape}')

        DT_MODEL_CONFIGURATION_NAME = 'model_6_depth_3'
        io.debug(f'DT_MODEL_CONFIGURATION_NAME={DT_MODEL_CONFIGURATION_NAME}')
        dt_model = configuration.model_from_configuration_name(DT_MODEL_CONFIGURATION_NAME)
        dt_model.fit(X_train, y_train)



        dfs += [metrics.get_metric_evaluations(dt_model,
                                                X_train,
                                                y_train,
                                                DT_MODEL_CONFIGURATION_NAME,
                                                experiment_config_name=experiment_configuration_name,
                                                description='TRAIN'
                                                ),
                metrics.get_metric_evaluations(dt_model,
                                                X_test,
                                                y_test,
                                                DT_MODEL_CONFIGURATION_NAME,
                                                experiment_config_name=experiment_configuration_name,
                                                description='TEST')]


        fig, ax = plt.subplots(figsize=(25,15))
        tree.plot_tree(dt_model,
                    feature_names=list(map(_capitalize_feature_name, features_names)),
                    class_names=['NR', 'R'],
                    fontsize=13,
                    impurity=False,
                    label='none',
                    filled=True,
                    node_ids=False,
                    )

        output_file = config['explainable_dt_figures'].replace('.jpg', f'_{experiment_configuration_description}_{DT_MODEL_CONFIGURATION_NAME}.jpg')

        fig.savefig(output_file, bbox_inches='tight')

    df = pd.concat(dfs)
    df.to_csv(config['explainable_dt_metrics'], index=None)
