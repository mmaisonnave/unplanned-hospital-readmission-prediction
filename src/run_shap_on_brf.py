import sys
import pickle
import joblib
import pandas as pd
sys.path.append('..')

from utilities import configuration
from utilities import health_data
from utilities import metrics
from utilities import io

import json
import shap

import matplotlib.pyplot as plt

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
    

if __name__ == "__main__":
    SAMPLE_SIZE_FOR_BACKGROUND_DISTRIBUTION=500
    SAMPLE_SIZE_TO_COMPUT_SHAP_ON=500
    EXPERIMENT_CONFIGURATION_NAME = 'configuration_93' # (N)+(C)+(I)+ Combined D (CD)
    MODEL_CONFIGURATION_NAME ="model_1" # BRF, balanced

    config = configuration.get_config()
    experiment_configurations = json.load(open(config['experiments_config'], encoding='utf-8'))
    X_train, y_train, X_test, y_test, feature_names = health_data.Admission.get_train_test_matrices(experiment_configurations[EXPERIMENT_CONFIGURATION_NAME])


    io.debug(f'X_train.shape={X_train.shape}')
    io.debug(f'y_train.shape={X_train.shape}')

    io.debug(f'X_test.shape= {X_test.shape}')
    io.debug(f'y_test.shape= {y_test.shape}')




    brf = configuration.model_from_configuration_name(MODEL_CONFIGURATION_NAME)
    io.debug('Training from scratch....')
    brf.fit(X_train, y_train)

    df = pd.concat([metrics.get_metric_evaluations(brf,
                                        X_train,
                                        y_train,
                                        MODEL_CONFIGURATION_NAME,
                                        experiment_config_name=EXPERIMENT_CONFIGURATION_NAME,
                                        description='TRAIN'
                                        ),
                    metrics.get_metric_evaluations(brf,
                                        X_test,
                                        y_test,
                                        MODEL_CONFIGURATION_NAME,
                                        experiment_config_name=EXPERIMENT_CONFIGURATION_NAME,
                                        description='TEST')])
    
    csv_name = config['shap_metrics'][:-len(".csv")]
    csv_name = csv_name + f'_BGD={SAMPLE_SIZE_FOR_BACKGROUND_DISTRIBUTION}'
    csv_name = csv_name + f'_SS={SAMPLE_SIZE_TO_COMPUT_SHAP_ON}.csv'
    df.to_csv(csv_name, index=False)

    io.debug(df[['Precision', 'Recal', 'F1-Score', 'AUC']])

    sampled_X = shap.utils.sample(X_test.toarray(),
                                  SAMPLE_SIZE_FOR_BACKGROUND_DISTRIBUTION
                                  )  # No. of instances for use as the background distribution

    io.debug(f'Sampled X (for background noise) shape={sampled_X.shape}')

    io.debug('Creating explainer ...')
    explainer = shap.Explainer(brf.predict,
                               sampled_X,
                               feature_names=list(map(_capitalize_feature_name, feature_names)),
                               )

    io.debug('Calculating SHAP VALUES')
    shap_values = explainer(X_test[:SAMPLE_SIZE_TO_COMPUT_SHAP_ON,:].toarray(),
                            max_evals=2*X_test.shape[1]+1,
                            )


    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    # ~ Default beeswarm plot ~
    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    fig, ax = plt.gcf(), plt.gca()
    io.debug(f'shap_values.shape={shap_values.shape}')
    shap.plots.beeswarm(shap_values,
                        max_display=30,
                        )



    figure_name = config['shap_figures'][:-len(".jpg")]
    figure_name = figure_name + f'_BGD={SAMPLE_SIZE_FOR_BACKGROUND_DISTRIBUTION}'
    figure_name = figure_name + f'_SS={SAMPLE_SIZE_TO_COMPUT_SHAP_ON}.jpg'

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)

    fig.savefig(figure_name,
                bbox_inches='tight')
    

    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    # ~ Absolute beeswarm plot  ~
    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    fig, ax = plt.gcf(), plt.gca()
    shap.plots.beeswarm(shap_values.abs,
                        max_display=30,
                        color="shap_red")

    figure_name = config['shap_figures'][:-len(".jpg")]
    figure_name = figure_name + f'_BGD={SAMPLE_SIZE_FOR_BACKGROUND_DISTRIBUTION}'
    figure_name = figure_name + f'_SS={SAMPLE_SIZE_TO_COMPUT_SHAP_ON}_abs.jpg'


    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)

    fig.savefig(figure_name,
                bbox_inches='tight')
    

    # ~ ~ ~ ~ ~ ~ ~ 
    # ~ bar plot  ~
    # ~ ~ ~ ~ ~ ~ ~
    fig, ax = plt.gcf(), plt.gca()
    shap.plots.bar(shap_values.abs.mean(0),
                    max_display=30
                   )


    figure_name = config['shap_figures'][:-len(".jpg")]
    figure_name = figure_name + f'_BGD={SAMPLE_SIZE_FOR_BACKGROUND_DISTRIBUTION}'
    figure_name = figure_name + f'_SS={SAMPLE_SIZE_TO_COMPUT_SHAP_ON}_bar.jpg'


    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)

    fig.savefig(figure_name,
                bbox_inches='tight')
