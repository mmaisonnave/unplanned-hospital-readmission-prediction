import numpy as np
import json

import sys
sys.path.append('..')
from utilities import configuration
from utilities import health_data
from utilities import io

if __name__ == "__main__":
    config = configuration.get_config()

    EXPERIMENT_CONFIGURATION_NAME='configuration_31' # All binary features, min_df=2, with feature selection 7500
    experiment_configurations = json.load(open(config['experiments_config'], encoding='utf-8'))

    X_train, y_train, X_test, y_test, features_names = health_data.Admission.get_train_test_matrices(experiment_configurations[EXPERIMENT_CONFIGURATION_NAME])

    frequency = np.sum(X_train.toarray()>0, axis=0)
    io.debug(f'frequency.shape={frequency.shape}')
    assert frequency.shape[0] == len(features_names)

    diagnosis_mapping = health_data.Admission.get_diagnoses_mapping()
    intervention_mapping = health_data.Admission.get_intervention_mapping()

    code2frequency = {('D', code.lower()): 0 for code in diagnosis_mapping.keys()}
    code2frequency = code2frequency | {('I', code.lower()): 0 for code in intervention_mapping.keys()}

    for feature_name, freq in zip(features_names, frequency):
        if ('D',feature_name.lower()) in code2frequency:
            assert not ('I',feature_name.lower()) in code2frequency
            assert code2frequency[('D',feature_name.lower())]==0
            code2frequency[('D',feature_name.lower())]=freq
        if ('I',feature_name.lower()) in code2frequency:
            assert not ('D',feature_name.lower()) in code2frequency
            assert code2frequency[('I',feature_name.lower())]==0
            code2frequency[('I',feature_name.lower())]=freq

    sorted_results = [(code,freq) for code, freq in code2frequency.items()]
    sorted_results = sorted(sorted_results, key=lambda x: x[1], reverse=True)

    with open(config['intervention_and_diagnosis_frequencies'], 'w', encoding='utf-8') as writer:
        writer.write('\n'.join(f'{code},{freq}' for code, freq in sorted_results))
    

