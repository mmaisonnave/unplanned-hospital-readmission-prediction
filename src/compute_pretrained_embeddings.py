"""
This scripts takes config/gensim.json as an input, and computes each one of the 
embedding models described in the file. It skips all already  computed embeddings (those 
which are already stored in disk).

The results are stored (an read from) gensim/models/

"""
import json
import os
import sys
sys.path.append('..')

from utilities import health_data
from utilities import configuration
from utilities import io

if __name__=='__main__':
    config = configuration.get_config()

    # Opening JSON file 
    with open(config['gensim_config'], encoding='utf-8')  as f:
        gensim_config = json.load(f) 

    train, test = health_data.Admission.get_training_testing_data()

    io.debug(f"Working with INTERVENTION embeddings ({len(gensim_config['intervention_configs'])})")
    for config_name in gensim_config['intervention_configs']:
        if os.path.exists(os.path.join(config['gensim_model_folder'], config_name )):
            io.debug(f'Computing {config_name}')
            health_data.Admission.intervention_embeddings(train+test, model_name=config_name)
        else:
            io.debug(f'Skipping: {config_name}')


    io.debug(f"Working with DIAGNOSIS embeddings ({len(gensim_config['diagnosis_configs'])})")
    for config_name in gensim_config['diagnosis_configs']:
        if os.path.exists(os.path.join(config['gensim_model_folder'], config_name)):
            io.debug(f'Computing {config_name}')
            health_data.Admission.diagnosis_embeddings(train+test, model_name=config_name)
        else:
            io.debug(f'Skipping: {config_name}')