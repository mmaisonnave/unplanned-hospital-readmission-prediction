"""
This script processes the configuration file `config/gensim.json` to compute 
embedding models. It generates embeddings based on the specified configurations, 
skipping any models that have already been computed and stored on disk.

The generated embeddings are stored in the directory `gensim/models/`, and the 
script reads from this directory to check for existing models.

Workflow:
1. The script loads the configurations from `config/gensim.json`.
2. It retrieves the training and testing data.
3. For each intervention embedding model specified in the configuration file:
   - If the model has not been computed yet, the script generates the embeddings 
     and stores them.
   - If the model has already been computed and stored, the script skips it.
4. The same process is repeated for the diagnosis embedding models.

The script ensures that no redundant computations are performed by checking if 
the embeddings already exist on disk before proceeding with model computation.
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