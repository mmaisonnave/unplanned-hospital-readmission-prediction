"""
Module: gensim_model_evaluation

Description:
------------
This module evaluates the performance of pre-trained Gensim Doc2Vec models on the 
task of diagnosis and intervention similarity. It loads models specified in the 
configuration file, applies them to a sample of data, and computes the ranking of 
the correct diagnosis or intervention codes. The results are saved in CSV files for 
further analysis.

Usage:
------
python gensim_model_evaluation.py --considered-ranks=<int> --random-sample-size=<int>

Arguments:
----------
--considered-ranks : int [Required]
    The number of top similar documents to consider for ranking evaluation.

--random-sample-size : int [Required]
    The number of random samples to use for evaluation from the total dataset.

Functionality:
--------------
1. **Loading Configuration**:
   - Reads model configurations from 'gensim_config' specified in the configuration file.

2. **Data Preparation**:
   - Loads training and testing data using the `health_data.Admission.get_training_testing_data()` method.
   - Prepares diagnosis and intervention data for evaluation.

3. **Model Evaluation**:
   - For each model listed in the configuration file, the script:
     - Loads the model from the specified folder.
     - Selects a random sample of data.
     - Computes the similarity rankings of the diagnosis or intervention codes.
     - Records the rank of the correct code in the list of most similar codes.

4. **Results**:
   - Generates CSV files containing the evaluation results:
     - `diagnosis_results.csv`: Evaluation results for diagnosis models.
     - `intervention_results.csv`: Evaluation results for intervention models.
   - Files are saved in the 'gensim_results_folder' specified in the configuration file.

5. **Output Format**:
   - The CSV files have columns for ranks, with each model's performance reported.

Warnings:
---------
- Models must be pre-trained and available in the 'gensim_model_folder' for evaluation.
- The CSV files will be overwritten if they already exist in the results folder.
"""

import gensim
import numpy as np
import collections
import os
import pandas as pd
import argparse
import json

import sys
sys.path.append('..')

from utilities import health_data
from utilities import configuration
from utilities import io


def evaluate_diagnosis_models(considered_ranks, random_sample_size):
    config = configuration.get_config()

    with open(config['gensim_config'])  as f:
        gensim_config = json.load(f)

    train, testing = health_data.Admission.get_training_testing_data()


    diagnosis_data = [gensim.models.doc2vec.TaggedDocument(admission.diagnosis.codes, [ix]) 
                                                        for ix,admission in enumerate(train+testing)]

    model_names = [filename_ for filename_ in gensim_config['diagnosis_configs'].keys()]
    io.debug(model_names)


    model_names = [model_name
                   for model_name in model_names
                   if os.path.isfile(os.path.join(config['gensim_model_folder'], model_name))]
    io.debug(model_names)


    df = pd.DataFrame([-1]+list(range(1,considered_ranks+1)), columns=['Rank'])

    for model_name in sorted(model_names):
        io.debug(f'Working on {model_name}')
        model = gensim.models.doc2vec.Doc2Vec.load(os.path.join(config['gensim_model_folder'], model_name))

        rng = np.random.default_rng(seed=1299137873036141205)

        random_sample_ix = rng.choice(range(len(diagnosis_data)), size=random_sample_size, replace=False)
        random_sample = [diagnosis_data[ix] for ix in random_sample_ix]


        ranks = []
        for doc_id, words in [(item.tags[0],item.words) for item in random_sample]:
            inferred_vector = model.infer_vector(words)
            sims = model.dv.most_similar([inferred_vector], topn=considered_ranks)
            if doc_id in [docid for docid, sim in sims]:
                rank = [docid for docid, sim in sims].index(doc_id)+1
            else:
                rank = -1
            ranks.append(rank)

        freq = collections.Counter(ranks)

        frequencies = []
        for rank in range(-1,considered_ranks+1):
            if rank==0:
                continue
            if rank not in freq:
                freq[rank]=0
            frequencies.append(freq[rank])

        df[model_name]=frequencies
    df.to_csv(os.path.join(config['gensim_results_folder'], 'diagnosis_results.csv'), sep=';', index=False)


def evaluate_intervention_models(considered_ranks, random_sample_size):
    config = configuration.get_config()

    with open(config['gensim_config'], encoding='utf-8')  as f:
        gensim_config = json.load(f) 

    train, testing = health_data.Admission.get_training_testing_data()

    interventions_data = [gensim.models.doc2vec.TaggedDocument(admission.intervention_code, [ix]) 
                                                        for ix,admission in enumerate(train+testing)]

    model_names = [filename_ for filename_ in gensim_config['intervention_configs'].keys()]

    model_names = [model_name 
                   for model_name in model_names 
                   if os.path.isfile(os.path.join(config['gensim_model_folder'], model_name))]

    df = pd.DataFrame([-1]+list(range(1,considered_ranks+1)), columns=['Rank'])

    for model_name in sorted(model_names):
        io.debug(f'Working on {model_name}')

        model = gensim.models.doc2vec.Doc2Vec.load(os.path.join(config['gensim_model_folder'], model_name))

        rng = np.random.default_rng(seed=1299137873036141205)

        random_sample_ix = rng.choice(range(len(interventions_data)), size=random_sample_size, replace=False)
        random_sample = [interventions_data[ix] for ix in random_sample_ix]


        ranks = []
        for doc_id, words in [(item.tags[0],item.words) for item in random_sample]:
            inferred_vector = model.infer_vector(words)
            sims = model.dv.most_similar([inferred_vector], topn=considered_ranks)
            if doc_id in [docid for docid, sim in sims]:
                rank = [docid for docid, sim in sims].index(doc_id)+1
            else:
                rank = -1
            ranks.append(rank)

        freq = collections.Counter(ranks)

        frequencies = []
        for rank in range(-1,considered_ranks+1):
            if rank==0:
                continue
            if rank not in freq:
                freq[rank]=0
            frequencies.append(freq[rank])

        df[model_name]=frequencies
    df.to_csv(os.path.join(config['gensim_results_folder'], 'intervention_results.csv'), sep=';', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--considered-ranks', dest="considered_ranks", type=int, required=True)
    parser.add_argument('--random-sample-size', dest="random_sample_size", type=int, required=True)

    args = parser.parse_args()   

    considered_ranks= args.considered_ranks
    random_sample_size = args.random_sample_size



    # evaluate_diagnosis_models(considered_ranks, random_sample_size)
    evaluate_intervention_models(considered_ranks, random_sample_size)