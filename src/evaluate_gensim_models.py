"""This module takes the name of all the models present in gensim.json, and for all the models that were already
trained, it loads them and try their performance on the task of diagnosis similarity. 

It saves the results in gensim/results/{diagnosis_results.csv and intervention_results.csv}

Only evaluates models that are already trained and available to load from disk in the gensim/models/ folder.

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