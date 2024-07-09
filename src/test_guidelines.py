import numpy as np
import pandas as pd

import os

import sys

sys.path.append('..')

import matplotlib.pyplot as plt

from utilities import health_data
from utilities import configuration
from utilities import metrics
from utilities import io

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

def _get_guideline_feature_names():
    return [  
              # Group I
              'acute_days',
              'cmg',
              'New Acute Patient',
              'Unplanned Readmit',

              # Group II
              'alc_days',
              'Day Surgery Entry',
              'Emergency Entry',
              'General Surgery',
              'level 1 comorbidity',
              'transfusion given',
              'urgent admission',

              # Group III
              'age',
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
              'Urology',

              # Group I D+i
              'j441',
              'i500',
              'z515',
              'z38000',
              '5md50aa',
              ]



def create_train_test_df(experiment_configuration='configuration_93' ,
                         to_disk=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Get train test matrices using health_data.Admission.configuration_from_configuration_name 
    and filtering down the features to the 34 features from the consensus analysis. Variables
    used to determine population are excluded (like urgent, unplanned readmit, day surgery).

    Args:
        experiment_configuration (str, optional): configuration name in experiment_configuration.json. 
                                                  Defaults to 'configuration_93'.
        to_disk (bool, optional): If True stores the two DFs to disk. Defaults to False.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: We return two data frames (train and test) with the 34 relevant
                                           features plus the target variable. 
    """    
    EXPERIMENT_CONFIGURATION=experiment_configuration
    FEATURES=_get_guideline_feature_names()
    io.debug(f'Using EXPERIMENT_CONFIGURATION={EXPERIMENT_CONFIGURATION}, obtaining params ...')
    params = configuration.configuration_from_configuration_name(EXPERIMENT_CONFIGURATION)

    io.debug('Retrieving train and test matrices ...')
    X_train, \
        y_train, \
            X_test, \
                y_test, \
                    columns =  health_data.Admission.get_train_test_matrices(params)
    
    io.debug(f'X_train.shape={X_train.shape}')
    io.debug(f'y_train.shape={y_train.shape}')

    io.debug(f'X_test.shape={X_test.shape}')
    io.debug(f'y_test.shape={y_test.shape}')



    io.debug(f'Removing irrelevant columns, only keeping the following features: {str(FEATURES)}')
    columns = list(columns)
    indexes = [ix for ix in [columns.index(feature) for feature in FEATURES]]

    
    X_train = X_train[:,indexes].toarray()
    X_test = X_test[:,indexes].toarray()

    io.debug(f'X_train.shape={X_train.shape}')
    io.debug(f'y_train.shape={y_train.shape}')

    io.debug(f'X_test.shape={X_test.shape}')
    io.debug(f'y_test.shape={y_test.shape}')

    io.debug('Building DataFrames ...')

    train_df = pd.DataFrame(np.hstack([X_train, y_train.reshape(-1,1)]),
                            columns=FEATURES+['target'])
    
    test_df = pd.DataFrame(np.hstack([X_test, y_test.reshape(-1,1)]),
                           columns=FEATURES+['target'])
    
    if to_disk:
        train_df.to_csv('train_df.csv', index=None)
        test_df.to_csv('test_df.csv', index=None)

    return train_df, test_df




class Guideline():
    """Contains the four decision trees that constitute the guideline. Implements the methods
         - fit:             train all four DTs.
         - predict:         predicts using all four DTs.
         - plot_trees:      plots all four DTs.
         - get_masks:       masks that define the population.
         - get_mask_names:  name for those masks.
         
    """    
    def __init__(self,max_depth=3):
        io.debug(f'New Guideline created(max_depth={max_depth})')
        io.debug(f'Masks names: {str(Guideline.get_mask_names())}')
        self.max_depth=max_depth
        self.models = []
        self.trained=False
        self.columns=None

    def fit(self, train:pd.DataFrame) -> None:
        assert 'target' in train.columns, 'Label data not available for training'

        io.debug(f'Training Guidelines. train data shape={train.shape} ({type(train)})')

        masks = Guideline.get_masks(train)
        mask_names = Guideline.get_mask_names()
        self.columns = train.drop(columns=['target']).columns

        for mask, mask_name in zip(masks, mask_names):
            io.debug(f'Mask name={mask_name}')
            population_data = train[mask]

            io.debug(f'population_data.shape={population_data.shape}')

            dt = DecisionTreeClassifier(max_depth=self.max_depth, class_weight='balanced')
            io.debug(f'Model={str(dt)}')
            
            dt.fit(population_data.drop(columns=['target']).values,
                   population_data['target'].values)
            
            self.models.append(dt)
        self.trained=True

    def predict(self, data:pd.DataFrame) -> np.ndarray:
        assert self.trained
        assert all(self.columns== data.drop(columns=['target']).columns)

        io.debug(f'Evaluating guidelines. data.shape={data.shape} ({type(data)})')

        all_ytrues=[]
        all_yscores=[]
        results=None
        for model, mask, mask_name in zip(self.models, 
                                          Guideline.get_masks(data), 
                                          Guideline.get_mask_names()):
            
            io.debug(f'Evaluating on population = {mask_name}')
            population_data = data[mask]

            io.debug(f'Population size = {population_data.shape} ({type(population_data)})')

            # yhat = model.predict(population_data.drop(columns=['target']).values,)
            y_score = model.predict_proba(population_data.drop(columns=['target']).values,)[:,1]
            ytrue = population_data['target'].values
            
            # io.debug(f'y_score={str(y_score)}')
            # io.debug(f'ytrue={str(ytrue)}')

            io.debug(f'type(y_score)={type(y_score)}')
            io.debug(f'type(ytrue)={type(ytrue)}')

            all_yscores.append(y_score)
            all_ytrues.append(ytrue)
            
            new_results = metrics.get_metric_evaluations_from_yscore(ytrue,
                                                                     y_score,
                                                                     description=mask_name)
            results = new_results if results is None else pd.concat([results, new_results])

            io.info(f'New metrics computed: {new_results.iloc[0,:]}')

        new_results = metrics.get_metric_evaluations_from_yscore(np.hstack(all_ytrues),
                                                                 np.hstack(all_yscores),
                                                                 description='all')
        results = pd.concat([results, new_results])
        io.info(f'New metrics computed: {new_results.iloc[0,:]}')

        io.debug('Saving performance to disk:')
        io.debug(pd.DataFrame(results))
        return results

    def plot_trees(self,):    
        config = configuration.get_config()
        for model, mask_name in zip(self.models, self.get_mask_names()):
            io.debug(f'Printing tree {mask_name}')
            fig, ax = plt.subplots(figsize=(25,15))
            tree.plot_tree(model,
                        feature_names=list(self.columns),
                        class_names=['NR', 'R'],
                        fontsize=11,
                        impurity=False,
                        label='none',
                        filled=True,
                        node_ids=False,
                        )
            
            base_name = config['guidelines_dts_figures']
            figure_filename = f"{base_name}_{mask_name.lower().replace(' ','_')}_dt.jpg"
            io.debug(f'Saving tree in {figure_filename}')
            fig.savefig(figure_filename, bbox_inches='tight')

    @staticmethod
    def get_masks(data:pd.DataFrame) -> list[np.ndarray]:
        return [(data['urgent admission']==0),
                (data['urgent admission']==1)&(data['Unplanned Readmit']==0),
                (data['urgent admission']==1)&(data['Unplanned Readmit']==1)&(data['Day Surgery Entry']==0),
                (data['urgent admission']==1)&(data['Unplanned Readmit']==1)&(data['Day Surgery Entry']==1),
                ]
    @staticmethod
    def get_mask_names() -> list[str]:
        return ['urgent=0',
                'Urgent=1 -> Unplanned Readmit=0',
                'Urgent=1 -> Unplanned Readmit=1 -> day surgery=0',
                'Urgent=1 -> Unplanned Readmit=1 -> day surgery=1',
                ]

def get_heldout(experiment_configuration='configuration_93')->pd.DataFrame:
    """
    Retrieves the development and heldout matrices (shared number of columns) using the method 
    health_data.Admission.get_development_and_held_out_matrices. Removing all variables except the 
    34 relevant variables from the consensus analyisis (N+C groups I to III and D+I group I). Variables
    used to determine population are excluded (like urgent, unplanned readmit, day surgery).

    Args:
        experiment_configuration (str, optional): configuration name in experiment_configuration.json. 
                                                  Defaults to 'configuration_93'.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: We return two data frames (development and heldout) with the 
                                           34 relevant features plus the target variable. 
    """    
    EXPERIMENT_CONFIGURATION=experiment_configuration
    FEATURES=_get_guideline_feature_names()
    
    io.debug(f'Using EXPERIMENT_CONFIGURATION={EXPERIMENT_CONFIGURATION}, obtaining params ...')
    params = configuration.configuration_from_configuration_name(EXPERIMENT_CONFIGURATION)

    io.debug('Retrieving train and test matrices ...')
    X_development, \
        y_development, \
            X_heldout, \
                y_heldout, \
                    columns =  health_data.Admission.get_development_and_held_out_matrices(params)
    
    io.debug(f'X_development.shape={X_development.shape} ({type(X_development)})')
    io.debug(f'y_development.shape={y_development.shape} ({type(y_development)})')

    io.debug(f'X_heldout.shape={X_heldout.shape} ({type(X_heldout)})')
    io.debug(f'y_heldout.shape={y_heldout.shape} ({type(y_heldout)})')


    io.debug(f'Removing irrelevant columns, only keeping the following features: {str(FEATURES)}')
    columns = list(columns)
    indexes = [ix for ix in [columns.index(feature) for feature in FEATURES]]

    
    X_development = X_development[:,indexes].toarray()
    X_heldout = X_heldout[:,indexes].toarray()

    io.debug(f'X_development.shape={X_development.shape} ({type(X_development)})')
    io.debug(f'y_development.shape={y_development.shape} ({type(y_development)})')

    io.debug(f'X_heldout.shape={X_heldout.shape} ({type(X_heldout)})')
    io.debug(f'y_heldout.shape={y_heldout.shape} ({type(y_heldout)})')

    io.debug('Building DataFrames ...')

    development_df = pd.DataFrame(np.hstack([X_development, y_development.reshape(-1,1)]),
                            columns=FEATURES+['target'])
    
    heldout_df = pd.DataFrame(np.hstack([X_heldout, y_heldout.reshape(-1,1)]),
                           columns=FEATURES+['target'])

    return development_df, heldout_df

def _get_guideline_only_features()->list[str]:
    return ['urgent admission',
            'acute_days',
            'cmg',
            'case_weight',
            'OBS Delivered',
            'Unplanned Readmit',
            'Day Surgery Entry',
            'alc_days',
            'New Acute Patient',
            'Urology',
            'General Medicine'
            ]


def main():
    config = configuration.get_config()

    # Obtaining train and test DataFrames. 
    EXPERIMENT_CONFIGURATION_NAME='configuration_93'
    train_df, test_df = create_train_test_df(experiment_configuration=EXPERIMENT_CONFIGURATION_NAME)

    # Create and training model
    guideline = Guideline()
    guideline.fit(train_df)

    # Evaluating on train and test
    train_performance = guideline.predict(train_df)
    test_performance = guideline.predict(test_df)

    # Saving trees to disk
    guideline.plot_trees()

    # Obtaining and evaluating on heldout
    development_df, heldout_df = get_heldout(experiment_configuration=EXPERIMENT_CONFIGURATION_NAME)
    heldout_performance = guideline.predict(heldout_df)


    # Saving results
    train_performance['split'] = ['train']*train_performance.shape[0]
    test_performance['split'] = ['test']*test_performance.shape[0]
    heldout_performance['split'] = ['heldout']*heldout_performance.shape[0]

    all_results = pd.concat([train_performance,
                             test_performance,
                             heldout_performance])

    performance_results_filename = config['guideline_dts_performances']
    pd.DataFrame(all_results).to_csv(performance_results_filename,index=None)

    io.ok('Done!')


if __name__ == '__main__':
    main()
