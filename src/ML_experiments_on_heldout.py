import pandas as pd 
import joblib
import os
import sys
sys.path.append('..')

from utilities import configuration
from utilities import health_data
from utilities import metrics
from utilities import io

from dataclasses import dataclass

@dataclass
class Experiment:
    model_name: str
    experiment_configuration_name: str
    model_configuration_name: str



def get_full_list_of_experiments() -> list[Experiment]:
    """
    Reurn a list with all eight models we need to test in held-out.
    The list has the following format:
        (Name, experiment_configuration_name, model_configuraiton_name, )
    """
    best_model_configurations = [Experiment(model_name='BRF1',
                                            experiment_configuration_name='configuration_93',
                                            model_configuration_name='model_1'),
                                Experiment(model_name='BRF2',
                                           experiment_configuration_name='configuration_91',
                                           model_configuration_name='model_2'),
                                Experiment(model_name='MLP1',
                                           experiment_configuration_name='configuration_67',
                                           model_configuration_name='model_5'),
                                Experiment(model_name='DT1',
                                           experiment_configuration_name='configuration_93',
                                           model_configuration_name='model_6'),
                                Experiment(model_name='DT2',
                                           experiment_configuration_name='configuration_91',
                                           model_configuration_name='model_7'),
                                Experiment(model_name='LR',
                                           experiment_configuration_name='configuration_93',
                                           model_configuration_name='model_8'),
                                Experiment(model_name='BNB1',
                                           experiment_configuration_name='configuration_94',
                                           model_configuration_name='model_10'),
                                Experiment(model_name='BNB2',
                                           experiment_configuration_name='configuration_93',
                                           model_configuration_name='model_10'),
                                Experiment(model_name='BRF3',
                                           experiment_configuration_name='configuration_93_bis',
                                           model_configuration_name='model_1'),
                                ]
    return best_model_configurations

def append_new_results_to_file(new_results:pd.DataFrame)->None:
    """It takes new results in pandas data frame format, and stores it to disk 
    filename: "ml_performance_on_held_out.csv"
    if it doensn't exist, it creates the file. If it exists, appends new results 
    (columns must match)

    Args:
        new_results (pd.DataFrame): The new results to write (or append) to disk.
    """    
    config = configuration.get_config()
    if heldout_results_available():
        existing_results =  get_heldout_results()
        concatenated_results = pd.concat([existing_results, new_results])
    else:
        concatenated_results = new_results
    
    concatenated_results.to_csv(config['ml_performance_on_held_out'],
                                index=None,
                                )

def get_heldout_results() -> pd.DataFrame:
    """Recovers from disk the results of the held-outs experiments and returns it as a dataframe

    Returns:
        pd.DataFrame: returns a dataframe with all the results from the experiments in the held-out
    """
    config = configuration.get_config()
    results_df = pd.read_csv(config['ml_performance_on_held_out'],)
    return results_df

def heldout_results_available()->bool:
    """Check if there are results from the held-out experiments in disk.

    Returns:
        bool: Returns true if are already computed results from the held-out experiments or false otherwise
    """    
    config = configuration.get_config()
    return os.path.isfile(config['ml_performance_on_held_out'])

def get_already_trained_model_names()->list[str]:
    """Get the names of the models already trained to test on the held-out

    Returns:
        list[str]: name list of the models already trained.
    """
    results_df = get_heldout_results()
    trained_model_names = [model_name \
                           for model_name in results_df['Description'] \
                           if 'On training' in model_name]
    
    return [elem.replace(' - On training','') for elem in trained_model_names]

def get_already_tested_model_names()->list[str]:
    """Get the names of the models already tested on the held-out

    Returns:
        list[str]: name list of the models already tested on the held-out.
    """
    results_df = get_heldout_results()
    tested_model_names = [model_name \
                          for model_name in results_df['Description'] \
                          if 'On heldout' in model_name]
    
    return [elem.replace(' - On heldout','') for elem in tested_model_names]

def get_pending_to_train_list_of_experiments() -> list[Experiment]:
    """Returns a list with Experiments (dataclass) with all the experiments that are pending to be trained
       (no performance metrics are found in disk).

       It uses: get_already_trained_model_names (to get names), 
                get_full_list_of_experiments    (to get Experiments) and
                heldout_results_available       (do check if there is any results in disk)

    Returns:
        list[Experiment]: List of the experiments pending for training.
    """    
    already_computed = set(get_already_trained_model_names()) if heldout_results_available() else []

    all_experiments = get_full_list_of_experiments()
    pending_experiments = [expleriment for expleriment in all_experiments 
                                if expleriment.model_name not in already_computed]
    return pending_experiments

def get_list_of_trained_experiments() -> list[Experiment]:
    """Returns a list with Experiments (dataclass) with all the experiments that are already trained
       (performance metrics found in disk).

       It uses: get_already_trained_model_names (to get names), 
                get_full_list_of_experiments    (to get Experiments) and
                heldout_results_available       (do check if there is any results in disk)
    """    
    already_computed = set(get_already_trained_model_names()) if heldout_results_available() else []
    all_experiments = get_full_list_of_experiments()
    completed_experiments = [experiment for experiment in all_experiments 
                                if experiment.model_name in already_computed]
    return completed_experiments


def get_pending_to_validate_list_of_experiments() -> list[Experiment]:
    """Returns a list with Experiments (dataclass) with all the experiments that are pending to be 
       evaluated on the heldout.
       (no performance metrics are found in disk for the heldouts).

       It uses: get_already_tested_model_names  (to get names of already tested), 
                get_full_list_of_experiments    (to get all Experiments) and
                heldout_results_available       (do check if there is any results in disk)
    """   
    already_tested = set(get_already_tested_model_names()) if heldout_results_available() else []

    all_experiments = get_full_list_of_experiments()
    pending_experiments = [experiment for experiment in all_experiments 
                                if experiment.model_name not in already_tested]
    return pending_experiments

def create_and_store_models():
    """
    This procedure 
    1. obtains the pending models to be trained (get_pending_to_train_list_of_experiments) and
    one by one, 
        I.   it obtains the data (development and heldout), 
        II.  trains the model on the development set,
        III. computes performance metrics 
        IV.  stores disk and other info (columns)
        V.   stores performance metrics 

    """    
    io.debug('---------- '*10)
    io.debug(' ~ ~ ~ ~ create_and_store_models ~ ~ ~ ~ ')
    pending_experiments = get_pending_to_train_list_of_experiments()
    config = configuration.get_config()

    results_df = get_heldout_results() if heldout_results_available() else None
    if not results_df is None:
        io.debug(f'Previous experiments found (results.shape={results_df.shape})')

    io.debug(f'Pending experiments found (len(pending_experiments)={len(pending_experiments)})')

    io.debug('PENDING EXPERIMENTS:')
    for experiment in pending_experiments:
        io.debug(str(experiment))
        
    for experiment in pending_experiments:
        model_name = experiment.model_name
        experiment_configuration = experiment.experiment_configuration_name
        model_configuration = experiment.model_configuration_name
        io.debug(f'model_name=               {model_name}')
        io.debug(f'experiment_configuration= {experiment_configuration}')
        io.debug(f'model_configuration=      {model_configuration}')

        # DATA
        params = configuration.configuration_from_configuration_name(experiment_configuration)

        X_development, y_development,  X_heldout, y_heldout, columns = health_data.Admission.get_development_and_held_out_matrices(params)
        X=X_development
        y=y_development
        # X, y, columns = health_data.Admission.get_both_train_test_matrices(params)

        io.debug('Training Data:')
        io.debug(f'X.shape=      {str(X.shape):20} ({type(X)})')
        io.debug(f'y.shape=      {str(y.shape):20} ({type(y)})')
        io.debug(f'columns.shape={str(columns.shape):20} ({type(columns)})')

        # MODEL
        model = configuration.model_from_configuration_name(model_configuration)
        io.debug(f'Creating and Training model={str(model)} ...')

        model.fit(X,y)
        new_results_df = metrics.get_metric_evaluations(model,
                                                    X,
                                                    y,
                                                    model_configuration,
                                                    experiment_configuration,
                                                    description=f'{model_name} - On training',
                                                    )
        io.debug('Results computed on training.')
        io.debug(f'new_results_df.shape={new_results_df.shape}')
        io.info(f'new_results_df.iloc[0,:]={new_results_df.iloc[0,:]}')

        # Store metrics, store models.
        if results_df is None:
            results_df = new_results_df
        else:
            results_df = pd.concat([results_df, new_results_df])

        
        model_path = f"{config['checkpoint_folder']}/{model_name}.joblib"
        columns_path = f"{config['checkpoint_folder']}/{model_name}_columns.joblib"
        joblib.dump(model, model_path)
        joblib.dump(columns, columns_path)
        io.debug('Saving models in: {model_path}')
        io.debug('Saving columns in: {columns_path}')
        

        io.debug(f'Saving new results (results_df.shape={results_df.shape})')
        results_df.to_csv(config['ml_performance_on_held_out'],
                        index=None,
                        )
        
def validate_and_run_experiments_on_heldout():
    """
    This procedure 
    1. obtains the pending models to be evaluated (get_pending_to_validate_list_of_experiments) and
    one by one, 
        I.   it obtains the data (development and heldout), 
        II.  retrieves models from disk
        III. computes performance metrics on development (to check model load properly)
        IV.  compares TP,TN,FN,FP to check model and data load properly (compares with results in disk)
        V.   Computes performance on held-out, and saves new results.

    """    
    io.debug('---------- '*10)
    io.debug('RUNNING ```validate_and_run_experiments_on_heldout``` method')

    assert heldout_results_available(), 'No results found, nothing to validate'

    config = configuration.get_config()

    completed_experiments = get_list_of_trained_experiments()
    io.debug(f'Obtaining description of trained experiments (len: {len(completed_experiments)}) ({str([e.model_name for e in completed_experiments])}).')

    pending_to_validate = get_pending_to_validate_list_of_experiments()
    io.debug(f'Obtaining description of pending to validate experiments (len: {len(pending_to_validate)}) ({str([e.model_name for e in pending_to_validate])}).')

    results_df = get_heldout_results()
    io.debug(f'Results read from disk ({results_df.shape[0]} row, {results_df.shape[1]} columns).')

    for experiment in pending_to_validate:
        io.debug(f'Running experiment={experiment}')
        model_name = experiment.model_name
        experiment_configuration = experiment.experiment_configuration_name
        model_configuration = experiment.model_configuration_name

        # DATA
        params = configuration.configuration_from_configuration_name(experiment_configuration)

        X_development, y_development,  X_heldout, y_heldout, columns = health_data.Admission.get_development_and_held_out_matrices(params)

        io.debug(f'X_development.shape={str(X_development.shape):20} (type={type(X_development)})')
        io.debug(f'y_development.shape={str(y_development.shape):20} (type={type(y_development)})\n')

        io.debug(f'X_heldout.shape={str(X_heldout.shape):20} (type={type(X_heldout)})')
        io.debug(f'y_heldout.shape={str(y_heldout.shape):20} (type={type(y_heldout)})\n')

        io.debug(f'columns.shape={str(columns.shape):20} (type={type(columns)})\n')



        io.debug('Loading models from disk ...')
        trained_model = joblib.load(f"{config['checkpoint_folder']}/{model_name}.joblib")
        model_columns = joblib.load(f"{config['checkpoint_folder']}/{model_name}_columns.joblib")
        
        io.debug(f'trained_model={str(trained_model)}')
        io.debug(f'model_columns.shape={str(model_columns.shape):20} (type={type(model_columns)})')

        io.debug('computing performance on development set ...')
        new_results_df = metrics.get_metric_evaluations(trained_model,
                                                    X_development,
                                                    y_development,
                                                    model_configuration, 
                                                    experiment_configuration, 
                                                    description=f'{model_name} - On training',
                                                    )
        
        io.debug(f'new_results_df.shape={new_results_df.shape} (type={type(new_results_df)})')

        io.debug('Checking both columns (stored and recently computed) are the same (same features) ...')
        # for var1,var2 in zip(columns, model_columns):
        #     io.debug(f'{var1:20} --- {var2:20}')

        assert all(columns==model_columns)
        io.debug('[  OK   ] Correct columns.')

        io.debug('Checking the experiment and model configuration names matches the one in the retrieved df')

        io.debug(f"Obtaining results for model name={model_name} ({model_name} - On training)")
        stored_results=results_df[results_df['Description']==f'{model_name} - On training'].iloc[0,:]
        io.debug(f'stored_results.shape={stored_results.shape} ({type(stored_results)})')

        assert results_df[results_df['Description']==f'{model_name} - On training'].shape[0]==1, 'More than one stored results for the same model.'
        assert stored_results['Experiment config']==experiment_configuration, 'Validating wrong model'
        assert stored_results['Model config']==model_configuration, 'Validating wrong model'

        io.debug('Comparing performance metrics')
        io.debug(f'stored_results.values={stored_results.values}')
        io.debug(f'new_results_df.iloc[0,:].values={new_results_df.iloc[0,:].values}')
        

        io.debug(f'stored_results.shape={stored_results.shape} ({type(stored_results)})')
        io.debug(f'new_results_df.shape={new_results_df.shape} ({type(new_results_df)})')

        assert stored_results['TP']==new_results_df.iloc[0,:]['TP']
        assert stored_results['TN']==new_results_df.iloc[0,:]['TN']
        assert stored_results['FP']==new_results_df.iloc[0,:]['FP']
        assert stored_results['FN']==new_results_df.iloc[0,:]['FN']

        # assert all(stored_results.values == new_results_df.iloc[0,:].values)
        io.debug('[  OK   ] Correct values.')


        io.ok(f'Model model_name={model_name} validated!')

        io.debug('Running models on held-out and computing metrics')

        heldout_results = metrics.get_metric_evaluations(trained_model,
                                                         X_heldout,
                                                         y_heldout,
                                                         model_configuration,
                                                         experiment_configuration,
                                                         description=f'{model_name} - On heldout',
                                                         )
        
        append_new_results_to_file(heldout_results)
        

        io.debug(heldout_results.iloc[0,:])

def main():
    create_and_store_models()
    validate_and_run_experiments_on_heldout()

if __name__ == '__main__':
    main()