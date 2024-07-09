import pandas as pd
import numpy as np
import sys
sys.path.append('..')

from utilities import configuration
from utilities import health_data
from utilities import io

if __name__ == '__main__':
    CONFIGURATION_NAME = 'configuration_87'
    config = configuration.get_config()

    data = {
        'Population': ['Readmitted', 'Not Readmitted'],
    }

    params = configuration.configuration_from_configuration_name(CONFIGURATION_NAME)

    X, y, columns = health_data.Admission.get_both_train_test_matrices(params)

    io.debug(f'type(X)={type(X)}')
    io.debug(f'type(y)={type(y)}')


    io.debug(f'X.shape={X.shape}')
    io.debug(f'y.shape={y.shape}')

    df = pd.DataFrame(X.toarray(), columns=columns)

    io.debug(f'df.shape={df.shape}')

    X_R = df.iloc[y==1,:]
    X_NR = df.iloc[y==0,:]

    io.debug(f'type(X_R)={type(X_R)}')
    io.debug(f'X_R.shape={X_R.shape}')

    io.debug(f'type(X_NR)={type(X_NR)}')
    io.debug(f'X_NR.shape={X_NR.shape}')


    data['Total Count'] = [X_R.shape[0], 
                           X_NR.shape[0]]

    data['Male'] = [np.sum(X_R['male']==1), 
                    np.sum(X_NR['male']==1)]

    data['Female'] = [np.sum(X_R['female']==1), 
                      np.sum(X_NR['female']==1)]

    data['ALC'] = [np.sum(X_R['is alc']==1), 
                   np.sum(X_NR['is alc']==1)]

    data['Rural'] = [np.sum(X_R['is central zone']==0), 
                     np.sum(X_NR['is central zone']==0)]

    data['Urgent Admission'] = [np.sum(X_R['urgent admission']==1), 
                                np.sum(X_NR['urgent admission']==1)]
    
    data['Elective Admission'] = [np.sum(X_R['elective admission']==1), 
                                  np.sum(X_NR['elective admission']==1)]

    data['Newborn Admission'] = [np.sum((X_R['elective admission']==0) & (X_R['urgent admission']==0)), 
                                 np.sum((X_NR['elective admission']==0) & (X_NR['urgent admission']==0))]

    data['Clinic Entry'] = [np.sum(X_R['Clinic Entry']==1), 
                            np.sum(X_NR['Clinic Entry']==1)]


    data['Direct Entry'] = [np.sum(X_R['Direct Entry']==1), 
                            np.sum(X_NR['Direct Entry']==1)]
    

    data['Emergency Entry'] = [np.sum(X_R['Emergency Entry']==1), 
                            np.sum(X_NR['Emergency Entry']==1)]
    

    data['New Acute Patient'] = [np.sum(X_R['New Acute Patient']==1), 
                                 np.sum(X_NR['New Acute Patient']==1)]
    
    data['Day Surgery Entry'] = [np.sum(X_R['Day Surgery Entry']==1), 
                                 np.sum(X_NR['Day Surgery Entry']==1)]
    
    data['Planned Readmit'] = [np.sum(X_R['Panned Readmit']==1), 
                                 np.sum(X_NR['Panned Readmit']==1)]

    data['Unplanned Readmit'] = [np.sum(X_R['Unplanned Readmit']==1), 
                                 np.sum(X_NR['Unplanned Readmit']==1)]

    data['Age 0-18'] = [np.sum(X_R['age'] <= 18), 
                        np.sum(X_NR['age'] <= 18)]
    

    data['Age 19-35'] = [np.sum((1 <= X_R['age']) & (X_R['age'] <= 35)), 
                        np.sum((19 <= X_NR['age']) & (X_NR['age'] <= 35))]

    
    data['Age 36-50'] = [np.sum((36 <= X_R['age']) & (X_R['age'] <= 50)), 
                         np.sum((36 <= X_NR['age']) & (X_NR['age'] <= 50))]
    
    data['Age 51-65'] = [np.sum((51 <= X_R['age']) & (X_R['age'] <= 65)), 
                         np.sum((51 <= X_NR['age']) & (X_NR['age'] <= 65))]
    
    data['Age 66+'] = [np.sum(66 <= X_R['age']), 
                       np.sum(66 <= X_NR['age'])]

    io.debug(data)
    results = pd.DataFrame(data)
    results.T.to_csv(config['dataset_statistics_table'], index=False)
