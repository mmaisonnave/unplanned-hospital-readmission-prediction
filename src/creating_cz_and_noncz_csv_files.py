"""
>>
USAGE: python creating_cz_and_noncz_files.py --save-to-disk=[True,False]
>>

if --save-to-disk==False no changes are done (it can be considered as a test run).
if --save-to-disk==True files are written to disk, can be overwriting files, cannot be undone.

This modules carries out three tasks:
1. Processing 32 Central Zone (CZ) files and saves to disk as an unified file.
2. Processing 32 Non Central Zonce (Non CZ) files and saves to disk as an unified file.
3. Takes the newly processed unified Non CZ file. It reads it, fix date format, and saves it again. 

In the input files, multiple rows represent the same entry. This script combines them into a single entry:

EXAMPLE ORIGINAL DATA:
patient id, diagnosis,
123456,diagnosis1_patient_1
,diagnosis2_patient_1
,diagnosis3_patient_1
123457,diagnosis_1_patient_2
,diagnosis_2_patient_2
,diagnosis_3_patient_2
,diagnosis_4_patient_2
...

EXAMPLE RESULT of first and second task:
123456,[diagnosis1_patient_1,diagnosis2_patient_1,diagnosis3_patient_1]
123457,[diagnosis_1_patient_2,diagnosis_2_patient_2,diagnosis_3_patient_2,diagnosis_4_patient_2]
...

INPUT:
------
~For task no. 1~
 --------------
The data is taken from individual CSV files describing the data quarter by quarter (from 2015Q1 until 2022Q4).
cz_files:
 - 2015/ALC Machine 2015Q1 - coded HCN.csv
 - 2015/ALC Machine 2015Q2 - coded HCN.csv
 ...
 -
 - 2022/ALC Machine 2022Q3 - coded HCN.csv
 - 2022/ALC Machine 2022Q4 - coded HCN.csv

 ~For task no. 2~
 --------------
The data is taken from individual CSV files describing the data quarter by quarter (from 2015Q1 until 2022Q4).
noncz_files:
 - 2015/2015 Non CZ - coded HCN/noncz 2015Q1.csv
 - 2015/2015 Non CZ - coded HCN/noncz 2015Q2.csv
 ...
 - 2022/2022 Non CZ - coded HCN/noncz 2022Q3.csv
 - 2022/2022 Non CZ - coded HCN/noncz 2022Q4.csv

Total: 32 CZ files (task no. 1), and 32 Non CZ files (task no. 2).

 ~For task no. 3~
 --------------
The input of the 3rd task is the output of task no. 2.

The path to all input files is retrieved from the config/path.yaml file 
(no argument required to receive the path to the inputs)

OUTPUT:
-------
The output is to files (taken from the config/path.yaml).
For task no. 1:
--------------
unified_merged_file_cz: 
 - /Users/marianomaisonnave/Documents/CBU Postdoc/Grant Data/Merged/2015_2022/full_cz_database.csv

For task no. 2:
--------------
unified_merged_file_noncz: 
 - /Users/marianomaisonnave/Documents/CBU Postdoc/Grant Data/Merged/2015_2022/full_noncz_database.csv

Task no. 3:
----------
Reads full_noncz_database.csv file, corrects dates, and saves it again. 
The script transform the date into the format: year-month-day. Examples: 2014-12-22, 2015-1-1, ...

"""

import os
import pandas as pd
import numpy as np
import re
import argparse

import sys
sys.path.append('..')
from utilities import io
from utilities import configuration


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-to-disk', 
                        dest='save_to_disk', 
                        required=True, 
                        help='wether to save to disk or not',
                        type=str,
                        choices=['True', 'False']
                        )
    args = parser.parse_args()

    params = {'save_to_disk' : args.save_to_disk=='True'}

    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
    # CZ FILES  (TASK no. 1)
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------    
    config = configuration.get_config()
    io.debug('\n')
    io.debug('Starting with CZ FILES')

    if params['save_to_disk']:
        io.warning('Results will be stored in risk, changes cannot be undone.')
    else:
        io.debug('Reults will **NOT** be stored in disk, this is a test run.')

    filepaths = [os.path.join(config['data_path'], file) for file in config['cz_files']]
    assert all([os.path.isfile(filepath) for filepath in filepaths])
    io.debug(f'Found {len(filepaths):,} CZ files to transform, reading and converting to DataFrames ...')

    all_dfs = []
    for filepath in (filepaths):
        df = pd.read_csv(filepath, ) #low_memory=False)
        all_dfs.append(df)


    column_count = all_dfs[0].shape[1]
    assert all([df.shape[1]==column_count for df in all_dfs])
    io.debug(f'Number of columns found in all files={column_count}')
    io.debug(f"Shapes of all DataFrames: {'; '.join([str(df.shape) for df in all_dfs])}")


    for column_ix in range(column_count):
        io.debug(f'{column_ix:2}: {set([df.columns[column_ix] for df in all_dfs])}')
        assert len(set([df.columns[column_ix] for df in all_dfs]))==1

    df = pd.concat(all_dfs)

    io.debug(f'Concating all DataFrames, resulting shape= {df.shape[0]:,} x {df.shape[1]}')


    data = {column:[] for column in df.columns}


    row_ix=0
    while row_ix<df.shape[0]:
        end = row_ix + 1
        while end<df.shape[0] and np.isnan(df['Institution Number:'].iloc[end]):
            end+=1

        auxdf = df.iloc[row_ix:end, :].copy()
        
        for column in auxdf:
            content = [elem for ix,elem in enumerate(auxdf[column]) if not auxdf[column].isna().iloc[ix]]

            # If list only has one element, then store the element, else store the list.
            content = content[0] if len(content)==1 else content
            data[column].append(content)

        row_ix=end   
        
    newdf=pd.DataFrame(data)

    # For large numbers >1,000 ; we remove the ',' from the string:
    newdf['ALCDays'] = [re.sub(',','',alcdays) if isinstance(alcdays,str) else alcdays for alcdays in newdf['ALCDays']]
    newdf['Acute Days'] = [re.sub(',','',alcdays) if isinstance(alcdays,str) else alcdays for alcdays in newdf['Acute Days']]

    # Mapping to change some columns to new types (from str to int).
    io.debug('Changing Institution Number, Patient Age, ALCDays, and Acute Days to int.')

    newdf = newdf.astype({'Institution Number:': 'int',  
                        'Patient Age:': 'int', 
                        'ALCDays':'int',
                        'Acute Days': 'int',
                        })

    # Making Case Weight a float
    io.debug('Transform Case Weight into a float type (can contain NaN values).')
    new_case_weight=[]
    for elem in newdf['Case Weight']:
        if elem==[]:
            newelem=np.nan
        elif isinstance(elem, str):
            assert '.' in elem
            newelem =  float(elem.replace(',',''))
        else:
            assert isinstance(elem, float)
            newelem = elem
        new_case_weight.append(newelem)
    newdf['Case Weight'] = np.array(new_case_weight)

    io.debug('Displaying first item:')
    for column,value in zip(newdf.columns, newdf.iloc[0,:]):
        io.debug(f'{column:20} ({str(type(value)):23}):   {value}')

    if params['save_to_disk']:
        io.debug(f"Saving to disk. Destination: {config['unified_merged_file_cz']}")
        newdf.to_csv(config['unified_merged_file_cz'], index=None)


    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
    # Non-CZ FILES (task no. 2)
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
    io.debug('\n')
    io.debug('Working on Non-CZ filesxs')

    filepaths = [os.path.join(config['data_path'], file) for file in config['noncz_files']]
    assert all([os.path.isfile(filepath) for filepath in filepaths])
    io.debug(f'Found {len(filepaths):,} Non-CZ files to transform, reading and converting to DataFrames ...')

    all_dfs = []
    for filepath in (filepaths):
        df = pd.read_csv(filepath, )#low_memory=False)
        all_dfs.append(df)


    column_count = all_dfs[0].shape[1]
    assert all([df.shape[1]==column_count for df in all_dfs])
    io.debug(f'Number of columns found in all files={column_count}')
    io.debug(f"Shapes of all DataFrames: {'; '.join([str(df.shape) for df in all_dfs])}")


    for column_ix in range(column_count):
        io.debug(f'{column_ix:2}: {set([df.columns[column_ix] for df in all_dfs])}')
        assert len(set([df.columns[column_ix] for df in all_dfs]))==1, set([df.columns[column_ix] for df in all_dfs])

    df = pd.concat(all_dfs)

    io.debug(f'Concating all DataFrames, resulting shape= {df.shape[0]:,} x {df.shape[1]}')


    data = {}
    for column in df.columns:
        data[column]=[]
    row_ix=0
    while row_ix<df.shape[0]:
        end = row_ix + 1
        while end<df.shape[0] and np.isnan(df['Institution Number:'].iloc[end]) :
            end+=1

        auxdf = df.iloc[row_ix:end, :].copy()
        
        for column in auxdf:
            content = [elem for ix,elem in enumerate(auxdf[column]) if not auxdf[column].isna().iloc[ix]]
            content = content[0] if len(content)==1 else content
            data[column].append(content)

        row_ix=end   
        
    newdf=pd.DataFrame(data)


    # Removing ',' from large numbers (>1,000)
    newdf['ALCDays'] = [re.sub(',','',alcdays) if isinstance(alcdays,str) else alcdays for alcdays in newdf['ALCDays']]
    newdf['Acute Days'] = [re.sub(',','',alcdays) if isinstance(alcdays,str) else alcdays for alcdays in newdf['Acute Days']]
    
    io.debug('Changing Institution Number, Patient Age, ALCDays, and Acute Days to int.')

    newdf = newdf.astype({'Institution Number:': 'int',  
                        'Patient Age:': 'int', 
                        'ALCDays':'int',
                        'Acute Days': 'int',
                        })
    
    io.debug('Displaying first item:')
    for column,value in zip(newdf.columns, newdf.iloc[0,:]):
        io.debug(f'{column:20} ({str(type(value)):23}):   {value}')

    if params['save_to_disk']:
        io.debug(f"Saving to disk. Destination: {config['unified_merged_file_noncz']}")
        newdf.to_csv(config['unified_merged_file_noncz'], index=None)

    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    # CORRECTING DATES IN NON-CZ FILES (task no. 3)
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    df = pd.read_csv('/Users/marianomaisonnave/Documents/CBU Postdoc/Grant Data/Merged/2015_2022/full_noncz_database_TEMP.csv')

    admit_dates=[]
    for date in df['Admit Date:']:
        assert len(date.split('/'))==3 or date=='**', date
        if len(date.split('/'))==3:
            month, day, year = date.split('/')    
            if 1<=int(year) and int(year)<=23:
                newdate=f'20{year}-{month}-{day}'
            else:
                newdate=f'19{year}-{month}-{day}'
        else:
            assert date=='**'
            newdate=date
        admit_dates.append(newdate)
    df['Admit Date:']=admit_dates

    disch_dates=[]
    for date in df['Disch Date:']:
        assert len(date.split('/'))==3 or date=='**', date
        if len(date.split('/'))==3:
            month, day, year = date.split('/')    
            assert 15<=int(year) and int(year)<=22
            newdate=f'20{year}-{month}-{day}'

        else:
            assert date=='**'
            newdate=date
        disch_dates.append(newdate)
    df['Disch Date:']=disch_dates
    if params['save_to_disk']:
        io.debug(f"Saving to disk. Destination: {config['unified_merged_file_cz']}")
        df.to_csv(config['unified_merged_file_noncz'], index=None)

