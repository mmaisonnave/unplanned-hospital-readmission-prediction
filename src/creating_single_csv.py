"""
>>
USAGE: python creating_cz_and_noncz_files.py --save-to-disk=[True,False]
>>

if --save-to-disk==False no changes are done (it can be considered as a test run).
if --save-to-disk==True files are written to disk, can be overwriting files, cannot be undone.


This scripts take two files describing the database:
1. unified_merged_file_cz: full_cz_database.csv
2. unified_merged_file_noncz: full_noncz_database.csv

and combines the two files into one:
3. unified_merged_file: 
    - /Users/marianomaisonnave/Documents/CBU Postdoc/Grant Data/Merged/2015_2022/full_database.csv

A couple of derived columns are built, but so far we haven't use them.

All path to the two input files and the output file are take from the config/paths.yaml file. 
No need to any input except the param "save-to-disk". If false, the run is a test run 
with no changes.

"""
import pandas as pd
from datetime import date
import datetime
import sys
sys.path.append('..')
from utilities import io
from utilities import configuration
import argparse

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

 
    config = configuration.get_config()
    io.debug('\n')
    io.debug('Unifying two files (CZ and Non-CZ) into one. Script: creating_single_csv.py')

    if params['save_to_disk']:
        io.warning('Results will be stored in risk, changes cannot be undone.')
    else:
        io.debug('Reults will **NOT** be stored in disk, this is a test run.')

    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
    # READING CZ FILES
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------   
    cz_df = pd.read_csv(config['unified_merged_file_cz'])
    io.debug(f'CZ files read. Shape={cz_df.shape}')
    io.debug('Adding additional columns (derived from others): is ALC, Total days in hospital, etc.')

    cz_df['is ALC Patient'] = cz_df['ALCDays'] > 0
    cz_df['Admit Date:'] = [date.fromisoformat(date_.replace('/','-')) for date_ in cz_df['Admit Date:']]
    cz_df['Disch Date:'] = [date.fromisoformat(date_.replace('/','-')) for date_ in cz_df['Disch Date:']]
    cz_df['Disch Date (year-month):'] = [str(date_)[:7] for date_ in cz_df['Disch Date:']]
    cz_df['Total Days in Hospital'] = [1 if (discharge-admit).days==0 else (discharge-admit).days  
                                    for admit,discharge in zip(cz_df['Admit Date:'], cz_df['Disch Date:'])]
    cz_df[['Admit Date:', 'Disch Date:', 'Patient Age:', 'ALCDays','Disch Date (year-month):']]

    cz_df['CZ status']=['cz']*cz_df.shape[0]

    io.debug(f"All entries for dataset 'CZ' - found:    {cz_df.shape[0]:9,} entries")
    io.debug(f'Final shape of CZ files dataframe = {cz_df.shape}')

    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
    # READING Non-CZ FILES
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------   
    noncz_df = pd.read_csv(config['unified_merged_file_noncz'])
    io.debug(f'Non-CZ files read. Shape={noncz_df.shape}')
    io.debug('Adding additional columns (derived from others): is ALC, Total days in hospital, etc.')


    noncz_df['is ALC Patient'] = noncz_df['ALCDays'] > 0
    noncz_df['Admit Date:'] = [None if date_=='**' else datetime.datetime.strptime(date_, "%Y-%m-%d") for date_ in noncz_df['Admit Date:']]
    noncz_df['Disch Date:'] = [datetime.datetime.strptime(date_, "%Y-%m-%d")  for date_ in noncz_df['Disch Date:']]
    noncz_df['Disch Date (year-month):'] = [str(date_)[:7] for date_ in noncz_df['Disch Date:']]
    noncz_df['Total Days in Hospital'] = [1 if (discharge-admit).days==0 else (discharge-admit).days  
                                    for admit,discharge in zip(noncz_df['Admit Date:'], noncz_df['Disch Date:'])]
    noncz_df[['Admit Date:', 'Disch Date:', 'Patient Age:', 'ALCDays','Disch Date (year-month):']]

    io.debug(f"All entries for dataset 'Non-CZ' - found:    {noncz_df.shape[0]:9,} entries")
    io.debug(f'Final shape of Non-CZ files dataframe = {noncz_df.shape}')

    noncz_df['CZ status']=['Non-cz']*noncz_df.shape[0]

    noncz_df = noncz_df.rename(columns={'Inst Type 2018':'Institution Type', 'Nursing Unit:': 'Discharge Nurse Unit'})

    full_df = pd.concat([cz_df, noncz_df])
    if params['save_to_disk']:
        full_df.to_csv(config['unified_merged_file'], index=False)
        io.debug(f"Combining Non-cz and cz data into a single file: {config['unified_merged_file']}")

