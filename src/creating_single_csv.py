"""
Script: creating_cz_and_noncz_files.py

Description:
------------
This script merges two patient admission data files into a single unified dataset, 
generating derived columns and storing the combined result in a CSV file. The input 
files represent two distinct datasets: one for Central Zone (CZ) and another for 
Non-Central Zone (Non-CZ) patients. The paths for these input and output files are
specified in the 'config/paths.yaml' file.

Usage:
------
python creating_cz_and_noncz_files.py --save-to-disk=[True|False]

Arguments:
----------
--save-to-disk : str [Required]
    - 'True'  : Saves the merged dataset to disk. Be aware that this may 
                overwrite existing files.
    - 'False' : No data is saved to disk; this is a dry run for testing.

Functionality:
--------------
1. **Reads Input Files**: 
   - CZ Dataset: `full_cz_database.csv`
   - Non-CZ Dataset: `full_noncz_database.csv`
   - Input paths are defined in the 'config/paths.yaml' file.

2. **Processes the Data**:
   - Generates derived columns such as:
     - **is ALC Patient**: Identifies patients with Alternate Level of Care (ALC) days.
     - **Total Days in Hospital**: Calculates the total duration of hospital stay.
     - **Discharge Date (year-month)**: Extracts the year and month from the discharge date.
   - Adds a **CZ Status** column to differentiate between CZ and Non-CZ data.

3. **Merges the Data**:
   - Combines the processed CZ and Non-CZ datasets into a single dataframe.

4. **Output**:
   - If `--save-to-disk=True`, the merged dataset is saved to the specified output path in 'config/paths.yaml'. 
   - Example output: `/Users/marianomaisonnave/Documents/CBU Postdoc/Grant Data/Merged/2015_2022/full_database.csv`

Warnings:
---------
- When `--save-to-disk=True`, the output CSV will be written to disk, potentially overwriting 
  existing files. This action is irreversible.
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

