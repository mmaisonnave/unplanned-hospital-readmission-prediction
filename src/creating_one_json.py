"""
Script: creating_cz_and_noncz_files.py

Description:
------------
This script processes patient admission data from an input CSV 
file and converts it into a JSON format. The input and output 
file paths are specified in the 'config/paths.yaml' 
configuration file. 

The script offers two modes of operation:
1. **Test Run**: No changes are saved to disk (useful for validation 
   purposes).
2. **Save to Disk**: The JSON output is saved to disk, potentially 
   overwriting existing files. This action is irreversible.

Usage:
------
python creating_cz_and_noncz_files.py --save-to-disk=[True|False]

Arguments:
----------
--save-to-disk : str [Required]
    - 'True'  : Saves the generated JSON data to disk.
    - 'False' : No data is saved to disk; this is a dry run for testing.

Input:
------
- unified_merged_file : Path to the input CSV file containing the full database.
    (Configured in config/paths.yaml)

Output:
-------
- json_file : Path to the output JSON file.
    (Configured in config/paths.yaml)
    - Example: /full_database.json

Details:
--------
The script reads the input CSV file and transforms each row into a dictionary, 
which is then stored in a JSON file. The following fields are extracted 
and processed:

1. **Coded HCN** (Health Card Number)
2. **Institution Number**
3. **Admit Date** and **Discharge Date**
4. **Readmission Code**
5. **Patient Age**
6. **Gender**
7. **MRDx** (Most Responsible Diagnosis)
8. **Postal Code**
9. **Diagnosis Code** and **Diagnosis Long Text**
10. **Diagnosis Type**
11. **Intervention Code** and **Px Long Text**
12. **Admit Category**
13. **Entry Code**
14. **Transfusion Given**
15. **Main Pt Service** (Main Patient Service)
16. **CMG** (Case Mix Group)
17. **Comorbidity Level**
18. **Case Weight**
19. **ALC Days** (Alternate Level of Care Days)
20. **Acute Days**
21. **Institution To** and **Institution From**
22. **Institution Type**
23. **Discharge Nurse Unit**
24. **CZ Status** (Central Zone Status)

Warnings:
---------
- When `--save-to-disk=True`, the output JSON will be written to disk, which may 
  overwrite existing files. This action cannot be undone.
"""
import pandas as pd
import datetime
import numpy as np
import re
import json
import ast
import os
import sys
import argparse
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

 
    config = configuration.get_config()

    io.debug('\n')
    io.debug('Reading unified CSV file to transfor to JSON ')

    if params['save_to_disk']:
        io.warning('Results will be stored in risk, changes cannot be undone.')
    else:
        io.debug('Reults will **NOT** be stored in disk, this is a test run.')


    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    # READING UNIFIED FILE (CSV)
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    df = pd.read_csv(config['unified_merged_file'])
    io.debug(f"Read from disk: {config['unified_merged_file']}")
    io.debug(f'Shape of Data Frame = {df.shape}')

    data={}
    io.debug('Transforming each row of DataFrame into a dict to store as JSON ...')
    for ix in range(df.shape[0]):
        row = df.iloc[ix,:]
        # --------- #
        # Coded HCN #
        # --------- #

        coded_hcn = row['Coded HCN']
        if  '[' in coded_hcn and ']' in coded_hcn: # Coded HCN is a list.
            if coded_hcn=='[]': # Empty list
                coded_hcn=None
            else: # list with at least one element.
                coded_hcn = re.findall('([0-9\.][0-9\.]*)',coded_hcn)
                assert str(coded_hcn).replace("'",'')==row['Coded HCN'], str(ix)+':'+str(coded_hcn)
                coded_hcn = list(filter(lambda elem: elem!=170805, map(int, map(float,coded_hcn))))
                assert len(coded_hcn)<=1
                coded_hcn = None if len(coded_hcn)==0 else coded_hcn[0]
        else: # Coded HCN is a single number
            coded_hcn = int(float(coded_hcn))
            assert coded_hcn==float(row['Coded HCN'])
        data[ix]={'HCN code':coded_hcn}

        # ------------------ #
        # Institution Number #
        # ------------------ #
        institution_number = int(row['Institution Number:'])
        data[ix]['Institution Number'] = institution_number

        # ------------------------ #
        # Admit and Discharge Date #
        # ------------------------ #
        admit_date = datetime.datetime.fromisoformat(row['Admit Date:'][:10]) if not isinstance(row['Admit Date:'], float) else None
        discharge_date = datetime.datetime.fromisoformat(row['Disch Date:'][:10])
        data[ix]['Admit Date'] = admit_date
        data[ix]['Discharge Date'] = discharge_date

        readmission_code = row['Readmission Code:'] if row['Readmission Code:']!='**' else None
        assert readmission_code is None or (not '[' in readmission_code and not ']' in readmission_code and not '**' in readmission_code, row['Readmission Code:'])
        data[ix]['Readmission Code']=readmission_code

        # ----------- #
        # Patient Age #
        # ----------- #
        patient_age = int(row['Patient Age:'])
        data[ix]['Patient Age'] = patient_age

        # ------ #
        # Gender #
        # ------ #
        gender = row['Gender:'] if row['Gender:']!='**' else None
        data[ix]['Gender'] = gender


        # ---- #
        # MRDx #
        # ---- #
        mrdx = row['MRDx']
        assert not mrdx.startswith('[') and not '**' in mrdx, mrdx
        data[ix]['MRDx'] = mrdx

        postal_code = row['Postal Code:'] if row['Postal Code:']!='**' else None
        assert postal_code is None or len(postal_code)==6,postal_code
        assert postal_code is None or postal_code.isalnum(), postal_code
        data[ix]['Postal Code'] = postal_code

        # --------- #
        # Diagnosis #
        # --------- #
        diagnosis = row['Diagnosis:']
        if '[' in diagnosis:
            if diagnosis=='[]':
                diagnosis=[]
            else:
                diagnosis = ast.literal_eval(diagnosis)
                assert str(diagnosis)==row['Diagnosis:']
        elif diagnosis=='**':
            diagnosis = []
        else:
            diagnosis = [diagnosis]
        assert diagnosis!='**',diagnosis
        data[ix]['Diagnosis Code']= diagnosis

        # ------------------- #
        # Diagnosis Long Text #
        # ------------------- #
        diagnosis_long_text = row['Diagnosis Long Text']
        if '**'== diagnosis_long_text or '[]'==diagnosis_long_text:
            diagnosis_long_text=[]
        elif diagnosis_long_text.startswith('[')  and diagnosis_long_text.endswith(']'):
            diagnosis_long_text = ast.literal_eval(diagnosis_long_text)
            assert str(diagnosis_long_text)==row['Diagnosis Long Text'],str(ix)+':'+str(diagnosis_long_text)
        else:
            diagnosis_long_text = [diagnosis_long_text]
        data[ix]['Diagnosis Long Text']= diagnosis_long_text

        # -------------- #
        # Diagnosis type #
        # -------------- #
        diagnosis_type = row['Diagnosis Type']
        if '**'== diagnosis_type or '[]'==diagnosis_type:
            diagnosis_type=[]
        elif '[' in diagnosis_type and ']' in diagnosis_type:
            diagnosis_type = ast.literal_eval(diagnosis_type)
            assert str(diagnosis_type)==row['Diagnosis Type']
        else:
            diagnosis_type = [diagnosis_type]
        data[ix]['Diagnosis Type']= diagnosis_type
        
        # ----------------- #
        # Intervention Code #
        # ----------------- #
        intervention_code = row['Intervention Code']
        if '**'== intervention_code or '[]'==intervention_code:
            intervention_code=[]
        elif '[' in intervention_code and ']' in intervention_code:
            intervention_code = ast.literal_eval(intervention_code)
            assert str(intervention_code) == row['Intervention Code']
        else:
            intervention_code = [intervention_code]
        data[ix]['Intervention Code']= intervention_code

        # ------------ #
        # Px Long Text #
        # ------------ #
        px_long_text = row['Px Long Text']
        if '**'== px_long_text or '[]'==px_long_text:
            px_long_text=[]
        elif px_long_text.startswith('[') and px_long_text.endswith(']'):
            px_long_text = ast.literal_eval(px_long_text)
            assert str(px_long_text)==row['Px Long Text'], str(ix)+':'+str(px_long_text)
        else:
            px_long_text = [px_long_text]
        data[ix]['Px Long Text']= px_long_text
        
        # ------------- #
        # Admit Ctegory #
        # ------------- #
        admit_category = row['Admit Category:'] if row['Admit Category:']!='**' else None
        assert admit_category!='**' and admit_category!='[]', admit_category
        data[ix]['Admit Category']=admit_category

        # ---------- #
        # Entry Code #
        # ---------- #
        entry_code = row['Entry Code:'] if row['Entry Code:']!='**' else None
        assert entry_code!='**' and entry_code!='[]', entry_code
        data[ix]['Entry Code']=entry_code

        # ----------------- #
        # Transfusion Given #
        # ----------------- #
        transfusion_given = row['Transfusion Given']  if row['Transfusion Given']!='**' else None
        assert transfusion_given!='**' and transfusion_given!='[]', transfusion_given
        data[ix]['Transfusion Given']=transfusion_given

        # ---------------- #
        # Main Pt Service: #
        # ---------------- #
        main_pt_service = row['Main Pt Service:'] if row['Main Pt Service:']!='**' else None
        assert main_pt_service!='**' and main_pt_service!='[]', main_pt_service
        data[ix]['Main Pt Service']=main_pt_service

        # --- #
        # CMG # 
        # --- #
        cmg = float(row['CMG']) if row['CMG']!='**' else None
        assert cmg!='**' and cmg!='[]', cmg
        data[ix]['CMG'] = cmg

        # ----------------- #
        # Comorbidity Level #
        # ----------------- #
        comorbidity_level = row['Comorbidity Level'] if row['Comorbidity Level']!='**' else None
        assert comorbidity_level!='**' and comorbidity_level!='[]', comorbidity_level
        data[ix]['Comorbidity Level']=comorbidity_level

        # ----------- #
        # Case Weight #
        # ----------- #
        case_weight = row['Case Weight']
        if case_weight=='1,946.89':
            case_weight=1946.89
        case_weight = float(case_weight) if case_weight!='**' else None
        
        assert case_weight!='**' and case_weight!='[]', case_weight
        data[ix]['Case Weight']=case_weight

        # ------- #
        # ALCDays #
        # ------- #
        alcdays = int(row['ALCDays']) if row['ALCDays']!='**' else None
        assert alcdays!='**' and alcdays!='[]', alcdays
        data[ix]['ALC Days']=alcdays

        # ---------- #
        # Acute Days #
        # ---------- # 
        acute_days = int(row['Acute Days']) if row['Acute Days']!='**' else None
        assert acute_days!='**' and acute_days!='[]', acute_days
        data[ix]['Acute Days']=acute_days

        # -------------- #
        # Institution To #
        # -------------- # 
        institution_to = (row['Institution To']) if row['Institution To']!='**' else None
        assert institution_to!='**' and institution_to!='[]', institution_to
        data[ix]['Institution To']=institution_to


        # ---------------- #
        # Institution From #
        # ---------------- #
        institution_from = (row['Institution From']) if row['Institution From']!='**' else None
        assert institution_from!='**' and institution_from!='[]', institution_from
        data[ix]['Institution From']=institution_from


        # ---------------- #
        # Institution Type #
        # ---------------- #
        institution_type = (row['Institution Type']) if row['Institution Type']!='**' else None
        assert institution_type!='**' and institution_type!='[]', institution_type
        data[ix]['Institution Type']=institution_type


        # -------------------- #
        # Discharge Nurse Unit #
        # -------------------- #
        discharge_nurse_unit = (row['Discharge Nurse Unit']) if row['Discharge Nurse Unit']!='**' else None
        assert discharge_nurse_unit!='**' and discharge_nurse_unit!='[]', discharge_nurse_unit
        data[ix]['Discharge Nurse Unit']=discharge_nurse_unit

        # --------- #
        # CZ Status #
        # --------- #
        cz_status = row['CZ status']
        assert cz_status!='**' and cz_status!='[]'
        data[ix]['CZ Status']=cz_status

    if params['save_to_disk']:
        with open(config['json_file'], 'w') as f:
            json.dump(data, f, default=str)
        io.debug(f"JSON stored to disk: {config['json_file']}")