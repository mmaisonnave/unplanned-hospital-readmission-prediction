"""
This script generates two mappings:
1. A mapping of intervention codes to their descriptions.
2. A mapping of diagnosis codes to their descriptions.

Main Methods:
- `build_intervention_dict`: Creates a dictionary where each intervention code 
  from the dataset is mapped to its corresponding description.
- `build_diagnosis_dict`: Creates a dictionary where each diagnosis code from 
  the dataset is mapped to its corresponding description.

The script reads data from a JSON file (`full_database.json`), which contains 
all admissions in the dataset. It processes the data to generate the two 
mappings and stores the results in the following files:
- `Merged/2015_2022/intervention_dict.json`: Stores the intervention code-to-description mapping.
- `Merged/2015_2022/diagnosis_dict.json`: Stores the diagnosis code-to-description mapping.

Command-line Arguments:
- `--save-to-disk`: Specifies whether to save the generated mappings to disk.
- `--compute-diagnoses`: Specifies whether to generate and save the diagnosis code mapping.
- `--compute-interventions`: Specifies whether to generate and save the intervention code mapping.

If the user chooses not to save the results, the mappings will be generated but not stored.
"""
import argparse
import json

import sys
sys.path.append('..')
from utilities import health_data
from utilities import configuration
from utilities import io

def build_intervention_dict(admissions: list[health_data.Admission]) -> dict:
    code2description = {}
    total_entries=0
    invalid_entries=0
    for admission in admissions:
        codes = admission.intervention_code
        texts = admission.px_long_text
        for code,text in zip(codes,texts):
            total_entries+=1
            if not code in code2description:
                code2description[code]={text}
            else:
                if not text in code2description[code]:
                    invalid_entries+=1
                    code2description[code].add(text)
    io.debug(f'Duplicated or problematic entries (interventions):  {invalid_entries:,}/{total_entries:,}  ({invalid_entries/total_entries:.4%})')
    return code2description

def build_diagnosis_dict(admissions: list[health_data.Admission]) -> dict:
    code2description = {}
    total_entries=0
    invalid_entries=0
    for admission in admissions:
        codes = admission.diagnosis.codes
        texts = admission.diagnosis.texts
        for code,text in zip(codes,texts):
            total_entries+=1
            if not code in code2description:
                code2description[code]={text}
            else:
                if not text in code2description[code]:
                    invalid_entries+=1
                    code2description[code].add(text)
    io.debug(f'Duplicated or problematic entries (diagnoses):      {invalid_entries:,}/{total_entries:,}  ({invalid_entries/total_entries:.4%})')
    return code2description


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-to-disk', 
                        dest='save_to_disk', 
                        required=True, 
                        help='wether to save to disk or not',
                        type=str,
                        choices=['True', 'False']
                        )
    parser.add_argument('--compute-diagnoses', 
                        dest='compute_diagnoses', 
                        required=True, 
                        help='wether to compute diagnosis code dict or not',
                        type=str,
                        choices=['True', 'False']
                        )
    parser.add_argument('--compute-interventions', 
                        dest='compute_interventions', 
                        required=True, 
                        help='wether to compute intervention code dict or not',
                        type=str,
                        choices=['True', 'False']
                        )
    args = parser.parse_args()

    params = {'save_to_disk' : args.save_to_disk=='True',
              'compute_interventions' : args.compute_interventions=='True',
              'compute_diagnoses' : args.compute_diagnoses=='True',
              }


    config = configuration.get_config()



    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    # Retriving all data JSON file (including held-out)
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    f = open(config['json_file'], encoding='utf-8')
    all_data = json.load(f)
    io.debug(f'Running interventions and diagnoses map generation script using {len(all_data):,} entries.')
    io.debug(f'params: {str(params)}')

    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    # Converting JSON to DataClasses
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    all_admissions = []
    for ix in all_data:
        all_admissions.append(
            health_data.Admission.from_dict_data(admit_id=int(ix), admission=all_data[ix])
            )

    print(f'len(all_admissions)={len(all_admissions):,}')

    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    # INTERVENTIONS
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    if params['compute_interventions']:    
        io.debug(f'Computing intervention mapping over {len(all_admissions)} entries.')

        interventions_dict = build_intervention_dict(all_admissions)

        if params['save_to_disk']:
            with open(config['intervention_dict'], 'w', encoding='utf-8') as f:
                json.dump(interventions_dict, f, default=str)
        else:
            io.debug('Intervention mapping computed but not stored')
    else:
        io.debug('Skipping interventions ...')
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    # DIAGNOSES
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
    if params['compute_diagnoses']: 
        diagnosis_dict = build_diagnosis_dict(all_admissions)

        if params['save_to_disk']:
            with open(config['diagnosis_dict'], 'w', encoding='utf-8') as f:
                json.dump(diagnosis_dict, f, default=str)
        else:
            io.debug('Diagnosis mapping computed but not stored')
    else:
        io.debug('Skipping diagnoses ...')