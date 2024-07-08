"""
This scripts takes a JSON file with all the data and split it into three:
1. training and test set
2. held-out set
3. unused set.

>>
USAGE: python creating_cz_and_noncz_files.py --save-to-disk=[True,False]
>>

if --save-to-disk==False no changes are done (it can be considered as a test run).
if --save-to-disk==True files are written to disk, can be overwriting files, cannot be undone.

*INPUT*
json_file: full_database.json

*OUTPUT*
train_val_json: train_validation.json
heldout_json: heldout.json
unused_after_heldout_json: unused_after_held_out.json



"""

from collections import defaultdict
import numpy as np
import sys
import json
sys.path.append('..')
from utilities import io
from utilities import configuration
from utilities import health_data
import argparse
import datetime

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
    # ...
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------    
    config = configuration.get_config()
    io.debug('\n')
    io.debug('Starting script to split one JSON file with all data into train/test, heldout and unused.')

    if params['save_to_disk']:
        io.warning('Results will be stored in risk, changes cannot be undone.')
    else:
        io.debug('Reults will **NOT** be stored in disk, this is a test run.')


    io.debug('Reading JSON ...')
    f = open(config['json_file'])
    data = json.load(f)
    io.debug(f'Number of entries found: {len(data):,}')

    all_admissions = []
    for ix in data:
        all_admissions.append(
            health_data.Admission.from_dict_data(admit_id=int(ix), admission=data[ix])
            )
        
    io.debug('Transforming JSON entries into Admission DataClass, readmission target no computed.')

    # print
    # # Dictionary organizing data by patient
    # patient2admissions = defaultdict(list)
    # for admission in all_admissions:
    #     code = admission.code
    #     patient2admissions[code].append(admission)

    # # Ordering patient list by discharge date (from back )
    # for patient_code in patient2admissions:
    #     admissions_list = patient2admissions[patient_code]
    #     admissions_list = sorted(admissions_list, key=lambda admission: admission.discharge_date, reverse=False)
    #     assert all([admissions_list[i].discharge_date <= admissions_list[i+1].discharge_date for i in range(len(admissions_list)-1)])
    #     patient2admissions[patient_code] = admissions_list
    # print(len(patient2admissions))

    # patient_count=0
    # valid_readmission_count=0
    # for patient_code in patient2admissions:
    #     patient_admissions = patient2admissions[patient_code]
    #     ix = 0 
    #     while ix < len(patient_admissions):
    #         readmission_code = patient_admissions[ix].readmission_code
    #         if health_data.ReadmissionCode.is_readmit(readmission_code):
    #             # Either is not the first admission (ix>0) or 
    #             # we don't have the patient previous admition (readmission close to begining of dataset) (admit-(2015-01-01))<28 days
    #             # assert ix>0 or (patient_admissions[ix].admit_date - datetime.datetime.fromisoformat('2015-01-01')).days<365
    #             if ix>0 and  patient_admissions[ix-1].is_valid_readmission(patient_admissions[ix]):
    #                 patient_admissions[ix-1].add_readmission(patient_admissions[ix])
    #                 valid_readmission_count+=1
    #         ix+=1
    #     patient_count+=1
    # valid_readmission_count


    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    # HELD OUT BOUNDARIES 
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    # Computing average LOS
    length_of_stays=[]
    for admission in all_admissions:
        if not admission.admit_date is None:
            days = (admission.discharge_date - admission.admit_date).days
            length_of_stays.append(days)

    io.debug(f'Computing average length of stay: {np.average(length_of_stays):.4f}')
    io.debug(f'Computing std length of stay:     {np.std(length_of_stays):.4f}')

    # Assuming normal distribution of LOS, mean LOS + one std to each side will contain 68 % of instances. 
    # So, from 0 to mean LOS + one std has 84 %  (until the mean has 50% of instances, + 68%/2 for the mean LOS to (mean LOS + one std))
    # We will round up to a length of stay of 60 days. 



    held_out_size = 365
    readmission_timeframe = 30
    time_for_discharge_to_happen = 60   # For us to have the readmission, the readmission and the discharge has to happen before the end of our data (Dec 31st, 2022)
                                        # So, for us to see the discharge it has to happen in October 2nd, 2022, 30 days after the readmission happens (Nov 1st, 2022), 60 after the discharge 
                                        # happens (on Dec 31st, 2022), so we will have the full entry of the readmission, because the discharge happened before the end of our dataset.

    latest_date = max([admission.discharge_date for admission in all_admissions])
    begining_dataset = min([admission.discharge_date for admission in all_admissions])


    start_heldout=latest_date - datetime.timedelta(days=held_out_size+readmission_timeframe+time_for_discharge_to_happen)
    end_heldout = latest_date - datetime.timedelta(days=readmission_timeframe+time_for_discharge_to_happen) 

    io.debug(f'Begining:                              {begining_dataset}')
    io.debug(f'Start held-out:                        {start_heldout}')
    io.debug(f'End held-out:                          {end_heldout}')
    io.debug(f'End data (usable 30 days prior):       {latest_date}')



    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    # CREATING THREE JSONS
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    held_out_data = {}
    train_val_data = {}
    unused_after_heldout={}
    for ix in data:
        discharge_date = datetime.datetime.fromisoformat(data[ix]['Discharge Date'])
        if begining_dataset <= discharge_date and discharge_date < start_heldout:
            train_val_data[ix]=data[ix]
        elif start_heldout<= discharge_date and discharge_date <= end_heldout:
            held_out_data[ix]=data[ix]
        else:
            unused_after_heldout[ix]=data[ix]


    io.debug(f'held out:      {len(held_out_data):7,} entries')
    io.debug(f'train and dev: {len(train_val_data):7,} entries')
    io.debug(f'unused:        {len(unused_after_heldout):7,} entries')

    # Training JSON
    if params['save_to_disk']:
        with open(config['train_val_json'], 'w') as f:
            json.dump(train_val_data, f)

        # Held-out JSON
        with open(config['heldout_json'], 'w') as f:
            json.dump(held_out_data, f)

        # Unused JSON
        with open(config['unused_after_heldout_json'], 'w') as f:
            json.dump(unused_after_heldout, f)
