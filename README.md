# unplanned-hospital-readmission-prediction

### Database Creation
1. bash/creating_database.sh: 
it Runs:
- creating_cz_and_noncz_csv_files.py
- creating_single_csv.py 
- creating_one_json.py

2. bash/creating_held_out.sh
it runs:
1. creating_held_out.py (takes a JSON file with all the data and split it into development,heldout,unused).


### All cross validation experiments:
- bash/running_all_experiments_cv.sh
it runs:
1. src/running_all_experiments_cv.py


### Explainability Experiments:
- Decision trees: run_explainable_dt.{sh|py}
- Logistic Regression: run_explainable_logreg.{sh|py}
- SHAP: run_shap_on_brf.{sh|py}
- PFI: compute_permutation_feature_importance.{sh|py}


### Experiments on heldout:
- test_guidelines{sh|py}



### Note Regarding Embeddings
- For some experiments precomputed embeddings are required, I cannot upload them because of size. I am uploading them in parts (see gensim/models). To join them back use the script merge_interv_conf_2_embeddings.sh. The only file it was to big was (diag_conf_11_embeddings.npy) I splitted into six parts (interv_conf_2_embeddings.npy.tgz.part1 ...).