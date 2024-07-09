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