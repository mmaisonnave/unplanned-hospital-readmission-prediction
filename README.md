


# Unplanned Hospital Readmission Prediction Project

![Contributors](https://img.shields.io/github/contributors/mmaisonnave/unplanned-hospital-readmission-prediction?style=plastic)
![Forks](https://img.shields.io/github/forks/mmaisonnave/unplanned-hospital-readmission-prediction)
![Stars](https://img.shields.io/github/stars/mmaisonnave/unplanned-hospital-readmission-prediction)
![GitHub](https://img.shields.io/github/license/mmaisonnave/unplanned-hospital-readmission-prediction?style=round-square)
![Issues](https://img.shields.io/github/issues/mmaisonnave/unplanned-hospital-readmission-prediction)


This repository contains all source code, configuration, pre-computed models, and results submitted to the [Journal of Biomedical Informatics](https://www.sciencedirect.com/journal/journal-of-biomedical-informatics) as part of our submission titled "Explainable Machine Learning to Identify Risk Factors for Unplanned Hospital Readmissions in Nova Scotian Hospitals" by Mariano Maisonnave, Enayat Rajabi, Majid Taghavi, and Peter VanBerkel. Due to privacy and confidentiality agreements, the datasets used in this study cannot be provided.


**Methodology**: As part of this work, we explore how machine learning can predict unplanned hospital readmission in the province of Nova Scotia using the [Discharge Abstract Database (DAD)](https://www.cihi.ca/en/discharge-abstract-database-dad-metadata). We explore different predictive models and preprocessing and feature engineering methodologies to identify patients at risk of readmission using information available at discharge. Using explainability tools, we tap into the decision boundary of the best models to provide insights to medical professionals about relevant risk factors for the patients at risk of readmission and offer a practical guideline for early screening of patients at risk of readmission.



**Results**: 
Our findings suggest that explainability tools can offer insights into risk factors contributing to patients' risk of readmission. We found several risk factors that we corroborated with the existing literature, and we created a guideline that allows for an early screening of patients, allowing target follow-up on a reduced population (less than half the patients) while steel capturing the majority of the readmissions (more than 72% recall).


## Installation
```
git clone https://github.com/mmaisonnave/unplanned-hospital-readmission-prediction
cd unplanned-hospital-readmission-prediction
pip install -r requirements.txt
```

## Repository Structure

- `bash/`: We run all experiments in the [Digital Research Alliance of Canada](https://alliancecan.ca/) clusters using the [Slurm](https://slurm.schedmd.com/documentation.html) scheduler. In this folder, we stored all the Bash/Slurm scripts we used to run all experiments. Each bash script calls one or more Python scripts to perform specific tasks. For example, the 'creating_database.sh' script calls three Python scripts to do three distinct tasks: (1) Creating a central zone and not central zone CSV files, (2) unifying both CSV files, and (3) transforming the data into the JSON format.
- `config/`: This folder contains all configuration scripts for all experiments. We define model hyperparameters and architectures in configuration files (for example, in the model.json file). Similarly, we define the preprocessing task applied to our data in each experiment in the experiment_configuration.json configuration file.
- `src/`: In this folder, We have Python scripts for running experiments, transforming the data, creating embedding models, and other tasks. We organized the experiments into Bash/Slurm scripts that invoke the Python scripts in this folder.
- `results/`: Some Python scripts store results in CSV files or figures. The Python scripts store those results in the results folder.
- `utilities/`: In this folder, we store additional functionality required for this project. For example, we have Python modules for (1) managing standard output (`io.py`), (1) handling the data (`health_data.py`), handling encryption of the data (`crypto.py`), and other functionalities. This external repository is linked and stored in the `utilities/` folder of this repository.
- `gensim/`: We pre-computed embedding models to use in some of the machine learning models. We stored the pre-computed models in this folder. We stored some models in parts to overcome the GitHub file size limitation. We provide a Bash script to merge back parts into a single file. We stored the script to merge the files in the same `gensim/` folder.



## Dependencies
We ran all experiments using the dependencies and libraries listed in the `requirements.txt` file.



## Running the Experiments

**Database Creation**
To store the data in a more computer-friendly format, we took the original data, which was divided into multiple files across multiple folders and merged into a single JSON file. The `bash/creating_database.sh` script handles this procedure. It offloads the task to three Python scripts:

1. `src/creating_cz_and_noncz_csv_files.py`
2. `src/creating_single_csv.py`
3. `src/creating_one_json.py`


**Database Creation** 
The `bash/creating_held_out.sh` splits the data (stored in a single JSON file) to create a held-out using the last year of the data. It uses the `src/creating_held_out.py` Python script. 


**Cross-Validation experiments and evaluation on held-out** To run all cross-validation experiments (200+ models), we used the `bash/running_all_experiments_cv.sh` script that calls the `src/running_all_experiments_cv.py` Python script. We used the `bash/ML_experiments_on_heldout.sh` bash script, which calls the `src/ML_experiments_on_heldout.py` Python script to evaluate the best models on the held-out split.




**Explainability Experiments** We have individual Python and Bash/Slurm scripts for all explainability experiments. We used the `run_shap_on_brf.{sh|py}` files to run the SHAP experiments. We used the `compute_permutation_feature_importance.{sh|py}` files to run the permutation feature importance experiments. We used the `run_explainable_logreg.{sh|py}` files to run the explainability experiments using the Logistic Regression model. We used the `run_explainable_dt.{sh|py}` files to run the experiments with shallow (max-depth=3) decision trees and save the figures. Finally, we used the `consensus_analysis_dt.{sh|py}` files to run the interaction analysis experiments. The interaction analysis consisted of running an explainable decision tree on top of a dataset that contains the best features found with all the other explainability tools.


**Miscellaneous** We used the `build_dataset_statistics_table.{sh|py}` files to build a table with dataset statistics. We build a mapping of intervention and diagnosis codes to their corresponding description using the files in `building_diag_and_interv_code_map.{py
sh}`. We counted the frequency of each intervention and diagnosis code in the dataset using the `compute_D_and_I_frequencies.{sh|py}` files. 

**Gensim embedding models** To build and test the embedding models we built using Gensim, we used the `compute_pretrained_embeddings.{sh|py}` and `evaluate_gensim_models.{sh|py}` files.


**Guidelines** To build and evaluate the guidelines (including on the held-out data), we use the `test_guidelines.{py|sh}` files.


## Authors

* [Mariano Maisonnave](https://github.com/mmaisonnave): conceptualization, data curation, formal analysis, investigation, methodology, software, validation, visualization, writing (initial and review and editing).
* [Enayat Rajabi](https://erajabi.github.io/): conceptualization, funding acquisition, resources, supervision, writing (review and editing).
* [Majid Taghavi](https://www.smu.ca/researchers/sobey/profiles/majid-taghavi.html): conceptualization, funding acquisition, resources, supervision, writing (review and editing).
* [Peter VanBerkel](https://www.dal.ca/faculty/engineering/industrial/faculty-staff/our-faculty1/professors/peter-vanberkel.html): conceptualization, funding acquisition, resources, supervision, writing (review and editing).


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


## Contributing
We welcome contributions from the community. Please open an issue or submit a pull request for any improvements or suggestions.

## Acknowledgments
We thank Research Nova Scotia, which partially funded this research, grant number RNS-NHIG-2021-1968. We also thank the Nova Scotia Health Research Ethics Board (REB) and the Nova Scotia Health Authority, particularly Ashley Boyc and Michele deSteCroix, for their prompt and effective support throughout our research. Lastly, this research was enabled in part by the advanced computing resources provided by ACENET ([https://www.ace-net.ca/](ace-net.ca)) and the Digital Research Alliance of Canada ([https://alliancecan.ca/](alliancecan.ca)).

