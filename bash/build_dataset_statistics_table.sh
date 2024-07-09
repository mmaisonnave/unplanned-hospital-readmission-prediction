#!/bin/bash
#SBATCH --time=0-00:30:0
#SBATCH --account=def-erajabi
#SBATCH --ntasks=1 
#SBATCH --nodes=1 
#SBATCH --mem=24GB 
#SBATCH --cpus-per-task=1
#SBATCH --job-name=build_dataset_statistics_table
#SBATCH --output=/home/maiso/cbu/slurm/output/%x-%j.out

echo Running script at $(pwd)

ENV=alc
REPOSITORY_PATH=$(cat ../config/paths.yaml | grep repository\_path: | grep -v ^# | sed 's/^repository\_path:\ //g')
PYTHON_SCRIPTS_FOLDER=src
PYTHON_SCRIPT_NAME=build_dataset_statistics_table.py

PYTHON_SCRIPT=$REPOSITORY_PATH/$PYTHON_SCRIPTS_FOLDER/$PYTHON_SCRIPT_NAME


echo $(date) - Running python file: $PYTHON_SCRIPT
echo $(date) - Using python: $(python --version)
echo $(date) - Which python: $(which python)

# IF PYTHON SCRIPT NOT FOUND, EXIT
if [ ! -f $PYTHON_SCRIPT ]; then
    echo "Python script not found ($PYTHON_SCRIPT_NAME)"
fi


# NO VIRTUAL ENV, EXIT
if [ -z "${VIRTUAL_ENV}" ];
then 
    echo Scripts expect conda environment set "$ENV"
    exit 1
fi

# DIFFERENT VRITUAL ENV FROM EXPECTED, EXIT
if [ $(echo $VIRTUAL_ENV | sed 's/.*\/\(.*\)$/\1/g' | sed 's/\n\n*//g')  != $ENV ]; 
then 
    echo Scripts expect conda environment set "$ENV"
    exit 1
fi

CUSTOM_COMMAND="$PYTHON_SCRIPT"
echo [RUNNING] python $CUSTOM_COMMAND
python $CUSTOM_COMMAND

