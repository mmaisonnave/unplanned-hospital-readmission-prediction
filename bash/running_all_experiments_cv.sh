#!/bin/bash
#SBATCH --time=1-0:00:0
#SBATCH --account=def-erajabi
#SBATCH --ntasks=1 
#SBATCH --nodes=1 
#SBATCH --mem=32GB 
#SBATCH --cpus-per-task=1 
#SBATCH --job-name=running_all_experiments_cv
#SBATCH --output=/home/maiso/cbu/slurm/output/%x-%j.out

ENV=alc
REPOSITORY_PATH=$(cat ../config/paths.yaml | grep repository\_path: | grep -v ^# | sed 's/^repository\_path:\ //g')
BASH_SCRIPTS_FOLDER=src
SCRIPT_NAME=running_all_experiments_cv.py

PYTHON_SCRIPT=$REPOSITORY_PATH/$BASH_SCRIPTS_FOLDER/$SCRIPT_NAME

# IF PYTHON SCRIPT NOT FOUND, EXIT
if [ ! -f $PYTHON_SCRIPT ]; then
    echo "Python script not found ($PYTHON_SCRIPT)"
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

SIMULATION=False
CUSTOM_CONFIGURATION=all
# CUSTOM_CONFIGURATION=configuration_27

CUSTOM_COMMAND="$PYTHON_SCRIPT --simulation=$SIMULATION --experiment-configuration=$CUSTOM_CONFIGURATION"
echo [RUNNING] python $CUSTOM_COMMAND
python $CUSTOM_COMMAND
