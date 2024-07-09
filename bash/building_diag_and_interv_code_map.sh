ENV=alc
REPOSITORY_PATH=$(cat ../config/paths.yaml | grep repository\_path: | grep -v ^# | sed 's/^repository\_path:\ //g')
BASH_SCRIPTS_FOLDER=src
SCRIPT_NAME=building_diag_and_interv_code_map.py

FULL_SCRIPT_PATH=$REPOSITORY_PATH/$BASH_SCRIPTS_FOLDER/$SCRIPT_NAME

# IF PYTHON SCRIPT NOT FOUND, EXIT
if [ ! -f $FULL_SCRIPT_PATH ]; then
    echo "Python script not found ($FULL_SCRIPT_PATH)"
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

# If SAVE_CHANGES_TO_DISK==False, then running this script does change anything, can be safely run multiple times.
SAVE_CHANGES_TO_DISK=True
COMPUTE_INTERVENTIONS=True
COMPUTE_DIAGNOSES=True

CUSTOM_COMMAND="$FULL_SCRIPT_PATH --save-to-disk=$SAVE_CHANGES_TO_DISK --compute-diagnoses=$COMPUTE_DIAGNOSES --compute-interventions=$COMPUTE_INTERVENTIONS"
echo [RUNNING] python $CUSTOM_COMMAND
python $CUSTOM_COMMAND
echo