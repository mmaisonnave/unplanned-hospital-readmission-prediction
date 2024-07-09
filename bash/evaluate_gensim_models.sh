ENV=alc
REPOSITORY_PATH=$(cat ../config/paths.yaml | grep repository\_path: | grep -v ^# | sed 's/^repository\_path:\ //g')
PYTHON_SCRIPTS_FOLDER=src
PYTHON_SCRIPT_NAME=evaluate_gensim_models.py

PYTHON_SCRIPT=$REPOSITORY_PATH/$PYTHON_SCRIPTS_FOLDER/$PYTHON_SCRIPT_NAME


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

RANDOM_SAMPLE_SIZE=5000
CONSIDERED_RANKS=20


CUSTOM_COMMAND="$PYTHON_SCRIPT --considered-ranks=$CONSIDERED_RANKS --random-sample-size=$RANDOM_SAMPLE_SIZE"
echo [RUNNING] python $CUSTOM_COMMAND
python $CUSTOM_COMMAND

