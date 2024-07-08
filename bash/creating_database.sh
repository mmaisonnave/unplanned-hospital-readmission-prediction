ENV=alc
REPOSITORY_PATH=$(cat ../config/paths.yaml | grep repository\_path: | grep -v ^# | sed 's/^repository\_path:\ //g')
BASH_SCRIPTS_FOLDER=src
SCRIPT1_CREATE_TWO_CSV=creating_cz_and_noncz_csv_files.py
SCRIPT2_CREATE_ONE_CSV=creating_single_csv.py
SCRIPT3_CREATE_TWO_JSON=creating_one_json.py

PYTHON_SCRIPT1=$REPOSITORY_PATH/$BASH_SCRIPTS_FOLDER/$SCRIPT1_CREATE_TWO_CSV
PYTHON_SCRIPT2=$REPOSITORY_PATH/$BASH_SCRIPTS_FOLDER/$SCRIPT2_CREATE_ONE_CSV
PYTHON_SCRIPT3=$REPOSITORY_PATH/$BASH_SCRIPTS_FOLDER/$SCRIPT3_CREATE_TWO_JSON

# IF PYTHON SCRIPT NOT FOUND, EXIT
for PYTHON_SCRIPT in $PYTHON_SCRIPT1 $PYTHON_SCRIPT2 $PYTHON_SCRIPT3
do 
    if [ ! -f $PYTHON_SCRIPT ]; then
        echo "Python script not found ($PYTHON_SCRIPT)"
    fi
done

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
SAVE_CHANGES_TO_DISK=False

CUSTOM_COMMAND="$PYTHON_SCRIPT1 --save-to-disk=$SAVE_CHANGES_TO_DISK"
echo [RUNNING] python $CUSTOM_COMMAND
python $CUSTOM_COMMAND
echo

CUSTOM_COMMAND="$PYTHON_SCRIPT2 --save-to-disk=$SAVE_CHANGES_TO_DISK"
echo [RUNNING] python $CUSTOM_COMMAND
python $CUSTOM_COMMAND
echo

CUSTOM_COMMAND="$PYTHON_SCRIPT3 --save-to-disk=$SAVE_CHANGES_TO_DISK"
echo [RUNNING] python $CUSTOM_COMMAND
python $CUSTOM_COMMAND
echo