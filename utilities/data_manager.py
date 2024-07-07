from utilities import configuration
import json
from utilities import crypto

def get_train_test_json_content() -> dict:
        """This retrieves the training and validating data from an encrypted JSON file and returns the 
        resulting dictionary. It asks for user input (it request the encryption key). The methods uses the
        utilities.crypto package. 

        Returns:
            dict: Content of train and validation JSON
        """
        config = configuration.get_config()

        # ---------- ---------- ---------- ---------- 
        # Retriving train testing data from JSON file
        # ---------- ---------- ---------- ---------- 
        f = crypto.get_content_of_encrypted_file(config['encrypted_train_val_json'])

        return json.loads(f)


def get_heldout_json_content() -> dict:
        """This retrieves the held-out data from an encrypted JSON file and returns the 
        resulting dictionary. It asks for user input (it request the encryption key). The methods uses the
        utilities.crypto package. 

        Returns:
            dict: Content of train and validation JSON
        """
        config = configuration.get_config()

        # ---------- ---------- ---------- ---------- 
        # Retriving held-out data from JSON file (encrypted)
        # ---------- ---------- ---------- ---------- 
        f = crypto.get_content_of_encrypted_file(config['encrypted_heldout_json'])

        return json.loads(f)