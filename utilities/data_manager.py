"""
Module for retrieving and decrypting JSON data.

This module provides functions to retrieve training, validation, and held-out data from encrypted JSON files.
It relies on the `utilities.crypto` package for decryption and the `utilities.configuration` package
to obtain file paths from configuration settings. The data is returned as Python dictionaries after decryption
and JSON parsing.

Functions:
    get_train_test_json_content() -> dict:
        Retrieves training and validation data from an encrypted JSON file and returns it as a dictionary.

    get_heldout_json_content() -> dict:
        Retrieves and decrypts held-out data from a JSON file, returning its content as a dictionary.

Dependencies:
    - utilities.configuration: For accessing configuration settings.
    - utilities.crypto: For decrypting encrypted JSON files.
    - json: For parsing JSON data into dictionaries.
"""
from utilities import configuration
import json
from utilities import crypto

def get_train_test_json_content() -> dict:
    """
    Retrieves training and validation data from an encrypted JSON file and returns it as a dictionary.

    This method prompts the user to provide the encryption key needed to decrypt the JSON file. The file path
    to the encrypted JSON is obtained from the configuration settings, and the decryption is handled using the
    `utilities.crypto` package. The decrypted content is then parsed from JSON format into a Python dictionary.

    Returns:
        dict: A dictionary containing the training and validation data loaded from the JSON file. The structure
              of the dictionary depends on the contents of the JSON file.
    """
    config = configuration.get_config()

    # ---------- ---------- ---------- ---------- 
    # Retriving train testing data from JSON file
    # ---------- ---------- ---------- ---------- 
    f = crypto.get_content_of_encrypted_file(config['encrypted_train_val_json'])

    return json.loads(f)


def get_heldout_json_content() -> dict:
    """
    Retrieves and decrypts held-out data from a JSON file, and returns its content as a dictionary.

    This method prompts the user for an encryption key to decrypt the JSON file. It relies on the
    `utilities.crypto` package to handle decryption. The JSON file is expected to contain both training
    and validation data.

    Returns:
        dict: A dictionary containing the content of the decrypted JSON file, which includes training
              and validation data.

    Raises:
        KeyError: If the configuration does not contain the required 'encrypted_heldout_json' key.
        FileNotFoundError: If the encrypted JSON file cannot be found.
        json.JSONDecodeError: If the JSON file cannot be parsed correctly.
        EncryptionError: If there is an error during the decryption process.
    """
    config = configuration.get_config()

    # ---------- ---------- ---------- ---------- 
    # Retriving held-out data from JSON file (encrypted)
    # ---------- ---------- ---------- ---------- 
    f = crypto.get_content_of_encrypted_file(config['encrypted_heldout_json'])

    return json.loads(f)