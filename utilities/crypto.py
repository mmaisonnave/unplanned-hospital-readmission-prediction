"""
A module to handle the encryption and decrpytion of data. 

Method:
-------
    - encrpypt_file(input_file: str, output_file:str , ) -> None
    - decrypt_file(input_file: str, output_file:str, ) -> None
    - get_content_of_encrypted_file(input_file: str,) -> str:
    - main()->None

The main calls:
    - encrpypt_file and
    - encrpypt_file
    

This module can be called from the command line (main method) to encrypt or decrypt files.

I use the command line functionality to encrypt and decrypt the data the first time.

The experiments retrieved the encrypted data calling the `get_content_of_encrypted_file` method.


It requires a key which I hande with a Singleton: `SingletonKey`. So, we read the key from 
disk once and all parts of the software can use it. The key is removed immediately after it 
is read. So, every time you need to run the pgoram you need to provide the key file `config/alc.key` 
and it will be read and deleted every time.  
"""
from cryptography.fernet import Fernet
import argparse
import os
from getpass import getpass

# configuration module needs to be imported from the package `utilities`
# Dependeing if the module was running from the utilities folder or the repository folder, where 
# to find the module configurations changes. 
if os.getcwd() == 'utilities':
    import configuration
else:
    from utilities import configuration

import os


class SingletonKey(object):
    """
    Basic singleton tempate to have the key only once in memory.
    """    
    def __new__(cls, key:str=None):
        """
        It returns a SingletonKey instance. It could be an existing one or a new one.
        Args:
            key (str, optional): Key to store. Typically used reading key from file:
                                 keyholder = SingletonKey(key=reader.read()). 
                                 Defaults to None.

        Returns:
            _type_: SingletonKey instance (new or existing).
        """        
        if not hasattr(cls, 'instance'):
            # Creating new instance with empty Key
            cls.instance = super(SingletonKey, cls).__new__(cls)
            cls.key=None

        # No matter if new or existing singleton, if the key!=None then add new key
        if not key is None:
            cls.instance.key=key
            
        return cls.instance

def encrypt_file(input_file: str, output_file:str , ) -> None:
    """
    Receives a input file name and an output file name. It reads the content in the input file,
    It prompts the user for a key for encryption, encrypts and stores the results in the
    output file name.

    Args:
        input_file (str): Name of the file to encrypt
        output_file (str): name of the file where to store the encrypted content (method did not
                           check if the file already exists. It will override any existing file).
    """    
    # using the generated key
    fernet = Fernet(getpass('Enter key for encryption: '))
    
    # opening the original file to encrypt
    with open(input_file, 'rb') as file:
        original = file.read()
        
    # encrypting the file
    encrypted = fernet.encrypt(original)
    
    # opening the file in write mode and 
    # writing the encrypted data
    with open(output_file, 'wb') as encrypted_file:
        encrypted_file.write(encrypted)


def decrypt_file(input_file: str, output_file:str, ) -> None:
    """Method to descrypt a file.

    Args:
        input_file (str): name of the file to decrypt
        output_file (str): destination file name. If the file exists it overrides it.
    """
    fernet = Fernet(getpass('Enter key for decryption: '))

    # opening the encrypted file
    with open(input_file, 'rb') as enc_file:
        encrypted = enc_file.read()
    
    # decrypting the file
    decrypted = fernet.decrypt(encrypted)
    
    # opening the file in write mode and
    # writing the decrypted data
    with open(output_file, 'wb') as dec_file:
        dec_file.write(decrypted)


def get_content_of_encrypted_file(input_file: str,) -> str:
    """Reads an encrypted file, and it returns its content (string).

    Args:
        input_file (str): Name of the encrypted file

    Returns:
        str: Content of the decrypted file.
    """    
    config = configuration.get_config()

    if SingletonKey().key is None:
        # If we don't have the key we need to get it from a file or from the user...
        if os.path.isfile(config['keyfile']):
            # Get key from file.
            with open(config['keyfile'], 'r') as reader:
                keyholder = SingletonKey(key=reader.read())
            os.remove(config['keyfile'])
        else:
            # Get key from user if everything else fails
            keyholder = SingletonKey(key=getpass('Enter key for decryption: '))

    fernet = Fernet(SingletonKey().key)

    # opening the encrypted file
    with open(input_file, 'rb') as enc_file:
        encrypted = enc_file.read()
    
    # decrypting the file
    decrypted = fernet.decrypt(encrypted)

    return decrypted    



def main()->None:
    """
    Method used to encrypt train and validation JSON data file.

    Usage for encrypting:
        python crypto.py 
        --action=encrypt
        --input=train_validation.json 
        --output=train_validation.json.encryption

    Usage for decrypting:
        python crypto.py 
        --action=decrypt
        --input=train_validation.json.encryption
        --output=train_validation.json

    WARNING: The method doesn't check if the output file exists. It will override existing files.
    """
    parser = argparse.ArgumentParser(
                        prog='Encrypt/Decrypt',
                        description='What the program does',
                        epilog='Text at the bottom of help')
    parser.add_argument('--action', 
                        required=True,
                        type=str,
                        choices=['encrypt', 'decrypt'],
                        dest='action',
                        )
    parser.add_argument('--input', 
                        required=True,
                        type=str,
                        dest='input',
                        )
    parser.add_argument('--output', 
                        required=True,
                        type=str,
                        dest='output',
                        )
    args = parser.parse_args()

    assert os.path.isfile(args.input)
    assert not os.path.isfile(args.output)

    if args.action == 'encrypt':
        encrypt_file(input_file=args.input, output_file=args.output)
    else:
        assert args.action == 'decrypt'
        decrypt_file(input_file=args.input, output_file=args.output)

if __name__ == '__main__':
    main()