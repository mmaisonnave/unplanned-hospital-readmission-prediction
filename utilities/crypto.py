from cryptography.fernet import Fernet
import argparse
import os
from getpass import getpass

if os.getcwd() == 'utilities':
    import configuration
else:
    from utilities import configuration

import os
class SingletonKey(object):
    def __new__(cls, key=None):
        if not hasattr(cls, 'instance'):
            # Creating new instance with empty Key
            cls.instance = super(SingletonKey, cls).__new__(cls)
            cls.key=None

        # No matter if new or existing singleton, if the key!=None then add new key
        if not key is None:
            cls.instance.key=key
            
        return cls.instance

def encrpypt_file(input_file: str, output_file:str , ) -> None:
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
    config = configuration.get_config()

    if SingletonKey().key is None:
        # If we don't have the key we need to get it from a file or from the user...
        if os.path.isfile(config['keyfile']):
            # Get key from file.
            with open(config['keyfile'], 'r') as reader:
                keyholder = SingletonKey(key=reader.read())
            # os.remove(config['keyfile'])
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



if __name__ == '__main__':
    """
    Used to encrypt train and validation JSON data file:
    python crypto.py 
     --action=encrypt 
     --input=/Users/marianomaisonnave/Documents/CBU Postdoc/Grant Data/Merged/2015_2022/train_validation.json 
     --output=train_validation.json.encryption
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
        encrpypt_file(input_file=args.input, output_file=args.output)
    else:
        assert args.action == 'decrypt'
        decrypt_file(input_file=args.input, output_file=args.output)



