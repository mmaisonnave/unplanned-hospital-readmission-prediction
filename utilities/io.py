
import datetime

def _format(message:str)->str:
    if not message.endswith('.') and not message.endswith('!'):
        message=f'{message}.'
    return message

def done()->None:
    ok('Done!')

def ok(message:str)->None:
    print(f'{str(datetime.datetime.now())} [  OK   ] {_format(message)}')

def warning(message:str)->None:
    print(f'{str(datetime.datetime.now())} [WARNING] {_format(message)}')

def debug(message:str)->None:
    print(f'{str(datetime.datetime.now())} [ DEBUG ] {_format(message)}')

def info(message:str)->None:
    print(f'{str(datetime.datetime.now())} [ INFO  ] {_format(message)}')

def error(message:str)->None:
    print(f'{str(datetime.datetime.now())} [ ERROR ] {_format(message)}')