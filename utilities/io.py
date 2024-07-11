
import datetime

def _format(message:str)->str:
    if not message.endswith('.') and not message.endswith('!'):
        message=f'{message}.'
    return message

def done()->None:
    ok('Done!')

def ok(obj)->None:
    if isinstance(obj,str):
        message = obj
    else:
        message =str(obj)
    print(f'{str(datetime.datetime.now())} [  OK   ] {_format(message)}')

def warning(obj:str)->None:
    if isinstance(obj,str):
        message = obj
    else:
        message =str(obj)
    print(f'{str(datetime.datetime.now())} [WARNING] {_format(message)}')

def debug(obj:str)->None:
    if isinstance(obj,str):
        message = obj
    else:
        message =str(obj)
    print(f'{str(datetime.datetime.now())} [ DEBUG ] {_format(message)}')

def info(obj:str)->None:
    if isinstance(obj,str):
        message = obj
    else:
        message =str(obj)
    print(f'{str(datetime.datetime.now())} [ INFO  ] {_format(message)}')

def error(obj:str)->None:
    if isinstance(obj,str):
        message = obj
    else:
        message =str(obj)
    print(f'{str(datetime.datetime.now())} [ ERROR ] {_format(message)}')