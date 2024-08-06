"""
Python module to handle standard output formating. It produces formated output that includes the
type of message and date. The formated output looks like the following:

2024-07-31 12:15:18.388333 [ DEBUG ] {MESSAGE}
2024-07-31 12:15:56.711550 [WARNING] {MESSAGE}
2024-07-31 12:15:56.711589 [ INFO  ] {MESSAGE}
...


"""
import datetime

def _format(message:str)->str:
    """
    It receives a message and appends a dot if it was not ending with one.

    Args:
        message (str): original message

    Returns:
        str: message with a dot appended if it was not one in the original message.
    """
    if not message.endswith('.') and not message.endswith('!'):
        message=f'{message}.'
    return message


def done()->None:
    """It prints a formatted "Done" message.
    """
    ok('Done!')

def ok(obj)->None:
    """
    It receives a message and prints it in the format of the module (with date and message type).
    if the method receives a str message it prints it as is. If it receives and objects, it prints
    the str(object).

    The methods prints the method as an "ok" type of message.

    Args:
        obj: A message or object to print. If str print as is, if object prints str(obj).
    """    
    if isinstance(obj,str):
        message = obj
    else:
        message =str(obj)
    print(f'{str(datetime.datetime.now())} [  OK   ] {_format(message)}')

def warning(obj:str)->None:
    """
    It receives a message and prints it in the format of the module (with date and message type).
    if the method receives a str message it prints it as is. If it receives and objects, it prints
    the str(object).

    The methods prints the method as an "warning" type of message.

    Args:
        obj: A message or object to print. If str print as is, if object prints str(obj).
    """  
    if isinstance(obj,str):
        message = obj
    else:
        message =str(obj)
    print(f'{str(datetime.datetime.now())} [WARNING] {_format(message)}')

def debug(obj:str)->None:
    """
    It receives a message and prints it in the format of the module (with date and message type).
    if the method receives a str message it prints it as is. If it receives and objects, it prints
    the str(object).

    The methods prints the method as an "debug" type of message.

    Args:
        obj: A message or object to print. If str print as is, if object prints str(obj).
    """  
    if isinstance(obj,str):
        message = obj
    else:
        message =str(obj)
    print(f'{str(datetime.datetime.now())} [ DEBUG ] {_format(message)}')

def info(obj:str)->None:
    """
    It receives a message and prints it in the format of the module (with date and message type).
    if the method receives a str message it prints it as is. If it receives and objects, it prints
    the str(object).

    The methods prints the method as an "info" type of message.

    Args:
        obj: A message or object to print. If str print as is, if object prints str(obj).
    """  
    if isinstance(obj,str):
        message = obj
    else:
        message =str(obj)
    print(f'{str(datetime.datetime.now())} [ INFO  ] {_format(message)}')

def error(obj:str)->None:
    """
    It receives a message and prints it in the format of the module (with date and message type).
    if the method receives a str message it prints it as is. If it receives and objects, it prints
    the str(object).

    The methods prints the method as an "error" type of message.

    Args:
        obj: A message or object to print. If str print as is, if object prints str(obj).
    """  
    if isinstance(obj,str):
        message = obj
    else:
        message =str(obj)
    print(f'{str(datetime.datetime.now())} [ ERROR ] {_format(message)}')