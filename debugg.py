
DEBUG_GLOBAL_FLAG = True

def debug(*args, **kwargs):
    """ Prints accordingly to the defined variable in this file """
    if DEBUG_GLOBAL_FLAG:
        print(*args, **kwargs)
