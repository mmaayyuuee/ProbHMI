

def this_is_wrapper(params:dict):
    def wrapper(cls):
        for key, value in params.items():
            setattr(cls, key, value)
        return cls
    return wrapper