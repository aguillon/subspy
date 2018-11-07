from enum import Flag, auto
from functools import wraps
from copy import deepcopy

class Verbosity(Flag):
    NONE = 0
    COST_FUNCTION = auto()
    CENTERS = auto()
    MEMBERSHIPS = auto()
    ALL = COST_FUNCTION | CENTERS | MEMBERSHIPS

def verbose_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "verbose" in kwargs:
            verbose = kwargs["verbose"]
            if verbose is False or verbose is None:
                verbose = Verbosity.NONE
            if verbose is True:
                verbose = Verbosity.ALL
            kwargs["verbose"] = verbose
        return func(*args, **kwargs)
    return wrapper

__default_logger = set()

# TODO: does this impact performances?
def logging(flag, name, all_names = __default_logger):
    __default_logger.add(name)
    __default_logger.add("verbose")
    def logging_decorator(func):
        @wraps(func)
        def func_wrapper(*args, **kwargs):
            if "verbose" in kwargs and  kwargs["verbose"] & flag:
                assert name in kwargs, "Flag {} is set but {} is absent from **kwargs".format(flag, name)
                t = kwargs[name]
                for n in all_names:
                    if n in kwargs:
                        kwargs.pop(n)
                res = func(*args, **kwargs)
                t.append(deepcopy(res))
                return res
            else:
                for n in all_names:
                    if n in kwargs:
                        kwargs.pop(n)
                return func(*args, **kwargs)
        return func_wrapper
    return logging_decorator

