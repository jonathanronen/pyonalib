"""
Model with functions for dealing with functions
"""
from functools import partial

def execute_serial(data, functions):
    """
    Execute the functional composition of `functions` on data.
    The first function will be run on data, the second function on the output of the first, and the final return value is returned.
    """
    ret = data
    for f in functions:
        ret = f(ret)
    return ret

def compose_serial(*functions):
    """
    Returns a callable that is the functional composition of functions.
    """
    return partial(execute_serial, functions=functions)

def execute_many(data, functions):
    """
    Call all the functions with the same data.
    No return value.
    """
    for f in functions:
        f(data)

def compose_parallel(*functions):
    """
    Returns a callable that executes all `functions` on same, single argument data.
    'Parallel' here does not mean parallel execution, only that all the functions use the same input data.
    """
    return partial(execute_many, functions=functions)
