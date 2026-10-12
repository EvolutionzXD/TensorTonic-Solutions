import numpy as np

def expected_value_discrete(x: list, p: list) -> float:
    """
    Returns the expected value as a Python float.
    """
    x = np.array(x, dtype = float)
    p = np.array(p, dtype = float)

    return np.sum(x*p)
    pass