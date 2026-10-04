import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    xrr = np.array(x)
    y = 1/(1 + np.exp(-xrr))
    return y
    pass