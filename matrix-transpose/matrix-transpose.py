import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    A_arr = np.array(A)
    return A_arr.T
    pass
