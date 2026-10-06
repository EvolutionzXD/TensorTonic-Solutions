import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    PE = np.full((seq_len, d_model), 0, dtype = float)
    for pos in range(seq_len):
        for i in range(d_model):
            if i % 2 == 0:
                PE[pos, i] = np.sin(pos/(base**(i/d_model)))
            else:
                PE[pos, i] = np.cos(pos/(base**((i-1)/d_model)))
    return PE
    pass