import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    pos = np.arange(seq_len)[:,np.newaxis]
    i = np.arange(d_model)
    base_arr = (base**((i-(i%2))/d_model)) 

    angle = pos/base_arr
    
    PE = np.full((seq_len, d_model), 0, dtype = float)
    PE[:, 0::2]=np.sin(angle[:, 0::2])
    PE[:, 1::2]=np.cos(angle[:, 1::2])

    return PE
    pass