import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    N = len(seqs)
    if max_len is None:
        L = max((len(seq) for seq in seqs), default = 0)
    else:
        L = max_len
    
    padding = np.full((N, L), pad_value, dtype=int)

    for i, seq in enumerate(seqs):
        M = min(len(seq), L)
        padding[i,0 : M] = seq[0 : M]

    return padding
    pass