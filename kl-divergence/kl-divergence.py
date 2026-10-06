import numpy as np

def kl_divergence(p: list, q: list, eps: float = 1e-12) -> float:
    """
    Returns the divergence as a float.
    """
    p_arr = np.array(p, dtype = float)
    q_arr = np.array(q, dtype = float)

    
    D_KL = np.sum(p_arr[p_arr > 0]*np.log(p_arr[p_arr > 0]/q_arr[p_arr > 0]))

    return float(D_KL)
    pass