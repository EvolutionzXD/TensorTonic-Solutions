import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """

    s_old=np.array(s);
    g_arr=np.array(g);
    w_old=np.array(w);

    s_new=beta*s_old+(1-beta)*g_arr**2
    w_new=w_old-lr/(np.sqrt(s_new+eps))*g_arr

    return (w_new, s_new)
    pass