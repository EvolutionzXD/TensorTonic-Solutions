import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    m_np = np.array(m)
    v_np = np.array(v)
    g = np.array(grad)
    
    m_new = beta1*m_np + (1 - beta1)*g
    v_new = beta2*v_np + (1 - beta2)*g*g

    m_hat = m_new/(1 - beta1**t)
    v_hat = v_new/(1 - beta2**t)

    p = np.array(param)
    p = p - lr * m_hat/(np.sqrt(v_hat) + eps)

    return (p, m_new, v_new)
    pass