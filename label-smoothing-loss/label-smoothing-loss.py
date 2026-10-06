import math

def label_smoothing_loss(predictions: list, target: int, epsilon: float) -> float:
    """
    Returns cross-entropy loss for the smoothed target distribution.
    """
    # preds = np.array(predictions, dtype=float)
    # q = np.full(epsilon/len(preds))
    # q[target] += (1 - epsilon)

    # L = np.sum(q*np.ln(preds))

    # return float(L)

    p = [float(x) for x in predictions]
    q = [epsilon/len(p)]*len(p)
    q[target] += 1 - epsilon

    loss = 0.0

    for qi, pi in zip(q, p):
        loss += - qi*math.log(pi)

    return loss
    pass