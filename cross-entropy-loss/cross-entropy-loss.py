import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Returns the mean multiclass cross-entropy loss as a Python float.
    """
    # Write code here
    arr_true = np.array(y_true)
    arr_pred = np.array(y_pred)

    correct_log = arr_pred[np.arange(len(arr_true)), arr_true]

    loss = -np.mean(np.log(correct_log))

    return loss
    pass