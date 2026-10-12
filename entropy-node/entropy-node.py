import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """

    labels = {}
    for node in y:
        labels[node] = labels.get(node, 0)+1

    H = 0.0
    for count in labels.values():
        p = count/len(y)
        
        if p == 0: 
            H += 0
        else:
            H += -p*np.log2(p)

    return H
    
    pass