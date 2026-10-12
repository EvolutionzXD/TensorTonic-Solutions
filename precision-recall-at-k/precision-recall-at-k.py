def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    if k == 0:
        return [0, 0]
        
    set_relevant = set(relevant)
    recommended = recommended[:k]
    corected = sum(1 for i in recommended if i in set_relevant)

    precision = corected/k
    recall = corected/len(relevant)

    return [precision, recall]
    pass