def top_k_recommendations(scores: list, rated_indices: list, k: int) -> list:
    """
    Returns the highest-scoring unrated item indices.
    """
    # Write code here

    rated_set = set(rated_indices)
    unrated_indices = [
        i for i in range(len(scores)) if i not in rated_set
    ]

    unrated_indices.sort(key=lambda x: -scores[x])

    return unrated_indices[:k]
    pass