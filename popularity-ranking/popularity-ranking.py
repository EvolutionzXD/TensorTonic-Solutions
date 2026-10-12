def popularity_ranking(items: list, min_votes: int, global_mean: float) -> list:
    """
    Returns the weighted rating for every item.
    """
    score = [(items[i][1]*items[i][0]+min_votes*global_mean)/(items[i][1]+min_votes) for i in range(len(items))]

    return score
    pass