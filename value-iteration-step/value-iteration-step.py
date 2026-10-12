def value_iteration_step(values: list, transitions: list, rewards: list, gamma: float) -> list[float]:
    """
    Returns one updated floating-point value for every state.
    """

    new_value = []

    for transition, reward in zip(transitions, rewards):
        max_a = 0.0
        for choice, reward_ in zip(transition, reward):
            tmp = reward_+gamma*sum([choice[i]*values[i] for i in range(len(values))])
            max_a = max(max_a, tmp)
        new_value.append(max_a)
        
    return new_value
    pass