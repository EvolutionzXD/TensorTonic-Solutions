def memory_accountant(
    param_shapes: list[list[int]], param_bytes_per_element: int,
    grad_bytes_per_element: int, activation_shapes: list[list[int]],
    activation_bytes_per_element: int, optimizer: str,
    optimizer_bytes_per_element: int,
) -> dict:
    """
    Returns integer byte counts for parameters, gradients, activations, optimizer_state, and total.
    """
    param_element=0 
    activation_element=0
    
    for param_shape in param_shapes:
        param_element = param_element + math.prod(param_shape)
    
    for activation_shape in activation_shapes:
        activation_element = activation_element + math.prod(activation_shape)
    
    parameters = param_element*param_bytes_per_element
    gradients = param_element*grad_bytes_per_element
    activations = activation_element*activation_bytes_per_element
    optimizers = 0
    if optimizer == "sgd":
        optimizers = 0
    elif optimizer == "adagrad":
        optimizers = param_element * optimizer_bytes_per_element
    elif optimizer == "adam":
        optimizers = 2*param_element * optimizer_bytes_per_element
    total = parameters+gradients+activations+optimizers
    return { "parameters": parameters,"gradients":gradients,"activations":activations,"optimizer_state":optimizers,"total":total
    }
    pass
