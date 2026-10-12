import torch

def rmsnorm(x: torch.Tensor, g: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Returns the RMS-normalized tensor with the same shape and dtype as x.
    """
    d = x.shape[-1]
    variance = torch.mean(x**2, dim = -1, keepdim = True)
    new_x = x/torch.sqrt(variance+ epsilon)*g

    return new_x
    
    pass
