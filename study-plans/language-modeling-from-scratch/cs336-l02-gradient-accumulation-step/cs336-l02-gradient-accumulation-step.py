import torch

def gradient_accumulation_step(
    param: torch.Tensor, microbatch_inputs: list[torch.Tensor],
    microbatch_targets: list[torch.Tensor], lr: float,
) -> dict:
    """
    Returns new_param and full_grad tensors in a dictionary.
    """
    sum_grad = torch.zeros_like(param)
    L = 0
    N = 0
    grad = 0
    
    for input, target in zip(microbatch_inputs, microbatch_targets):
        n_m = input.shape[0]
        mb_grad = 2*(input.T@(input@param-target))
        sum_grad = sum_grad+mb_grad
        N = N + n_m
        
    full_grad=sum_grad/N
    new_param = param - lr*full_grad

    return {"new_param": new_param, "full_grad": full_grad}
    pass
