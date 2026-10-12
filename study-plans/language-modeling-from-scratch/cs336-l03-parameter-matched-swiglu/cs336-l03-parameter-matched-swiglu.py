import torch

def parameter_matched_swiglu(
    x: torch.Tensor, w_g: torch.Tensor, w_v: torch.Tensor,
    w_o: torch.Tensor, base_params: int,
) -> dict:
    """
    Returns output, hidden_width, and parameter_count in a dictionary.
    """

    D = x.shape[-1]
    H = w_g.shape[-1]
    
    h = min(H, max(1, int(base_params/(3*D) +0.5)))

    def SiLU(z):
        return z/(1+torch.exp(-z))

    w_g_sub = w_g[:, :h]
    w_v_sub = w_v[:, :h]
    w_o_sub = w_o[:h, :]

    gate = SiLU(x @ w_g_sub)
    value = x @ w_v_sub
    
    output = (gate*value)@w_o_sub
    parmeter = 3*D*h

    return {
        "output": output,
        "hidden_width": h,
        "parameter_count": parmeter
    }
    pass
