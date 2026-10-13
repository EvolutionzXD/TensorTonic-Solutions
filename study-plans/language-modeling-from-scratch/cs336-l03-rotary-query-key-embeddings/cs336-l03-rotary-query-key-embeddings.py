import torch

def rotary_embed(
    q: torch.Tensor, k: torch.Tensor,
    positions: torch.Tensor, inv_freq: torch.Tensor,
) -> dict:
    """
    Returns q_rotated and k_rotated tensors in a dictionary.
    """
    if positions.ndim == 2:
        positions = positions.unsqueeze(1)
    out = torch.einsum("...i,d->...id", positions, inv_freq)
    cos = torch.cos(out)
    sin = torch.sin(out)
    q_new = q.clone()
    q_new[...,0::2] = q[...,0::2]*cos - q[...,1::2]*sin
    q_new[...,1::2] = q[...,0::2]*sin + q[...,1::2]*cos

    k_new = k.clone()
    k_new[...,0::2] = k[...,0::2]*cos - k[...,1::2]*sin
    k_new[...,1::2] = k[...,0::2]*sin + k[...,1::2]*cos

    return {
        "q_rotated": q_new,
        "k_rotated": k_new
    }
    pass
