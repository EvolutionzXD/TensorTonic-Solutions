import torch

def causal_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Returns the attention tensor with the same shape and dtype as q.
    """
    batch, q_head, sequence, head_witdth = q.shape
    _, kv_head, _, _ = k.shape

    G = q_head//kv_head
    q_v = q.view(batch, kv_head, G, sequence, head_witdth)
    M = torch.full((sequence, sequence), float('-inf'), device = q.device, dtype = q.dtype)
    M = torch.triu(M, diagonal = 1)
    
    score = torch.matmul(q_v, k.transpose(-1, -2).unsqueeze(2))
    score = score / (head_witdth**0.5)
    score = score + M;
    score = torch.matmul(torch.softmax(score, dim=-1),v.unsqueeze(2))
    score = score.view(batch, q_head, sequence, head_witdth)
    return score
    pass
