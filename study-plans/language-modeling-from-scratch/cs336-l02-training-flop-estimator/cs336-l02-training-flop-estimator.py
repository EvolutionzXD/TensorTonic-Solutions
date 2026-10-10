def flop_estimator(matmuls: list[list[int]], attention_flops: int = 0) -> dict:
    """
    Returns integer forward_flops, backward_flops, and total_flops in a dictionary.
    """

    forward_flops=sum([2*a*b*c for a,b,c in matmuls])+attention_flops
    backward_flops=2*forward_flops
    total_flops=forward_flops+backward_flops

    return {
        "forward_flops": forward_flops,
        "backward_flops": backward_flops,
        "total_flops": total_flops
    }
    pass
