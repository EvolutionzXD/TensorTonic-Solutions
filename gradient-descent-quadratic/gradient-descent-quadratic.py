def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Returns the final scalar x after the requested iterations.
    """
    x = x0
    for step in range(steps):
        fx = a*x**2+b*x+c
        fxgradient = 2*a*x + b
        x = x - fxgradient*lr

    return x
    pass