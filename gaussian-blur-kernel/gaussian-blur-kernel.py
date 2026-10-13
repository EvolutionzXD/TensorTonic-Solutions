import math

def gaussian_kernel(size: int, sigma: float) -> list:
    """
    Returns a square two-dimensional list.
    """
    # Write code here
    kennel = []
    sum = 0.0
    for _x in range(size):
        _kenel = []
        for _y in range(size):
            x = _x + 1 - int(size/2+0.5)
            y = _y + 1 - int(size/2+0.5)
            G = math.exp(-(x**2 + y**2)/(2*sigma**2))
            sum += G
            _kenel.append(G)
        kennel.append(_kenel)

    kennel = [[val/sum for val in row] for row in kennel]
    return kennel
    pass