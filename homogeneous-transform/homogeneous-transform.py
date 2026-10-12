import numpy as np

def apply_homogeneous_transform(T: list, points: list) -> np.ndarray:
    """
    Returns transformed points with shape (3,) or (N, 3).
    """
    
    points = np.array(points)

    if points.ndim == 1:
        points = points.reshape(1, -1)
    one = np.ones((len(points), 1))
    T = np.array(T)
    
    points = np.hstack([points, one])

    new_points = (T@points.T).T

    if len(new_points) == 1:
        return new_points[0][:3]
    return new_points[:, :3]
    pass