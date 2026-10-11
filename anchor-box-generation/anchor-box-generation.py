def generate_anchors(feature_size: int, image_size: float, scales: list[float], aspect_ratios: list[float]) -> list[list[float]]:
    """
    Returns a list of [x1, y1, x2, y2] anchor boxes.
    """

    stride = image_size/feature_size
    box = []
    
    for i in range(feature_size):
        for j in range(feature_size):
            cx = (j+0.5)*stride
            cy = (i+0.5)*stride

            for scale in scales:
                for aspect_ratio in aspect_ratios:
                    w = scale*aspect_ratio**0.5
                    h = scale*aspect_ratio**-0.5
                    box.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2])  
    return box
    pass