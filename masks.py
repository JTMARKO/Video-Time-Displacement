import numpy as np
from scipy.ndimage import gaussian_filter


def create_blurred_mask():
    height, width = 480, 640
    threshold = 480 - 260

    # Create a 2D grid of y coordinates
    y = np.arange(height).reshape(-1, 1)

    # Create initial mask: 1 where y <= threshold, 0 otherwise
    mask = (y >= threshold).astype(float)

    # Broadcast mask to full width
    mask = np.repeat(mask, width, axis=1)

    # Apply Gaussian blur to soften the edge (adjust sigma for blur amount)
    blurred_mask = gaussian_filter(mask, sigma=30)

    return blurred_mask


def create_bridge_blurred_mask():
    def line_eq(p1, p2, x):
        x1, y1 = p1
        x2, y2 = p2
        # Avoid division by zero for vertical lines
        if x2 == x1:
            return np.inf
        # y = mx + b
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return m * x + b

    y_cushion = -60
    # y_cushion = 0
    # Points defining the lines
    p1_line1 = (48 - y_cushion, 230)
    p2_line1 = (405 - y_cushion, 136)
    p1_line2 = (10 + y_cushion, 122)
    p2_line2 = (353 + y_cushion, 119)

    # Create a grid of (x, y) coordinates
    height, width = 480, 640
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)

    # Get y values on lines for every x
    y_on_line1 = line_eq(p1_line1, p2_line1, X)
    y_on_line2 = line_eq(p1_line2, p2_line2, X)

    # Left area of intersection means y between the two lines (because line2 is mostly horizontal near 119)
    # and inside the polygons formed by the intersection. We define mask as pixels where Y >= max(y_on_line1, y_on_line2)
    mask = ((Y <= np.maximum(y_on_line1, y_on_line2)) & (Y > y_on_line2)).astype(float)
    blurred_mask = gaussian_filter(mask, sigma=30)
    return blurred_mask


def create_blurred_mask_plane():
    height, width = 480, 640
    threshold = 210

    # Create a 2D grid of y coordinates
    y = np.arange(height).reshape(-1, 1)

    # Create initial mask: 1 where y <= threshold, 0 otherwise
    mask = (y <= threshold).astype(float)

    # Broadcast mask to full width
    mask = np.repeat(mask, width, axis=1)

    # Apply Gaussian blur to soften the edge (adjust sigma for blur amount)
    blurred_mask = gaussian_filter(mask, sigma=30)

    return blurred_mask


def create_train_blurred_mask():
    def line_eq(p1, p2, x):
        x1, y1 = p1
        x2, y2 = p2
        # Avoid division by zero for vertical lines
        if x2 == x1:
            return np.inf
        # y = mx + b
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return m * x + b

    y_cushion = -60
    # y_cushion = 0
    # Points defining the lines
    p1_line1 = (105 - y_cushion, 364)
    p2_line1 = (316 - y_cushion, 278)
    p1_line2 = (131 + y_cushion, 30)
    p2_line2 = (329 + y_cushion, 235)

    # Create a grid of (x, y) coordinates
    height, width = 480, 640
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)

    # Get y values on lines for every x
    y_on_line1 = line_eq(p1_line1, p2_line1, X)
    y_on_line2 = line_eq(p1_line2, p2_line2, X)

    # Left area of intersection means y between the two lines (because line2 is mostly horizontal near 119)
    # and inside the polygons formed by the intersection. We define mask as pixels where Y >= max(y_on_line1, y_on_line2)
    mask = ((Y <= np.maximum(y_on_line1, y_on_line2)) & (Y > y_on_line2)).astype(float)
    blurred_mask = gaussian_filter(mask, sigma=30)

    xx, yy = np.meshgrid(x, y)

    # Create the desired array with value x + 1 at each (x, y)
    result = xx + 1

    # print(result)

    return blurred_mask * (result * 0.04)


def create_blurred_mask_wall():
    height, width = 480, 640
    threshold = 210

    # Create a 2D grid of y coordinates
    x = np.arange(width).reshape(1, -1)

    # Create initial mask: 1 where y <= threshold, 0 otherwise
    mask = (x <= threshold).astype(float)

    # Broadcast mask to full width
    mask = np.repeat(mask, height, axis=0)

    # Apply Gaussian blur to soften the edge (adjust sigma for blur amount)
    blurred_mask = gaussian_filter(mask, sigma=30)

    return blurred_mask
