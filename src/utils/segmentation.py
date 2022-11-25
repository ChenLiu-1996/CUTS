import numpy as np


def point_hint_seg(label_pred: np.array, label_true: np.array) -> np.array:
    '''
    Use a single point from ground truth as a hint provider.
    Find the desired label index by finding the "middle point" of the foreground,
    defined as the foreground point closest to the foreground centroid.
    '''
    foreground_xys = np.argwhere(label_true)  # shape: [2, num_points]
    centroid_xy = np.mean(foreground_xys, axis=0)
    distances = ((foreground_xys - centroid_xy)**2).sum(axis=1)
    middle_point_xy = foreground_xys[np.argmin(distances)]
    label_id = label_pred[middle_point_xy[0], middle_point_xy[1]]
    seg = label_pred == label_id

    return seg