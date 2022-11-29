import numpy as np
from skimage import measure, morphology


def point_hint_seg(label_pred: np.array,
                   label_true: np.array,
                   dataset_name: str = None) -> np.array:
    '''
    Use a single point from ground truth as a hint provider.
    Find the desired label index by finding the "middle point" of the foreground,
    defined as the foreground point closest to the foreground centroid.

    To avoid sampling from the edge, we only look at the biggest connected component.
    '''

    if dataset_name in ['brain']:
        # For the brain dataset, erode the label and get the centroid of the biggest CC.
        found = False
        radius = 15
        while not found:
            disk = morphology.disk(radius=radius)
            try:
                eroded_label = morphology.erosion(label_true, footprint=disk)
                largestCC = largest_connected_component(eroded_label)
                found = True
            except:
                radius -= 2

        label_true = eroded_label

    largestCC = largest_connected_component(label_true)

    foreground_xys = np.argwhere(largestCC)  # shape: [2, num_points]
    centroid_xy = np.mean(foreground_xys, axis=0)
    distances = ((foreground_xys - centroid_xy)**2).sum(axis=1)
    middle_point_xy = foreground_xys[np.argmin(distances)]
    label_id = label_pred[middle_point_xy[0], middle_point_xy[1]]
    seg = label_pred == label_id

    return seg


def largest_connected_component(label: np.array) -> np.array:
    # Connected component code from https://stackoverflow.com/a/55110923/20242127.
    connected_components = measure.label(label)
    assert (connected_components.max() != 0)  # assume at least 1 CC
    largestCC = connected_components == np.argmax(
        np.bincount(connected_components.flat)[1:]) + 1
    return largestCC