import numpy as np
from skimage import measure, morphology


def label_hint_seg(label_pred: np.array, label_true: np.array) -> np.array:
    '''
    Use every point from the ground truth as a hint provider.
    Find the desired label index by finding the most frequent label index
    corresponding to the foreground.

    This is effectively estimating the most likely sampled label index,
    if we randomly sample a point from the ground truth label.
    So it is in fact less arbitrary than point_hint_seg!
    '''

    foreground_xys = np.argwhere(label_true)
    if len(foreground_xys) > 0:
        label_counts = {}
        for (x, y) in foreground_xys:
            label_id = label_pred[x, y]
            if label_id not in label_counts.keys():
                label_counts[label_id] = 1
            else:
                label_counts[label_id] += 1

        label_id, max_count = 0, 0
        for i in label_counts.keys():
            count = label_counts[i]
            if count > max_count:
                max_count = count
                label_id = i

        seg = label_pred == label_id

    else:
        seg = np.zeros(label_pred.shape)

    return seg


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