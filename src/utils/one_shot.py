import numpy as np


class OneShotClusterEstimator(object):
    """
    In a real-world scenario, we would probably ask for the user to provide
    (at least) one single instance of annotation so that we can know which
    cluster in the latent space we shall retrieve for segmentation.

    `OneShotClusterEstimator` provides a series of methods to approximate the perfect case.
    """

    def __init__(self, oneshot_prior: str = 'point') -> None:
        self.oneshot_prior = oneshot_prior

    def estimate_cluster(self, label: np.array = None, clusters: np.array = None) -> int:
        """
        Dimension of `label` and `clusters` are expected to be [H, W].
        """
        assert self.oneshot_prior in ['point', 'mask']
        if self.oneshot_prior == 'point':
            return self.__est_cluster_with_point(label, clusters)
        else:
            return self.__est_cluster_with_mask(label, clusters)

    def __est_cluster_with_point(self, label: np.array = None, clusters: np.array = None) -> int:
        """
        In this method, we use **one point** from the ground truth segmentation of **each image in the dataset**
        as the prior. We will use this point to estimate the desired cluster ID for each image.

        We will find the foreground point that is closest to the cluster centroid.

        Dimension of `label` and `clusters` are expected to be [H, W].

        NOTE: This is currently done such that a point needs to be provided for each image.
        May need to improve this!
        """
        assert len(label.shape) == 2
        assert label.shape == clusters.shape

        # Find the "middle point" of the foreground,
        # defined as the foreground point closest to the foreground centroid.
        foreground_xys = np.argwhere(label)  # shape: [2, num_points]
        centroid_xy = np.mean(foreground_xys, axis=0)
        distances = ((foreground_xys - centroid_xy)**2).sum(axis=1)
        middle_point_xy = foreground_xys[np.argmin(distances)]

        cluster_id = clusters[middle_point_xy[0], middle_point_xy[1]]
        return cluster_id

    def __est_cluster_with_mask(self, label: np.array = None, clusters: np.array = None) -> int:
        """
        In this method, we use the ground truth segmentation **mask** of **one image over the entire dataset**
        as the prior. We will use this mask to estimate the desired cluster ID for the entire dataset.

        Dimension of `label` and `clusters` are expected to be [H, W].

        NOTE: Assuming entries in `clusters` are non-negative integers.
        """
        assert len(label.shape) == 2
        assert label.shape == clusters.shape
        assert np.sum(clusters < 0) == 0
        # TODO: How to elegantly assert entries in `clusters` are integers?

        foreground_xy = np.argwhere(label)  # shape: [2, num_points]

        cluster_ids = []
        for xy in foreground_xy:
            cluster_ids.append(clusters[xy[0], xy[1]])

        # Find the most frequent number.
        cluster_id = np.bincount(cluster_ids).argmax()
        return cluster_id
