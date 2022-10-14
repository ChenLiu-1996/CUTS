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
        assert self.oneshot_prior in ['point', 'mask']

    def estimate_cluster(self, label: np.array = None, clusters: np.array = None) -> int:
        """
        Dimension of `label` and `clusters` are expected to be [H, W].
        """
        if self.oneshot_prior == 'point':
            return self.__est_cluster_with_point(label, clusters)
        elif self.oneshot_prior == 'mask':
            return self.__est_cluster_with_mask(label, clusters)
        else:
            raise NotImplementedError

    def __est_cluster_with_point(self, label: np.array = None, clusters: np.array = None) -> int:
        """
        In this method, we use one point from one ground truth segmentation
        as the prior. We will use this point to estimate the desired cluster ID.

        Since the exact centroid is not necessarily in the segmentation foreground,
        we would rather use the "middle point" in the sorted x-y coordinates.

        Dimension of `label` and `clusters` are expected to be [H, W].
        """
        assert len(label.shape) == 2
        assert label.shape == clusters.shape

        foreground_xy = np.argwhere(label)  # shape: [2, num_points]
        middle_point_xy = foreground_xy[len(foreground_xy)//2]

        cluster_id = clusters[middle_point_xy[0], middle_point_xy[1]]
        return cluster_id

    def __est_cluster_with_mask(self, label: np.array = None, clusters: np.array = None) -> int:
        """
        In this method, we use one ground truth segmentation mask
        as the prior. We will use this mask to estimate the desired cluster ID.

        Dimension of `label` and `clusters` are expected to be [H, W].

        Assuming entries in `clusters` are non-negative integers.
        """
        assert len(label.shape) == 2
        assert label.shape == clusters.shape

        foreground_xy = np.argwhere(label)  # shape: [2, num_points]

        cluster_ids = []
        for xy in foreground_xy:
            cluster_ids.append(clusters[xy[0], xy[1]])

        return np.bincount(cluster_ids).argmax()
