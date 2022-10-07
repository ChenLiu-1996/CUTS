from typing import Iterable, Union

import numpy as np
import torch


class OneShotPrior(object):
    """
    In a real-world scenario, we would probably ask for the user to provide
    (at least) one single instance of annotation so that we can know which
    cluster in the latent space we shall retrieve for segmentation.

    `OneShotPrior` provides a series of methods to approximate the perfect case.
    """

    def __init__(self,
                 random_seed: int = 1,
                 label: Union[np.array, Iterable[np.array], torch.Tensor, Iterable[torch.Tensor]] = None):
        self.random_seed = random_seed
        self.label = label
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

    def mark_one_centroid(self):
        """
        In this method, we use the centroid of one random ground truth segmentation
        as the prior. We will use this centroid to find the cluster in the latent space.
        """
        label_mask = np.zeros_like(self.label)

        raise NotImplementedError

        for i in range(self.label.shape[0]):
            label_argwhere = np.argwhere(self.label[i])  # shape: [2, x]
            median_x_coord = np.median(label_argwhere[0, :]).reshape(
                (1, 1))  # shape: [1, 1]
            median_y_coord = np.median(label_argwhere[1, :]).reshape(
                (1, 1))  # shape: [1, 1]
            middle_pt = np.concatenate(
                [median_x_coord, median_y_coord], axis=0)  # shape: [2, 1]
            dist_to_middle_pt = ((label_argwhere - middle_pt)
                                 ** 2).sum(axis=0)  # shape: [x]
            argmin = np.argmin(dist_to_middle_pt)  # shape: [] (it's a scalar)
            # this sets the "middle pixel" of the ground truth to 1
            label_mask[i, label_argwhere[:, argmin]
                       [0], label_argwhere[:, argmin][1]] = 1

            label_mask = torch.from_numpy(label_mask).float()

    def mark_one_segmentation(self):
        raise NotImplementedError

    def find_cluster():
        raise NotImplementedError
