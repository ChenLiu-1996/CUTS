from typing import Tuple

import numpy as np
from glob import glob
import cv2
from torch.utils.data import Dataset


class CellHistology(Dataset):

    def __init__(self,
                 base_path: str = '../../data/cell_seg/',
                 out_shape: Tuple[int] = (128, 128)):

        self.base_path = base_path
        self.jpg_paths = sorted(glob('%s/%s' % (base_path, '*.jpg')))
        self.out_shape = out_shape

    def __len__(self) -> int:
        return len(self.jpg_paths)

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        image = cv2.imread(self.jpg_paths[idx], cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        assert len(image.shape) == 3
        assert image.shape[-1] == 3

        # Pseudo-label.
        # Significant BLUE channel.
        label = np.zeros_like(image[..., 0])
        label[image[..., 2] >= np.percentile(image[..., 2], 95)] = 1

        # Resize to `out_shape`. Be careful with labels.
        if not (self.out_shape[0] == image.shape[0]
                and self.out_shape[1] == image.shape[1]):
            resize_factor = np.array(self.out_shape) / image.shape[:2]
            dsize = np.int16(resize_factor.min() * np.float16(image.shape[:2]))
            image = cv2.resize(src=image,
                               dsize=dsize,
                               interpolation=cv2.INTER_CUBIC)
            label = cv2.resize(src=label,
                               dsize=dsize,
                               interpolation=cv2.INTER_NEAREST)

        # NOTE: Assuming binary label.
        assert len(np.unique(label)) <= 2
        label = label != np.unique(label)[0]

        # Rescale image.
        epsilon = 1e-12
        image = 2 * (image - image.min() +
                     epsilon) / (image.max() - image.min() + epsilon) - 1

        # Dimension fix.
        # Channel first to comply with Torch.
        assert image.shape[:2] == self.out_shape
        image = image.transpose(2, 0, 1)
        assert label.shape == self.out_shape
        label = label[None, :, :]

        return image, label

    def num_image_channel(self) -> int:
        # NOTE: temporary fix!
        return 3

    def num_classes(self) -> int:
        # NOTE: temporary fix!
        return 1
