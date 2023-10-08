from glob import glob
from typing import Tuple
import cv2

import numpy as np
from torch.utils.data import Dataset


class MouseBrain(Dataset):

    def __init__(self,
                 base_path: str = '../../data/mouse_brain',
                 out_shape: Tuple[int] = (128, 128)):
        self.out_shape = out_shape

        # This dataset contains a single numpy file.
        self.img_path = glob('%s/*' % (base_path))
        assert len(self.img_path) == 1
        self.img_path = self.img_path[0]

        data = np.load(self.img_path)
        self.image_names = [key for key in sorted(data.keys())]

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        data = np.load(self.img_path)
        image = data[self.image_names[idx]]

        epsilon = 1e-12
        image = 2 * (image - image.min() +
                     epsilon) / (image.max() - image.min() + epsilon) - 1

        assert len(image.shape) == 2

        # Resize to `out_shape`.
        resize_factor = np.array(self.out_shape) / image.shape
        dsize = np.int16(resize_factor.min() * np.float16(image.shape))
        image = cv2.resize(src=image,
                           dsize=dsize,
                           interpolation=cv2.INTER_CUBIC)

        # channel last to channel first to comply with Torch.
        image = image[None, ...]

        # Use an `np.nan` as a placeholder for the non-existent label.
        return image, np.nan

    def num_image_channel(self) -> int:
        # [B, C, H, W]
        return 1

    def num_classes(self) -> int:
        return None
