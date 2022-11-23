from typing import Tuple

import numpy as np
from torch.utils.data import Dataset


class BerkeleySegmentation(Dataset):
    def __init__(self,
                 base_path: str = '../../data/berkeley_image_segmentation/'):

        # Pre-load all the data to CPU. Saves time.
        with open('%s/prepared_data.npz' % base_path, 'rb') as f:
            npzfile = np.load(f)
            self.data_image = npzfile['all_images']
            self.data_label = npzfile['all_images_segmented']

        self.data_image = (self.data_image * 2) - 1
        # channel last to channel first to comply with Torch.
        self.data_image = np.moveaxis(self.data_image, -1, 1)
        self.data_label = self.data_label.squeeze(-1)

        # Sanity check.
        assert self.data_image.shape[0] == self.data_label.shape[0], \
            'BerkeleySegmentation Dataset have non-matching number of images (%s) and labels (%s)' \
            % (self.data_image.shape[0], self.data_label.shape[0])

    def __len__(self) -> int:
        return self.data_image.shape[0]

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        image = self.data_image[idx]
        label = self.data_label[idx]
        return image, label

    def num_image_channel(self) -> int:
        # [B, C, H, W]
        return self.data_image.shape[1]