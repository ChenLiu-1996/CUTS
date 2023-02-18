from typing import Tuple

import numpy as np
from torch.utils.data import Dataset


class PolyP(Dataset):
    def __init__(self,
                 base_path: str = '../../data/polyp/'):

        # Pre-load all the data to CPU. Saves time.
        with open('%s/selected_imgs_resized.npz' % base_path, 'rb') as f:
            npzfile = np.load(f)
            self.data_image = npzfile['imgs']
            self.data_label = npzfile['labels'][:, :, :, 0]

        self.data_image = (self.data_image * 2) - 1
        self.data_label = np.where(self.data_label > .5, 1, 0)
        # channel last to channel first to comply with Torch.
        self.data_image = np.moveaxis(self.data_image, -1, 1)
        self.data_label = np.moveaxis(self.data_label, -1, 1)

        # Sanity check.
        assert self.data_image.shape[0] == self.data_label.shape[0], \
            'DiabeticMacularEdema Dataset have non-matching number of images (%s) and labels (%s)' \
            % (self.data_image.shape[0], self.data_label.shape[0])

    def __len__(self) -> int:
        return len(self.img_path)

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        image = self.data_image[idx]
        label = self.data_label[idx]
        return image, label
