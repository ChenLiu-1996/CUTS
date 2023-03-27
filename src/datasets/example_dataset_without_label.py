from glob import glob
from typing import Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ExampleDatasetWithoutLabel(Dataset):

    def __init__(self,
                 base_path: str = '../../data/retina',
                 image_folder: str = 'image_with_GA'):

        # Load file paths.
        self.img_path = glob('%s/%s/*' % (base_path, image_folder))

        self.imgs = sorted([img for img in self.img_path])

        # Pre-load all the data to CPU. Saves time.
        # It works for this dataset since the dataset is not huge.
        self.data_image = []
        for img in self.imgs:
            self.data_image.append(np.array(Image.open(img)))
        self.data_image = np.array(self.data_image)

        self.data_image = (self.data_image / 255 * 2) - 1
        # channel last to channel first to comply with Torch.
        self.data_image = np.moveaxis(self.data_image, -1, 1)

    def __len__(self) -> int:
        return len(self.img_path)

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        image = self.data_image[idx]
        # Use an `np.nan` as a placeholder for the non-existent label.
        return image, np.nan

    def all_images(self) -> np.array:
        return self.data_image

    def num_image_channel(self) -> int:
        # [B, C, H, W]
        return self.data_image.shape[1]

    def num_classes(self) -> int:
        return None
