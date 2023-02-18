from glob import glob
from typing import Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class Retina(Dataset):
    def __init__(self,
                 base_path: str = '../../data/retina',
                 image_folder: str = 'selected_128',
                 label_folder: str = 'label_128'):
        # Load file paths.
        self.img_path = glob('%s/%s/*' % (base_path, image_folder))
        self.label_path = glob('%s/%s/*' % (base_path, label_folder))

        # Find and keep the matching subset between imgs and labels.
        img_ids = set([img.split('/')[-1].split('.')[0][5:]
                       for img in self.img_path])
        label_ids = set([label.split('/')[-1].split('.')[0][5:]
                         for label in self.label_path])
        matching_ids = set([_id for _id in img_ids if _id in label_ids])

        self.imgs = sorted([img for img in self.img_path if img.split(
            '/')[-1].split('.')[0][5:] in matching_ids])
        self.labels = sorted([label for label in self.label_path if label.split(
            '/')[-1].split('.')[0][5:] in matching_ids])

        # Sanity check.
        assert len(self.imgs) == len(self.labels), \
            'Retina Dataset have non-matching number of images (%s) and labels (%s)' \
            % (len(self.imgs), len(self.labels))

        # Pre-load all the data to CPU. Saves time.
        # It works for this dataset since the dataset is not huge.
        self.data_image, self.data_label = [], []
        for img in self.imgs:
            self.data_image.append(np.array(Image.open(img)))
        self.data_image = np.array(self.data_image)

        self.data_image = (self.data_image / 255 * 2) - 1
        # channel last to channel first to comply with Torch.
        self.data_image = np.moveaxis(self.data_image, -1, 1)
        self.data_label = np.moveaxis(self.data_label, -1, 1)

        for label in self.labels:
            self.data_label.append(np.load(label))
        self.data_label = np.array(self.data_label)


    def __len__(self) -> int:
        return len(self.img_path)

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        image = self.data_image[idx]
        label = self.data_label[idx]
        return image, label

    def all_images(self) -> np.array:
        return self.data_image

    def num_image_channel(self) -> int:
        # [B, C, H, W]
        return self.data_image.shape[1]

    def num_classes(self) -> int:
        return len(np.unique(self.data_label)) - 1
