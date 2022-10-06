from glob import glob

import numpy as np
# from data import ContrastiveViewGenerator
from PIL import Image
from torch.utils.data import Dataset


class Retina(Dataset):
    def __init__(self,
                 #  contrastive: bool = False,
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
        self.data_image, self.data_label = [], []
        for img in self.imgs:
            self.data_image.append(np.array(Image.open(img)))
        self.data_image = np.array(self.data_image)

        # NOTE: Made this modification
        # self.data_image = (self.data_image / 255 * 2) - 1
        self.data_image = (self.data_image / 255)
        # channel last to channel first to comply with Torch.
        self.data_image = np.moveaxis(self.data_image, -1, 1)

        for label in self.labels:
            self.data_label.append(np.load(label))
        self.data_label = np.array(self.data_label)

        # self.contrastive_gen = None
        # if contrastive:
        #     self.contrastive_gen = ContrastiveViewGenerator()

    def __len__(self) -> int:
        return len(self.img_path)

    def __getitem__(self, idx) -> tuple[np.array, np.array]:
        print(self.data_image.shape)
        image = self.data_image[idx]
        label = self.data_label[idx]
        # if self.contrastive_gen is not None:
        #     image = self.contrastive_gen(image)
        return image, label

    def all_images(self) -> np.array:
        return self.data_image
