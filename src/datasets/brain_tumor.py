from glob import glob
from typing import Tuple

import cv2
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset


class BrainTumor(Dataset):

    def __init__(self,
                 base_path: str = '../../data/brain_tumor/',
                 out_shape: Tuple[int] = (128, 128)):

        self.base_path = base_path
        self.label_paths = sorted(glob('%s/%s' % (base_path, '*_seg.nii.gz')))
        self.image_paths = [
            _str.replace('_seg', '') for _str in self.label_paths
        ]
        self.out_shape = out_shape

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        image_nii = nib.load(self.image_paths[idx])
        label_nii = nib.load(self.label_paths[idx])
        image = image_nii.get_fdata()
        label = label_nii.get_fdata()

        if len(image.shape) == 3:
            assert image.shape[-1] == 1
            image = image.squeeze(-1)
        if len(label.shape) == 3:
            assert label.shape[-1] == 1
            label = label.squeeze(-1)

        # Resize to 128x128. Be careful with labels.
        assert image.shape == label.shape
        assert len(image.shape) in [2, 3]
        if len(image.shape) == 3:
            assert image.shape[2] == 1
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

        # NOTE: Assuming vaccuum/background pixels are zero for images and labels!
        image = crop_or_pad(image,
                            in_shape=image.shape,
                            out_shape=self.out_shape)
        label = crop_or_pad(label,
                            in_shape=label.shape,
                            out_shape=self.out_shape)

        # Rescale image and label.
        epsilon = 1e-12
        image = 2 * (image - image.min() +
                     epsilon) / (image.max() - image.min() + epsilon) - 1

        # Dimension fix.
        # Channel first to comply with Torch.
        assert image.shape == self.out_shape
        assert label.shape == self.out_shape
        image = image[None, :, :]
        label = label[None, :, :]

        return image, label

    def num_image_channel(self) -> int:
        # NOTE: temporary fix!
        return 1

    def num_classes(self) -> int:
        # NOTE: temporary fix!
        return 1


def crop_or_pad(in_image: np.array,
                in_shape: Tuple[int],
                out_shape: Tuple[int],
                pad_value: float = 0) -> np.array:
    assert len(in_shape) == len(out_shape)
    D = len(in_shape)

    out_shape_min = [
        int(np.floor((out_shape[i] - in_shape[i]) /
                     2)) if out_shape[i] >= in_shape[i] else 0
        for i in range(D)
    ]
    out_shape_max = [
        out_shape_min[i] + in_shape[i] if out_shape[i] >= in_shape[i] else None
        for i in range(D)
    ]

    in_shape_min = [
        0 if out_shape[i] >= in_shape[i] else int(
            np.floor((in_shape[i] - out_shape[i]) / 2)) for i in range(D)
    ]
    in_shape_max = [
        None if out_shape[i] >= in_shape[i] else in_shape_min[i] + out_shape[i]
        for i in range(D)
    ]

    in_slicer = tuple(
        [slice(i, j) for (i, j) in zip(in_shape_min, in_shape_max)])
    out_slicer = tuple(
        [slice(i, j) for (i, j) in zip(out_shape_min, out_shape_max)])

    out_image = np.ones(out_shape) * pad_value
    out_image[out_slicer] = in_image[in_slicer]

    return out_image
