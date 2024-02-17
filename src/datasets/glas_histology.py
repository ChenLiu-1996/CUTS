from glob import glob
from typing import Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset
import cv2


class GlasHistology(Dataset):

    def __init__(self,
                 base_path: str = '../../data/glas_histology/',
                 out_shape: Tuple[int] = (128, 128)):

        self.base_path = base_path
        self.label_paths = sorted(glob('%s/%s' % (base_path, '*_anno.bmp')))
        self.image_paths = [
            _str.replace('_anno', '') for _str in self.label_paths
        ]
        self.out_shape = out_shape

        # Quick pass on the entire dataset to check dimension.
        self._num_image_channel = None
        self._num_classes = None
        unique_labels = set()
        for i in range(len(self.image_paths)):
            image = cv2.cvtColor(cv2.imread(self.image_paths[i]),
                                 cv2.COLOR_BGR2RGB)
            label = cv2.cvtColor(cv2.imread(self.label_paths[i]),
                                 cv2.COLOR_BGR2RGB)
            if i > 0:
                assert self._num_image_channel == image.shape[2], \
                'Images in `GlasHistology` have inconsistent channels'
            else:
                self._num_image_channel = image.shape[2]
                # `|` is set union.
                unique_labels = unique_labels | set(np.unique(label))
        self._num_classes = len(unique_labels) - 1

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        image = cv2.cvtColor(cv2.imread(self.image_paths[idx]),
                             cv2.COLOR_BGR2RGB)
        label = cv2.cvtColor(cv2.imread(self.label_paths[idx]),
                             cv2.COLOR_BGR2RGB)

        # Make sure `label` has shape [H, W] instead of [H, W, C].
        if len(label.shape) == 3:
            assert (label[:, :, 0]
                    == label[:, :, 1]).all() and (label[:, :, 0]
                                                  == label[:, :, 2]).all()
        label = label[:, :, 0]

        # Resize to `out_shape`. Be careful with labels.
        assert image.shape[:2] == label.shape
        assert len(image.shape) == 3
        resize_factor = np.array(self.out_shape) / image.shape[:2]
        dsize = np.int16(resize_factor.min() * np.float16(image.shape[:2]))
        image = cv2.resize(src=image,
                           dsize=dsize,
                           interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(src=label,
                           dsize=dsize,
                           interpolation=cv2.INTER_NEAREST)

        # NOTE: Assuming vaccuum/background pixels are zero for images and labels!
        image = crop_or_pad(image,
                            in_shape=image.shape,
                            out_shape=(*self.out_shape, image.shape[2]))
        label = crop_or_pad(label,
                            in_shape=label.shape,
                            out_shape=self.out_shape)

        # Rescale image.
        epsilon = 1e-12
        image = 2 * (image - image.min() +
                     epsilon) / (image.max() - image.min() + epsilon) - 1

        # Dimension fix.
        # Channel first to comply with Torch.
        assert image.shape[:2] == self.out_shape
        assert label.shape == self.out_shape
        image = np.moveaxis(image, -1, 0)
        label = label[None, :, :]
        label = label.astype(np.uint8)

        return image, label

    def num_image_channel(self) -> int:
        return self._num_image_channel

    def num_classes(self) -> int:
        return self._num_classes


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
