import argparse
import os
import sys
from glob import glob

import cv2
import numpy as np
import scipy
import scipy.ndimage
import skimage.feature
import skimage.segmentation
import yaml
from tqdm import tqdm

sys.path.append('../')
from utils.attribute_hashmap import AttributeHashmap
from utils.parse import parse_settings
from utils.seed import seed_everything


def get_baseline_predictions(img: np.array, method: str, dataset_name:str = None):
    img = (img * 255).astype(np.uint8)

    if len(img.shape) == 2:
        # (H, W) to (H, W, 1)
        img = img[..., None]

    if method == 'watershed':
        '''
        NOTE: This is written as "watershed", but in reality it's just Otsu's thresholding.
        '''
        if img.shape[-1] == 1:
            # (H, W, 1) to (H, W, 3)
            img = np.repeat(img, 3, axis=-1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, threshed = cv2.threshold(gray, 128, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        label_pred = gray > threshed

    elif method == 'felzenszwalb':
        param_map = {
            'retina': (200, 1.0),
            'brain_ventricles': (1000, 2.0),
            'brain_tumor': (100, 0.8),
            'default': (100, 0.8)
        }
        if dataset_name in param_map.keys():
            params = param_map[dataset_name]
        else:
            params = param_map['default']
        label_pred = skimage.segmentation.felzenszwalb(img, scale=params[0], sigma=params[1], min_size=10)

    elif method == 'slic':
        if img.shape[-1] == 1:
            # (H, W, 1) to (H, W, 3)
            img = np.repeat(img, 3, axis=-1)
        label_pred = skimage.segmentation.slic(img, n_segments=10)

    else:
        raise Exception('cannot parse METHOD: {}'.format(method))

    return label_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Path to config yaml file.',
                        required=True)
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config.config_file_name = args.config
    config = parse_settings(config, log_settings=False)
    dataset_name = config.dataset_name

    files_folder = '%s/%s' % (config.output_save_path, 'numpy_files')
    np_files_path = sorted(glob('%s/%s' % (files_folder, '*.npz')))
    save_path_numpy = '%s/%s' % (config.output_save_path,
                                 'numpy_files_seg_baselines')
    os.makedirs(save_path_numpy, exist_ok=True)

    methods = ['random', 'watershed', 'felzenszwalb', 'slic']

    seed_everything(config.random_seed)

    for image_idx in tqdm(range(len(np_files_path))):
        numpy_array = np.load(np_files_path[image_idx])
        image = numpy_array['image']
        label_true = numpy_array['label']

        image = (image + 1) / 2

        H, W = label_true.shape[:2]
        if len(label_true.shape) == 3:
            assert label_true.shape[-1] == 1
            label_true = label_true.squeeze(-1)

        label_pred_random = label_pred_watershed = label_pred_felzenszwalb = label_pred_slic = \
            np.full(label_true.shape, np.nan)

        for method in methods:
            if method == 'random':
                if np.isnan(label_true.max()):
                    label_pred_random = np.random.randint(
                        0, 2, label_true.shape)
                else:
                    label_pred_random = np.random.randint(
                        0,
                        label_true.max() + 1, label_true.shape)

            elif method == 'watershed':
                label_pred_watershed = get_baseline_predictions(
                    image, method='watershed').astype(np.uint8)

            elif method == 'felzenszwalb':
                label_pred_felzenszwalb = get_baseline_predictions(
                    image, method='felzenszwalb', dataset_name=dataset_name).astype(np.uint8)
            elif method == 'slic':
                label_pred_slic = get_baseline_predictions(
                    image, method='slic').astype(np.uint8)

        with open(
                '%s/%s' %
            (save_path_numpy, 'sample_%s.npz' % str(image_idx).zfill(5)),
                'wb+') as f:
            np.savez(f,
                     image=image,
                     label=label_true,
                     label_random=label_pred_random,
                     label_watershed=label_pred_watershed,
                     label_felzenszwalb=label_pred_felzenszwalb,
                     label_slic=label_pred_slic)
