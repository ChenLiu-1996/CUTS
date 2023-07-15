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
from utils.metrics import dice_coeff


def get_baseline_predictions(img: np.array, method: str):
    img = (img * 255).astype(np.uint8)

    if len(img.shape) == 2:
        # (H, W) to (H, W, 1)
        img = img[..., None]

    if method == 'watershed':
        if img.shape[-1] == 1:
            # (H, W, 1) to (H, W, 3)
            img = np.repeat(img, 3, axis=-1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((3, 3),
                                                              dtype=int))

        _, threshed = cv2.threshold(gray, 128, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        distance = scipy.ndimage.distance_transform_edt(threshed)
        coords = skimage.feature.peak_local_max(distance,
                                                labels=threshed,
                                                threshold_rel=0.9)

        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = scipy.ndimage.label(mask)

        label_pred = skimage.segmentation.watershed(-distance,
                                                    markers,
                                                    mask=threshed)
    elif method == 'felzenszwalb':
        label_pred = skimage.segmentation.felzenszwalb(img, scale=1500)

    elif method == 'slic':
        if img.shape[-1] == 1:
            # (H, W, 1) to (H, W, 3)
            img = np.repeat(img, 3, axis=-1)
        label_pred = skimage.segmentation.slic(img, n_segments=10)

    else:
        raise Exception('cannot parse METHOD: {}'.format(method))

    return label_pred


def invert_if_better(label_, label_true):
    inverted_ = 1 - label_
    if dice_coeff(label_, label_true) > dice_coeff(inverted_, label_true):
        return label_
    else:
        return inverted_


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
                    image, method='watershed')
                if label_true.max() in [0, 1]:
                    label_pred_watershed = (
                        label_pred_watershed
                        > np.mean(label_pred_watershed)).astype(np.uint8)
                    label_pred_watershed = invert_if_better(
                        label_pred_watershed, label_true)

            elif method == 'felzenszwalb':
                label_pred_felzenszwalb = get_baseline_predictions(
                    image, method='felzenszwalb')
                if label_true.max() in [0, 1]:
                    label_pred_felzenszwalb = (
                        label_pred_felzenszwalb
                        > np.mean(label_pred_felzenszwalb)).astype(np.uint8)
                    label_pred_felzenszwalb = invert_if_better(
                        label_pred_felzenszwalb, label_true)

            elif method == 'slic':
                label_pred_slic = get_baseline_predictions(image,
                                                           method='slic')
                if label_true.max() in [0, 1]:
                    label_pred_slic = (label_pred_slic
                                       > np.mean(label_pred_slic)).astype(
                                           np.uint8)
                    label_pred_slic = invert_if_better(label_pred_slic,
                                                       label_true)

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
