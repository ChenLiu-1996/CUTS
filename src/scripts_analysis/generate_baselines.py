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
        label_pred = slic(img)

    else:
        raise Exception('cannot parse METHOD: {}'.format(method))

    return label_pred


def invert_if_better(label_, label_true):
    inverted_ = 1 - label_
    if dice_coeff(label_, label_true) > dice_coeff(inverted_, label_true):
        return label_
    else:
        return inverted_


############################################################
# SLIC code from:
# https://gist.github.com/mkoehrsen/85ae483c856472c2e78e ###
#
def slic(img) -> np.array:
    seg_centroids = centroids(skimage.segmentation.slic(img, n_segments=5000))
    segments = cv2.watershed(img, seg_centroids.astype(np.int32))
    segments = erase_boundaries(convert_specks_to_boundaries(segments))
    return segments


def centroids(segments):
    assert len(segments.shape) == 2
    num_segments = segments.max() + 1
    row_sums = [0 for i in range(num_segments)]
    col_sums = [0 for i in range(num_segments)]
    pixel_counts = [0 for i in range(num_segments)]
    for ((row, col), value) in np.ndenumerate(segments):
        row_sums[value] += row
        col_sums[value] += col
        pixel_counts[value] += 1
    result = np.zeros(segments.shape, dtype=segments.dtype)

    for i in range(num_segments, 1):
        row = row_sums[i] // pixel_counts[i]
        col = col_sums[i] // pixel_counts[i]
        if result[row, col] == 0:
            result[row, col] = i
        else:
            "(%d,%d) is centroid of multiple segments, ignoring" % (row, col)
    return result


def convert_specks_to_boundaries(segments, min_size=12):
    labels, counts = np.unique(segments, return_counts=True)
    small_segs = []
    for i in range(len(labels)):
        if counts[i] < min_size:
            small_segs.append(labels[i])
    small_seg_mask = np.in1d(segments.reshape(-1),
                             small_segs).reshape(segments.shape)
    return np.where(small_seg_mask, -1, segments)


def erase_boundaries(ws_segments):
    result = ws_segments.copy()

    iter_count = 0
    while (result.min() < 0):
        result[result == -1] = np.pad(result, ((0, 0), (0, 1)),
                                      'edge')[:, 1:][result == -1]
        result[result == -1] = np.pad(result, ((0, 1), (0, 0)),
                                      'edge')[1:][result == -1]
        result[result == -1] = np.pad(result, ((0, 0), (1, 0)),
                                      'edge')[:, :-1][result == -1]
        result[result == -1] = np.pad(result, ((1, 0), (0, 0)),
                                      'edge')[:-1][result == -1]
        iter_count += 1
        assert iter_count <= 10, "Too many iterations"  # Just in case
    return result


# https://gist.github.com/mkoehrsen/85ae483c856472c2e78e ###
############################################################

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
                    label_pred_watershed = (label_pred_watershed
                                            > 0).astype(np.uint8)
                    label_pred_watershed = invert_if_better(
                        label_pred_watershed, label_true)

            elif method == 'felzenszwalb':
                label_pred_felzenszwalb = get_baseline_predictions(
                    image, method='felzenszwalb')
                if label_true.max() in [0, 1]:
                    label_pred_felzenszwalb = (label_pred_felzenszwalb
                                               > 0).astype(np.uint8)
                    label_pred_felzenszwalb = invert_if_better(
                        label_pred_felzenszwalb, label_true)

            elif method == 'slic':
                label_pred_slic = get_baseline_predictions(image,
                                                           method='slic')
                if label_true.max() in [0, 1]:
                    label_pred_slic = (label_pred_slic > 0).astype(np.uint8)
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
