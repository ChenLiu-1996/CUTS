import argparse
import os
import sys
import warnings
from glob import glob
from tqdm import tqdm
from typing import Dict

import numpy as np
import scprep
import cv2
import yaml
from matplotlib import pyplot as plt
import multiscale_phate
from sklearn.preprocessing import normalize

sys.path.append('../')
from utils.attribute_hashmap import AttributeHashmap
from utils.diffusion_condensation import continuous_renumber, get_persistent_structures
from utils.parse import parse_settings
from utils.segmentation import label_hint_seg

warnings.filterwarnings("ignore")


def grayscale_3channel(image: np.array) -> np.array:
    if len(image.shape) == 2:
        image = image[..., None]
    assert len(image.shape) == 3
    if image.shape[-1] == 1:
        image = np.repeat(image, repeats=3, axis=-1)
    return image


def pop_blue_channel(label: np.array) -> np.array:
    assert label.min() >= 0 and label.max() <= 1
    if len(label.shape) == 3:
        label = label.squeeze(-1)
    assert len(label.shape) == 2

    # (R, G, B, A) where A := alpha (opacity).
    output = np.zeros((*label.shape, 4))
    output[..., 2] = label

    # Transparent at background. Opaque at foreground.
    output[label > 0, 3] = 0.6
    return np.uint8(255 * output)


def find_nearest_idx(arr: np.array, num: float) -> int:
    return np.abs(arr - num).argmin()


def plot_comparison(fig: plt.figure, num_samples: int, sample_idx: int,
                    data_hashmap: Dict, image_grayscale: bool,
                    label_binary: bool):
    # 1 row, 12 columns.
    # 1-st row are the images, labels, segmentations.
    # 2-nd row are the msphate plots if applicable.

    H, W = data_hashmap['label_true'].shape[:2]

    label_keys = [
        'label_true',
        'seg_kmeans' if label_binary else 'label_kmeans',
        'seg_persistent' if label_binary else 'label_persistent',
        'label_random',
        'label_watershed',
        'label_felzenszwalb',
        'label_slic',
        'label_dfc',
        'label_stego',
        'label_sam',
        'label_supervised_unet',
        'label_supervised_nnunet',
    ]
    num_labels = len(label_keys)
    num_cols = num_labels + 1

    ##### 1-st row!
    ax = fig.add_subplot(num_samples, num_cols, 1 + num_cols * sample_idx)
    ax.imshow(data_hashmap['image'], cmap='gray' if image_grayscale else None)
    ax.set_axis_off()

    for (figure_idx, key) in zip(range(2, 2 + num_labels), label_keys):
        ax = fig.add_subplot(num_samples, num_cols,
                             figure_idx + num_cols * sample_idx)
        ax.imshow(data_hashmap[key], cmap='gray' if label_binary else 'tab20')
        ax.set_axis_off()

    return fig


def plot_overlaid_comparison(fig: plt.figure,
                             num_samples: int,
                             sample_idx: int,
                             data_hashmap: Dict,
                             image_grayscale: bool,
                             pred_color: str = 'blue'):
    # 1 row, 12 columns.
    # 1-st row are the images, labels, segmentations.
    # 2-nd row are the msphate plots if applicable.
    H, W = data_hashmap['label_true'].shape[:2]

    label_keys = [
        'seg_kmeans',
        'seg_persistent',
        'label_random',
        'label_watershed',
        'label_felzenszwalb',
        'label_slic',
        'label_dfc',
        'label_stego',
        'label_sam',
        'label_supervised_unet',
        'label_supervised_nnunet',
    ]
    num_labels = len(label_keys)
    num_cols = num_labels + 1

    true_color = (0, 255, 0)
    if pred_color == 'blue':
        pred_color = (0, 0, 255)
    elif pred_color == 'red':
        pred_color = (255, 0, 0)

    ##### 1-st row!
    ax = fig.add_subplot(num_samples, num_cols, 1 + num_cols * sample_idx)

    image = np.uint8(255 * data_hashmap['image'].copy())
    if image_grayscale:
        image = grayscale_3channel(image)
    label_true = data_hashmap['label_true']

    # Contour of ground truth label
    true_contours, _hierarchy = cv2.findContours(np.uint8(label_true),
                                                 cv2.RETR_TREE,
                                                 cv2.CHAIN_APPROX_NONE)
    for contour in true_contours:
        cv2.drawContours(image, contour, -1, true_color, 1)
    ax.imshow(image)
    ax.set_axis_off()

    for (figure_idx, key) in zip(range(2, 2 + num_labels), label_keys):
        image = np.uint8(255 * data_hashmap['image'].copy())
        if image_grayscale:
            image = grayscale_3channel(image)

        for contour in true_contours:
            cv2.drawContours(image, contour, -1, true_color, 4)
        pred_contours, _hierarchy = cv2.findContours(
            np.uint8(data_hashmap[key]), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for contour in pred_contours:
            if key == 'label_random':
                weight = 1
            else:
                weight = 4
            cv2.drawContours(image, contour, -1, pred_color, weight)
        ax = fig.add_subplot(num_samples, num_cols,
                             figure_idx + num_cols * sample_idx)
        ax.imshow(image)
        ax.set_axis_off()

    return fig


def plot_results(fig: plt.figure,
                 num_samples: int,
                 sample_idx: int,
                 data_hashmap: Dict,
                 image_grayscale: bool,
                 num_granularities: int = 10):
    # 2 rows, 12 columns.
    # 1-st row are the images, kmeans, and segmentations.
    # 2-nd row are the recon, persistent, and msphate plots if applicable.

    assert num_granularities >= 0
    num_cols = num_granularities + 2

    H, W = data_hashmap['label_true'].shape[:2]
    latent = data_hashmap['latent']

    msphate_op = multiscale_phate.Multiscale_PHATE(knn=50,
                                                   landmarks=100,
                                                   random_state=0,
                                                   n_jobs=1)
    msphate_op.fit(normalize(latent, axis=1))
    levels = msphate_op.levels
    assert levels[0] == 0
    levels = levels[1:]  # Ignore finest resolution of all-distinct labels.
    msphate_granularities = [len(msphate_op.NxTs) + lvl for lvl in levels]
    diffusion_granularities = np.arange(len(data_hashmap['labels_diffusion']))

    if num_granularities > 0:
        diffusion_idx_selected = [
            find_nearest_idx(diffusion_granularities, num) for num in
            np.linspace(diffusion_granularities[0],
                        diffusion_granularities[-1], num_granularities)
        ]
        msphate_idx_selected = [
            find_nearest_idx(msphate_granularities, num) for num in
            np.linspace(msphate_granularities[0], msphate_granularities[-1],
                        num_granularities)
        ]

    ##### 1st row!
    ax = fig.add_subplot(2 * num_samples, num_cols,
                         1 + 2 * num_cols * sample_idx)
    ax.imshow(data_hashmap['image'], cmap='gray' if image_grayscale else None)
    ax.set_axis_off()

    ax = fig.add_subplot(2 * num_samples, num_cols,
                         2 + 2 * num_cols * sample_idx)
    cmap = plt.get_cmap('tab20', len(np.unique(data_hashmap['label_kmeans'])))
    ax.imshow(data_hashmap['label_kmeans'], cmap=cmap)
    ax.set_axis_off()

    for i in range(num_granularities):
        ax = fig.add_subplot(2 * num_samples, num_cols,
                             3 + i + 2 * num_cols * sample_idx)
        __label = data_hashmap['labels_diffusion'][diffusion_idx_selected[i]]
        __label = __label.reshape((H, W))
        ax.imshow(continuous_renumber(__label), cmap=cmap)
        ax.set_axis_off()

    ##### 2-nd row!
    ax = fig.add_subplot(2 * num_samples, num_cols,
                         1 + num_cols + 2 * num_cols * sample_idx)
    ax.imshow(data_hashmap['recon'], cmap='gray' if image_grayscale else None)
    ax.set_axis_off()

    ax = fig.add_subplot(2 * num_samples, num_cols,
                         2 + num_cols + 2 * num_cols * sample_idx)
    cmap = plt.get_cmap('tab20',
                        len(np.unique(data_hashmap['label_persistent'])))
    ax.imshow(data_hashmap['label_persistent'], cmap=cmap)
    ax.set_axis_off()

    for i in range(num_granularities):
        ax = fig.add_subplot(2 * num_samples, num_cols,
                             3 + i + num_cols + 2 * num_cols * sample_idx)

        _embeddings, _clusters, _sizes = msphate_op.transform(
            visualization_level=levels[msphate_idx_selected[i]],
            cluster_level=levels[msphate_idx_selected[i]])

        cmap = plt.get_cmap('tab20', len(np.unique(_clusters)))
        scprep.plot.scatter2d(_embeddings,
                              c=_clusters,
                              s=_sizes,
                              ax=ax,
                              cmap=cmap,
                              title=None,
                              colorbar=False)
        ax.set_axis_off()
        if ax.get_legend() is not None: ax.get_legend().remove()

    return fig


if __name__ == '__main__':
    random_seed = 0
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Path to config yaml file.',
                        required=True)
    parser.add_argument('--image-idx',
                        help='Image index.',
                        type=int,
                        nargs='+',
                        required=True)
    parser.add_argument(
        '--grayscale',
        help='Use this flag if the image is expected to be grayscale.',
        action='store_true')
    parser.add_argument(
        '--binary',
        help='Use this flag if the label is expected to be binary.',
        action='store_true')
    parser.add_argument(
        '--comparison',
        help='Whether or not to include the comparison against other methods.',
        action='store_true')
    parser.add_argument(
        '--separate',
        help=
        'If true, do not overlay with contour, and show the segmentations separately. Default to true for multi-class segmentation',
        action='store_true')
    parser.add_argument(
        '-r',
        '--rerun',
        action='store_true',
        help=
        'If true, will rerun the script until succeeds to circumvent deadlock.'
    )
    parser.add_argument(
        '-t',
        '--max-wait-sec',
        help='Max wait time in seconds for each process (only relevant if `--rerun`).' + \
            'Consider increasing if you hit too many TimeOuts.',
        type=int,
        default=60)
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config.config_file_name = args.config
    config = parse_settings(config, log_settings=False)

    files_folder_raw = '%s/%s' % (config.output_save_path, 'numpy_files')
    files_folder_baselines = '%s/%s' % (config.output_save_path,
                                        'numpy_files_seg_baselines')
    files_folder_dfc = '%s/%s' % (config.output_save_path,
                                  'numpy_files_seg_DFC')
    files_folder_stego = '%s/%s' % (config.output_save_path,
                                    'numpy_files_seg_STEGO')
    files_folder_sam = '%s/%s' % (config.output_save_path,
                                  'numpy_files_seg_SAM')
    files_folder_supervised_unet = '%s/%s' % (
        config.output_save_path, 'numpy_files_seg_supervised_unet')
    files_folder_supervised_nnunet = '%s/%s' % (
        config.output_save_path, 'numpy_files_seg_supervised_nnunet')
    files_folder_kmeans = '%s/%s' % (config.output_save_path,
                                     'numpy_files_seg_kmeans')
    files_folder_diffusion = '%s/%s' % (config.output_save_path,
                                        'numpy_files_seg_diffusion')
    figure_folder = '%s/%s' % (config.output_save_path, 'paper_figure')

    os.makedirs(figure_folder, exist_ok=True)

    files_path_raw = sorted(glob('%s/%s' % (files_folder_raw, '*.npz')))
    files_path_baselines = sorted(
        glob('%s/%s' % (files_folder_baselines, '*.npz')))
    files_path_dfc = sorted(glob('%s/%s' % (files_folder_dfc, '*.npz')))
    files_path_stego = sorted(glob('%s/%s' % (files_folder_stego, '*.npz')))
    files_path_sam = sorted(glob('%s/%s' % (files_folder_sam, '*.npz')))
    files_path_supervised_unet = sorted(
        glob('%s/%s' % (files_folder_supervised_unet, '*.npz')))
    files_path_supervised_nnunet = sorted(
        glob('%s/%s' % (files_folder_supervised_nnunet, '*.npz')))
    files_path_kmeans = sorted(glob('%s/%s' % (files_folder_kmeans, '*.npz')))
    files_path_diffusion = sorted(
        glob('%s/%s' % (files_folder_diffusion, '*.npz')))

    # Now plot the sub-figures for each sample, one by one.
    num_samples = len(args.image_idx)

    if args.comparison:
        fig = plt.figure(figsize=(25, 2 * num_samples))
    else:
        fig = plt.figure(figsize=(25, 4 * num_samples))

    for sample_idx, image_idx in enumerate(tqdm(args.image_idx)):

        numpy_array_raw = np.load(files_path_raw[image_idx])

        image = numpy_array_raw['image']
        image = (image + 1) / 2

        recon = numpy_array_raw['recon']
        recon = (recon + 1) / 2

        label_true = numpy_array_raw['label']
        if np.isnan(label_true).all():
            print('\n\n[Major Warning !!!] We found that the true label is all `NaN`s.' + \
            '\nThis shall only happen if you are not providing labels. Please double check!\n\n')
        label_true = label_true.astype(np.int16)
        latent = numpy_array_raw['latent']

        # In case some results are not generated, we will placehold them with blank labels.
        try:
            numpy_array_baselines = np.load(files_path_baselines[image_idx])
            label_random = numpy_array_baselines['label_random']
            label_watershed = numpy_array_baselines['label_watershed']
            if args.binary:
                label_watershed = label_hint_seg(label_pred=label_watershed,
                                                 label_true=label_true)
            label_felzenszwalb = numpy_array_baselines['label_felzenszwalb']
            if args.binary:
                label_felzenszwalb = label_hint_seg(
                    label_pred=label_felzenszwalb, label_true=label_true)
            label_slic = numpy_array_baselines['label_slic']
            if args.binary:
                label_slic = label_hint_seg(label_pred=label_slic,
                                            label_true=label_true)
        except:
            print(
                'Warning! `baselines` results not found. Placeholding with blank labels.'
            )
            label_random = np.zeros_like(label_true)
            label_watershed = np.zeros_like(label_true)
            label_felzenszwalb = np.zeros_like(label_true)
            label_slic = np.zeros_like(label_true)

        try:
            numpy_array_kmeans = np.load(files_path_kmeans[image_idx])
            label_kmeans = numpy_array_kmeans['label_kmeans']
        except:
            print(
                'Warning! `CUTS + k-means` results not found. Placeholding with blank labels.'
            )
            label_kmeans = np.zeros_like(label_true)

        try:
            numpy_array_diffusion = np.load(files_path_diffusion[image_idx])
            labels_diffusion = numpy_array_diffusion['labels_diffusion']
        except:
            print(
                'Warning! `CUTS + diffusion condensation` results not found. Placeholding with blank labels.'
            )
            labels_diffusion = np.zeros_like(label_true)
            granularities = None

        try:
            numpy_array_dfc = np.load(files_path_dfc[image_idx])
            label_dfc = numpy_array_dfc['label_dfc']
            label_dfc = label_hint_seg(label_pred=label_dfc,
                                       label_true=label_true)
        except:
            print(
                'Warning! `DFC` results not found. Placeholding with blank labels.'
            )
            label_dfc = np.zeros_like(label_true)

        try:
            numpy_array_stego = np.load(files_path_stego[image_idx])
            label_stego = numpy_array_stego['label_stego']
            label_stego = label_hint_seg(label_pred=label_stego,
                                         label_true=label_true)
        except:
            print(
                'Warning! `STEGO` results not found. Placeholding with blank labels.'
            )
            label_stego = np.zeros_like(label_true)

        try:
            numpy_array_sam = np.load(files_path_sam[image_idx])
            label_sam = numpy_array_sam['label_sam']
        except:
            print(
                'Warning! `SAM` results not found. Placeholding with blank labels.'
            )
            label_sam = np.zeros_like(label_true)

        try:
            numpy_array_unet = np.load(files_path_supervised_unet[image_idx])
            label_supervised_unet = numpy_array_unet['label_pred']
        except:
            print(
                'Warning! `Supervised UNet` results not found. Placeholding with blank labels.'
            )
            label_supervised_unet = np.zeros_like(label_true)

        try:
            numpy_array_nnunet = np.load(
                files_path_supervised_nnunet[image_idx])
            label_supervised_nnunet = numpy_array_nnunet['label_pred']
        except:
            print(
                'Warning! `Supervised nn-UNet` results not found. Placeholding with blank labels.'
            )
            label_supervised_nnunet = np.zeros_like(label_true)

        H, W = label_true.shape[:2]
        B = labels_diffusion.shape[0]

        label_persistent = get_persistent_structures(
            labels_diffusion.reshape((B, H, W)))

        seg_kmeans = label_hint_seg(label_pred=label_kmeans,
                                    label_true=label_true)
        seg_persistent = label_hint_seg(label_pred=label_persistent,
                                        label_true=label_true)

        data_hashmap = {
            'image': image,
            'recon': recon,
            'latent': latent,
            'label_true': label_true,
            'label_random': label_random,
            'label_watershed': label_watershed,
            'label_felzenszwalb': label_felzenszwalb,
            'label_slic': label_slic,
            'label_dfc': label_dfc,
            'label_stego': label_stego,
            'label_sam': label_sam,
            'label_supervised_unet': label_supervised_unet,
            'label_supervised_nnunet': label_supervised_nnunet,
            'label_kmeans': label_kmeans,
            'seg_kmeans': seg_kmeans,
            'labels_diffusion': labels_diffusion,
            'label_persistent': label_persistent,
            'seg_persistent': seg_persistent,
        }

        if args.comparison:
            if args.separate:
                fig = plot_comparison(fig=fig,
                                      num_samples=num_samples,
                                      sample_idx=sample_idx,
                                      data_hashmap=data_hashmap,
                                      image_grayscale=args.grayscale,
                                      label_binary=args.binary)
            else:
                assert args.binary
                fig = plot_overlaid_comparison(
                    fig=fig,
                    num_samples=num_samples,
                    sample_idx=sample_idx,
                    data_hashmap=data_hashmap,
                    image_grayscale=args.grayscale,
                    pred_color='blue'
                    if config.dataset_name == 'retina' else 'red')
        else:
            fig = plot_results(fig=fig,
                               num_samples=num_samples,
                               sample_idx=sample_idx,
                               data_hashmap=data_hashmap,
                               image_grayscale=args.grayscale)

    figure_str = ''
    for image_idx in args.image_idx:
        figure_str += str(image_idx) + '-'
    figure_str = figure_str.rstrip('-')

    fig_path = '%s/sample_%s' % (figure_folder, figure_str)
    fig.tight_layout()
    # plt.subplots_adjust(wspace=0.05, hspace=0.05)

    if args.comparison:
        if args.separate:
            fig.savefig('%s_figure_plot_comparison_separate.png' % fig_path)
        else:
            fig.savefig('%s_figure_plot_comparison.png' % fig_path)
    else:
        fig.savefig('%s_figure_plot.png' % fig_path)
