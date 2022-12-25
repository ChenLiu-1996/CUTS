import argparse
import os
import sys
import warnings
from glob import glob

import numpy as np
import phate
import scprep
import yaml
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize

sys.path.append('../')
from utils.attribute_hashmap import AttributeHashmap
from utils.diffusion_condensation import continuous_renumber, get_persistent_structures
from utils.parse import parse_settings
from utils.segmentation import label_hint_seg

warnings.filterwarnings("ignore")


def find_nearest_idx(arr: np.array, num: float) -> int:
    return np.abs(arr - num).argmin()


def plot_comparison(fig: plt.figure, num_samples: int, sample_idx: int,
                    data_hashmap: dict, data_phate: np.array):
    # 2 rows, 10 columns.
    # 1-st row are the images, labels, segmentations.
    # 2-nd row are the PHATE plots if applicable.

    H, W = data_hashmap['label_true'].shape[:2]

    ##### 1-st row!
    ax = fig.add_subplot(2 * num_samples, 10, 1 + 20 * sample_idx)
    ax.imshow(data_hashmap['image'])
    ax.set_axis_off()

    for (figure_idx, key) in zip(range(2, 2 + 9), [
            'label_true',
            'seg_kmeans',
            'seg_persistent',
            'label_random',
            'label_watershed',
            'label_felzenszwalb',
            'label_stego',
            'label_supervised_unet',
            'label_supervised_nnunet',
    ]):
        ax = fig.add_subplot(2 * num_samples, 10, figure_idx + 20 * sample_idx)
        ax.imshow(data_hashmap[key], cmap='gray')
        ax.set_axis_off()

    ##### 2-nd row!
    ax = fig.add_subplot(2 * num_samples, 10, 11 + 20 * sample_idx)
    ax.imshow(data_hashmap['recon'])
    ax.set_axis_off()

    for (figure_idx, key) in zip(range(12, 12 + 9), [
            'label_true',
            'seg_kmeans',
            'seg_persistent',
            'label_random',
            'label_watershed',
            'label_felzenszwalb',
            'label_stego',
            'label_supervised_unet',
            'label_supervised_nnunet',
    ]):
        ax = fig.add_subplot(2 * num_samples, 10, figure_idx + 20 * sample_idx)
        scprep.plot.scatter2d(data_phate,
                              c=continuous_renumber(data_hashmap[key].reshape(
                                  (H * W, -1))),
                              ax=ax,
                              title=None,
                              colorbar=False,
                              s=3)
        ax.set_axis_off()
        if ax.get_legend() is not None: ax.get_legend().remove()

    return fig


def plot_results(fig: plt.figure, num_samples: int, sample_idx: int,
                 data_hashmap: dict, data_phate: np.array,
                 granularities: np.array):
    # 2 rows, 12 columns.
    # 1-st row are the images, labels, segmentations.
    # 2-nd row are the PHATE plots if applicable.

    H, W = data_hashmap['label_true'].shape[:2]

    idx_selected = [
        find_nearest_idx(granularities, num)
        for num in np.linspace(granularities[0], granularities[-1], 6)
    ]

    ##### 1-st row!
    ax = fig.add_subplot(2 * num_samples, 12, 1 + 24 * sample_idx)
    ax.imshow(data_hashmap['image'])
    ax.set_axis_off()

    for (figure_idx, key) in zip(range(2, 2 + 5), [
            'label_true', 'label_kmeans', 'seg_kmeans',
            'persistent_structures', 'seg_persistent'
    ]):
        if key == 'label_true' or 'seg_' in key:
            cmap = 'gray'
        else:
            cmap = 'tab20'
        ax = fig.add_subplot(2 * num_samples, 12, figure_idx + 24 * sample_idx)
        ax.imshow(data_hashmap[key], cmap=cmap)
        ax.set_axis_off()

    for i in range(6):
        ax = fig.add_subplot(2 * num_samples, 12, 7 + i + 24 * sample_idx)
        __label = data_hashmap['labels_diffusion'][idx_selected[i]]
        __label = __label.reshape((H, W))
        ax.imshow(continuous_renumber(__label), cmap='tab20')
        ax.set_axis_off()

    ##### 2-nd row!
    ax = fig.add_subplot(2 * num_samples, 12, 13 + 24 * sample_idx)
    ax.imshow(data_hashmap['recon'])
    ax.set_axis_off()

    for (figure_idx, key) in zip(range(14, 14 + 5), [
            'label_true', 'label_kmeans', 'seg_kmeans',
            'persistent_structures', 'seg_persistent'
    ]):
        ax = fig.add_subplot(2 * num_samples, 12, figure_idx + 24 * sample_idx)
        scprep.plot.scatter2d(data_phate,
                              c=continuous_renumber(data_hashmap[key].reshape(
                                  (H * W, -1))),
                              ax=ax,
                              title=None,
                              colorbar=False,
                              s=3)
        ax.set_axis_off()
        if ax.get_legend() is not None: ax.get_legend().remove()

    for i in range(6):
        ax = fig.add_subplot(2 * num_samples, 12, 19 + i + 24 * sample_idx)
        __label = data_hashmap['labels_diffusion'][idx_selected[i]]
        __label = __label.reshape((H * W, -1))
        scprep.plot.scatter2d(data_phate,
                              c=continuous_renumber(__label),
                              ax=ax,
                              title=None,
                              colorbar=False,
                              s=3)
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
        '--comparison',
        help='Whether or not to include the comparison against other methods.',
        action='store_true')
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config.config_file_name = args.config
    config = parse_settings(config, log_settings=False)

    files_folder_baselines = '%s/%s' % (config.output_save_path,
                                        'numpy_files_seg_baselines')
    files_folder_stego = '%s/%s' % (config.output_save_path,
                                    'numpy_files_seg_STEGO')
    files_folder_supervised_unet = '%s/%s' % (
        config.output_save_path, 'numpy_files_seg_supervised_unet')
    files_folder_supervised_nnunet = '%s/%s' % (
        config.output_save_path, 'numpy_files_seg_supervised_nnunet')
    files_folder_kmeans = '%s/%s' % (config.output_save_path,
                                     'numpy_files_seg_kmeans')
    files_folder_diffusion = '%s/%s' % (config.output_save_path,
                                        'numpy_files_seg_diffusion')
    figure_folder = '%s/%s' % (config.output_save_path, 'paper_figure')
    phate_folder = '%s/%s' % (config.output_save_path, 'numpy_files_phate')

    os.makedirs(figure_folder, exist_ok=True)
    os.makedirs(phate_folder, exist_ok=True)

    files_path_baselines = sorted(
        glob('%s/%s' % (files_folder_baselines, '*.npz')))
    files_path_stego = sorted(glob('%s/%s' % (files_folder_stego, '*.npz')))
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
        fig = plt.figure(figsize=(20, 4 * num_samples))
    else:
        fig = plt.figure(figsize=(22, 4 * num_samples))

    for sample_idx, image_idx in enumerate(args.image_idx):

        numpy_array_baselines = np.load(files_path_baselines[image_idx])
        numpy_array_kmeans = np.load(files_path_kmeans[image_idx])
        numpy_array_diffusion = np.load(files_path_diffusion[image_idx])

        image = numpy_array_diffusion['image']
        recon = numpy_array_diffusion['recon']
        label_true = numpy_array_diffusion['label'].astype(np.int16)
        latent = numpy_array_diffusion['latent']
        label_random = numpy_array_baselines['label_random']
        label_watershed = numpy_array_baselines['label_watershed']
        label_felzenszwalb = numpy_array_baselines['label_felzenszwalb']
        label_kmeans = numpy_array_kmeans['label_kmeans']
        granularities = numpy_array_diffusion['granularities_diffusion']
        labels_diffusion = numpy_array_diffusion['labels_diffusion']

        # We have provided scripts to generate all other results except for STEGO.
        # In case you have not generated STEGO results, we will skip its plotting.
        try:
            numpy_array_stego = np.load(files_path_stego[image_idx])
            label_stego = numpy_array_stego['label_stego']
        except:
            label_stego = np.zeros_like(label_true)

        try:
            numpy_array_unet = np.load(files_path_supervised_unet[image_idx])
            label_supervised_unet = numpy_array_unet['label_pred']
        except:
            label_supervised_unet = np.zeros_like(label_true)

        try:
            numpy_array_nnunet = np.load(
                files_path_supervised_nnunet[image_idx])
            label_supervised_nnunet = numpy_array_nnunet['label_pred']
        except:
            label_supervised_nnunet = np.zeros_like(label_true)

        H, W = label_true.shape[:2]
        B = labels_diffusion.shape[0]

        persistent_structures = get_persistent_structures(
            labels_diffusion.reshape((B, H, W)))
        seg_kmeans = label_hint_seg(label_pred=label_kmeans,
                                    label_true=label_true)
        seg_persistent = label_hint_seg(label_pred=persistent_structures,
                                        label_true=label_true)

        data_hashmap = {
            'image': image,
            'recon': recon,
            'label_true': label_true,
            'label_random': label_random,
            'label_watershed': label_watershed,
            'label_felzenszwalb': label_felzenszwalb,
            'label_stego': label_stego,
            'label_supervised_unet': label_supervised_unet,
            'label_supervised_nnunet': label_supervised_nnunet,
            'label_kmeans': label_kmeans,
            'seg_kmeans': seg_kmeans,
            'granularities': granularities,
            'labels_diffusion': labels_diffusion,
            'persistent_structures': persistent_structures,
            'seg_persistent': seg_persistent,
        }

        phate_path = '%s/sample_%s.npz' % (phate_folder,
                                           str(image_idx).zfill(5))
        if os.path.exists(phate_path):
            # Load the phate data if exists.
            data_phate_numpy = np.load(phate_path)
            data_phate = data_phate_numpy['data_phate']
        else:
            # Otherwise, generate the phate data.
            phate_op = phate.PHATE(random_state=random_seed,
                                   n_jobs=config.num_workers)

            data_phate = phate_op.fit_transform(normalize(latent, axis=1))
            with open(phate_path, 'wb+') as f:
                np.savez(f, data_phate=data_phate)

        if args.comparison:
            fig = plot_comparison(fig=fig,
                                  num_samples=num_samples,
                                  sample_idx=sample_idx,
                                  data_hashmap=data_hashmap,
                                  data_phate=data_phate)
        else:
            fig = plot_results(fig=fig,
                               num_samples=num_samples,
                               sample_idx=sample_idx,
                               data_hashmap=data_hashmap,
                               data_phate=data_phate,
                               granularities=granularities)

    figure_str = ''
    for image_idx in args.image_idx:
        figure_str += str(image_idx) + '-'
    figure_str = figure_str.rstrip('-')

    fig_path = '%s/sample_%s' % (figure_folder, figure_str)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.03, hspace=0.03)

    if args.comparison:
        fig.savefig('%s_figure_plot_comparison.png' % fig_path)
    else:
        fig.savefig('%s_figure_plot.png' % fig_path)
