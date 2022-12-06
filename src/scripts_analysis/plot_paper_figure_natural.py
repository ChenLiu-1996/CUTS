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
from tqdm import tqdm

sys.path.append('../')
from utils.attribute_hashmap import AttributeHashmap
from utils.diffusion_condensation import continuous_renumber, get_persistent_structures
from utils.parse import parse_settings
from utils.segmentation import label_hint_seg

warnings.filterwarnings("ignore")


def find_nearest_idx(arr: np.array, num: float) -> int:
    return np.abs(arr - num).argmin()


def plot_comparison(data_hashmap: dict, data_phate: np.array):
    # 2 rows, 8 columns.
    # 1-st row are the images, labels, segmentations.
    # 2-nd row are the PHATE plots if applicable.

    H, W = data_hashmap['label_true'].shape[:2]

    fig = plt.figure(figsize=(16, 4))

    ##### 1-st row!
    ax = fig.add_subplot(2, 8, 1)
    ax.imshow(data_hashmap['image'])
    ax.set_axis_off()

    for (figure_idx, key) in zip(range(2, 2 + 7), [
            'label_true', 'label_kmeans', 'persistent_structures',
            'label_random', 'label_watershed', 'label_felzenszwalb',
            'label_stego'
    ]):
        ax = fig.add_subplot(2, 8, figure_idx)
        ax.imshow(data_hashmap[key], cmap='tab20')
        ax.set_axis_off()

    ##### 2-nd row!
    ax = fig.add_subplot(2, 8, 9)
    ax.imshow(data_hashmap['recon'])
    ax.set_axis_off()

    for (figure_idx, key) in zip(range(10, 10 + 7), [
            'label_true',
            'label_kmeans',
            'persistent_structures',
            'label_random',
            'label_watershed',
            'label_felzenszwalb',
            'label_stego',
    ]):
        ax = fig.add_subplot(2, 8, figure_idx)
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


def plot_results(data_hashmap: dict, data_phate: np.array,
                 granularities: np.array):
    # 2 rows, 12 columns.
    # 1-st row are the images, labels, segmentations.
    # 2-nd row are the PHATE plots if applicable.

    H, W = data_hashmap['label_true'].shape[:2]

    fig = plt.figure(figsize=(22, 4))
    idx_selected = [
        find_nearest_idx(granularities, num)
        for num in np.linspace(granularities[0], granularities[-1], 8)
    ]

    ##### 1-st row!
    ax = fig.add_subplot(2, 12, 1)
    ax.imshow(data_hashmap['image'])
    ax.set_axis_off()

    for (figure_idx,
         key) in zip(range(2, 2 + 3),
                     ['label_true', 'label_kmeans', 'persistent_structures']):
        ax = fig.add_subplot(2, 12, figure_idx)
        ax.imshow(data_hashmap[key], cmap='tab20')
        ax.set_axis_off()

    for i in range(8):
        ax = fig.add_subplot(2, 12, 5 + i)
        __label = data_hashmap['labels_diffusion'][idx_selected[i]]
        __label = __label.reshape((H, W))
        ax.imshow(continuous_renumber(__label), cmap='tab20')
        ax.set_axis_off()

    ##### 2-nd row!
    ax = fig.add_subplot(2, 12, 13)
    ax.imshow(data_hashmap['recon'])
    ax.set_axis_off()

    for (figure_idx,
         key) in zip(range(14, 14 + 3),
                     ['label_true', 'label_kmeans', 'persistent_structures']):
        ax = fig.add_subplot(2, 12, figure_idx)
        scprep.plot.scatter2d(data_phate,
                              c=continuous_renumber(data_hashmap[key].reshape(
                                  (H * W, -1))),
                              ax=ax,
                              title=None,
                              colorbar=False,
                              s=3)
        ax.set_axis_off()
        if ax.get_legend() is not None: ax.get_legend().remove()

    for i in range(8):
        ax = fig.add_subplot(2, 12, 17 + i)
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
    files_path_kmeans = sorted(glob('%s/%s' % (files_folder_kmeans, '*.npz')))
    files_path_diffusion = sorted(
        glob('%s/%s' % (files_folder_diffusion, '*.npz')))

    numpy_array_baselines = np.load(files_path_baselines[args.image_idx])
    numpy_array_kmeans = np.load(files_path_kmeans[args.image_idx])
    numpy_array_diffusion = np.load(files_path_diffusion[args.image_idx])

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
        numpy_array_stego = np.load(files_path_stego[args.image_idx])
        label_stego = numpy_array_stego['label_stego']
    except:
        label_stego = np.zeros_like(label_true)

    H, W = label_true.shape[:2]
    B = labels_diffusion.shape[0]

    persistent_structures = get_persistent_structures(
        labels_diffusion.reshape((B, H, W)))
    seg_kmeans = label_hint_seg(label_pred=label_kmeans, label_true=label_true)

    data_hashmap = {
        'image': image,
        'recon': recon,
        'label_true': label_true,
        'label_random': label_random,
        'label_watershed': label_watershed,
        'label_felzenszwalb': label_felzenszwalb,
        'label_stego': label_stego,
        'label_kmeans': label_kmeans,
        'seg_kmeans': seg_kmeans,
        'granularities': granularities,
        'labels_diffusion': labels_diffusion,
        'persistent_structures': persistent_structures,
    }

    phate_path = '%s/sample_%s.npz' % (phate_folder, str(
        args.image_idx).zfill(5))
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
        fig = plot_comparison(data_hashmap=data_hashmap, data_phate=data_phate)
    else:
        fig = plot_results(data_hashmap=data_hashmap,
                           data_phate=data_phate,
                           granularities=granularities)

    fig_path = '%s/sample_%s' % (figure_folder, str(args.image_idx).zfill(5))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.03, hspace=0.03)

    if args.comparison:
        fig.savefig('%s_figure_plot_comparison.png' % fig_path)
    else:
        fig.savefig('%s_figure_plot.png' % fig_path)
