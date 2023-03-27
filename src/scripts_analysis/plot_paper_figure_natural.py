import argparse
import os
import subprocess
import sys
import time
import warnings
from glob import glob

import numpy as np
import scprep
import yaml
from matplotlib import pyplot as plt

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
            'label_kmeans',
            'persistent_structures',
            'label_random',
            'label_watershed',
            'label_felzenszwalb',
            'label_stego',
            'label_supervised_unet',
            'label_supervised_nnunet',
    ]):
        ax = fig.add_subplot(2 * num_samples, 10, figure_idx + 20 * sample_idx)
        ax.imshow(data_hashmap[key], cmap='tab20')
        ax.set_axis_off()

    ##### 2-nd row!
    ax = fig.add_subplot(2 * num_samples, 10, 11 + 20 * sample_idx)
    ax.imshow(data_hashmap['recon'])
    ax.set_axis_off()

    for (figure_idx, key) in zip(range(12, 12 + 9), [
            'label_true',
            'label_kmeans',
            'persistent_structures',
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
        for num in np.linspace(granularities[0], granularities[-1], 8)
    ]

    ##### 1-st row!
    ax = fig.add_subplot(2 * num_samples, 12, 1 + 24 * sample_idx)
    ax.imshow(data_hashmap['image'])
    ax.set_axis_off()

    for (figure_idx,
         key) in zip(range(2, 2 + 3),
                     ['label_true', 'label_kmeans', 'persistent_structures']):
        ax = fig.add_subplot(2 * num_samples, 12, figure_idx + 24 * sample_idx)
        ax.imshow(data_hashmap[key], cmap='tab20')
        ax.set_axis_off()

    for i in range(8):
        ax = fig.add_subplot(2 * num_samples, 12, 5 + i + 24 * sample_idx)
        __label = data_hashmap['labels_diffusion'][idx_selected[i]]
        __label = __label.reshape((H, W))
        ax.imshow(continuous_renumber(__label), cmap='tab20')
        ax.set_axis_off()

    ##### 2-nd row!
    ax = fig.add_subplot(2 * num_samples, 12, 13 + 24 * sample_idx)
    ax.imshow(data_hashmap['recon'])
    ax.set_axis_off()

    for (figure_idx,
         key) in zip(range(14, 14 + 3),
                     ['label_true', 'label_kmeans', 'persistent_structures']):
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

    for i in range(8):
        ax = fig.add_subplot(2 * num_samples, 12, 17 + i + 24 * sample_idx)
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
    parser.add_argument(
        '-t',
        '--max-wait-sec',
        help='Max wait time in seconds for each process.' + \
            'Consider increasing if you hit too many TimeOuts.',
        type=int,
        default=60)
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config.config_file_name = args.config
    config = parse_settings(config, log_settings=False)

    files_folder_raw = '%s/%s' % (config.output_save_path,
                                  'numpy_files')
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

    files_path_raw = sorted(
        glob('%s/%s' % (files_folder_raw, '*.npz')))
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

        numpy_array_raw = np.load(files_path_raw[image_idx])
        image = numpy_array_raw['image']
        recon = numpy_array_raw['recon']
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
            label_felzenszwalb = numpy_array_baselines['label_felzenszwalb']
        except:
            print('Warning! `baselines` results not found. Placeholding with blank labels.')
            label_random = np.zeros_like(label_true)
            label_watershed = np.zeros_like(label_true)
            label_felzenszwalb = np.zeros_like(label_true)

        try:
            numpy_array_kmeans = np.load(files_path_kmeans[image_idx])
            label_kmeans = numpy_array_kmeans['label_kmeans']
        except:
            print('Warning! `CUTS + k-means` results not found. Placeholding with blank labels.')
            label_kmeans = np.zeros_like(label_true)

        try:
            numpy_array_diffusion = np.load(files_path_diffusion[image_idx])
            granularities = numpy_array_diffusion['granularities_diffusion']
            labels_diffusion = numpy_array_diffusion['labels_diffusion']
        except:
            print('Warning! `CUTS + diffusion condensation` results not found. Placeholding with blank labels.')
            num_placeholder_granularities = 10
            granularities = np.arange(num_placeholder_granularities)
            labels_diffusion = np.zeros((num_placeholder_granularities, *label_true.shape))

        try:
            numpy_array_stego = np.load(files_path_stego[image_idx])
            label_stego = numpy_array_stego['label_stego']
        except:
            print('Warning! `STEGO` results not found. Placeholding with blank labels.')
            label_stego = np.zeros_like(label_true)

        try:
            numpy_array_unet = np.load(files_path_supervised_unet[image_idx])
            label_supervised_unet = numpy_array_unet['label_pred']
        except:
            print('Warning! `Supervised UNet` results not found. Placeholding with blank labels.')
            label_supervised_unet = np.zeros_like(label_true)

        try:
            numpy_array_nnunet = np.load(
                files_path_supervised_nnunet[image_idx])
            label_supervised_nnunet = numpy_array_nnunet['label_pred']
        except:
            print('Warning! `Supervised nn-UNet` results not found. Placeholding with blank labels.')
            label_supervised_nnunet = np.zeros_like(label_true)

        H, W = label_true.shape[:2]
        B = labels_diffusion.shape[0]

        persistent_structures = get_persistent_structures(
            labels_diffusion.reshape((B, H, W)))
        seg_kmeans = label_hint_seg(label_pred=label_kmeans,
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
        }

        phate_path = '%s/sample_%s.npz' % (phate_folder,
                                           str(image_idx).zfill(5))
        if os.path.exists(phate_path):
            # Load the phate data if exists.
            data_phate_numpy = np.load(phate_path)
            data_phate = data_phate_numpy['data_phate']
        else:
            # Otherwise, generate the phate data.
            '''
            Because of the frequent deadlock problem, I decided to use the following solution:
            kill and restart whenever a process is taking too long (likely due to deadlock).
            '''
            load_path = files_path_raw[image_idx]
            num_workers = config.num_workers
            folder = '/'.join(
                os.path.dirname(os.path.abspath(__file__)).split('/'))

            file_success = False
            while not file_success:
                start = time.time()
                while True:
                    try:
                        proc = subprocess.Popen([
                            'python3', folder + '/helper_run_phate.py',
                            '--load_path', load_path, '--phate_path',
                            phate_path, '--random_seed',
                            str(config.random_seed), '--num_workers',
                            str(num_workers)
                        ],
                                                stdout=subprocess.PIPE,
                                                stderr=subprocess.PIPE)
                        stdout, stderr = proc.communicate(
                            timeout=args.max_wait_sec)
                        stdout, stderr = str(stdout), str(stderr)
                        stdout = stdout.lstrip('b\'').rstrip('\'')
                        stderr = stderr.lstrip('b\'').rstrip('\'')
                        print(image_idx, stdout, stderr)

                        proc.kill()
                        # This is determined by the sys.stdout in `helper_run_phate.py`
                        if stdout[:8] == 'SUCCESS!':
                            file_success = True
                        break

                    except subprocess.TimeoutExpired:
                        print('Time out! Restart subprocess.')
                        proc.kill()

            data_phate_numpy = np.load(phate_path)
            data_phate = data_phate_numpy['data_phate']

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
