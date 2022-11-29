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
from utils.parse import parse_settings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    random_seed = 0

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Path to config yaml file.',
                        required=True)
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config.config_file_name = args.config
    config = parse_settings(config, log_settings=False)

    files_folder = '%s/%s' % (config.output_save_path,
                              'numpy_files_seg_kmeans')
    figure_folder = '%s/%s' % (config.output_save_path, 'figures')
    phate_folder = '%s/%s' % (config.output_save_path, 'numpy_files_phate')

    os.makedirs(figure_folder, exist_ok=True)
    os.makedirs(phate_folder, exist_ok=True)

    np_files_path = sorted(glob('%s/%s' % (files_folder, '*.npz')))

    for image_idx in tqdm(range(len(np_files_path))):
        numpy_array = np.load(np_files_path[image_idx])
        image = numpy_array['image']
        label_true = numpy_array['label'].astype(np.int16)
        latent = numpy_array['latent']
        label_kmeans = numpy_array['label_kmeans']
        seg_kmeans = numpy_array['seg_kmeans']

        H, W = label_true.shape[:2]

        # 1. PHATE plot.
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

        fig1 = plt.figure(figsize=(15, 4))
        ax = fig1.add_subplot(1, 3, 1)
        # Plot the ground truth.
        scprep.plot.scatter2d(data_phate,
                              c=label_true.reshape((H * W, -1)),
                              legend_anchor=(1, 1),
                              ax=ax,
                              title='Ground truth label',
                              xticks=False,
                              yticks=False,
                              label_prefix="PHATE",
                              fontsize=10,
                              s=3)
        ax = fig1.add_subplot(1, 3, 2)
        # Plot the kmeans.
        scprep.plot.scatter2d(data_phate,
                              c=label_kmeans.reshape((H * W, -1)),
                              legend_anchor=(1, 1),
                              ax=ax,
                              title='Spectral K-means',
                              xticks=False,
                              yticks=False,
                              label_prefix="PHATE",
                              fontsize=10,
                              s=3)
        ax = fig1.add_subplot(1, 3, 3)
        # Plot the segmented kmeans.
        scprep.plot.scatter2d(data_phate,
                              c=seg_kmeans.reshape((H * W, -1)),
                              legend_anchor=(1, 1),
                              ax=ax,
                              title='Spectral K-means',
                              xticks=False,
                              yticks=False,
                              label_prefix="PHATE",
                              fontsize=10,
                              s=3)

        # 2. Segmentation plot.
        fig2 = plt.figure(figsize=(20, 6))
        ax = fig2.add_subplot(1, 4, 1)
        ax.imshow(image)
        ax.set_axis_off()
        ax = fig2.add_subplot(1, 4, 2)
        gt_cmap = 'gray' if len(np.unique(label_true)) <= 2 else 'tab20'
        ax.imshow(label_true, cmap=gt_cmap)
        ax.set_axis_off()
        ax = fig2.add_subplot(1, 4, 3)
        ax.imshow(seg_kmeans, cmap='gray')
        ax.set_title('Spectral K-means')
        ax.set_axis_off()
        ax = fig2.add_subplot(1, 4, 4)
        ax.imshow(label_kmeans, cmap='tab20')
        ax.set_title('Spectral K-means')
        ax.set_axis_off()

        fig_path = '%s/sample_%s' % (figure_folder, str(image_idx).zfill(5))

        fig1.tight_layout()
        fig1.savefig('%s_phate_kmeans.png' % fig_path)

        fig2.tight_layout()
        fig2.savefig('%s_segmentation_kmeans.png' % fig_path)
