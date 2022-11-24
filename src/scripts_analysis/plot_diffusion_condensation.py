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
from tqdm import tqdm

sys.path.append('../')
from utils.attribute_hashmap import AttributeHashmap
from utils.diffusion_condensation import diffusion_condensation
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

    files_folder = '%s/%s' % (config.output_save_path, 'numpy_files')
    figure_folder = '%s/%s' % (config.output_save_path, 'figures')

    os.makedirs(figure_folder, exist_ok=True)

    np_files_path = sorted(glob('%s/%s' % (files_folder, '*.npz')))

    for image_idx in tqdm(range(len(np_files_path))):
        numpy_array = np.load(np_files_path[image_idx])
        image = numpy_array['image']
        recon = numpy_array['recon']
        label_true = numpy_array['label']
        latent = numpy_array['latent']

        image = (image + 1) / 2
        recon = (recon + 1) / 2

        H, W = label_true.shape[:2]

        clusters, (catch_op, levels, data) = diffusion_condensation(
            latent,
            height_width=(H, W),
            pos_enc_gamma=config.pos_enc_gamma,
            num_workers=config.num_workers,
            return_all=True)

        n_rows = (len(levels) + 1) // 2

        # 1. PHATE plot.
        phate_op = phate.PHATE(random_state=random_seed)
        data_phate = phate_op.fit_transform(data)
        fig1 = plt.figure(figsize=(15, 4 * n_rows))
        for i in range(-1, len(levels)):
            ax = fig1.add_subplot(n_rows + 1, 2, i + 2)
            if i == -1:
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
            else:
                scprep.plot.scatter2d(data_phate,
                                      c=catch_op.NxTs[levels[i]],
                                      legend_anchor=(1, 1),
                                      ax=ax,
                                      title='Granularity ' +
                                      str(len(catch_op.NxTs) + levels[i]),
                                      xticks=False,
                                      yticks=False,
                                      label_prefix="PHATE",
                                      fontsize=10,
                                      s=3)

        # 2. Segmentation plot.
        fig2 = plt.figure(figsize=(12, 4 * n_rows))
        for i in range(-2, len(levels)):
            ax = fig2.add_subplot(n_rows + 1, 2, i + 3)
            if i == -2:
                ax.imshow(image)
                ax.set_axis_off()
            elif i == -1:
                ax.imshow(label_true, cmap='gray')
                ax.set_axis_off()
            else:
                ax.imshow(catch_op.NxTs[levels[i]].reshape((H, W)),
                          cmap='tab20')
                ax.set_title('Granularity ' +
                             str(len(catch_op.NxTs) + levels[i]))
                ax.set_axis_off()

        # 3. Reconstruction sanity check plot.
        fig3 = plt.figure()
        ax = fig3.add_subplot(1, 2, 1)
        ax.imshow(image.reshape((H, W, -1)))
        ax.set_axis_off()
        ax.set_title('Image')
        ax = fig3.add_subplot(1, 2, 2)
        ax.imshow(recon.reshape((H, W, -1)))
        ax.set_axis_off()
        ax.set_title('Reconstruction')

        fig_path = '%s/sample_%s' % (figure_folder, str(image_idx).zfill(5))

        fig1.tight_layout()
        fig1.savefig('%s_phate.png' % fig_path)

        fig2.tight_layout()
        fig2.savefig('%s_segmentation.png' % fig_path)

        fig3.tight_layout()
        fig3.savefig('%s_recon.png' % fig_path)
