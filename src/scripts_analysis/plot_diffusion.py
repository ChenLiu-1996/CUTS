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
from utils.diffusion_condensation import continuous_renumber, most_persistent_structures
from utils.parse import parse_settings
from utils.segmentation import label_hint_seg

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
                              'numpy_files_seg_diffusion')
    figure_folder = '%s/%s' % (config.output_save_path, 'figures')
    phate_folder = '%s/%s' % (config.output_save_path, 'numpy_files_phate')

    os.makedirs(figure_folder, exist_ok=True)
    os.makedirs(phate_folder, exist_ok=True)

    np_files_path = sorted(glob('%s/%s' % (files_folder, '*.npz')))

    for image_idx in tqdm(range(len(np_files_path))):
        numpy_array = np.load(np_files_path[image_idx])
        image = numpy_array['image']
        recon = numpy_array['recon']
        label_true = numpy_array['label'].astype(np.int16)
        latent = numpy_array['latent']
        granularities = numpy_array['granularities_diffusion']
        labels_diffusion = numpy_array['labels_diffusion']

        H, W = label_true.shape[:2]
        B = labels_diffusion.shape[0]

        n_rows = (len(granularities) + 1) // 2

        persistent_structures, _ = most_persistent_structures(
            labels_diffusion.reshape((B, H, W)))
        seg = label_hint_seg(label_pred=persistent_structures,
                             label_true=label_true)

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

        fig1 = plt.figure(figsize=(15, 4 * n_rows))
        for i in range(-3, len(granularities)):
            ax = fig1.add_subplot(n_rows + 2, 2, i + 4)
            if i == -3:
                # Plot the ground truth.
                scprep.plot.scatter2d(data_phate,
                                      c=continuous_renumber(
                                          label_true.reshape((H * W, -1))),
                                      legend_anchor=(1, 1),
                                      ax=ax,
                                      title='Ground truth label',
                                      xticks=False,
                                      yticks=False,
                                      label_prefix="PHATE",
                                      fontsize=10,
                                      s=3)
            elif i == -2:
                # Plot the segmented persistent structures.
                scprep.plot.scatter2d(
                    data_phate,
                    c=continuous_renumber(seg.reshape((H * W, -1))),
                    legend_anchor=(1, 1),
                    ax=ax,
                    title='Persistent Structures (Segmented)',
                    xticks=False,
                    yticks=False,
                    label_prefix="PHATE",
                    fontsize=10,
                    s=3)
            elif i == -1:
                # Plot the persistent structures.
                scprep.plot.scatter2d(data_phate,
                                      c=continuous_renumber(
                                          persistent_structures.reshape(
                                              (H * W, -1))),
                                      legend_anchor=(1, 1),
                                      ax=ax,
                                      title='Persistent Structures',
                                      xticks=False,
                                      yticks=False,
                                      label_prefix="PHATE",
                                      fontsize=10,
                                      s=3)
            else:
                scprep.plot.scatter2d(
                    data_phate,
                    c=continuous_renumber(labels_diffusion[i]),
                    legend_anchor=(1, 1),
                    ax=ax,
                    title='Granularity ' + str(granularities[i]),
                    xticks=False,
                    yticks=False,
                    label_prefix="PHATE",
                    fontsize=10,
                    s=3)

        # 2. Segmentation plot.
        fig2 = plt.figure(figsize=(12, 4 * n_rows))
        for i in range(-4, len(granularities)):
            ax = fig2.add_subplot(n_rows + 2, 2, i + 5)
            if i == -4:
                ax.imshow(image)
                ax.set_axis_off()
            elif i == -3:
                ax.imshow(label_true, cmap='gray')
                ax.set_axis_off()
            elif i == -2:
                ax.imshow(seg, cmap='gray')
                ax.set_title('Persistent Structures (Segmented)')
                ax.set_axis_off()
            elif i == -1:
                ax.imshow(continuous_renumber(persistent_structures), cmap='tab20')
                ax.set_title('Persistent Structures')
                ax.set_axis_off()
            else:
                ax.imshow(continuous_renumber(labels_diffusion[i].reshape(
                    (H, W))),
                          cmap='tab20')
                ax.set_title('Granularity ' + str(granularities[i]))
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
