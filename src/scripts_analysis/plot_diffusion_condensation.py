import argparse
import os
import sys
import warnings
from glob import glob

import numpy as np
import yaml
from tqdm import tqdm

sys.path.append('../')
from utils.attribute_hashmap import AttributeHashmap
from utils.diffusion_condensation import cluster_indices_from_mask, diffusion_condensation
from utils.metrics import dice_coeff
from utils.parse import parse_settings

warnings.filterwarnings("ignore")

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
    figure_folder = '%s/%s' % (config.output_save_path, 'figures')

    os.makedirs(figure_folder, exist_ok=True)

    np_files_path = sorted(glob('%s/%s' % (files_folder, '*.npz')))

    # dice_list = []
    for image_idx in tqdm(range(len(np_files_path))):
        numpy_array = np.load(np_files_path[image_idx])
        image = numpy_array['image']
        recon = numpy_array['recon']
        label_true = numpy_array['label']
        latent = numpy_array['latent']

        image = (image + 1) / 2
        recon = (recon + 1) / 2

        H, W = label_true.shape[:2]
        X = latent

        clusters, (fig1, fig2, fig3) = diffusion_condensation(
            X,
            height_width=(H, W),
            pos_enc_gamma=config.pos_enc_gamma,
            image_recon_label=(image, recon, label_true),
            return_figures=True)

        # seg = clusters.reshape((H, W))
        # cluster_id = cluster_indices_from_mask(seg, label_true, top1_only=True)
        # # cluster_indices, dice_map = cluster_indices_from_mask(seg, label_true)
        # # label_pred = np.logical_or.reduce([seg == i for i in cluster_indices])
        # # dice_list.append(dice_coeff(label_pred, label_true))
        # label_pred = seg == cluster_id
        # print('dice', dice_coeff(label_pred, label_true))

        fig_path = '%s/sample_%s' % (figure_folder, str(image_idx).zfill(5))

        fig1.tight_layout()
        fig1.savefig('%s_phate.png' % fig_path)

        fig2.tight_layout()
        fig2.savefig('%s_segmentation.png' % fig_path)

        fig3.tight_layout()
        fig3.savefig('%s_recon.png' % fig_path)
