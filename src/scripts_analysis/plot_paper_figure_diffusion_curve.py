import argparse
import os
import sys
import warnings
from glob import glob

import numpy as np
import yaml
from matplotlib import pyplot as plt

sys.path.append('../')
from utils.attribute_hashmap import AttributeHashmap
from utils.parse import parse_settings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Path to config yaml file.',
                        required=True)
    parser.add_argument('--image-idx',
                        help='Image index.',
                        type=int,
                        required=True)
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config.config_file_name = args.config
    config = parse_settings(config, log_settings=False)

    figure_folder = '%s/%s' % (config.output_save_path, 'paper_figure')
    diffusion_folder = '%s/%s' % (config.output_save_path,
                                  'numpy_files_seg_diffusion')

    if os.path.exists(diffusion_folder):
        # Load the phate data if exists.
        data_diffusion_numpy = np.load(diffusion_folder)

        gradients = data_diffusion_numpy['gradients']
        levels = data_diffusion_numpy['levels']

    plt.rcParams["font.family"] = 'serif'
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.plot(gradients, color='black', linewidth=2)
    plt.scatter(len(gradients) + levels,
                gradients[levels + 1],
                c='white',
                linewidth=2,
                edgecolors='firebrick')
    plt.xlabel('Diffusion Condensation Iteration', fontsize=16)
    plt.ylabel('Topological Activity', fontsize=16)

    fig_path = '%s/diffusion_curve_sample_%s.png' % (
        figure_folder, str(args.image_idx).zfill(5))
    plt.savefig(fig_path)
