import argparse
import os
import sys
import warnings
from glob import glob
from typing import Tuple
from matplotlib import pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm

sys.path.append('../')
from utils.attribute_hashmap import AttributeHashmap
from utils.parse import parse_settings

warnings.filterwarnings("ignore")


def calc_row_col(num_figures):
    nrows = int(np.sqrt(num_figures))
    ncols = num_figures // nrows
    if nrows * ncols < num_figures:
        ncols += 1
    return nrows, ncols


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

    dataset_name = config.dataset_name

    files_folder = '%s/%s' % (config.output_save_path, 'numpy_files')
    np_files_path = sorted(glob('%s/%s' % (files_folder, '*.npz')))

    save_path_numpy = '%s/%s' % (config.output_save_path,
                                 'numpy_files_seg_diffusion')
    os.makedirs(save_path_numpy, exist_ok=True)

    num_rows, num_cols = calc_row_col(len(np_files_path))
    fig = plt.figure(figsize=(3 * num_cols, 3 * num_rows))

    for image_idx in tqdm(range(len(np_files_path))):
        numpy_array = np.load(np_files_path[image_idx])
        image = numpy_array['image']
        recon = numpy_array['recon']
        label_true = numpy_array['label']
        latent = numpy_array['latent']

        ax = fig.add_subplot(num_rows, num_cols, image_idx + 1)
        image = (image + 1) / 2
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(os.path.basename(np_files_path[image_idx]))

    fig.savefig('check_images_%s' % dataset_name)
