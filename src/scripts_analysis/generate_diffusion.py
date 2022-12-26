import argparse
import os
import subprocess
import sys
import time
import warnings
from glob import glob
from typing import Tuple

import numpy as np
import yaml
from tqdm import tqdm

sys.path.append('../')
from utils.attribute_hashmap import AttributeHashmap
from utils.diffusion_condensation import diffusion_condensation
from utils.parse import parse_settings

warnings.filterwarnings("ignore")


def generate_diffusion(
        shape: Tuple[int],
        latent: np.array,
        num_workers: int = 1,
        random_seed: int = 0) -> Tuple[float, np.array, np.array]:

    H, W, C = shape
    assert latent.shape == (H * W, C)

    _, (catch_op,
        levels) = diffusion_condensation(latent,
                                         height_width=(H, W),
                                         pos_enc_gamma=config.pos_enc_gamma,
                                         num_workers=num_workers,
                                         random_seed=random_seed,
                                         return_all=True)

    labels_pred = np.array([catch_op.NxTs[lvl] for lvl in levels])
    granularities = [len(catch_op.NxTs) + lvl for lvl in levels]

    return labels_pred, granularities, levels, catch_op.gradient


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
    np_files_path = sorted(glob('%s/%s' % (files_folder, '*.npz')))

    save_path_numpy = '%s/%s' % (config.output_save_path,
                                 'numpy_files_seg_diffusion')
    os.makedirs(save_path_numpy, exist_ok=True)

    for image_idx in tqdm(range(len(np_files_path))):
        '''
        Because of the frequent deadlock problem, I decided to
        use the following solution:
        kill process and restart whenever a process is taking too long.
        '''

        load_path = np_files_path[image_idx]
        save_path = '%s/%s' % (save_path_numpy,
                               'sample_%s.npz' % str(image_idx).zfill(5))

        folder = '/'.join(
            os.path.dirname(os.path.abspath(__file__)).split('/'))
        proc = subprocess.Popen([
            'python3', folder + '/generate_diffusion_helper.py', '--load_path',
            load_path, '--save_path', save_path
        ])

        max_wait_sec = 300
        interval_sec = 1
        file_success = False
        while not file_success:
            start = time.time()
            result = proc.poll()
            while True:
                if result is not None:
                    file_success = True
                    break
                if time.time() - start >= max_wait_sec:
                    file_success = False
                    print('Time out! Restart subprocess.')
                    break
                time.sleep(interval_sec)

    print('All diffusion results generated.')
