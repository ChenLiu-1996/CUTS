import argparse
import os
import subprocess
import sys
import time
import warnings
from glob import glob

import numpy as np
import yaml
from tqdm import tqdm

sys.path.append('../')
from utils.attribute_hashmap import AttributeHashmap
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
    np_files_path = sorted(glob('%s/%s' % (files_folder, '*.npz')))

    save_path_numpy = '%s/%s' % (config.output_save_path,
                                 'numpy_files_seg_kmeans')
    os.makedirs(save_path_numpy, exist_ok=True)

    dice_list = []
    for image_idx in tqdm(range(52,len(np_files_path))):
        '''
        Because of the frequent deadlock problem, I decided to use the following solution:
        kill and restart whenever a process is taking too long (likely due to deadlock).
        '''

        load_path = np_files_path[image_idx]
        save_path = '%s/%s' % (save_path_numpy,
                               'sample_%s.npz' % str(image_idx).zfill(5))
        num_workers = config.num_workers

        folder = '/'.join(
            os.path.dirname(os.path.abspath(__file__)).split('/'))

        max_wait_sec = 60
        file_success = False
        while not file_success:
            start = time.time()
            while True:
                try:
                    proc = subprocess.Popen([
                        'python3', folder + '/helper_generate_kmeans.py',
                        '--load_path', load_path, '--save_path', save_path,
                        '--num_workers', str(num_workers)
                    ],
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE)
                    stdout, stderr = proc.communicate(timeout=max_wait_sec)
                    stdout, stderr = str(stdout), str(stderr)
                    stdout = stdout.lstrip('b\'').rstrip('\'')
                    stderr = stderr.lstrip('b\'').rstrip('\'')
                    print(image_idx, stdout, stderr)

                    proc.kill()
                    # This is determined by the sys.stdout in `helper_generate_kmeans.py`
                    if stdout[:8] == 'SUCCESS!':
                        file_success = True
                        dice = float(stdout.split('dice:')[1])
                        dice_list.append(dice)
                    break

                except subprocess.TimeoutExpired:
                    print('Time out! Restart subprocess.')
                    proc.kill()

    print('All kmeans results generated.')
    print('Dice: %.3f \u00B1 %.3f.' %
          (np.mean(dice_list), np.std(dice_list) / np.sqrt(len(dice_list))))
