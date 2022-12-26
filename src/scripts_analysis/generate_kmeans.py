import argparse
import os
import subprocess
import sys
import time
import warnings
from glob import glob

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

    for image_idx in tqdm(range(31, len(np_files_path))):
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
            'python3', folder + '/generate_kmeans_helper.py', '--load_path',
            load_path, '--save_path', save_path
        ], stdout=subprocess.PIPE)

        max_wait_sec = 30
        interval_sec = 1
        file_success = False
        while not file_success:
            start = time.time()
            while True:
                # result = proc.poll()
                # result = proc.wait(max_wait_sec)
                result, stderr = proc.communicate()
                print('here', result)
                if result is not None:
                    file_success = True
                    break
                if time.time() - start >= max_wait_sec:
                    file_success = False
                    proc.terminate()
                    proc = subprocess.Popen([
                        'python3', folder + '/generate_kmeans_helper.py',
                        '--load_path', load_path, '--save_path', save_path
                    ],
                                            stdout=subprocess.PIPE)
                    print('Time out! Restart subprocess.')
                    break
                time.sleep(interval_sec)

    print('All kmeans results generated.')
