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
    parser.add_argument(
        '-t',
        '--max-wait-sec',
        help='Max wait time in seconds for each process.' + \
            'Consider increasing if you hit too many TimeOuts.',
        type=int,
        default=60)
    parser.add_argument('-r', '--rerun', action='store_true')
    parser.add_argument('-o', '--overwrite', action='store_true')
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
    for image_idx in tqdm(range(len(np_files_path))):

        load_path = np_files_path[image_idx]
        save_path = '%s/%s' % (save_path_numpy,
                               os.path.basename(np_files_path[image_idx]))
        num_workers = config.num_workers

        folder = '/'.join(
            os.path.dirname(os.path.abspath(__file__)).split('/'))

        if os.path.exists(save_path) and not args.overwrite:
            print('File already exists: %s' % save_path)
            print(
                'Skipping this file. If want to recompute and overwrite, use `-o`/`--overwrite`.'
            )
            continue

        if not args.rerun:
            '''
            In many cases this is enough. If you experience deadlock, you can try to use `-r`/`--rerun`.
            '''
            numpy_array = np.load(load_path)
            image = numpy_array['image']
            label_true = numpy_array['label']
            latent = numpy_array['latent']

            image = (image + 1) / 2

            H, W = label_true.shape[:2]
            C = latent.shape[-1]
            X = latent

            from helper_generate_kmeans import generate_kmeans
            dice_score, label_pred, seg_pred = generate_kmeans(
                (H, W, C), latent, label_true, num_workers=num_workers)

            with open(save_path, 'wb+') as f:
                np.savez(f,
                         image=image,
                         label=label_true,
                         latent=latent,
                         label_kmeans=label_pred,
                         seg_kmeans=seg_pred)

            print('SUCCESS! %s, dice: %s' %
                  (load_path.split('/')[-1], dice_score))
            dice_list.append(dice_score)

        else:
            '''
            Because of the frequent deadlock problem, I decided to use the following solution:
            kill and restart whenever a process is taking too long (likely due to deadlock).
            '''
            file_success = False
            while not file_success:
                start = time.time()
                while True:
                    try:
                        proc = subprocess.Popen([
                            'python3', folder + '/helper_generate_kmeans.py',
                            '--load_path', load_path, '--save_path', save_path,
                            '--num_workers',
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
                        # This is determined by the sys.stdout in `helper_generate_kmeans.py`
                        if 'SUCCESS!' in stdout:
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
