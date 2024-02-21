import argparse
import os
import sys
import warnings
from glob import glob
from typing import Tuple

import numpy as np
import yaml
from tqdm import tqdm

sys.path.append('../')
from utils.attribute_hashmap import AttributeHashmap
from utils.diffusion_condensation import diffusion_condensation_catch, diffusion_condensation_msphate
from utils.parse import parse_settings

warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1


def generate_diffusion(
        shape: Tuple[int],
        latent: np.array,
        knn: int = 100,
        num_workers: int = 1,
        random_seed: int = 0,
        use_msphate: bool = True) -> Tuple[float, np.array, np.array]:

    H, W, C = shape
    assert latent.shape == (H * W, C)

    if use_msphate:
        labels_pred, granularities = diffusion_condensation_msphate(
            X=latent, knn=knn, num_workers=num_workers, random_seed=random_seed)
    else:
        labels_pred, granularities = diffusion_condensation_catch(
            X=latent, knn=knn, num_workers=num_workers, random_seed=random_seed)

    return labels_pred, granularities


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Path to config yaml file.',
                        required=True)
    parser.add_argument('-o',
                        '--overwrite',
                        action='store_true',
                        help='If true, overwrite previously computed results.')
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config.config_file_name = args.config
    config = parse_settings(config, log_settings=False)

    files_folder = '%s/%s' % (config.output_save_path, 'numpy_files')
    np_files_path = sorted(glob('%s/%s' % (files_folder, '*.npz')))

    save_path_numpy = '%s/%s' % (config.output_save_path,
                                 'numpy_files_seg_pixel_diffusion')
    os.makedirs(save_path_numpy, exist_ok=True)

    for image_idx in tqdm(range(len(np_files_path))):
        save_path = '%s/%s' % (save_path_numpy,
                               os.path.basename(np_files_path[image_idx]))

        if os.path.exists(save_path) and not args.overwrite:
            print('File already exists: %s' % save_path)
            print(
                'Skipping this file. If want to recompute and overwrite, use `-o`/`--overwrite`.'
            )
            continue

        numpy_array = np.load(np_files_path[image_idx])
        image = numpy_array['image']
        recon = numpy_array['recon']
        label_true = numpy_array['label']

        image = (image + 1) / 2
        recon = (recon + 1) / 2

        H, W = label_true.shape[:2]

        # NOTE: directly run diffusion condensation on pixels
        latent = image.reshape(H*W, -1)
        C = latent.shape[-1]

        labels_pred, granularities = generate_diffusion(
            (H, W, C), latent, num_workers=1)

        with open(save_path, 'wb+') as f:
            np.savez(
                f,
                image=image,
                recon=recon,
                label=label_true,
                latent=latent,
                labels_diffusion=labels_pred,
                granularities_diffusion=granularities,
            )

    print('All diffusion results generated.')

    # Somehow the code may hang at this point...
    # Force exit.
    os._exit(0)
