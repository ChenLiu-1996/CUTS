import argparse
import os
import sys
import warnings
from glob import glob
from typing import Tuple

import numpy as np
import phate
import yaml
from tqdm import tqdm

sys.path.append('../')
from utils.attribute_hashmap import AttributeHashmap
from utils.metrics import dice_coeff
from utils.parse import parse_settings
from utils.segmentation import point_hint_seg

warnings.filterwarnings("ignore")


def generate_kmeans(shape: Tuple[int],
                    latent: np.array,
                    label_true: np.array,
                    random_seed: int = 1) -> Tuple[float, np.array, np.array]:

    H, W, C = shape
    assert latent.shape == (H * W, C)

    seg_true = label_true > 0

    # Perform PHATE clustering.
    phate_operator = phate.PHATE(
        n_components=3,
        knn=100,
        n_landmark=500,
        t=2,
        verbose=False,
        random_state=random_seed,
        #  n_jobs=config.num_workers)
        n_jobs=4)

    phate_operator.fit_transform(latent)
    clusters = phate.cluster.kmeans(phate_operator,
                                    n_clusters=10,
                                    random_state=random_seed)

    # [H x W, C] to [H, W, C]
    label_pred = clusters.reshape((H, W))

    seg_pred = point_hint_seg(label_pred=label_pred, label_true=label_true)

    return dice_coeff(seg_pred, seg_true), label_pred, seg_pred


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
    for image_idx in tqdm(range(len(np_files_path))):
        numpy_array = np.load(np_files_path[image_idx])
        image = numpy_array['image']
        label_true = numpy_array['label']
        latent = numpy_array['latent']

        image = (image + 1) / 2

        H, W = label_true.shape[:2]
        C = latent.shape[-1]
        X = latent

        dice_score, label_pred, seg_pred = generate_kmeans((H, W, C), latent,
                                                           label_true)

        with open(
                '%s/%s' %
            (save_path_numpy, 'sample_%s.npz' % str(image_idx).zfill(5)),
                'wb+') as f:
            np.savez(f,
                     image=image,
                     label=label_true,
                     latent=latent,
                     label_kmeans=label_pred,
                     seg_kmeans=seg_pred)

        print('image idx: ', image_idx, 'dice: ', dice_score)
        dice_list.append(dice_score)

    print('\n\nFinal dice coeff: %.3f \u00B1 %.3f' %
          (np.mean(dice_list), np.std(dice_list) / np.sqrt(len(dice_list))))
