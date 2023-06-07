import argparse
import sys
import warnings
from typing import Tuple

import numpy as np
import phate

sys.path.append('../')
from utils.attribute_hashmap import AttributeHashmap
from utils.metrics import per_class_dice_coeff
from utils.segmentation import label_hint_seg

warnings.filterwarnings("ignore")


def generate_kmeans(shape: Tuple[int],
                    latent: np.array,
                    label_true: np.array,
                    num_workers: int = 1,
                    random_seed: int = 1) -> Tuple[float, np.array, np.array]:

    H, W, C = shape
    assert latent.shape == (H * W, C)

    seg_true = label_true > 0

    # Very occasionally, SVD won't converge.
    try:
        clusters = phate_clustering(latent=latent,
                                    random_seed=random_seed,
                                    num_workers=num_workers)
    except:
        clusters = phate_clustering(latent=latent,
                                    random_seed=random_seed + 1,
                                    num_workers=num_workers)

    # [H x W, C] to [H, W, C]
    label_pred = clusters.reshape((H, W))

    seg_pred = label_hint_seg(label_pred=label_pred, label_true=label_true)

    return per_class_dice_coeff(seg_pred, seg_true), label_pred, seg_pred


def phate_clustering(latent: np.array, random_seed: int,
                     num_workers: int) -> np.array:
    phate_operator = phate.PHATE(n_components=3,
                                 knn=100,
                                 n_landmark=500,
                                 t=2,
                                 verbose=False,
                                 random_state=random_seed,
                                 n_jobs=num_workers)

    phate_operator.fit_transform(latent)
    clusters = phate.cluster.kmeans(phate_operator,
                                    n_clusters=10,
                                    random_state=random_seed)
    return clusters


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--num_workers', type=int, required=True)
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    numpy_array = np.load(args.load_path)
    image = numpy_array['image']
    label_true = numpy_array['label']
    latent = numpy_array['latent']

    image = (image + 1) / 2

    H, W = label_true.shape[:2]
    C = latent.shape[-1]
    X = latent

    dice_score, label_pred, seg_pred = generate_kmeans(
        (H, W, C), latent, label_true, num_workers=args.num_workers)

    with open(args.save_path, 'wb+') as f:
        np.savez(f,
                 image=image,
                 label=label_true,
                 latent=latent,
                 label_kmeans=label_pred,
                 seg_kmeans=seg_pred)

    sys.stdout.write('SUCCESS! %s, dice: %s' %
                     (args.load_path.split('/')[-1], dice_score))
