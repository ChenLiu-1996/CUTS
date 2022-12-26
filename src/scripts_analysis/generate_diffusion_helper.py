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
from utils.diffusion_condensation import diffusion_condensation
from utils.parse import parse_settings

warnings.filterwarnings("ignore")


def generate_diffusion(
        shape: Tuple[int],
        latent: np.array,
        num_workers: int = 1,
        random_seed: int = 1) -> Tuple[float, np.array, np.array]:

    H, W, C = shape
    assert latent.shape == (H * W, C)

    _, (catch_op,
        levels) = diffusion_condensation(latent,
                                         height_width=(H, W),
                                         pos_enc_gamma=0,
                                         num_workers=num_workers,
                                         random_seed=random_seed,
                                         return_all=True)

    labels_pred = np.array([catch_op.NxTs[lvl] for lvl in levels])
    granularities = [len(catch_op.NxTs) + lvl for lvl in levels]

    return labels_pred, granularities, levels, catch_op.gradient


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    numpy_array = np.load(args.load_path)
    image = numpy_array['image']
    recon = numpy_array['recon']
    label_true = numpy_array['label']
    latent = numpy_array['latent']

    image = (image + 1) / 2
    recon = (recon + 1) / 2

    H, W = label_true.shape[:2]
    C = latent.shape[-1]
    X = latent

    labels_pred, granularities, levels, gradients = generate_diffusion(
        (H, W, C), latent)

    with open(args.save_path, 'wb+') as f:
        np.savez(f,
                    image=image,
                    recon=recon,
                    label=label_true,
                    latent=latent,
                    granularities_diffusion=granularities,
                    labels_diffusion=labels_pred,
                    levels=levels,
                    gradients=gradients)

