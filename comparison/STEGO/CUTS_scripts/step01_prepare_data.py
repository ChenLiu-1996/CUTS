# In this file, we prepare the data for STEGO, a new SOTA unsupervised segmentation method.

import argparse
import os
import sys

import numpy as np
import yaml
import cv2
from tqdm import tqdm

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-4])
sys.path.insert(0, import_dir + '/src/')

from data_utils.prepare_dataset import prepare_dataset
from utils.attribute_hashmap import AttributeHashmap
from utils.parse import parse_settings


def process(image: np.array) -> np.array:
    assert image.shape[0] == 1
    image = image.squeeze(0)
    image = image.cpu().numpy()

    image = (image + 1) / 2 * 255
    image = np.moveaxis(image, 0, -1)

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Path to CUTS config yaml file.',
                        required=True)
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config.config_file_name = args.config
    config = parse_settings(config, log_settings=False)

    # Use batch size of 1.
    config.batch_size = 1

    train_set, val_set, _ = prepare_dataset(config)

    # Save the train and val images to the desired location.
    save_folder_train = '../data/%s/imgs/%s' % (config.dataset_name, 'train')
    save_folder_val = '../data/%s/imgs/%s' % (config.dataset_name, 'val')

    os.makedirs(save_folder_train, exist_ok=True)
    os.makedirs(save_folder_val, exist_ok=True)

    for idx, (image, _) in enumerate(tqdm(train_set)):
        image = process(image)
        cv2.imwrite('%s/train_%s.jpg' % (save_folder_train, str(idx).zfill(5)),
                    image)

    for idx, (image, _) in enumerate(tqdm(val_set)):
        image = process(image)
        cv2.imwrite('%s/val_%s.jpg' % (save_folder_val, str(idx).zfill(5)),
                    image)
