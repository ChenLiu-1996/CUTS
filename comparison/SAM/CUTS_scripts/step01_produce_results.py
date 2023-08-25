from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry

import argparse
import torch
import cv2
import sys
import numpy as np
import torch.nn.init

import os
import numpy as np

import torch.multiprocessing
import yaml
from tqdm import tqdm

# Import from our CUTS codebase.
import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-4])
sys.path.insert(0, import_dir + '/src/utils/')
from attribute_hashmap import AttributeHashmap
from output_saver import squeeze_excessive_dimension

sys.path.insert(0, import_dir + '/src/data_utils/')
from prepare_dataset import prepare_dataset

use_cuda = torch.cuda.is_available()


def parse_settings(config: AttributeHashmap) -> AttributeHashmap:
    CUTS_ROOT = '/'.join(os.path.realpath(__file__).split('/')[:-4])
    for key in config.keys():
        if type(config[key]) == str and '$CUTS_ROOT' in config[key]:
            config[key] = config[key].replace('$CUTS_ROOT', CUTS_ROOT)

    for key in config.keys():
        if type(config[key]) == list:
            for i, item in enumerate(config[key]):
                if type(item) == str and '$CUTS_ROOT' in item:
                    config[key][i] = item.replace('$CUTS_ROOT', CUTS_ROOT)

    if 'lr' in config.keys():
        config.lr = float(config.lr)
    return config


def run(test_set, save_path, mode) -> None:
    use_cuda = torch.cuda.is_available()

    sam = sam_model_registry["default"](
        checkpoint=os.path.join('../SAM_checkpoint/', "sam_vit_h_4b8939.pth"))
    if use_cuda:
        sam.cuda()

    if mode == 'generate':
        model = SamAutomaticMaskGenerator(sam)
    else:
        model = SamPredictor(sam)

    for i, (x, y) in enumerate(tqdm(test_set)):
        if use_cuda:
            x = x.cuda().float()
        else:
            x = x.float()
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = (x + 1) / 2

        # load image
        image = np.uint8(
            x.permute(0, 2, 3, 1).squeeze(0).cpu().detach().numpy() * 255)
        label = y.permute(0, 2, 3,
                          1).squeeze(0).squeeze(-1).cpu().detach().numpy()

        if mode == 'generate':
            preds = model.generate(image)
            label_sam = np.zeros_like(label)
            for mask_idx in range(len(preds)):
                label_sam[preds[mask_idx]['segmentation']] = mask_idx+1
        else:
            model.set_image(image)
            assert np.min(label) == 0
            if np.max(label) != 1:
                label_sam = np.zeros_like(label)

            else:
                # Following code block directly adapted from
                # https://github.com/mazurowski-lab/segment-anything-medical-evaluation/blob/main/prompt_gen_and_exec_v1.py
                padded_mask = np.uint8(np.pad(label, ((1, 1), (1, 1)), 'constant'))
                dist_img = cv2.distanceTransform(padded_mask,
                                                distanceType=cv2.DIST_L2,
                                                maskSize=5).astype(
                                                    np.float32)[1:-1, 1:-1]
                # NOTE: numpy and opencv have inverse definition of row and column
                # NOTE: SAM and opencv have the same definition
                cY, cX = np.where(dist_img == dist_img.max())
                # NOTE: random seems to change DC by +/-1e-4
                # Random sample one point with largest distance
                random_idx = np.random.randint(0, len(cX))
                cX, cY = int(cX[random_idx]), int(cY[random_idx])

                # point: farthest from the object boundary
                pc = [(cX, cY)]
                pl = [1]
                preds, _, _ = model.predict(point_coords=np.array(pc),
                                        point_labels=np.array(pl),
                                        multimask_output=False)

                label_sam = preds.squeeze(0)

        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        x = np.moveaxis(x, 1, -1)
        y = np.moveaxis(y, 1, -1)
        x = squeeze_excessive_dimension(x)
        y = squeeze_excessive_dimension(y)

        assert x.shape[0] == 1
        assert y.shape[0] == 1
        x = x.squeeze(0)
        y = y.squeeze(0)

        with open('%s/%s' % (save_path, 'sample_%s.npz' % str(i).zfill(5)),
                  'wb+') as f:
            np.savez(f, image=x, label=y, label_sam=label_sam)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Path to CUTS config yaml file.',
                        required=True)
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    config_file_name = args.config
    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config.config_file_name = config_file_name
    config = parse_settings(config)

    # Use batch size of 1.
    config.batch_size = 1

    test_set, _ = prepare_dataset(config, mode='test')

    save_path = '%s/%s' % (config.output_save_path, 'numpy_files_seg_SAM')

    os.makedirs(save_path, exist_ok=True)

    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    if config.dataset_name == 'berkeley':
        mode = 'generate'
    else:
        assert config.dataset_name in [
            'retina', 'brain_ventricles', 'brain_tumor'
        ]
        mode = 'predict'
    run(test_set, save_path, mode)
