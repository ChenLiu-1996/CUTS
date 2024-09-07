from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry

import argparse
import torch
import cv2
import sys
import numpy as np
import skimage
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

    medsam = sam_model_registry["vit_b"](
        checkpoint=os.path.join('../MedSAM_checkpoint/', "medsam_vit_b.pth"))
    if use_cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    medsam.to(device)
    medsam.eval()

    if mode == 'generate':
        model = SamAutomaticMaskGenerator(medsam)
    else:
        model = SamPredictor(medsam)

    for i, (x, y) in enumerate(tqdm(test_set)):
        x = x.to(device).float()
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = (x + 1) / 2

        # load image
        image = np.uint8(
            x.permute(0, 2, 3, 1).squeeze(0).cpu().detach().numpy() * 255)
        label = y.permute(0, 2, 3,
                          1).squeeze(0).squeeze(-1).cpu().detach().numpy()

        # Resize to best size for MedSAM
        H, W = image.shape[:2]
        image = skimage.transform.resize(image, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
        label = skimage.transform.resize(label, (1024, 1024), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)

        if mode == 'generate':
            preds = model.generate(image)
            label_medsam_box = np.zeros_like(label)
            for mask_idx in range(len(preds)):
                label_medsam_box[preds[mask_idx]['segmentation']] = mask_idx+1
        else:
            model.set_image(image)

            assert np.min(label) == 0
            if np.max(label) != 1:
                label_medsam_box = np.zeros_like(label)

            else:
                # Following code block directly adapted from
                # https://github.com/mazurowski-lab/segment-anything-medical-evaluation/blob/main/prompt_gen_and_exec_v1.py

                # find coordinates of points in the region
                row,col = np.argwhere(label == 1).T
                # find the four corner coordinates
                y0,x0 = row.min(),col.min()
                y1,x1 = row.max(),col.max()
                bbox = np.array([[x0, y0, x1, y1]])
                preds, _, _ = model.predict(box=bbox,
                                            multimask_output=False)
                label_medsam_box = preds.squeeze(0)

        label_medsam_box = skimage.transform.resize(label_medsam_box, (H, W), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)

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
            np.savez(f, image=x, label=y, label_medsam_box=label_medsam_box)


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

    save_path = '%s/%s' % (config.output_save_path, 'numpy_files_seg_MedSAM_box')

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
