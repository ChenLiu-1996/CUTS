import argparse
import os
import sys
from glob import glob
import numpy as np
import yaml

from tqdm import tqdm

# Import from our CUTS codebase.
import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-4])
sys.path.insert(0, import_dir + '/src/utils/')
from attribute_hashmap import AttributeHashmap
from output_saver import squeeze_excessive_dimension

sys.path.insert(0, import_dir + '/src/data_utils/')
from prepare_dataset import prepare_dataset

# Import from STEGO.
STEGO_import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, STEGO_import_dir + '/src/')
from step03_train_segmentation import LitUnsupervisedSegmenter
from data import *
from modules import *


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


def my_app(config: AttributeHashmap, cfg: AttributeHashmap) -> None:
    pytorch_data_dir = cfg.pytorch_data_dir
    result_dir = "../results/predictions/{}".format(cfg.experiment_name)
    os.makedirs(join(result_dir, "img"), exist_ok=True)
    os.makedirs(join(result_dir, "label"), exist_ok=True)
    os.makedirs(join(result_dir, "cluster"), exist_ok=True)
    os.makedirs(join(result_dir, "picie"), exist_ok=True)

    assert len(cfg.model_paths) == 1
    model_path = sorted(glob(cfg.model_paths[0] + '*.ckpt'))[-1]

    model = LitUnsupervisedSegmenter.load_from_checkpoint(model_path)

    test_set, _ = prepare_dataset(config, mode='test')

    save_path = '%s/%s' % (config.output_save_path, 'numpy_files_seg_STEGO')

    os.makedirs(save_path, exist_ok=True)

    model.eval().cuda()

    if cfg.use_ddp:
        par_model = torch.nn.DataParallel(model.net)
    else:
        par_model = model.net

    for i, (x, y) in enumerate(tqdm(test_set)):
        with torch.no_grad():
            img = x.cuda().float()
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)

            img = (img + 1) / 2
            label = y.cuda()

            feats, code1 = par_model(img)
            feats, code2 = par_model(img.flip(dims=[3]))
            code = (code1 + code2.flip(dims=[3])) / 2

            code = F.interpolate(code,
                                 label.shape[-2:],
                                 mode='bilinear',
                                 align_corners=False)

            linear_probs = torch.log_softmax(model.linear_probe(code), dim=1)
            cluster_probs = model.cluster_probe(code, 2, log_probs=True)

            linear_preds = linear_probs.argmax(1)
            cluster_preds = cluster_probs.argmax(1)

            x = x.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            linear_preds = linear_preds.cpu().detach().numpy()
            cluster_preds = cluster_preds.cpu().detach().numpy()

            x = np.moveaxis(x, 1, -1)
            y = np.moveaxis(y, 1, -1)
            x = squeeze_excessive_dimension(x)
            y = squeeze_excessive_dimension(y)

            assert x.shape[0] == 1
            assert y.shape[0] == 1
            assert linear_preds.shape[0] == 1
            assert cluster_preds.shape[0] == 1
            x = x.squeeze(0)
            y = y.squeeze(0)
            linear_preds = linear_preds.squeeze(0)
            cluster_preds = cluster_preds.squeeze(0)

            with open('%s/%s' % (save_path, 'sample_%s.npz' % str(i).zfill(5)),
                      'wb+') as f:
                np.savez(f,
                         image=x,
                         label=y,
                         label_stego=cluster_preds,
                         seg_stego=linear_preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Path to CUTS config yaml file.',
                        required=True)
    parser.add_argument('--eval-config',
                        help='Path to STEGO eval config yaml file.',
                        required=True)
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    config_file_name = args.config
    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config.config_file_name = config_file_name
    config = parse_settings(config)

    # Use batch size of 1.
    config.batch_size = 1

    cfg = AttributeHashmap(yaml.safe_load(open(args.eval_config)))
    cfg = parse_settings(cfg)
    my_app(config, cfg)
