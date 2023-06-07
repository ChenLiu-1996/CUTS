import argparse
import os
from os.path import join
import sys
import numpy as np
import torch.multiprocessing
import torch.multiprocessing
import torch.nn as nn
from tqdm import tqdm
import yaml

# Import from our CUTS codebase.
import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-4])
sys.path.insert(0, import_dir + '/src/utils/')

from attribute_hashmap import AttributeHashmap
from seed import seed_everything

# Import from STEGO.
STEGO_import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, STEGO_import_dir + '/src/')
from data import ContrastiveSegDataset
from modules import *


def update_CUTS_ROOT(config: AttributeHashmap) -> AttributeHashmap:
    CUTS_ROOT = '/'.join(os.path.realpath(__file__).split('/')[:-4])
    for key in config.keys():
        if type(config[key]) == str and '$CUTS_ROOT' in config[key]:
            config[key] = config[key].replace('$CUTS_ROOT', CUTS_ROOT)
    return config


def get_feats(model, loader):
    all_feats = []
    for pack in tqdm(loader):
        img = pack["img"]
        feats = F.normalize(model.forward(img.cuda()).mean([2, 3]), dim=1)
        all_feats.append(feats.to("cpu", non_blocking=True))
    return torch.cat(all_feats, dim=0).contiguous()


def my_app(cfg: AttributeHashmap) -> None:
    pytorch_data_dir = cfg.pytorch_data_dir
    data_dir = join(cfg.output_root, "data")
    log_dir = join(cfg.output_root, "logs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(join(pytorch_data_dir, "nns"), exist_ok=True)

    seed_everything(seed=0)

    image_sets = ["val", "train"]
    dataset_names = ["directory"]
    crop_types = [None]

    res = cfg.res
    n_batches = cfg.batch_size

    if cfg.arch == "dino":
        from modules import DinoFeaturizer, LambdaLayer
        no_ap_model = torch.nn.Sequential(
            DinoFeaturizer(20, cfg),  # dim doesent matter
            LambdaLayer(lambda p: p[0]),
        ).cuda()
    else:
        cut_model = load_model(cfg.model_type, join(cfg.output_root,
                                                    "data")).cuda()
        no_ap_model = nn.Sequential(*list(cut_model.children())[:-1]).cuda()
    par_model = torch.nn.DataParallel(no_ap_model)

    for crop_type in crop_types:
        for image_set in image_sets:
            for dataset_name in dataset_names:
                nice_dataset_name = cfg.dir_dataset_name if dataset_name == "directory" else dataset_name

                feature_cache_file = join(
                    pytorch_data_dir, "nns",
                    "nns_{}_{}_{}_{}_{}.npz".format(cfg.model_type,
                                                    nice_dataset_name,
                                                    image_set, crop_type, res))

                if not os.path.exists(feature_cache_file):
                    print("{} not found, computing".format(feature_cache_file))
                    dataset = ContrastiveSegDataset(
                        pytorch_data_dir=pytorch_data_dir,
                        dataset_name=dataset_name,
                        crop_type=crop_type,
                        image_set=image_set,
                        transform=get_transform(res, False, "center"),
                        target_transform=get_transform(res, True, "center"),
                        cfg=cfg,
                    )

                    loader = DataLoader(dataset,
                                        256,
                                        shuffle=False,
                                        num_workers=cfg.num_workers,
                                        pin_memory=False)

                    with torch.no_grad():
                        normed_feats = get_feats(par_model, loader)
                        all_nns = []
                        step = normed_feats.shape[0] // n_batches
                        print(normed_feats.shape)
                        for i in tqdm(range(0, normed_feats.shape[0], step)):
                            torch.cuda.empty_cache()
                            batch_feats = normed_feats[i:i + step, :]
                            pairwise_sims = torch.einsum(
                                "nf,mf->nm", batch_feats, normed_feats)
                            all_nns.append(torch.topk(pairwise_sims, 8)[1])
                            del pairwise_sims
                        nearest_neighbors = torch.cat(all_nns, dim=0)

                        np.savez_compressed(feature_cache_file,
                                            nns=nearest_neighbors.numpy())
                        print("Saved NNs", cfg.model_type, nice_dataset_name,
                              image_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-config',
                        help='Path to STEGO train config yaml file.',
                        required=True)
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    cfg = AttributeHashmap(yaml.safe_load(open(args.train_config)))
    cfg = update_CUTS_ROOT(cfg)

    my_app(cfg)
