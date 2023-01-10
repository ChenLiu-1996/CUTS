import argparse
import sys
import warnings
from glob import glob

import numpy as np
import yaml
from tqdm import tqdm

sys.path.append('../')
from utils.attribute_hashmap import AttributeHashmap
from utils.diffusion_condensation import get_persistent_structures
from utils.metrics import dice_coeff, ergas, guided_relabel, range_aware_ssim, rmse
from utils.parse import parse_settings
from utils.segmentation import label_hint_seg

warnings.filterwarnings("ignore")


def load_baselines(path: str) -> dict:
    numpy_array = np.load(path)
    hashmap = {}
    hashmap['image'] = numpy_array['image']
    hashmap['label_true'] = numpy_array['label']
    hashmap['label_random'] = numpy_array['label_random']
    hashmap['label_watershed'] = numpy_array['label_watershed']
    hashmap['label_felzenszwalb'] = numpy_array['label_felzenszwalb']
    return hashmap


def load_kmeans(path: str) -> dict:
    numpy_array = np.load(path)
    hashmap = {}
    hashmap['image'] = numpy_array['image']
    hashmap['label_true'] = numpy_array['label']
    hashmap['latent'] = numpy_array['latent']
    hashmap['label_kmeans'] = numpy_array['label_kmeans']
    return hashmap


def load_diffusion(path: str) -> dict:
    numpy_array = np.load(path)
    hashmap = {}
    hashmap['image'] = numpy_array['image']
    hashmap['recon'] = numpy_array['recon']
    hashmap['label_true'] = numpy_array['label']
    hashmap['latent'] = numpy_array['latent']
    hashmap['granularities_diffusion'] = numpy_array['granularities_diffusion']
    hashmap['labels_diffusion'] = numpy_array['labels_diffusion']
    return hashmap


def load_stego(path: str) -> dict:
    numpy_array = np.load(path)
    hashmap = {}
    hashmap['seg_stego'] = numpy_array['seg_stego']
    hashmap['label_stego'] = numpy_array['label_stego']
    return hashmap


def combine_hashmaps(*args: dict) -> dict:
    combined = {}
    for hashmap in args:
        for k in hashmap.keys():
            if k not in combined.keys():
                combined[k] = hashmap[k]
    return combined


def segment(hashmap: dict, label_name: str = 'kmeans') -> dict:
    label_true = hashmap['label_true']
    label_pred = hashmap['label_%s' % label_name]

    H, W = label_true.shape
    label_pred = label_pred.reshape((H, W))

    seg = label_hint_seg(label_pred=label_pred, label_true=label_true)
    hashmap['seg_%s' % label_name] = seg

    return hashmap


def segment_every_diffusion(hashmap: dict) -> dict:
    label_true = hashmap['label_true']
    labels_pred = hashmap['labels_diffusion']

    B = labels_pred.shape[0]
    H, W = label_true.shape
    segs = np.zeros_like(labels_pred)
    for i in range(B):
        label_pred = labels_pred[i, ...].reshape((H, W))

        seg = label_hint_seg(label_pred=label_pred, label_true=label_true)

        segs[i, ...] = seg

    hashmap['segs_diffusion'] = segs

    return hashmap


def persistent_structures(hashmap: dict) -> dict:
    label_true = hashmap['label_true']
    labels_diffusion = hashmap['labels_diffusion']

    H, W = label_true.shape
    B = labels_diffusion.shape[0]
    labels_diffusion = labels_diffusion.reshape((B, H, W))
    persistent_label = get_persistent_structures(labels_diffusion)

    hashmap['labels_diffusion'] = labels_diffusion
    hashmap['label_diffusion-persistent'] = persistent_label

    return hashmap


# def metric_permuted_label(fn, mode: str, permutee: np.array,
#                           other_array: np.array) -> List[np.array]:
#     '''
#     Return the (min or max) metric:
#         fn(permutee, other_array)
#     as we permute the label indices of `permutee`.

#     NOTE: Nice try. But it takes too long as it runs each metric several million times.
#     '''
#     indices_from = sorted(list(set(np.unique(permutee)) - set([0])))

#     assert mode in ['min', 'max']

#     if mode == 'min':
#         best_metric = np.inf
#     elif mode == 'max':
#         best_metric = -np.inf

#     for indices_to in itertools.permutations(indices_from):
#         permuted = np.zeros_like(permutee)
#         assert len(indices_from) == len(indices_to)
#         for (i, j) in zip(indices_from, indices_to):
#             permuted[permutee == i] = j

#         metric = fn(permuted, other_array)
#         if mode == 'min':
#             best_metric = min(best_metric, metric)
#         elif mode == 'max':
#             best_metric = max(best_metric, metric)

#     return best_metric

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

    if config.dataset_name == 'retina':
        hparams = AttributeHashmap({
            'is_binary': True,
        })
    elif config.dataset_name == 'berkeley':
        hparams = AttributeHashmap({
            'is_binary': False,
        })
    elif config.dataset_name == 'brain_ventricles':
        hparams = AttributeHashmap({
            'is_binary': True,
        })

    files_folder_baselines = '%s/%s' % (config.output_save_path,
                                        'numpy_files_seg_baselines')
    files_folder_kmeans = '%s/%s' % (config.output_save_path,
                                     'numpy_files_seg_kmeans')
    files_folder_diffusion = '%s/%s' % (config.output_save_path,
                                        'numpy_files_seg_diffusion')
    files_folder_stego = '%s/%s' % (config.output_save_path,
                                    'numpy_files_seg_STEGO')

    np_files_path_baselines = sorted(
        glob('%s/%s' % (files_folder_baselines, '*.npz')))
    np_files_path_kmeans = sorted(
        glob('%s/%s' % (files_folder_kmeans, '*.npz')))
    np_files_path_diffusion = sorted(
        glob('%s/%s' % (files_folder_diffusion, '*.npz')))
    np_files_path_stego = sorted(glob('%s/%s' % (files_folder_stego, '*.npz')))

    num_files = max([
        len(np_files_path_baselines),
        len(np_files_path_kmeans),
        len(np_files_path_diffusion),
        len(np_files_path_stego)
    ])

    assert len(np_files_path_baselines) == num_files or len(
        np_files_path_baselines) == 0
    assert len(np_files_path_kmeans) == num_files or len(
        np_files_path_kmeans) == 0
    assert len(np_files_path_diffusion) == num_files or len(
        np_files_path_diffusion) == 0
    assert len(np_files_path_stego) == num_files or len(
        np_files_path_stego) == 0

    has_baselines = True if len(
        np_files_path_baselines) == num_files else False
    has_kmeans = True if len(np_files_path_kmeans) == num_files else False
    has_diffusion = True if len(
        np_files_path_diffusion) == num_files else False
    has_stego = True if len(np_files_path_stego) == num_files else False

    entity_tuples = []
    if has_baselines:
        entity_tuples.extend([
            ('random', 'label_true', 'label_random'),
            ('watershed', 'label_true', 'label_watershed'),
            ('felzenszwalb', 'label_true', 'label_felzenszwalb'),
        ])
    if has_stego:
        entity_tuples.extend([
            ('STEGO', 'label_true', 'label_stego'),
        ])
    if has_kmeans:
        entity_tuples.extend([
            ('ours (kmeans, multiclass)', 'label_true', 'label_kmeans'),
            ('ours (kmeans, binary)', 'label_true', 'seg_kmeans'),
        ])
    if has_diffusion:
        entity_tuples.extend([
            ('ours (diffusion-persistent, multiclass)', 'label_true',
             'label_diffusion-persistent'),
            ('ours (diffusion-persistent, binary)', 'label_true',
             'seg_diffusion-persistent'),
            ('ours (diffusion-best, multiclass)', 'label_true',
             'label_diffusion-best'),
            ('ours (diffusion-best, binary)', 'label_true',
             'seg_diffusion-best'),
        ])

    metrics = {
        'dice': {tup[0]: []
                 for tup in entity_tuples},
        'ssim': {tup[0]: []
                 for tup in entity_tuples},
        'ergas': {tup[0]: []
                  for tup in entity_tuples},
        'rmse': {tup[0]: []
                 for tup in entity_tuples},
    }

    for image_idx in tqdm(range(num_files)):
        baselines_hashmap, kmeans_hashmap, diffusion_hashmap, stego_hashmap = {}, {}, {}, {}
        if has_baselines:
            baselines_hashmap = load_baselines(
                np_files_path_baselines[image_idx])
        if has_kmeans:
            kmeans_hashmap = load_kmeans(np_files_path_kmeans[image_idx])
        if has_diffusion:
            diffusion_hashmap = load_diffusion(
                np_files_path_diffusion[image_idx])
        if has_stego:
            stego_hashmap = load_stego(np_files_path_stego[image_idx])

        hashmap = combine_hashmaps(baselines_hashmap, kmeans_hashmap,
                                   diffusion_hashmap, stego_hashmap)

        if has_kmeans:
            hashmap = segment(hashmap, label_name='kmeans')
        if has_diffusion:
            hashmap = persistent_structures(hashmap)
            hashmap = segment(hashmap, label_name='diffusion-persistent')
            hashmap = segment_every_diffusion(hashmap)

        # Re-label the label indices for multi-class labels.
        if not hparams.is_binary:
            # Relabel each of the diffusion labels.
            if has_diffusion:
                for i in range(hashmap['labels_diffusion'].shape[0]):
                    hashmap['labels_diffusion'][i, ...] = guided_relabel(
                        label_pred=hashmap['labels_diffusion'][i, ...],
                        label_true=hashmap['label_true'])

            for (_, _, p2) in entity_tuples:
                if p2 not in hashmap.keys():
                    continue
                else:
                    hashmap[p2] = guided_relabel(
                        label_pred=hashmap[p2],
                        label_true=hashmap['label_true'])

        for (entry, p1, p2) in entity_tuples:
            if p2 == 'label_diffusion-best':
                # Get the best among all diffusion labels.
                metrics['dice'][entry].append(
                    max([
                        dice_coeff(hashmap['label_true'],
                                   hashmap['labels_diffusion'][i, ...])
                        for i in range(hashmap['labels_diffusion'].shape[0])
                    ]))
                metrics['ssim'][entry].append(
                    max([
                        range_aware_ssim(hashmap['label_true'],
                                         hashmap['labels_diffusion'][i, ...])
                        for i in range(hashmap['labels_diffusion'].shape[0])
                    ]))
                metrics['ergas'][entry].append(
                    min([
                        ergas(hashmap['label_true'],
                              hashmap['labels_diffusion'][i, ...])
                        for i in range(hashmap['labels_diffusion'].shape[0])
                    ]))
                metrics['rmse'][entry].append(
                    min([
                        rmse(hashmap['label_true'],
                             hashmap['labels_diffusion'][i, ...])
                        for i in range(hashmap['labels_diffusion'].shape[0])
                    ]))
            elif p2 == 'seg_diffusion-best':
                # Get the best among all diffusion segmentations.
                metrics['dice'][entry].append(
                    max([
                        dice_coeff(hashmap['label_true'],
                                   hashmap['segs_diffusion'][i, ...])
                        for i in range(hashmap['segs_diffusion'].shape[0])
                    ]))
                metrics['ssim'][entry].append(
                    max([
                        range_aware_ssim(hashmap['label_true'],
                                         hashmap['segs_diffusion'][i, ...])
                        for i in range(hashmap['segs_diffusion'].shape[0])
                    ]))
                metrics['ergas'][entry].append(
                    min([
                        ergas(hashmap['label_true'],
                              hashmap['segs_diffusion'][i, ...])
                        for i in range(hashmap['segs_diffusion'].shape[0])
                    ]))
                metrics['rmse'][entry].append(
                    min([
                        rmse(hashmap['label_true'],
                             hashmap['segs_diffusion'][i, ...])
                        for i in range(hashmap['segs_diffusion'].shape[0])
                    ]))
            elif p2 not in hashmap.keys():
                metrics['dice'][entry].append(np.nan)
                metrics['ssim'][entry].append(np.nan)
                metrics['ergas'][entry].append(np.nan)
                metrics['rmse'][entry].append(np.nan)
            else:
                metrics['dice'][entry].append(
                    dice_coeff(hashmap[p1], hashmap[p2]))
                metrics['ssim'][entry].append(
                    range_aware_ssim(hashmap[p1], hashmap[p2]))
                metrics['ergas'][entry].append(ergas(hashmap[p1], hashmap[p2]))
                metrics['rmse'][entry].append(rmse(hashmap[p1], hashmap[p2]))

    if hparams.is_binary:
        print('\n\nDice Coefficient')
        for (entry, _, _) in entity_tuples:
            print('%s: %.3f \u00B1 %.3f' %
                  (entry, np.mean(
                      metrics['dice'][entry]), np.std(metrics['dice'][entry]) /
                   np.sqrt(len(metrics['dice'][entry]))))

    print('\n\nSSIM')
    for (entry, _, _) in entity_tuples:
        print('%s: %.3f \u00B1 %.3f' % (entry, np.mean(
            metrics['ssim'][entry]), np.std(metrics['ssim'][entry]) /
                                        np.sqrt(len(metrics['ssim'][entry]))))

    print('\n\nERGAS')
    for (entry, _, _) in entity_tuples:
        print('%s: %.3f \u00B1 %.3f' % (entry, np.mean(
            metrics['ergas'][entry]), np.std(metrics['ergas'][entry]) /
                                        np.sqrt(len(metrics['ergas'][entry]))))

    print('\n\nRMSE')
    for (entry, _, _) in entity_tuples:
        print('%s: %.3f \u00B1 %.3f' % (entry, np.mean(
            metrics['rmse'][entry]), np.std(metrics['rmse'][entry]) /
                                        np.sqrt(len(metrics['rmse'][entry]))))
