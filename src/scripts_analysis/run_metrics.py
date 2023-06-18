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
from utils.metrics import dice_coeff, ergas, guided_relabel, hausdorff, per_class_dice_coeff, per_class_hausdorff, \
    range_aware_ssim, rmse
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


def load_unet(path: str) -> dict:
    numpy_array = np.load(path)
    hashmap = {}
    hashmap['image'] = numpy_array['image']
    hashmap['label_true'] = numpy_array['label_true']
    hashmap['label_unet'] = numpy_array['label_pred']
    return hashmap


def load_nnunet(path: str) -> dict:
    numpy_array = np.load(path)
    hashmap = {}
    hashmap['image'] = numpy_array['image']
    hashmap['label_true'] = numpy_array['label_true']
    hashmap['label_nnunet'] = numpy_array['label_pred']
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Path to config yaml file.',
                        nargs='+',
                        required=True)
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    metric_name_map = {
        'dice': 'Dice Coefficient',
        'hausdorff': 'Hausdorff Distance',
        'ssim': 'SSIM',
        'ergas': 'ERGAS',
        'rmse': 'RMSE'
    }

    if len(args.config) > 1:
        META_ANALYSIS = True
        meta_metrics = {}
        print('Computing the metrics across different experiments.')
    else:
        META_ANALYSIS = False
        print('Computing the metrics in a single experiment.')

    for config_file in args.config:
        config = AttributeHashmap(yaml.safe_load(open(config_file)))
        config.config_file_name = config_file
        config = parse_settings(config, log_settings=False)

        hparams = AttributeHashmap({
            'is_binary': config.is_binary,
        })

        files_folder_baselines = '%s/%s' % (config.output_save_path,
                                            'numpy_files_seg_baselines')
        files_folder_kmeans = '%s/%s' % (config.output_save_path,
                                         'numpy_files_seg_kmeans')
        files_folder_diffusion = '%s/%s' % (config.output_save_path,
                                            'numpy_files_seg_diffusion')
        files_folder_stego = '%s/%s' % (config.output_save_path,
                                        'numpy_files_seg_STEGO')
        files_folder_unet = '%s/%s' % (config.output_save_path,
                                       'numpy_files_seg_supervised_unet')
        files_folder_nnunet = '%s/%s' % (config.output_save_path,
                                         'numpy_files_seg_supervised_nnunet')

        np_files_path_baselines = sorted(
            glob('%s/%s' % (files_folder_baselines, '*.npz')))
        np_files_path_kmeans = sorted(
            glob('%s/%s' % (files_folder_kmeans, '*.npz')))
        np_files_path_diffusion = sorted(
            glob('%s/%s' % (files_folder_diffusion, '*.npz')))
        np_files_path_stego = sorted(
            glob('%s/%s' % (files_folder_stego, '*.npz')))
        np_files_path_unet = sorted(
            glob('%s/%s' % (files_folder_unet, '*.npz')))
        np_files_path_nnunet = sorted(
            glob('%s/%s' % (files_folder_nnunet, '*.npz')))

        num_files = max([
            len(np_files_path_baselines),
            len(np_files_path_kmeans),
            len(np_files_path_diffusion),
            len(np_files_path_stego),
            len(np_files_path_unet),
            len(np_files_path_nnunet),
        ])

        assert len(np_files_path_baselines) == num_files or len(
            np_files_path_baselines) == 0
        assert len(np_files_path_kmeans) == num_files or len(
            np_files_path_kmeans) == 0
        assert len(np_files_path_diffusion) == num_files or len(
            np_files_path_diffusion) == 0
        assert len(np_files_path_stego) == num_files or len(
            np_files_path_stego) == 0
        assert len(np_files_path_unet) == num_files or len(
            np_files_path_unet) == 0
        assert len(np_files_path_nnunet) == num_files or len(
            np_files_path_nnunet) == 0

        has_baselines = True if len(
            np_files_path_baselines) == num_files else False
        has_kmeans = True if len(np_files_path_kmeans) == num_files else False
        has_diffusion = True if len(
            np_files_path_diffusion) == num_files else False
        has_stego = True if len(np_files_path_stego) == num_files else False
        has_unet = True if len(np_files_path_unet) == num_files else False
        has_nnunet = True if len(np_files_path_nnunet) == num_files else False

        entity_tuples = []
        if has_baselines:
            entity_tuples.extend([
                ('random', 'label_true', 'label_random'),
                ('watershed', 'label_true', 'label_watershed'),
                ('felzenszwalb', 'label_true', 'label_felzenszwalb'),
            ])
        if has_kmeans:
            if hparams.is_binary:
                entity_tuples.extend([
                    ('ours (kmeans, binary)', 'label_true', 'seg_kmeans'),
                ])
            else:
                entity_tuples.extend([
                    ('ours (kmeans, multiclass)', 'label_true',
                     'label_kmeans'),
                ])
        if has_diffusion:
            if hparams.is_binary:
                entity_tuples.extend([
                    ('ours (diffusion-persistent, binary)', 'label_true',
                     'seg_diffusion-persistent'),
                    ('ours (diffusion-best, binary)', 'label_true',
                     'seg_diffusion-best'),
                ])
            else:
                entity_tuples.extend([
                    ('ours (diffusion-persistent, multiclass)', 'label_true',
                     'label_diffusion-persistent'),
                    ('ours (diffusion-best, multiclass)', 'label_true',
                     'label_diffusion-best'),
                ])
        if has_stego:
            entity_tuples.extend([
                ('STEGO', 'label_true', 'label_stego'),
            ])
        if has_unet:
            entity_tuples.extend([
                ('Supervised UNet', 'label_true', 'label_unet'),
            ])
        if has_nnunet:
            entity_tuples.extend([
                ('Supervised nn-UNet', 'label_true', 'label_nnunet'),
            ])

        metrics = {
            'dice': {
                tup[0]: []
                for tup in entity_tuples
            },
            'hausdorff': {
                tup[0]: []
                for tup in entity_tuples
            },
            'ssim': {
                tup[0]: []
                for tup in entity_tuples
            },
            'ergas': {
                tup[0]: []
                for tup in entity_tuples
            },
            'rmse': {
                tup[0]: []
                for tup in entity_tuples
            },
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
            if has_unet:
                unet_hashmap = load_unet(np_files_path_unet[image_idx])
            if has_nnunet:
                nnunet_hashmap = load_nnunet(np_files_path_nnunet[image_idx])

            hashmap = combine_hashmaps(baselines_hashmap, kmeans_hashmap,
                                       diffusion_hashmap, stego_hashmap,
                                       unet_hashmap, nnunet_hashmap)

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
                # Relabel all the other predicted labels.
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
                    assert not hparams.is_binary
                    metrics['dice'][entry].append(
                        max([
                            per_class_dice_coeff(
                                label_true=hashmap['label_true'],
                                label_pred=hashmap['labels_diffusion'][i, ...])
                            for i in range(
                                hashmap['labels_diffusion'].shape[0])
                        ]))
                    metrics['hausdorff'][entry].append(
                        min([
                            per_class_hausdorff(
                                label_true=hashmap['label_true'],
                                label_pred=hashmap['labels_diffusion'][i, ...])
                            for i in range(
                                hashmap['labels_diffusion'].shape[0])
                        ]))
                    metrics['ssim'][entry].append(
                        max([
                            range_aware_ssim(
                                label_true=hashmap['label_true'],
                                label_pred=hashmap['labels_diffusion'][i, ...])
                            for i in range(
                                hashmap['labels_diffusion'].shape[0])
                        ]))
                    metrics['ergas'][entry].append(
                        min([
                            ergas(hashmap['label_true'],
                                  hashmap['labels_diffusion'][i, ...]) for i in
                            range(hashmap['labels_diffusion'].shape[0])
                        ]))
                    metrics['rmse'][entry].append(
                        min([
                            rmse(hashmap['label_true'],
                                 hashmap['labels_diffusion'][i, ...]) for i in
                            range(hashmap['labels_diffusion'].shape[0])
                        ]))
                elif p2 == 'seg_diffusion-best':
                    # Get the best among all diffusion segmentations.
                    assert hparams.is_binary
                    metrics['dice'][entry].append(
                        max([
                            dice_coeff(
                                label_true=hashmap['label_true'],
                                label_pred=hashmap['segs_diffusion'][i, ...])
                            for i in range(hashmap['segs_diffusion'].shape[0])
                        ]))
                    metrics['hausdorff'][entry].append(
                        min([
                            hausdorff(
                                label_true=hashmap['label_true'],
                                label_pred=hashmap['segs_diffusion'][i, ...])
                            for i in range(hashmap['segs_diffusion'].shape[0])
                        ]))
                    metrics['ssim'][entry].append(
                        max([
                            range_aware_ssim(
                                label_true=hashmap['label_true'],
                                label_pred=hashmap['segs_diffusion'][i, ...])
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
                    # nan-padding for unavailable measurements.
                    metrics['dice'][entry].append(np.nan)
                    metrics['hausdorff'][entry].append(np.nan)
                    metrics['ssim'][entry].append(np.nan)
                    metrics['ergas'][entry].append(np.nan)
                    metrics['rmse'][entry].append(np.nan)
                else:
                    if hparams.is_binary:
                        metrics['dice'][entry].append(
                            dice_coeff(label_true=hashmap[p1],
                                       label_pred=hashmap[p2]))
                        metrics['hausdorff'][entry].append(
                            hausdorff(label_true=hashmap[p1],
                                      label_pred=hashmap[p2]))
                    else:
                        metrics['dice'][entry].append(
                            per_class_dice_coeff(label_true=hashmap[p1],
                                                 label_pred=hashmap[p2]))
                        metrics['hausdorff'][entry].append(
                            per_class_hausdorff(label_true=hashmap[p1],
                                                label_pred=hashmap[p2]))
                    metrics['ssim'][entry].append(
                        range_aware_ssim(label_true=hashmap[p1],
                                         label_pred=hashmap[p2]))
                    metrics['ergas'][entry].append(
                        ergas(hashmap[p1], hashmap[p2]))
                    metrics['rmse'][entry].append(
                        rmse(hashmap[p1], hashmap[p2]))

        if META_ANALYSIS:
            # Take the mean value over subjects for each metric.
            # Aggregate them across experiments.
            for k1 in metrics.keys():
                for k2 in metrics[k1].keys():
                    try:
                        meta_metrics[k1][k2] += [np.nanmean(metrics[k1][k2])]
                    except:
                        try:
                            meta_metrics[k1][k2] = [
                                np.nanmean(metrics[k1][k2])
                            ]
                        except:
                            meta_metrics[k1] = {}
                            meta_metrics[k1][k2] = [
                                np.nanmean(metrics[k1][k2])
                            ]

        print('\n\nResults (mean \u00B1 sem) for', config_file)

        for key in metric_name_map.keys():
            print('\n\n', metric_name_map[key])
            for (entry, _, _) in entity_tuples:
                print('%s: %.3f \u00B1 %.3f' %
                      (entry, np.nanmean(metrics[key][entry]),
                       np.nanstd(metrics[key][entry]) /
                       np.sqrt(len(metrics[key][entry]))))

    if META_ANALYSIS:
        print('\n\n')
        print('=======================================')
        print('Meta Results (mean \u00B1 std) for', args.config)
        print('=======================================')

        for key in metric_name_map.keys():
            print('\n\n', metric_name_map[key])
            for (entry, _, _) in entity_tuples:
                print('%s: %.3f \u00B1 %.3f' %
                      (entry, np.nanmean(meta_metrics[key][entry]),
                       np.nanstd(meta_metrics[key][entry])))
