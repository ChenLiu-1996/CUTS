import argparse
import sys
import warnings
from glob import glob

import numpy as np
import yaml
from tqdm import tqdm

sys.path.append('../')
from utils.attribute_hashmap import AttributeHashmap
from utils.metrics import dice_coeff, ergas, rmse, ssim
from utils.parse import parse_settings
from utils.diffusion_condensation import most_persistent_structures

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
    hashmap['seg_kmeans'] = numpy_array['seg_kmeans']
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


def combine_hashmaps(*args: dict) -> dict:
    combined = {}
    for hashmap in args:
        for k in hashmap.keys():
            combined[k] = hashmap[k]
    return combined


def segment_diffusion(hashmap: dict) -> dict:
    '''
    Produce segmentation from the diffusion condensation results.
    '''
    label_true = hashmap['label_true']
    labels_diffusion = hashmap['labels_diffusion']

    H, W = label_true.shape
    B = labels_diffusion.shape[0]
    labels_diffusion = labels_diffusion.reshape((B, H, W))
    label =labels_diffusion[B//2, ...]
    # label, _ = most_persistent_structures(labels_diffusion)

    # Use a single point from ground truth as a hint provider.
    # Find the desired cluster id by finding the "middle point" of the foreground,
    # defined as the foreground point closest to the foreground centroid.
    foreground_xys = np.argwhere(label_true)  # shape: [2, num_points]
    centroid_xy = np.mean(foreground_xys, axis=0)
    distances = ((foreground_xys - centroid_xy)**2).sum(axis=1)
    middle_point_xy = foreground_xys[np.argmin(distances)]
    cluster_id = label[middle_point_xy[0], middle_point_xy[1]]
    seg = label == cluster_id

    hashmap['seg_diffusion'] = seg

    return hashmap


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

    files_folder_baselines = '%s/%s' % (config.output_save_path,
                                        'numpy_files_seg_baselines')
    files_folder_kmeans = '%s/%s' % (config.output_save_path,
                                     'numpy_files_seg_kmeans')
    files_folder_diffusion = '%s/%s' % (config.output_save_path,
                                        'numpy_files_seg_diffusion')

    np_files_path_baselines = sorted(
        glob('%s/%s' % (files_folder_baselines, '*.npz')))
    np_files_path_kmeans = sorted(
        glob('%s/%s' % (files_folder_kmeans, '*.npz')))
    np_files_path_diffusion = sorted(
        glob('%s/%s' % (files_folder_diffusion, '*.npz')))

    assert len(np_files_path_baselines) == len(np_files_path_kmeans)
    assert len(np_files_path_baselines) == len(np_files_path_diffusion)

    candidates = [
        'random', 'watershed', 'felzenszwalb', 'seg_kmeans', 'seg_diffusion'
    ]
    metrics = {
        'dice': {c: []
                 for c in candidates},
        'ssim': {c: []
                 for c in candidates},
        'ergas': {c: []
                  for c in candidates},
        'rmse': {c: []
                 for c in candidates},
    }

    for image_idx in tqdm(range(len(np_files_path_baselines))):
        baselines_hashmap = load_baselines(np_files_path_baselines[image_idx])
        kmeans_hashmap = load_kmeans(np_files_path_kmeans[image_idx])
        diffusion_hashmap = load_diffusion(np_files_path_diffusion[image_idx])

        assert (baselines_hashmap['image'] == kmeans_hashmap['image']
                ).all() and (baselines_hashmap['image']
                             == diffusion_hashmap['image']).all()
        assert (baselines_hashmap['label_true'] == kmeans_hashmap['label_true']
                ).all() and (baselines_hashmap['label_true']
                             == diffusion_hashmap['label_true']).all()

        hashmap = combine_hashmaps(baselines_hashmap, kmeans_hashmap,
                                   diffusion_hashmap)

        hashmap = segment_diffusion(hashmap)

        metrics['dice']['random'].append(
            dice_coeff(hashmap['label_true'], hashmap['label_random']))
        metrics['dice']['watershed'].append(
            dice_coeff(hashmap['label_true'], hashmap['label_watershed']))
        metrics['dice']['felzenszwalb'].append(
            dice_coeff(hashmap['label_true'], hashmap['label_felzenszwalb']))
        metrics['dice']['seg_kmeans'].append(
            dice_coeff(hashmap['label_true'], hashmap['seg_kmeans']))
        metrics['dice']['seg_diffusion'].append(
            dice_coeff(hashmap['label_true'], hashmap['seg_diffusion']))

        metrics['ssim']['random'].append(
            ssim(hashmap['label_true'], hashmap['label_random']))
        metrics['ssim']['watershed'].append(
            ssim(hashmap['label_true'], hashmap['label_watershed']))
        metrics['ssim']['felzenszwalb'].append(
            ssim(hashmap['label_true'], hashmap['label_felzenszwalb']))
        metrics['ssim']['seg_kmeans'].append(
            ssim(hashmap['label_true'], hashmap['seg_kmeans']))
        metrics['ssim']['seg_diffusion'].append(
            ssim(hashmap['label_true'], hashmap['seg_diffusion']))

        metrics['ergas']['random'].append(
            ergas(hashmap['label_true'], hashmap['label_random']))
        metrics['ergas']['watershed'].append(
            ergas(hashmap['label_true'], hashmap['label_watershed']))
        metrics['ergas']['felzenszwalb'].append(
            ergas(hashmap['label_true'], hashmap['label_felzenszwalb']))
        metrics['ergas']['seg_kmeans'].append(
            ergas(hashmap['label_true'], hashmap['seg_kmeans']))
        metrics['ergas']['seg_diffusion'].append(
            ergas(hashmap['label_true'], hashmap['seg_diffusion']))

        metrics['rmse']['random'].append(
            rmse(hashmap['label_true'], hashmap['label_random']))
        metrics['rmse']['watershed'].append(
            rmse(hashmap['label_true'], hashmap['label_watershed']))
        metrics['rmse']['felzenszwalb'].append(
            rmse(hashmap['label_true'], hashmap['label_felzenszwalb']))
        metrics['rmse']['seg_kmeans'].append(
            rmse(hashmap['label_true'], hashmap['seg_kmeans']))
        metrics['rmse']['seg_diffusion'].append(
            rmse(hashmap['label_true'], hashmap['seg_diffusion']))

    print('\n\nDice Coefficient')
    print('random: %.3f \u00B1 %.3f' % (np.mean(
        metrics['dice']['random']), np.std(metrics['dice']['random'])))
    print('watershed: %.3f \u00B1 %.3f' % (np.mean(
        metrics['dice']['watershed']), np.std(metrics['dice']['watershed'])))
    print('felzenszwalb: %.3f \u00B1 %.3f' %
          (np.mean(metrics['dice']['felzenszwalb']),
           np.std(metrics['dice']['felzenszwalb'])))
    print('ours (kmeans): %.3f \u00B1 %.3f' % (np.mean(
        metrics['dice']['seg_kmeans']), np.std(metrics['dice']['seg_kmeans'])))
    print('ours (diffusion): %.3f \u00B1 %.3f' %
          (np.mean(metrics['dice']['seg_diffusion']),
           np.std(metrics['dice']['seg_diffusion'])))

    print('\n\nSSIM')
    print('random: %.3f \u00B1 %.3f' % (np.mean(
        metrics['ssim']['random']), np.std(metrics['ssim']['random'])))
    print('watershed: %.3f \u00B1 %.3f' % (np.mean(
        metrics['ssim']['watershed']), np.std(metrics['ssim']['watershed'])))
    print('felzenszwalb: %.3f \u00B1 %.3f' %
          (np.mean(metrics['ssim']['felzenszwalb']),
           np.std(metrics['ssim']['felzenszwalb'])))
    print('ours (kmeans): %.3f \u00B1 %.3f' % (np.mean(
        metrics['ssim']['seg_kmeans']), np.std(metrics['ssim']['seg_kmeans'])))
    print('ours (diffusion): %.3f \u00B1 %.3f' %
          (np.mean(metrics['ssim']['seg_diffusion']),
           np.std(metrics['ssim']['seg_diffusion'])))

    print('\n\nERGAS')
    print('random: %.3f \u00B1 %.3f' % (np.mean(
        metrics['ergas']['random']), np.std(metrics['ergas']['random'])))
    print('watershed: %.3f \u00B1 %.3f' % (np.mean(
        metrics['ergas']['watershed']), np.std(metrics['ergas']['watershed'])))
    print('felzenszwalb: %.3f \u00B1 %.3f' %
          (np.mean(metrics['ergas']['felzenszwalb']),
           np.std(metrics['ergas']['felzenszwalb'])))
    print('ours (kmeans): %.3f \u00B1 %.3f' %
          (np.mean(metrics['ergas']['seg_kmeans']),
           np.std(metrics['ergas']['seg_kmeans'])))
    print('ours (diffusion): %.3f \u00B1 %.3f' %
          (np.mean(metrics['ergas']['seg_diffusion']),
           np.std(metrics['ergas']['seg_diffusion'])))

    print('\n\nRMSE')
    print('random: %.3f \u00B1 %.3f' % (np.mean(
        metrics['rmse']['random']), np.std(metrics['rmse']['random'])))
    print('watershed: %.3f \u00B1 %.3f' % (np.mean(
        metrics['rmse']['watershed']), np.std(metrics['rmse']['watershed'])))
    print('felzenszwalb: %.3f \u00B1 %.3f' %
          (np.mean(metrics['rmse']['felzenszwalb']),
           np.std(metrics['rmse']['felzenszwalb'])))
    print('ours (kmeans): %.3f \u00B1 %.3f' % (np.mean(
        metrics['rmse']['seg_kmeans']), np.std(metrics['rmse']['seg_kmeans'])))
    print('ours (diffusion): %.3f \u00B1 %.3f' %
          (np.mean(metrics['rmse']['seg_diffusion']),
           np.std(metrics['rmse']['seg_diffusion'])))
