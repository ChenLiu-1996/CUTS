import argparse
import os
import sys
import warnings
from glob import glob

import numpy as np
import multiscale_phate
import yaml
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
import scprep

sys.path.append('../')
from utils.attribute_hashmap import AttributeHashmap
from utils.diffusion_condensation import continuous_renumber, get_persistent_structures
from utils.parse import parse_settings
from utils.segmentation import label_hint_seg

warnings.filterwarnings("ignore")


def find_nearest_idx(arr: np.array, num: float) -> int:
    return np.abs(arr - num).argmin()


def plot_diffusion_graph(data_hashmap: dict, num_granularities: int = 20):

    label = data_hashmap['labels_diffusion']

    msphate_op = multiscale_phate.Multiscale_PHATE(knn=50,
                                                   landmarks=100,
                                                   random_state=0,
                                                   n_jobs=1)

    msphate_op.fit(normalize(latent, axis=1))
    levels = msphate_op.levels
    assert levels[0] == 0
    levels = levels[1:]  # Ignore finest resolution of all-distinct labels.
    msphate_granularities = [len(msphate_op.NxTs) + lvl for lvl in levels]
    diffusion_granularities = np.arange(len(data_hashmap['labels_diffusion']))

    if num_granularities > 0:
        msphate_idx_selected = [
            find_nearest_idx(msphate_granularities, num) for num in
            np.linspace(msphate_granularities[0], msphate_granularities[-1],
                        num_granularities)
        ]

    x_list, y_list, z_list, c_list, s_list = [], [], [], [], []
    x_lnk_list, y_lnk_list, z_lnk_list, c_lnk_list = [], [], [], []
    for i in range(num_granularities):

        # Record (x, y, z, c) at each granularity
        curr_embeddings, curr_clusters, curr_sizes = msphate_op.transform(
            visualization_level=levels[msphate_idx_selected[i]],
            cluster_level=levels[msphate_idx_selected[i]])

        x_list.extend(
            list(curr_embeddings[:, 0] /
                 (np.ptp(curr_embeddings[:, 0]) + 1e-9)))
        y_list.extend(
            list(curr_embeddings[:, 1] /
                 (np.ptp(curr_embeddings[:, 1]) + 1e-9)))
        z_list.extend([i for _ in range(len(curr_embeddings))])
        c_list.extend(list(curr_clusters))
        s_list.extend(list(curr_sizes / (np.ptp(curr_sizes) + 1e-9) * 1e3))

    x_list = np.array(x_list)
    y_list = np.array(y_list)
    z_list = np.array(z_list)
    c_list = np.array(c_list)
    s_list = np.array(s_list)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(projection='3d')
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 2))
    ax.scatter(x_list,
               y_list,
               z_list,
               c=c_list,
               s=s_list,
               alpha=0.5,
               cmap='tab20')
    ax.view_init(10, 0)
    fig.subplots_adjust(top=2.0, bottom=1.0)
    return fig


if __name__ == '__main__':
    random_seed = 0
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Path to config yaml file.',
                        required=True)
    parser.add_argument('--image-idx',
                        help='Image index.',
                        type=int,
                        required=True)
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config.config_file_name = args.config
    config = parse_settings(config, log_settings=False)

    files_folder_diffusion = '%s/%s' % (config.output_save_path,
                                        'numpy_files_seg_diffusion')
    figure_folder = '%s/%s' % (config.output_save_path, 'paper_figure')
    phate_folder = '%s/%s' % (config.output_save_path, 'numpy_files_phate')

    os.makedirs(figure_folder, exist_ok=True)
    os.makedirs(phate_folder, exist_ok=True)

    files_path_diffusion = sorted(
        glob('%s/%s' % (files_folder_diffusion, '*.npz')))

    numpy_array_diffusion = np.load(files_path_diffusion[args.image_idx])

    image = numpy_array_diffusion['image']
    label_true = numpy_array_diffusion['label'].astype(np.int16)
    latent = numpy_array_diffusion['latent']
    labels_diffusion = numpy_array_diffusion['labels_diffusion']

    H, W = label_true.shape[:2]
    B = labels_diffusion.shape[0]

    persistent_structures = get_persistent_structures(
        labels_diffusion.reshape((B, H, W)))
    seg_persistent = label_hint_seg(label_pred=persistent_structures,
                                    label_true=label_true)

    data_hashmap = {
        'image': image,
        'label_true': label_true,
        'labels_diffusion': labels_diffusion,
        'persistent_structures': persistent_structures,
        'seg_persistent': seg_persistent,
    }

    fig = plot_diffusion_graph(data_hashmap=data_hashmap)

    fig_path = '%s/diffusion_graph_sample_%s.png' % (
        figure_folder, str(args.image_idx).zfill(5))
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches='tight')
