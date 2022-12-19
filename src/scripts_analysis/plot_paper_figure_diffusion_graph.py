import argparse
import os
import sys
import warnings
from glob import glob

import numpy as np
import phate
import yaml
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize

sys.path.append('../')
from utils.attribute_hashmap import AttributeHashmap
from utils.diffusion_condensation import continuous_renumber, get_persistent_structures
from utils.parse import parse_settings
from utils.segmentation import label_hint_seg

warnings.filterwarnings("ignore")


def plot_diffusion_graph(data_hashmap: dict, data_phate: np.array):

    label = data_hashmap['labels_diffusion']

    iter_dim = np.max(data_phate) - np.min(data_phate)
    z_values = np.linspace(0, iter_dim, label.shape[0])

    Xs, Ys, Zs, Cs = [], [], [], []
    Xs_link, Ys_link, Zs_link, Cs_link = [], [], [], []
    # matrix = np.zeros((H, W, Z))
    for gran_idx, z in enumerate(z_values):
        curr_diffusion = label[gran_idx, ...]
        idx_arr = np.where(curr_diffusion)[0]
        Xs.extend(list(data_phate[idx_arr, 0]))
        Ys.extend(list(data_phate[idx_arr, 1]))
        Zs.extend([z for _ in range(len(idx_arr))])
        Cs.extend(
            list(continuous_renumber(curr_diffusion[idx_arr])))

        if gran_idx > 0 and gran_idx < len(z_values) - 1:
            prev_diffusion = label[gran_idx - 1, ...]
            prev_z = z_values[gran_idx - 1]
            curr_z = z_values[gran_idx]
            for cluster_id in np.unique(curr_diffusion):
                cluster_idx_arr_curr = np.where(
                    curr_diffusion == cluster_id)[0]
                cluster_idx_arr_prev = np.where(
                    prev_diffusion == cluster_id)[0]
                cluster_x_curr = np.mean(data_phate[cluster_idx_arr_curr, 0])
                cluster_y_curr = np.mean(data_phate[cluster_idx_arr_curr, 1])
                cluster_x_prev = np.mean(data_phate[cluster_idx_arr_prev, 0])
                cluster_y_prev = np.mean(data_phate[cluster_idx_arr_prev, 1])

                color = continuous_renumber(
                    curr_diffusion)[cluster_idx_arr_curr]
                assert len(np.unique(color)) == 1
                color = color[0]

                num_filler = 5
                Xs_link.extend(list(np.linspace(cluster_x_prev, cluster_x_curr, num_filler)))
                Ys_link.extend(list(np.linspace(cluster_y_prev, cluster_y_curr, num_filler)))
                Zs_link.extend(list(np.linspace(prev_z, curr_z, num_filler)))
                Cs_link.extend([color for _ in range(num_filler)])

    Xs = np.array(Xs)
    Ys = np.array(Ys)
    Zs = np.array(Zs)
    Cs = np.array(Cs)
    Xs_link = np.array(Xs_link)
    Ys_link = np.array(Ys_link)
    Zs_link = np.array(Zs_link)
    Cs_link = np.array(Cs_link)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(Xs, Ys, Zs, c=Cs, alpha=0.05, cmap='tab20')
    ax.scatter(Xs_link, Ys_link, Zs_link, c=Cs_link, alpha=0.5, cmap='tab20')
    ax.set_box_aspect((np.ptp(Xs), np.ptp(Ys), np.ptp(Zs)))
    ax.view_init(-5, 0)
    ax.set_axis_off()

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

    phate_path = '%s/sample_%s.npz' % (phate_folder, str(
        args.image_idx).zfill(5))
    if os.path.exists(phate_path):
        # Load the phate data if exists.
        data_phate_numpy = np.load(phate_path)
        data_phate = data_phate_numpy['data_phate']
    else:
        # Otherwise, generate the phate data.
        phate_op = phate.PHATE(random_state=random_seed,
                               n_jobs=config.num_workers)

        data_phate = phate_op.fit_transform(normalize(latent, axis=1))
        with open(phate_path, 'wb+') as f:
            np.savez(f, data_phate=data_phate)

    fig = plot_diffusion_graph(data_hashmap=data_hashmap,
                               data_phate=data_phate)

    fig_path = '%s/diffusion_graph_sample_%s.png' % (
        figure_folder, str(args.image_idx).zfill(5))
    fig.tight_layout()
    fig.savefig(fig_path)
