import os
import warnings
from multiprocessing import Lock, Pool, shared_memory
from typing import Tuple

import numpy as np
import phate
import torch
from tqdm import tqdm
from utils.diffusion_condensation import cluster_indices_from_mask, diffusion_condensation
from utils.metrics import dice_coeff

warnings.filterwarnings("ignore")


class LatentEvaluator(object):
    """
    Evaluate the latent representation of the unsupervised segmentation model.

    Metrics used:
        Dice coefficient.

    @params:
    `segmentation_paradigm`
        Options implemented

        1. 'kmeans_point':
            [Baseline as in WACV submission]
                In this method, the segmentation is performed using PHATE + K-means clustering.
                We are using some weak supervision: 1 pixel per image.
            Overview:
                Hypothetically, the users would provide 1 pixel of annotation per image,
                and our method will provide binary segmentation that "guesses" the desired foreground.
            Detail:
                For each image, we extract the pixel-level embeddings and perform K-means clustering,
                leading to a K-class label map.
                Then, we use the single-pixel annotation (center pixel of ground truth mask by implementation)
                to identify the desired label ID, and compare that label to the ground truth.

        2. 'diffusion_null':
            [New method with Diffusion Condensation]
                In this method, the segmentation is performed using diffusion condensation.
                We are not using any supervision at all (and hence not providing something too meaningful).
            Overview:
                Hypothetically, the users would provide no annotation,
                and our methods will provide multi-class segmentation that separates distinct features,
                without even trying to "guess" what the desired foreground is.
            Details:
                For each image, we extract the pixel-level embeddings and perform hierarchical clustering using
                using diffusion condensation. By formulating the pixels as nodes in a graph and defining
                connectivity using affinity, we can assign labels to each connected component, which can be
                seen as multi-class segmentation.

                The evaluation is hand-wavy and probably overrepresents the power of the method:
                We evaluate the segmentation individually using the "best matching" label IDs.

        3. 'diffusion_point':
            [New method with Diffusion Condensation]
                In this method, the segmentation is performed using diffusion condensation.
                We are using some weak supervision: 1 pixel per image.
            Overview:
                Hypothetically, the users would provide 1 pixel of annotation per image,
                and our method will provide binary segmentation that "guesses" the desired foreground.
            Details:
                For each image, we extract the pixel-level embeddings and perform hierarchical clustering using
                using diffusion condensation. By formulating the pixels as nodes in a graph and defining
                connectivity using affinity, we can assign labels to each connected component, which can be
                seen as multi-class segmentation.
                Then, we use the single-pixel annotation (center pixel of ground truth mask by implementation)
                to identify the desired label ID, and compare that label to the ground truth.

        4. 'diffusion_distill':
            NOTE: Out of these paradigms, this is the most realistic and useful,
            and meanwhile the most challenging.
            [New method with Diffusion Condensation]
                In this method, the segmentation is performed using diffusion condensation.
                We are using some weak supervision: 1 mask per dataset.
            Overview:
                Hypothetically, the users would provide 1 ground truth mask for 1 image,
                and our methods will provide the corresponding binary segmentation for the rest
                of the images in the dataset by "best matching" the desired foreground.
            Details:
                For each image, we extract the pixel-level embeddings and perform hierarchical clustering using
                using diffusion condensation. By formulating the pixels as nodes in a graph and defining
                connectivity using affinity, we can assign labels to each connected component, which can be
                seen as multi-class segmentation.
                For the 1 image whose ground truth mask is provided, we use its mask to estimate the desired label IDs.
                For the rest of the images in the dataset, we generate their respective multi-class segmetation
                label maps in the same manner, and try to estimate which label IDs are corresponding to the
                desired label IDs in the annotated image.

    `pos_enc_gamma`:
        Weighting for positional encoding.
        Only relevant to the Diffusion Condensation paradigms.
    `save_path`:
        Path to save numpy files and image files.
    `num_workers`:
        For multiprocessing.
    """

    def __init__(self,
                 segmentation_paradigm: str = 'diffusion_null',
                 pos_enc_gamma: float = 0.0,
                 save_path: str = None,
                 num_workers: int = 1,
                 multiprocessing: bool = False,
                 random_seed: int = None) -> None:
        self.segmentation_paradigm = segmentation_paradigm
        self.pos_enc_gamma = pos_enc_gamma
        self.num_workers = num_workers
        self.multiprocessing = multiprocessing
        self.random_seed = random_seed

        self.save_path_numpy = '%s/%s/' % (save_path, 'numpy_files')
        os.makedirs(self.save_path_numpy, exist_ok=True)

    def eval(
        self,
        image_batch: torch.Tensor,
        recon_batch: torch.Tensor,
        label_true_batch: torch.Tensor,
        latent_batch: torch.Tensor,
        metric: str = 'dice',
    ) -> Tuple[float, np.array, np.array]:
        """
        @params:
        `label_true`: ground truth segmentation map.
        `latent`: latent embedding by the model.

        `label_true` and `latent` are expected to have dimension [B, H, W, C].
        """

        image_batch = image_batch.cpu().detach().numpy()
        recon_batch = recon_batch.cpu().detach().numpy()
        label_true_batch = label_true_batch.cpu().detach().numpy()
        latent_batch = latent_batch.cpu().detach().numpy()
        # channel-first to channel-last
        image_batch = np.moveaxis(image_batch, 1, -1)
        recon_batch = np.moveaxis(recon_batch, 1, -1)
        latent_batch = np.moveaxis(latent_batch, 1, -1)

        B, H, W, C = latent_batch.shape
        metrics = []
        if metric == 'dice':
            metric_fn = dice_coeff
        else:
            metric_fn = None

        # Save the images, labels, and latent embeddings as numpy files for future reference.
        for image_idx in tqdm(range(B)):
            self.save_as_numpy(
                image_idx=image_idx,
                image=image_batch[image_idx, ...],
                recon=recon_batch[image_idx, ...],
                label=label_true_batch[image_idx, ...],
                # [H, W, C] to [H x W, C]
                latent=latent_batch[image_idx, ...].reshape((H * W, C)))

        # Multiprocessing for speedup. This may require a good server.
        if self.multiprocessing and \
            self.segmentation_paradigm in ['diffusion_null', 'diffusion_point']:
            # Prepare lock and shared memory.
            global lock
            lock = Lock()
            shm_pointer, shm_array = create_shared_block(shape=(B, H, W))
            signals = [
                (shm_pointer.name, image_idx, B, H, W, self.pos_enc_gamma,
                 latent_batch[image_idx, ...].reshape(
                     (H * W, C)), self.random_seed) for image_idx in range(B)
            ]
            # Run diffusion condensation in parallel.
            with Pool(self.num_workers) as p:
                list(
                    tqdm(p.imap(call_diffusion_condensation, signals),
                         total=len(signals)))
            # Move the numpy array out of the shared memory.
            label_pred_arr = np.zeros_like(shm_array, dtype=np.uint8)
            label_pred_arr[:] = shm_array[:]
            del shm_array
            shm_pointer.close()
            shm_pointer.unlink()

        if metric_fn is not None:
            if self.segmentation_paradigm == 'kmeans_point':
                '''
                1. Perform k-means clustering on the latent embeddings,
                producing unsupervised multi-class clusters.
                2. Estimate which clusters IDs correspond to the desired foreground by
                using the 1 pixel prior per image.
                '''

                for image_idx in tqdm(range(B)):
                    # [H, W, C] to [H x W, C]
                    latent = latent_batch[image_idx, ...].reshape((H * W, C))
                    seg_true = label_true_batch[image_idx, ...] > 0

                    # Perform PHATE clustering.
                    phate_operator = phate.PHATE(n_components=3,
                                                 knn=100,
                                                 n_landmark=500,
                                                 t=2,
                                                 verbose=False,
                                                 random_state=self.random_seed)
                    phate_operator.fit_transform(latent)
                    clusters = phate.cluster.kmeans(
                        phate_operator,
                        n_clusters=10,
                        random_state=self.random_seed)

                    # [H x W, C] to [H, W, C]
                    label_pred = clusters.reshape((H, W))

                    # Find the desired cluster id by finding the "middle point" of the foreground,
                    # defined as the foreground point closest to the foreground centroid.
                    foreground_xys = np.argwhere(
                        label_true_batch[image_idx,
                                         ...])  # shape: [2, num_points]
                    centroid_xy = np.mean(foreground_xys, axis=0)
                    distances = ((foreground_xys - centroid_xy)**2).sum(axis=1)
                    middle_point_xy = foreground_xys[np.argmin(distances)]
                    cluster_id = label_pred[middle_point_xy[0],
                                            middle_point_xy[1]]

                    seg_pred = label_pred == cluster_id

                    if metric_fn is not None:
                        metrics.append(metric_fn(seg_pred, seg_true))
                    else:
                        metrics.append(None)

                    print('image idx: ', image_idx, 'dice: ',
                          metrics[image_idx])

            elif self.segmentation_paradigm == 'diffusion_null':
                '''
                1. Perform diffusion condensation on the latent embeddings,
                producing unsupervised multi-class clusters.
                2. Estimate which clusters IDs correspond to the desired foreground by
                cross-comparing against the ground truth segmentation mask.
                NOTE: Again, this is somewhat cheating. Please accept it.
                '''

                for image_idx in tqdm(range(B)):
                    # [H, W, C] to [H x W, C]
                    latent = latent_batch[image_idx, ...].reshape((H * W, C))
                    seg_true = label_true_batch[image_idx, ...] > 0

                    if self.multiprocessing:
                        label_pred = label_pred_arr[image_idx, ...]
                    else:
                        clusters = diffusion_condensation(
                            X_orig=latent,
                            height_width=(H, W),
                            pos_enc_gamma=self.pos_enc_gamma,
                            random_seed=self.random_seed)
                        label_pred = clusters.reshape((H, W))

                    # NOTE:
                    # This is ALMOST like cheating, as we infer the label IDs from the ground truth mask.
                    # With that said, the multi-class segmentation `label_map` is not affected regardless.
                    cluster_indices, _ = cluster_indices_from_mask(
                        label_pred, seg_true)

                    seg_pred = np.logical_or.reduce(
                        [label_pred == i for i in cluster_indices])

                    if metric_fn is not None:
                        metrics.append(metric_fn(seg_pred, seg_true))
                    else:
                        metrics.append(None)

                    print('image idx: ', image_idx, 'dice: ',
                          metrics[image_idx])

            elif self.segmentation_paradigm == 'diffusion_point':
                '''
                1. Perform diffusion condensation on the latent embeddings,
                producing unsupervised multi-class clusters.
                2. Estimate which clusters IDs correspond to the desired foreground by
                using the 1 pixel prior per image.
                '''

                for image_idx in tqdm(range(B)):
                    # [H, W, C] to [H x W, C]
                    latent = latent_batch[image_idx, ...].reshape((H * W, C))
                    seg_true = label_true_batch[image_idx, ...] > 0

                    if self.multiprocessing:
                        label_pred = label_pred_arr[image_idx, ...]
                    else:
                        clusters = diffusion_condensation(
                            X_orig=latent,
                            height_width=(H, W),
                            pos_enc_gamma=self.pos_enc_gamma,
                            random_seed=self.random_seed)
                        label_pred = clusters.reshape((H, W))

                    # Find the desired cluster id by finding the "middle point" of the foreground,
                    # defined as the foreground point closest to the foreground centroid.
                    foreground_xys = np.argwhere(
                        label_true_batch[image_idx,
                                         ...])  # shape: [2, num_points]
                    centroid_xy = np.mean(foreground_xys, axis=0)
                    distances = ((foreground_xys - centroid_xy)**2).sum(axis=1)
                    middle_point_xy = foreground_xys[np.argmin(distances)]
                    cluster_id = label_pred[middle_point_xy[0],
                                            middle_point_xy[1]]

                    seg_pred = label_pred == cluster_id

                    if metric_fn is not None:
                        metrics.append(metric_fn(seg_pred, seg_true))
                    else:
                        metrics.append(None)

                    print('image idx: ', image_idx, 'dice: ',
                          metrics[image_idx])

            elif self.segmentation_paradigm == 'diffusion_distill':
                raise NotImplementedError

            else:
                raise NotImplementedError

        return metrics

    def save_as_numpy(self, image_idx: int, image: np.array, recon: np.array,
                      label: np.array, latent: np.array) -> None:
        with open(
                '%s/%s' %
            (self.save_path_numpy, 'sample_%s.npz' % str(image_idx).zfill(5)),
                'wb+') as f:
            np.savez(f, image=image, recon=recon, label=label, latent=latent)


def create_shared_block(shape: Tuple[int]):
    '''
    Adapted from https://stackoverflow.com/a/59257364.
    '''
    # Create an empty numpy array.
    temp = np.zeros(shape=shape, dtype=np.uint8)
    # Create shared memory.
    shm_pointer = shared_memory.SharedMemory(create=True, size=temp.nbytes)
    # Create a numpy array backed by shared memory.
    shm_array = np.ndarray(temp.shape, dtype=np.uint8, buffer=shm_pointer.buf)
    shm_array[:] = temp[:]  # Copy the original data into shared memory
    return shm_pointer, shm_array


def call_diffusion_condensation(input_arg_tuple: Tuple) -> None:
    '''
    We have to use 1 single input argument for python multiprocessing.
    The shared memory part is adapted from https://stackoverflow.com/a/59257364.
    '''
    shm_name, image_idx, B, H, W, pos_enc_gamma, latent, random_seed = input_arg_tuple

    existing_shm_pointer = shared_memory.SharedMemory(name=shm_name)
    existing_shm_array = np.ndarray((B, H, W),
                                    dtype=np.uint8,
                                    buffer=existing_shm_pointer.buf)

    clusters = diffusion_condensation(X_orig=latent,
                                      height_width=(H, W),
                                      pos_enc_gamma=pos_enc_gamma,
                                      random_seed=random_seed)
    label_pred = clusters.reshape((H, W))

    lock.acquire()
    existing_shm_array[image_idx, ...] = label_pred.astype(np.uint8)
    lock.release()
    existing_shm_pointer.close()
