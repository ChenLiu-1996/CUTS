import heapq
from typing import List, Tuple

import numpy as np
import pandas as pd
import multiscale_phate
from CATCH import catch
from sklearn.preprocessing import normalize
from utils.metrics import dice_coeff


def diffusion_condensation_catch(X: np.array,
                                 knn: int,
                                 num_workers: int = 1,
                                 random_seed: int = 0) -> np.array:
    '''
    `X` : [N, C] feature matrix,
        where N := number of feature vectors
              C := number of features
    '''
    X = normalize(X, axis=1)
    data = pd.DataFrame(X)

    # Very occasionally, SVD won't converge.
    success = False
    for seed_increment in range(5):
        try:
            catch_op = catch.CATCH(knn=knn,
                                   random_state=random_seed + seed_increment,
                                   n_jobs=num_workers)
            catch_op.fit(data)
            success = True
        except:
            pass
        if success:
            break

    levels = catch_op.transform()
    labels_pred = np.array([catch_op.NxTs[lvl] for lvl in levels])
    granularities = [len(catch_op.NxTs) + lvl for lvl in levels]
    return labels_pred, granularities


def diffusion_condensation_msphate(X: np.array,
                                   knn: int,
                                   num_workers: int = 1,
                                   random_seed: int = 0) -> np.array:
    '''
    `X` : [N, C] feature matrix,
        where N := number of feature vectors
              C := number of features
    '''
    msphate_op = multiscale_phate.Multiscale_PHATE(knn=knn,
                                                   random_state=random_seed,
                                                   n_jobs=num_workers)
    msphate_op.fit(normalize(X, axis=1))
    levels = msphate_op.levels
    assert levels[0] == 0
    levels = levels[1:]  # Ignore finest resolution of all-distinct labels.

    labels_pred = np.array([msphate_op.NxTs[lvl] for lvl in levels])
    granularities = [len(msphate_op.NxTs) + lvl for lvl in levels]
    return labels_pred, granularities


# def pos_enc_sinusoid(shape: Tuple[int]) -> np.array:
#     '''
#     Positional encoding for 2D image.
#     Encoding scheme: Sinusoidal.

#     `shape`: [H, W, C].
#     '''

#     H, W, C = shape
#     freq = C // 2

#     pos_enc = np.zeros((H, W, C))
#     multiplier = np.exp(np.arange(0, freq, 2) * -(np.log(1e5) / freq))

#     pos_H = np.arange(H)[:, None]
#     pos_W = np.arange(W)[:, None]

#     pos_enc[:, :, :freq:2] = np.sin(pos_H * multiplier)[:, :, None].transpose(
#         0, 2, 1).repeat(W, axis=1)
#     pos_enc[:, :, 1:freq:2] = np.cos(pos_H * multiplier)[:, :, None].transpose(
#         0, 2, 1).repeat(W, axis=1)
#     pos_enc[:, :, freq::2] = np.sin(pos_W * multiplier)[:, :, None].transpose(
#         2, 0, 1).repeat(H, axis=0)
#     pos_enc[:, :,
#             freq + 1::2] = np.cos(pos_W * multiplier)[:, :, None].transpose(
#                 2, 0, 1).repeat(H, axis=0)

#     return pos_enc

# def diffusion_condensation_simple(X_orig: np.array,
#                                   height_width: Tuple[int] = (128, 128),
#                                   pos_enc_gamma: float = 0.0,
#                                   similarity_thr: float = 0.95,
#                                   convergence_ratio: float = 1e-4,
#                                   return_all_segs: bool = False) -> np.array:
#     '''
#     `X_orig` : [N, C] feature matrix,
#         where N := number of feature vectors
#               C := number of features
#     `pos_enc_gamma`: weighting for positional encoding.
#     `similarity_thr`: a threshold above which nodes are considered connected.
#     `convergence_ratio`: if values in the affinitiy matrix A
#                          are mostly (judged by this ratio) identical,
#                          we consider the diffusion condensation to have converged.

#     Returns the clusters (distinct connected components in the converged affinity matrix).
#     `clusters`: [N,] non-negative integers.

#     Math:
#         `A`: affinity matrix. Here defined as a scaled version of cosine similarity.
#         `D`: degree matrix.
#         `P`: diffusion matrix. P := D^-1 A.
#     '''
#     N, C = X_orig.shape

#     X = normalize(X_orig)
#     if pos_enc_gamma > 0:
#         pos_enc = pos_enc_sinusoid((*height_width, C))
#         pos_enc = pos_enc.reshape((-1, C))
#         X += pos_enc_gamma * normalize(pos_enc)

#     A_prev = np.zeros((X.shape[0], X.shape[0]))

#     if return_all_segs:
#         all_segs = []
#     else:
#         all_segs = None

#     converged = False
#     while not converged:
#         A = 1 / 2 + 1 / 2 * cosine_similarity(X)
#         A[A < similarity_thr] = 0

#         MAE = np.sum(np.abs(A - A_prev))
#         A_prev = A

#         D_inv = np.diag(1.0 / np.sum(A, axis=1))
#         D_inv_sparse = sparse.csr_matrix(D_inv)
#         A_sparse = sparse.csr_matrix(A)
#         P_sparse = D_inv_sparse @ A_sparse

#         num_connected, clusters = sparse.csgraph.connected_components(
#             csgraph=A_sparse, directed=False, return_labels=True)

#         X = P_sparse @ X

#         if return_all_segs:
#             seg = clusters.reshape(*height_width)
#             all_segs.append(seg)

#         if MAE < convergence_ratio * N**2:
#             converged = True

#     return clusters, all_segs



def cluster_indices_from_mask(
        labels: np.array,
        mask: np.array,
        top1_only: bool = False) -> Tuple[List[int], dict]:
    '''
    `labels` is a label map from unsupervised clustering.
    `mask` is the ground truth mask of binary segmentation.
    This function estimates the list of cluster indices that corresponds to the mask.
    The current implementation uses a greedy algorithm.
    '''

    all_cluster_indices = np.unique(labels)

    # Use a max heap to track the dice scores.
    # By default, heapq maintains a min heap.
    # Hence we negate the dice score to "mimic" a max heap.
    dice_heap = []

    # Single-cluster dice scores.
    for cluster_idx in all_cluster_indices:
        heapq.heappush(dice_heap,
                       (-dice_coeff(labels == cluster_idx, mask), cluster_idx))

    if top1_only:
        best_dice, best_cluster_idx = heapq.heappop(dice_heap)
        return best_cluster_idx

    else:
        # Combine the different clusters.
        # Assuming the cluster with highest dice is definitely in the foreground.
        # Then try to merge in other clusters to see if dice increases.
        best_dice, best_cluster_idx = heapq.heappop(dice_heap)
        best_dice = -best_dice
        best_cluster_indices = [best_cluster_idx]
        dice_map = {best_cluster_idx: best_dice}
        for _ in range(len(dice_heap)):
            curr_dice, cluster_idx = heapq.heappop(dice_heap)
            dice_map[cluster_idx] = -curr_dice
            cluster_idx_candidate = best_cluster_indices + [cluster_idx]
            label_candidate = np.logical_or.reduce(
                [labels == i for i in cluster_idx_candidate])
            dice_candidate = dice_coeff(label_candidate, mask)
            if dice_candidate > best_dice:
                best_dice = dice_candidate
                best_cluster_indices = cluster_idx_candidate

        return best_cluster_indices, dice_map


def get_persistent_structures(labels: np.array) -> np.array:
    '''
    Given a set of B labels on the same image, with shape [B, H, W]
    Return a label with the most persistent structures, with shape [H, W]
    '''
    min_area_ratio = 1e-2
    B, H, W = labels.shape
    min_area = min_area_ratio * H * W

    persistent_label = np.zeros((H, W), dtype=np.int16)
    persistence_tuple = []  # (persistence, area, label_idx, frame_idx)

    for label_idx in np.unique(labels):
        curr_persistence, max_persistence, max_area, best_frame = 0, 0, 0, None
        for frame_idx in range(B - 1):
            curr_area = np.sum(labels[frame_idx, ...] == label_idx)
            if curr_area > 0 and best_frame is None:
                best_frame = frame_idx
                max_area = curr_area
            if curr_area > 0:
                curr_persistence += 1
                if curr_persistence > max_persistence:
                    max_persistence = curr_persistence
                if curr_area > max_area:
                    max_area = curr_area
                    best_frame = frame_idx
        area = np.sum(labels[best_frame, ...] == label_idx)
        if area < min_area:
            continue
        if max_persistence < 2:
            continue
        persistence_tuple.append(
            (max_persistence, area, label_idx, best_frame))

    persistence_tuple = sorted(persistence_tuple, key=lambda x: (-x[1], -x[0]))

    for (_, area, label_idx, frame_idx) in persistence_tuple:
        loc = labels[frame_idx, ...] == label_idx
        persistent_label[loc] = label_idx

    return persistent_label


# def get_persistent_structures(labels: np.array,
#                               min_frame_ratio: float = 1 / 2,
#                               min_area_ratio: float = 1 / 200) -> np.array:
#     '''
#     Given a set of B labels on the same image, with shape [B, H, W]
#     Return a label with the most persistent structures, with shape [H, W]
#     '''
#     B, H, W = labels.shape
#     K = int(min_frame_ratio * B)
#     filtered_labels = labels.copy()
#     persistent_label = np.zeros((H, W), dtype=np.int16)

#     # Assign persistence score to each label index.
#     # Persistence score:
#     #   K-th smallest dice coefficient.

#     # Use a min heap to track the persistence scores.
#     persistence_heap = []
#     for label_idx in np.unique(filtered_labels):
#         sum_area_ratio, num_frames = 0, 0
#         # Use some number > 1 (highest possible dice) as initialization.
#         min_dice_heap = [2 for _ in range(K)]
#         existent_frames = np.sum(filtered_labels == label_idx, axis=(1, 2)) > 0
#         for i in range(B - 1):
#             if existent_frames[i] and existent_frames[i + 1]:
#                 num_frames += 1
#                 dice = dice_coeff(filtered_labels[i, ...] == label_idx,
#                                   filtered_labels[i + 1, ...] == label_idx)
#                 put_back_list = []
#                 for _ in range(K - 1):
#                     put_back_list.append(heapq.heappop(min_dice_heap))
#                 Kth_smallest_dice = heapq.heappop(min_dice_heap)
#                 put_back_list.append(min(dice, Kth_smallest_dice))
#                 for _ in range(K):
#                     heapq.heappush(min_dice_heap, put_back_list.pop(0))
#                 sum_area_ratio += np.sum(
#                     filtered_labels[i, ...] == label_idx) / (H * W)

#         if num_frames < K:
#             Kth_smallest_dice = 0
#         else:
#             assert len(min_dice_heap) == K
#             for _ in range(K - 1):
#                 _ = heapq.heappop(min_dice_heap)
#             Kth_smallest_dice = heapq.heappop(min_dice_heap)
#         mean_area_ratio = 0 if num_frames == 0 else sum_area_ratio / num_frames
#         # TODO: Can we do better?
#         persistence_score = Kth_smallest_dice

#         if mean_area_ratio < min_area_ratio or num_frames < K:
#             filtered_labels[filtered_labels == label_idx] = 0
#         else:
#             heapq.heappush(persistence_heap, (persistence_score, label_idx))

#     # Re-color the label map, with label index with higher persistence score taking priority.
#     for _ in range(len(persistence_heap)):
#         persistence_score, label_idx = heapq.heappop(persistence_heap)
#         # Ignore the background index.
#         if label_idx == 0:
#             continue
#         loc = np.sum(filtered_labels == label_idx, axis=0) > 0
#         persistent_label[loc] = label_idx

#     # Re-number as continuous non-neg integers.
#     persistent_label = continuous_renumber(persistent_label)

#     return persistent_label

# def get_persistent_structures(
#     labels: np.array,
# ) -> np.array:
#     '''
#     Given a set of B labels on the same image, with shape [B, H, W]
#     Return a label with the most persistent structures, with shape [H, W]
#     '''
#     B, H, W = labels.shape
#     # K = int(min_frame_ratio * B)
#     filtered_labels = labels.copy()
#     persistent_label = np.zeros((H, W), dtype=np.int16)

#     # Assign persistence score to each label index.
#     #
#     # Persistence score:
#     #   Average Negative Length-Normalized Hausdorff distance of the same label across frames.
#     # Use a min heap to track the persistence scores.
#     #
#     persistence_heap = []
#     for label_idx in np.unique(labels):
#         num_frames, sum_normalized_hausdorff = 0, 0
#         existent_frames = np.sum(labels == label_idx, axis=(1, 2)) > 0
#         stable_frames = []
#         for i in range(B - 1):
#             if existent_frames[i] and existent_frames[i + 1]:
#                 image1 = ((labels[i, ...] == label_idx) * 255).astype(np.uint8)
#                 contour1, _ = cv2.findContours(image1, cv2.RETR_TREE,
#                                                cv2.CHAIN_APPROX_NONE)
#                 image2 = ((labels[i + 1, ...] == label_idx) * 255).astype(
#                     np.uint8)
#                 contour2, _ = cv2.findContours(image2, cv2.RETR_TREE,
#                                                cv2.CHAIN_APPROX_NONE)
#                 contour_length = np.sum([
#                     cv2.arcLength(arc, True) for arc in contour1
#                 ]) + np.sum([cv2.arcLength(arc, True) for arc in contour2])

#                 hausdorff = 1 + directed_hausdorff(
#                     np.vstack(contour1).squeeze(1),
#                     np.vstack(contour2).squeeze(1))[0]
#                 normalized_hausdorff = hausdorff / contour_length
#                 if normalized_hausdorff < 1:
#                     stable_frames.append(i)
#                     sum_normalized_hausdorff += normalized_hausdorff
#                     num_frames += 1
#         mean_normalized_hausdorff = sum_normalized_hausdorff / num_frames if num_frames > 0 else np.inf

#         persistence_score = -mean_normalized_hausdorff
#         heapq.heappush(persistence_heap,
#                        (persistence_score, label_idx, stable_frames))

#     # Re-color the label map, with label index with higher persistence score taking priority.
#     for _ in range(len(persistence_heap)):
#         persistence_score, label_idx, stable_frames = heapq.heappop(
#             persistence_heap)
#         # Ignore the background index.
#         if label_idx == 0:
#             continue
#         # Only consider the stable frames.
#         loc = np.sum(
#             ((filtered_labels == label_idx)[stable_frames, ...]), axis=0) > 0
#         persistent_label[loc] = label_idx

#     # Re-number as continuous non-neg integers.
#     persistent_label = continuous_renumber(persistent_label)

#     return persistent_label


def continuous_renumber(label_orig: np.array) -> np.array:
    '''
    Renumber the entries of a label map as continous non-negative integers.
    '''
    label = label_orig.copy()
    val_before = np.unique(label_orig)
    val_after = np.arange(len(val_before))
    for (a, b) in zip(val_before, val_after):
        label[label_orig == a] = b

    return label


def associate_frames(labels: np.array) -> np.array:
    ordered_labels = labels.copy()

    B, H, W = labels.shape

    # Find the best-matching label indices pairs between adjacent frames.
    # Update the next frame using the matching label indices from the previous frame.
    for image_idx in range(B - 1):
        label_prev = ordered_labels[image_idx, ...]
        label_next = ordered_labels[image_idx + 1, ...]

        label_vec_prev = np.array(
            [label_prev.reshape(H * W) == i for i in np.unique(label_prev)],
            dtype=np.int16)
        label_vec_next = np.array(
            [label_next.reshape(H * W) == i for i in np.unique(label_next)],
            dtype=np.int16)

        # Use matrix multiplication to get intersection matrix.
        intersection_matrix = np.matmul(label_vec_prev, label_vec_next.T)

        # Use matrix multiplication to get union matrix.
        union_matrix = H * W - np.matmul(1 - label_vec_prev,
                                         (1 - label_vec_next).T)

        iou_matrix = intersection_matrix / union_matrix

        for i, label_idx_next in enumerate(np.unique(label_next)):
            # loc: pixels corresponding to `label_idx_next` in the next frame.
            loc = ordered_labels[image_idx + 1, ...] == label_idx_next
            if np.sum(iou_matrix[..., i]) > 0:
                label_idx_prev = np.unique(label_prev)[np.argmax(
                    iou_matrix[..., i])]
                ordered_labels[image_idx + 1, loc] = label_idx_prev
            else:
                ordered_labels[image_idx + 1, loc] = 0

    return ordered_labels