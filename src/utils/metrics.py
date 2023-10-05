import numpy as np
import sewar
import torch
import torch.nn.functional as F
from skimage.metrics import hausdorff_distance, structural_similarity
from sklearn.metrics import accuracy_score


def ssim(a: np.array, b: np.array, **kwargs) -> float:
    '''
    Please make sure the data are provided in [H, W, C] shape.
    '''
    assert a.shape == b.shape

    H, W = a.shape[:2]

    if min(H, W) < 7:
        win_size = min(H, W)
        if win_size % 2 == 0:
            win_size -= 1
    else:
        win_size = None

    if len(a.shape) == 3:
        channel_axis = -1
    else:
        channel_axis = None

    return structural_similarity(a,
                                 b,
                                 channel_axis=channel_axis,
                                 win_size=win_size,
                                 **kwargs)


def range_aware_ssim(label_true: np.array, label_pred: np.array) -> float:
    '''
    Surprisingly, skimage ssim infers data range from data type...
    It's okay within our neural network training since the scale is
    quite close to its guess (-1 to 1 for float numbers), but
    not okay in many other places.
    '''
    if isinstance(label_true.max(), bool):
        label_true = label_true.astype(np.float32)
        label_pred = label_pred.astype(np.float32)
    data_range = label_true.max() - label_true.min()

    if data_range == 0:
        data_range = 1.0

    return ssim(a=label_true, b=label_pred, data_range=data_range)


def ergas(a: np.array, b: np.array) -> float:
    return sewar.full_ref.ergas(a, b)


def rmse(a: np.array, b: np.array) -> float:
    return sewar.full_ref.rmse(a, b)


def dice_coeff(label_pred: np.array, label_true: np.array) -> float:
    epsilon = 1e-12
    intersection = np.logical_and(label_pred, label_true).sum()
    dice = (2 * intersection + epsilon) / (label_pred.sum() +
                                           label_true.sum() + epsilon)
    return dice


def per_class_dice_coeff(label_pred: np.array, label_true: np.array) -> float:
    dice_list = []
    # Iterate over all non-background classes.
    for class_id in np.unique(label_true)[1:]:
        dice_list.append(
            dice_coeff(label_pred=label_pred == class_id,
                       label_true=label_true == class_id))
    return np.mean(dice_list)


def hausdorff(label_pred: np.array, label_true: np.array) -> float:
    if np.sum(label_pred) == 0 or np.sum(label_true) == 0:
        # If `label_pred` or `label_true` is all zeros,
        # return the max Euclidean distance.
        H, W = label_true.shape
        return np.sqrt((H**2 + W**2))
    else:
        return hausdorff_distance(label_pred, label_true)


def per_class_hausdorff(label_pred: np.array, label_true: np.array) -> float:
    hausdorff_list = []
    # Iterate over all non-background classes.
    for class_id in np.unique(label_true):
        hausdorff_list.append(
            hausdorff(label_pred=label_pred == class_id,
                      label_true=label_true == class_id))
    return np.mean(hausdorff_list)


def contrastive_acc(z_anchor: torch.Tensor,
                    z_positives: torch.Tensor) -> float:
    B, S, _ = z_anchor.shape

    z_anchor = F.normalize(input=z_anchor, p=2, dim=-1)
    z_positives = F.normalize(input=z_positives, p=2, dim=-1)

    auroc_list = []
    for batch_idx in range(B):
        sim_matrix = torch.matmul(z_anchor[batch_idx, ...],
                                  z_positives[batch_idx, ...].T)
        pos_pair_ij = torch.diag(torch.ones(S)).bool()
        label_pred = torch.cat(
            (sim_matrix[pos_pair_ij], sim_matrix[~pos_pair_ij]), dim=0)
        label_true = torch.cat((torch.ones(S), torch.zeros(S**2 - S)), dim=0)
        label_pred = label_pred.cpu().detach().numpy()
        label_true = label_true.cpu().detach().numpy()
        auroc_list.append(accuracy_score(label_true, label_pred > 0.5))
    return np.mean(auroc_list)


def recon_ssim(x: torch.Tensor, x_recon: torch.Tensor) -> float:
    x = x.cpu().detach().numpy()
    x_recon = x_recon.cpu().detach().numpy()

    B, S, _, _, _ = x.shape

    ssim_list = []
    for batch_idx in range(B):
        for sample_idx in range(S):
            img1 = x[batch_idx, sample_idx, ...]
            img2 = x_recon[batch_idx, sample_idx, ...]
            # Channel first to channel last to accommodate SSIM.
            img1 = np.moveaxis(img1, 0, -1)
            img2 = np.moveaxis(img2, 0, -1)
            ssim_list.append(range_aware_ssim(img1, img2))
    return np.mean(ssim_list)


def guided_relabel(label_pred: np.array, label_true: np.array) -> np.array:
    '''
    Relabel (i.e., update label index) `label_pred` such that it best matches `label_true`.

    For each label index, assign an one-hot vector (flattened pixel values),
    and compute the IOU among each pair of such one-hot vectors b/w `label_pred` and `label_true`.
    '''
    assert label_pred.shape == label_true.shape
    H, W = label_pred.shape

    label_pred_vec = np.array(
        [label_pred.reshape(H * W) == i for i in np.unique(label_pred)],
        dtype=np.int16)
    label_true_vec = np.array(
        [label_true.reshape(H * W) == i for i in np.unique(label_true)],
        dtype=np.int16)

    # Use matrix multiplication to get intersection matrix.
    intersection_matrix = np.matmul(label_pred_vec, label_true_vec.T)

    # Use matrix multiplication to get union matrix.
    union_matrix = H * W - np.matmul(1 - label_pred_vec,
                                     (1 - label_true_vec).T)

    iou_matrix = intersection_matrix / union_matrix

    renumbered_label_pred = np.zeros_like(label_pred)

    for i, label_pred_idx in enumerate(np.unique(label_pred)):
        pix_loc = label_pred == label_pred_idx
        label_true_idx = np.unique(label_true)[np.argmax(iou_matrix[i, :])]
        renumbered_label_pred[pix_loc] = label_true_idx

    return renumbered_label_pred
