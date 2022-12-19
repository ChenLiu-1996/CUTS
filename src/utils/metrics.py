import numpy as np
import sewar
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity
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
    data_range = label_true.max() - label_true.min()

    return ssim(a=label_true, b=label_pred, data_range=data_range)


def ergas(a: np.array, b: np.array) -> float:
    return sewar.full_ref.ergas(a, b)


def rmse(a: np.array, b: np.array) -> float:
    return sewar.full_ref.rmse(a, b)


def dice_coeff(pred: np.array, label: np.array) -> float:
    intersection = np.logical_and(pred, label).sum()
    dice = (2 * intersection) / (pred.sum() + label.sum())
    return dice


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
            ssim_list.append(ssim(img1, img2))
    return np.mean(ssim_list)
