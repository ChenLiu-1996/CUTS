import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import accuracy_score


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
            ssim_list.append(ssim(img1, img2, channel_axis=-1))
    return np.mean(ssim_list)


