from glob import glob

import cv2
import numpy as np
import scipy
import scipy.ndimage
import sewar
import skimage.feature
import skimage.segmentation
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim


def get_baseline_predictions(data, METHOD):
    predicted_segmentations_baseline = []

    for i, img in enumerate(data):
        if (i % 10) == 0: print(i)

        img = (img.numpy() + 1) / 2
        img = (img * 255).astype(np.uint8)

        if METHOD == 'watershed':
            if img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN,
                                    np.ones((3, 3), dtype=int))

            _, threshed = cv2.threshold(
                gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            distance = scipy.ndimage.distance_transform_edt(threshed)
            coords = skimage.feature.peak_local_max(distance,
                                                    labels=threshed,
                                                    threshold_rel=.9)

            mask = np.zeros(distance.shape, dtype=bool)
            mask[tuple(coords.T)] = True
            markers, _ = scipy.ndimage.label(mask)

            labels = skimage.segmentation.watershed(-distance,
                                                    markers,
                                                    mask=threshed)
        elif METHOD == 'felzenszwalb':
            labels = skimage.segmentation.felzenszwalb(img, scale=1500)
        else:
            raise Exception('cannot parse METHOD: {}'.format(METHOD))

        predicted_segmentations_baseline.append(labels)

    predicted_segmentations_baseline = np.stack(
        predicted_segmentations_baseline, axis=0)

    return predicted_segmentations_baseline


def dice_coeff(pred, label):

    # inter = torch.dot(pred.view(-1), label.view(-1))
    # union = torch.sum(pred) + torch.sum(label)
    # dice = (2 * inter.float() ) / union.float()
    inter = pred.reshape(-1).dot(label.reshape(-1))
    union = pred.sum() + label.sum()
    dice = (2. * inter) / union
    return dice


MODELS = [
    'ours', 'random', 'watershed', 'felzenszwalb', 'ours_no_patchify',
    'ours_no_contrastive'
]
STATS = ['ssim', 'ergas', 'rmse', 'gt']
DATAS = ['berkeley', 'retina', 'brain']

MODEL = MODELS[0]
STAT = STATS[0]
DATA = DATAS[0]

SAMPLE = True
k = 25

if MODEL == 'ours':
    fn = 'output_{}/output.npz'.format(DATA)
elif MODEL == 'ours_no_contrastive':
    fn = 'output_{}_no_contrastive/output.npz'.format(DATA)
elif MODEL == 'ours_no_patchify':
    fn = 'output_{}_no_patchify/output.npz'.format(DATA)
else:
    fn = 'output_{}/output.npz'.format(DATA)

with open(fn, 'rb') as f:
    npzfile = np.load(f)
    data = npzfile['data']
    label = npzfile['label']
    predicted_segmentations = npzfile['predicted_segmentations']
data = torch.from_numpy(data)

if MODEL in ['ours', 'ours_no_patchify', 'ours_no_contrastive']:
    preds = predicted_segmentations
elif MODEL == 'random':
    predicted_segmentations_random = np.random.randint(
        0,
        predicted_segmentations.max() + 1, predicted_segmentations.shape)
    preds = predicted_segmentations_random
elif MODEL == 'watershed':
    predicted_segmentations_watershed = get_baseline_predictions(
        data, METHOD='watershed')
    preds = predicted_segmentations_watershed
elif MODEL == 'felzenszwalb':
    predicted_segmentations_felzenszwalb = get_baseline_predictions(
        data, METHOD='felzenszwalb')
    preds = predicted_segmentations_felzenszwalb

if STAT == 'gt':
    label_mask = np.zeros_like(label)
    pctile = 50 if DATA != 'brain' else 25
    for i in range(label.shape[0]):
        label_argwhere = np.argwhere(label[i]).T  # shape: [2, x]
        median_x_coord = np.percentile(label_argwhere[0, :], pctile).reshape(
            (1, 1)
        )  # shape: [1, 1] median_x_coord = np.median(label_argwhere[0, :]).reshape((1, 1)) # shape: [1, 1]
        median_y_coord = np.percentile(label_argwhere[1, :], pctile).reshape(
            (1, 1)
        )  # shape: [1, 1] median_y_coord = np.median(label_argwhere[1, :]).reshape((1, 1)) # shape: [1, 1]
        middle_pt = np.concatenate([median_x_coord, median_y_coord],
                                   axis=0)  # shape: [2, 1]
        dist_to_middle_pt = ((label_argwhere - middle_pt)**2).sum(
            axis=0)  # shape: [x]
        argmin = np.argmin(dist_to_middle_pt)  # shape: [] (it's a scalar)
        label_mask[
            i, label_argwhere[:, argmin][0], label_argwhere[:, argmin]
            [1]] = 1  # this sets the "middle pixel" of the ground truth to 1

    dcs = []
    for i in range(data.shape[0]):
        l = label[i]
        lm = label_mask[i]
        p = preds[i]

        true_cluster = np.argwhere(lm).T
        true_cluster = p[true_cluster[0], true_cluster[1]]
        p_chosen = np.where(p == true_cluster, 1., 0.)

        clusts_ = p_chosen.reshape([-1])

        dc = dice_coeff(clusts_, l.reshape([-1]))

        dcs.append(dc)
    dcs = np.mean(dcs)
    print('{} / {}: {:.3f}'.format(MODEL, STAT, dcs))

else:
    all_within_clust_ssim = []
    all_without_clust_ssim = []

    for i in range(data.shape[0]):
        r = np.arange(128 * 128)
        if SAMPLE:
            r = np.random.choice(128 * 128, 100, replace=False)
        # patches
        channels = data.shape[-1]
        tmp = data[i].permute(2, 0, 1)
        tmp = F.pad(tmp, pad=[k // 2, k // 2, k // 2, k // 2], value=0)
        tmp = tmp.unfold(1, k, 1).unfold(2, k,
                                         1).reshape(channels, -1, k,
                                                    k).permute(1, 2, 3, 0)
        data_ = tmp[r]

        clusts_ = preds[i].reshape([-1])[r]

        within_clust_ssim = []
        without_clust_ssim = []

        data_ = (data_.numpy() + 1) / 2
        for ii in range(data_.shape[0]):
            for kk in range(ii + 1, data_.shape[0]):

                if STAT == 'ssim':
                    s = ssim(data_[ii], data_[kk], multichannel=True)
                elif STAT == 'ergas':
                    s = sewar.full_ref.ergas(data_[ii], data_[kk])
                elif STAT == 'rmse':
                    s = sewar.full_ref.rmse(data_[ii], data_[kk])

                if np.isnan(s): continue

                if clusts_[ii] == clusts_[kk]:
                    within_clust_ssim.append(s)
                else:
                    without_clust_ssim.append(s)

        if within_clust_ssim:
            all_within_clust_ssim.append(np.mean(within_clust_ssim))

        if without_clust_ssim:
            all_without_clust_ssim.append(np.mean(without_clust_ssim))

        print('{}: {:.3f} {:.3f}'.format(i, np.mean(all_within_clust_ssim),
                                         np.mean(all_without_clust_ssim)))

    all_within_clust_ssim = np.stack(all_within_clust_ssim)
    all_without_clust_ssim = np.stack(all_without_clust_ssim)

    print('{} / {}: {:.3f} {:.3f}'.format(MODEL, STAT,
                                          np.mean(all_within_clust_ssim),
                                          np.mean(all_without_clust_ssim)))
