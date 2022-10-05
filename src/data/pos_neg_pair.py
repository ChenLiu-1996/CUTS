import numpy as np
import torch

from ..util import check_ssim


def select_near_positive(img_data, full_version=True):
    """
    Select positive pairs based on similarity.
    """
    pool = []
    for k in range(len(img_data)):
        if full_version or k == 0:
            img = img_data[k]
            img_pool = []
            for j in range(16384):
                if j % 1000 == 0:
                    print('{} / {}: {}'.format(k, len(img_data), j))
                temp = []
                r0, c0 = _get_row_col_id(j)
                best_sim = 0

                m = np.random.choice([num_ for num_ in range(
                    max(0, r0 - 1), min(r0 + 2, 128))], 1)[0]
                n = np.random.choice([num_ for num_ in range(
                    max(0, c0 - 1), min(c0 + 2, 128)) if ((m != r0) or (num_ != c0))], 1)[0]
                r, c = m, n

                if full_version:
                    for m in range(max(0, r0 - 1), min(r0 + 2, 128)):
                        for n in range(max(0, c0 - 1), min(c0 + 2, 128)):
                            if best_sim > 0 and np.random.binomial(1, .5) < .75:
                                continue
                            if m != r0 or n != c0:
                                temp_v = check_ssim(img, r0, c0, m, n)
                                if temp_v > best_sim:
                                    best_sim = temp_v
                                    r, c = m, n
                temp.append(128 * c + r)
                img_pool.append(temp)
        else:
            pass
        pool.append(img_pool)
    pool = torch.LongTensor(pool)
    return pool


def select_negative_random(img_data, ssim_thresh=.5, full_version=True):
    """
    Randomly select negative pairs.
    """
    pool = []
    for k in range(len(img_data)):
        if full_version or k == 0:
            img = img_data[k]
            img_pool = []
            for j in range(img_data.shape[-2] * img_data.shape[-1]):
                if j % 1000 == 0:
                    print('{} / {}: {}'.format(k, len(img_data), j))
                temp = []
                found_one = False
                r0, c0 = _get_row_col_id(j)
                while not found_one:
                    a = np.random.randint(
                        0, img_data.shape[-2] * img_data.shape[-1])
                    r, c = _get_row_col_id(a)
                    if a != j and abs(r0 - r) > 20 and abs(c0 - c) > 20:
                        if full_version and check_ssim(img, r0, c0, r, c) > ssim_thresh:
                            continue
                        temp.append(a)
                        found_one = True
                img_pool.append(temp)
        else:
            pass
        pool.append(img_pool)
    pool = torch.LongTensor(pool)
    return pool


def _get_row_col_id(v):

    return v % 128, v // 128
