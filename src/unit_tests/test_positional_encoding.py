import os
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


def pos_enc_sinusoid(shape: Tuple[int]) -> np.array:
    '''
    Positional encoding for 2D image.
    Encoding scheme: Sinusoidal.

    `shape`: [H, W, C].
    '''

    H, W, C = shape
    freq = C // 2

    pos_enc = np.zeros((H, W, C))
    multiplier = np.exp(np.arange(0, freq, 2) * -(np.log(1e5) / freq))

    pos_H = np.arange(H)[:, None]
    pos_W = np.arange(W)[:, None]

    pos_enc[:, :, :freq:2] = np.sin(pos_H * multiplier)[:, :, None].transpose(
        0, 2, 1).repeat(W, axis=1)
    pos_enc[:, :, 1:freq:2] = np.cos(pos_H * multiplier)[:, :, None].transpose(
        0, 2, 1).repeat(W, axis=1)
    pos_enc[:, :, freq::2] = np.sin(pos_W * multiplier)[:, :, None].transpose(
        2, 0, 1).repeat(H, axis=0)
    pos_enc[:, :,
            freq + 1::2] = np.cos(pos_W * multiplier)[:, :, None].transpose(
                2, 0, 1).repeat(H, axis=0)

    return pos_enc


def pos_enc_coord(shape: Tuple[int]) -> np.array:
    '''
    Positional encoding for 2D image.
    Encoding scheme: Coordinates.

    `shape`: [H, W, C].
    '''

    H, W, C = shape

    pos_enc = np.zeros((H, W, C))
    pos_enc[:, :, ::2] = np.arange(-H // 2,
                                   H // 2)[:, None,
                                           None].repeat(W,
                                                        axis=1).repeat(C // 2,
                                                                       axis=2)
    pos_enc[:, :, 1::2] = np.arange(-W // 2,
                                    W // 2)[None, :,
                                            None].repeat(H,
                                                         axis=0).repeat(C // 2,
                                                                        axis=2)

    return pos_enc


def check_similarity(enc: np.array, anchor_pos: np.array):
    H, W, C = enc.shape

    anchor_vec = enc[anchor_pos[0], anchor_pos[1], :]

    euc_distances = []
    similarities = []

    for i in range(H):
        for j in range(W):
            euc_distances.append(np.linalg.norm(anchor_pos - np.array([i, j])))
            similarities.append(
                cosine_similarity(anchor_vec.reshape(1, -1),
                                  enc[i, j, :].reshape(1, -1)).item())

    return euc_distances, similarities


if __name__ == '__main__':
    figure_folder = './figures/'
    os.makedirs(figure_folder, exist_ok=True)

    H, W, C = 100, 100, 128

    pos_enc = pos_enc_sinusoid((H, W, C))

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    anchor_pos = np.array([0, 0])
    euc_distances, similarities = check_similarity(pos_enc, anchor_pos)
    ax.scatter(euc_distances, similarities, alpha=0.1, c='firebrick')
    ax = fig.add_subplot(1, 2, 2)
    anchor_pos = np.array([H // 2, W // 2])
    euc_distances, similarities = check_similarity(pos_enc, anchor_pos)
    ax.scatter(euc_distances, similarities, alpha=0.1, c='firebrick')
    fig.savefig('%s/%s' % (figure_folder, 'positional_encoding.png'))
