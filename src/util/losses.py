import numpy as np
import torch
import torch.nn.functional as F


def local_nce_loss_fast(features: np.array, negative_pool: torch.Tensor, positive_pool: torch.Tensor) -> torch.Tensor:
    temperature = 0.25

    # feature1 = features[:2]
    # feature3 = features[2:4]
    # feature4 = features[4:6]

    num_splits = features.shape[0] // 3

    feature1, feature3, feature4 = torch.split(features, num_splits, dim=0)

    feature3 = feature3.reshape(
        feature3.shape[0], 128, 128, -1).detach().numpy()
    feature3 = np.flip(feature3, 1)
    feature3 = feature3.reshape(feature3.shape[0], 16384, -1)
    feature3 = torch.from_numpy(feature3)

    feature4 = feature4.reshape(
        feature4.shape[0], 128, 128, -1).detach().numpy()
    feature4 = np.flip(feature4, 2)
    feature4 = feature4.reshape(feature4.shape[0], 16384, -1)
    feature4 = torch.from_numpy(feature4)

    positives = torch.arange(0, 16384).reshape(16384, 1).repeat(2, 1, 1)
    combined_pos_neg = torch.cat((positives, positive_pool, negative_pool), 2)

    feature3_1 = feature3[0][combined_pos_neg[0]].unsqueeze(0)
    feature3_2 = feature3[1][combined_pos_neg[1]].unsqueeze(0)
    feature3_new = torch.cat((feature3_1, feature3_2), 0)

    feature4_1 = feature4[0][combined_pos_neg[0]].unsqueeze(0)
    feature4_2 = feature4[1][combined_pos_neg[1]].unsqueeze(0)
    feature4_new = torch.cat((feature4_1, feature4_2), 0)

    feature1_new = torch.unsqueeze(feature1, 3)
    feature1_norm = torch.norm(feature1_new, dim=2).reshape(2, -1, 1, 1)
    feature3_norm = torch.norm(feature3_new, dim=3).unsqueeze(2)
    feature4_norm = torch.norm(feature4_new, dim=3).unsqueeze(2)

    feature13_norm = torch.matmul(feature1_norm, feature3_norm).squeeze(2)
    feature14_norm = torch.matmul(feature1_norm, feature4_norm).squeeze(2)

    similarity_matrix2 = torch.matmul(
        feature3_new, feature1_new).squeeze(3) / feature13_norm
    similarity_matrix3 = torch.matmul(
        feature4_new, feature1_new).squeeze(3) / feature14_norm

    logits1 = torch.cat([similarity_matrix2, similarity_matrix3], dim=1)
    logits1 = logits1.view(logits1.shape[0] * logits1.shape[1], -1)
    logits1 = logits1 / temperature

    out = F.softmax(logits1, dim=1)
    out_ = out[:, 0] + out[:, 1]
    loss = -torch.log(out_).mean()
    return loss
