import os
import time

import numpy as np
import phate
import torch
from data import collate_contrastive, select_near_positive, select_negative_random
from datasets import BerkeleySegmentation, BrainArman, DiabeticMacularEdema, PolyP, Retina
from loss import local_nce_loss_fast
from metric import dice_coeff
from model import CUTSEncoder
from torch import optim
from torch.utils.data import DataLoader

####################
# hyperparameters
DATA = 'brain'
MODEL = 'full'  # 'full', 'no_patchify', 'no_contrastive'
LOAD_POOLS = True
####################


if DATA == 'retina':
    data = Retina()
elif DATA == 'berkeley':
    data = BerkeleySegmentation
elif DATA == 'polyp':
    data = PolyP()
elif DATA == 'macular_edema':
    data = DiabeticMacularEdema()
elif DATA == 'brain':
    data = BrainArman()
else:
    raise Exception('fix DATA option')

DATA_FOLDER = 'output_{}_{}'.format(DATA, MODEL)
if not os.path.exists(DATA_FOLDER):
    os.mkdir(DATA_FOLDER)
####################


####################


def train(model, iterator, optimizer, negative_pool, positive_pool, lambda_contrastive_loss=1, lambda_patchify_loss=10):
    loss = 0
    feature_all = []
    model.train()
    t = time.time()
    for i, (train_batch, train_labels) in enumerate(iterator):
        if i and i % 10 == 0:
            print('  iter {} loss: {:.3f}/{:.3f} ({:.1f} sec)'.format(i,
                  train_loss_.item(), patchify_loss, time.time() - t))
            t = time.time()
        optimizer.zero_grad()

        features, patchify_loss = model(train_batch)
        train_loss_ = local_nce_loss_fast(
            features, negative_pool[2*i:2*(i+1)], positive_pool[2*i:2*(i+1)])
        train_loss = lambda_contrastive_loss * \
            train_loss_ + lambda_patchify_loss * patchify_loss

        train_loss.backward()
        optimizer.step()

        loss += train_loss.item()
        feature_all.append(features.cpu())

    loss = loss / len(iterator)

    print('loss: {:.3f}'.format(loss))

    return loss, feature_all


def save_pools(positive_pool, negative_pool, DATA):
    with open('saved_pools/{}.npz'.format(DATA), 'wb+') as f:
        np.savez(f, positive_pool=positive_pool, negative_pool=negative_pool)


def load_pools(DATA):
    with open('saved_pools/{}.npz'.format(DATA), 'rb') as f:
        npzfile = np.load(f)
        positive_pool = npzfile['positive_pool']
        negative_pool = npzfile['negative_pool']
    return positive_pool, negative_pool


####################
# model hyperparameters
learning_rate = 0.0001
max_epochs = 25
batch_size = 2  # right now must stay at 2
lambda_contrastive_loss = 1
lambda_patchify_loss = 10

# for ablation test
if MODEL == 'no_patchify':
    lambda_patchify_loss = 0
if MODEL == 'no_contrastive':
    lambda_contrastive_loss = 0
####################


####################
# build the dataset object
train_index = np.arange(data.shape[0]).tolist()

training_set = CUTSDataset(train_index, data_input, label)
training_generator = DataLoader(
    training_set, batch_size=batch_size, shuffle=True, collate_fn=collate_contrastive)
eval_generator = DataLoader(training_set, batch_size=batch_size, shuffle=False)
####################


####################
# build the model
model = CUTSEncoder()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
####################


####################
# calculate the pools of positive and negative examples for contrastive loss
if LOAD_POOLS:
    positive_pool, negative_pool = load_pools(DATA)
    positive_pool = torch.from_numpy(positive_pool)
    negative_pool = torch.from_numpy(negative_pool)
else:
    t = time.time()
    positive_pool = select_near_positive(data_input)
    print('positive pool done')
    negative_pool = select_negative_random(data_input)
    print('negative pool done')

    save_pools(positive_pool, negative_pool, DATA)
    print('saved pools')
    print('{:.1f} seconds'.format(time.time() - t))
####################

####################
# train the model
model.train()
for epoch in range(max_epochs):
    train_loss, features = train(model, training_generator, optimizer, negative_pool, positive_pool,
                                 lambda_contrastive_loss=lambda_contrastive_loss, lambda_patchify_loss=lambda_patchify_loss)
    if epoch % 2 == 0:
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, max_epochs, train_loss))
####################

####################
# get model predictions out (latent feature space)
model.eval()
features = []
for i, (eval_batch, _) in enumerate(eval_generator):
    feats_, _ = model(eval_batch)
    features.append(feats_)
features_all = np.concatenate(
    [feats[:2].detach() for feats in features], axis=0).reshape((-1, 128, 128, 128))
####################


####################
# cluster latent feature space and produce pixel-level predictions


ncluster = 6
predicted_segmentations = []
dice_coeffs = []
for i in range(len(features_all)):

    feature = features_all[i]
    try:
        fn_ = names[i].split('/')[-1].split('.')[0]
    except:
        fn_ = names[i]
    print(fn_)

    feature = feature.reshape((-1, 128))

    phate_operator = phate.PHATE(n_components=3, knn=100, n_landmark=500, t=2)
    phate_data = phate_operator.fit_transform(feature)
    cluster_ = phate.cluster.kmeans(phate_operator, n_clusters=ncluster)

    predicted_segmentation = cluster_.reshape([128, 128])
    predicted_segmentations.append(predicted_segmentation)

    if DATA in ['retina', 'polyp', 'brain']:
        true_cluster = np.argwhere(np.logical_and(label_mask[i], label[i]))
        true_cluster = predicted_segmentation[true_cluster[0], true_cluster[1]]

        final_preds = np.where(predicted_segmentation == true_cluster, 1., 0.)
    else:
        final_preds = np.zeros_like(predicted_segmentation)

    score = dice_coeff(final_preds, label[i].numpy())
    dice_coeffs.append(score)
    print('{}: Dice coef: {:.3f} (mean: {:.3f})'.format(
        i, score, np.mean(dice_coeffs)))
####################


####################
# write out results
with open('{}/output.npz'.format(DATA_FOLDER), 'wb+') as f:
    np.savez(f, predicted_segmentations=predicted_segmentations, features=features,
             data=data, label=label, names=names, dice_coeffs=dice_coeffs)
####################
