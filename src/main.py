import argparse
import time

import numpy as np
import phate
import torch
import yaml
from data import collate_contrastive, select_near_positive, select_negative_random
from datasets import BerkeleySegmentation, BrainArman, DiabeticMacularEdema, PolyP, Retina
from model import CUTSEncoder
from torch import optim
from torch.utils.data import DataLoader, random_split
from util import AttributeHashmap, dice_coeff, local_nce_loss_fast

# DATA_FOLDER = 'output_{}_{}'.format(DATA, MODEL)
# if not os.path.exists(DATA_FOLDER):
#     os.mkdir(DATA_FOLDER)
####################


####################


# def save_pools(positive_pool, negative_pool, DATA):
#     with open('saved_pools/{}.npz'.format(DATA), 'wb+') as f:
#         np.savez(f, positive_pool=positive_pool, negative_pool=negative_pool)


# def load_pools(DATA):
#     with open('saved_pools/{}.npz'.format(DATA), 'rb') as f:
#         npzfile = np.load(f)
#         positive_pool = npzfile['positive_pool']
#         negative_pool = npzfile['negative_pool']
#     return positive_pool, negative_pool


####################
# model hyperparameters


####################


# ####################
# build the dataset object
# train_index = np.arange(data.shape[0]).tolist()

# training_set = CUTSDataset(train_index, data_input, label)
# training_generator = DataLoader(
#     training_set, batch_size=batch_size, shuffle=True, collate_fn=collate_contrastive)
# eval_generator = DataLoader(training_set, batch_size=batch_size, shuffle=False)
# ####################


# ####################
# ####################


# ####################
# # calculate the pools of positive and negative examples for contrastive loss
# # if LOAD_POOLS:
# #     positive_pool, negative_pool = load_pools(DATA)
# #     positive_pool = torch.from_numpy(positive_pool)
# #     negative_pool = torch.from_numpy(negative_pool)
# # else:
# #     t = time.time()
# #     positive_pool = select_near_positive(data_input)
# #     print('positive pool done')
# #     negative_pool = select_negative_random(data_input)
# #     print('negative pool done')

# #     save_pools(positive_pool, negative_pool, DATA)
# #     print('saved pools')
# #     print('{:.1f} seconds'.format(time.time() - t))
# ####################

# t = time.time()
# positive_pool = select_near_positive(data_input)
# print('positive pool done')
# negative_pool = select_negative_random(data_input)
# print('negative pool done')
# print('{:.1f} seconds'.format(time.time() - t))


# ####################
# # train the model


# def train(model, iterator, optimizer, negative_pool, positive_pool, lambda_contrastive_loss=1, lambda_patchify_loss=10):
#     loss = 0
#     feature_all = []
#     model.train()
#     t = time.time()
#     for i, (train_batch, train_labels) in enumerate(iterator):
#         if i and i % 10 == 0:
#             print('  iter {} loss: {:.3f}/{:.3f} ({:.1f} sec)'.format(i,
#                   train_loss_.item(), patchify_loss, time.time() - t))
#             t = time.time()
#         optimizer.zero_grad()

#         features, patchify_loss = model(train_batch)
#         train_loss_ = local_nce_loss_fast(
#             features, negative_pool[2*i:2*(i+1)], positive_pool[2*i:2*(i+1)])
#         train_loss = lambda_contrastive_loss * \
#             train_loss_ + lambda_patchify_loss * patchify_loss

#         train_loss.backward()
#         optimizer.step()

#         loss += train_loss.item()
#         feature_all.append(features.cpu())

#     loss = loss / len(iterator)

#     print('loss: {:.3f}'.format(loss))

#     return loss, feature_all


# for epoch in range(max_epochs):
#     train_loss, features = train(model, training_generator, optimizer, negative_pool, positive_pool,
#                                  lambda_contrastive_loss=lambda_contrastive_loss, lambda_patchify_loss=lambda_patchify_loss)
#     if epoch % 2 == 0:
#         print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, max_epochs, train_loss))
# ####################

# ####################
# # get model predictions out (latent feature space)
# model.eval()
# features = []
# for i, (eval_batch, _) in enumerate(eval_generator):
#     feats_, _ = model(eval_batch)
#     features.append(feats_)
# features_all = np.concatenate(
#     [feats[:2].detach() for feats in features], axis=0).reshape((-1, 128, 128, 128))
# ####################


# ####################
# # cluster latent feature space and produce pixel-level predictions


# ncluster = 6
# predicted_segmentations = []
# dice_coeffs = []
# for i in range(len(features_all)):

#     feature = features_all[i]
#     try:
#         fn_ = names[i].split('/')[-1].split('.')[0]
#     except:
#         fn_ = names[i]
#     print(fn_)

#     feature = feature.reshape((-1, 128))

#     phate_operator = phate.PHATE(n_components=3, knn=100, n_landmark=500, t=2)
#     phate_data = phate_operator.fit_transform(feature)
#     cluster_ = phate.cluster.kmeans(phate_operator, n_clusters=ncluster)

#     predicted_segmentation = cluster_.reshape([128, 128])
#     predicted_segmentations.append(predicted_segmentation)

#     if dataset_name in ['retina', 'polyp', 'brain']:
#         true_cluster = np.argwhere(np.logical_and(label_mask[i], label[i]))
#         true_cluster = predicted_segmentation[true_cluster[0], true_cluster[1]]

#         final_preds = np.where(predicted_segmentation == true_cluster, 1., 0.)
#     else:
#         final_preds = np.zeros_like(predicted_segmentation)

#     score = dice_coeff(final_preds, label[i].numpy())
#     dice_coeffs.append(score)
#     print('{}: Dice coef: {:.3f} (mean: {:.3f})'.format(
#         i, score, np.mean(dice_coeffs)))
# ####################


def parse_settings(config: AttributeHashmap):
    # type issues
    config.learning_rate = float(config.learning_rate)
    config.weight_decay = float(config.weight_decay)
    # for ablation test
    if config.model_setting == 'no_patchify':
        config.lambda_patchify_loss = 0
    if config.model_setting == 'no_contrastive':
        config.lambda_contrastive_loss = 0
    return config


def prepare_dataset(config: AttributeHashmap, mode: str = 'train'):
    # Read dataset.
    if config.dataset_name == 'retina':
        dataset = Retina(base_path=config.dataset_path)
    elif config.dataset_name == 'berkeley':
        dataset = BerkeleySegmentation(base_path=config.dataset_path)
    elif config.dataset_name == 'polyp':
        dataset = PolyP(base_path=config.dataset_path)
    elif config.dataset_name == 'macular_edema':
        dataset = DiabeticMacularEdema(base_path=config.dataset_path)
    elif config.dataset_name == 'brain':
        dataset = BrainArman(base_path=config.dataset_path)
    else:
        raise Exception('fix DATA option')

    # Train/Validation/Test Split
    ratios = [float(c) for c in config.train_val_test_ratio.split(':')]
    ratios = [c/sum(ratios) for c in ratios]

    train_len = int(len(dataset) * ratios[0])
    validation_len = int(len(dataset) * ratios[1])

    train_set, remaining = random_split(dataset, lengths=[train_len, len(dataset)-train_len],
                                        generator=torch.Generator().manual_seed(config.random_seed))
    validation_set, test_set = random_split(remaining, lengths=[validation_len, len(remaining)-validation_len],
                                            generator=torch.Generator().manual_seed(config.random_seed))

    # Load into DataLoader
    if mode == 'train':
        train_set = DataLoader(
            dataset=train_set, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
        validation_set = DataLoader(
            dataset=validation_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
        return train_set, validation_set, dataset.all_images()
    else:
        test_set = DataLoader(
            dataset=test_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
        return test_set


def train(config: AttributeHashmap):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_set, validation_set, all_train_images = prepare_dataset(
        config=config, mode='train')

    # Build the model
    model = CUTSEncoder().to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(config.max_epochs):
        epoch_loss = 0
        feature_all = []
        model.train()
        t = time.time()
        for i, (x_train, _y_train) in enumerate(train_set):
            if i and i % 10 == 0:
                print('iter {} loss: {:.3f}/{:.3f} ({:.1f} sec)'.format(i,
                                                                        train_loss.item(), patchify_loss, time.time() - t))
                t = time.time()
            x_train, _y_train = x_train.to(device).type(
                torch.FloatTensor), _y_train.to(device).type(torch.FloatTensor)
            optimizer.zero_grad()
            features, patchify_loss = model(x_train)
            # train_loss_ = local_nce_loss_fast(
            #     features, negative_pool[2*i:2*(i+1)], positive_pool[2*i:2*(i+1)])

            train_loss = loss_fn(features, _y_train)

            # train_loss = lambda_contrastive_loss * \
            #     train_loss_ + lambda_patchify_loss * patchify_loss

            train_loss.backward()
            optimizer.step()

            epoch_loss += train_loss.item()
            feature_all.append(features.cpu())

        epoch_loss = epoch_loss / len(train_set)

        print('loss: {:.3f}'.format(epoch_loss))

        return epoch_loss, feature_all

    return


def test(config: AttributeHashmap):
    test_set = prepare_dataset(config=config, mode='test')

    return

####################
# write out results
# with open('{}/output.npz'.format(DATA_FOLDER), 'wb+') as f:
#     np.savez(f, predicted_segmentations=predicted_segmentations, features=features,
#              data=data, label=label, names=names, dice_coeffs=dice_coeffs)
####################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entry point to run CUTS.')
    parser.add_argument(
        '--train', help='Flag for running train & validation.', action='store_true')
    parser.add_argument(
        '--test', help='Flag for running test.', action='store_true')
    parser.add_argument(
        '--config', help='Path to config yaml file.', required=True)
    args = vars(parser.parse_args())

    args = AttributeHashmap(args)
    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config = parse_settings(config)

    # Currently only supports 2 modes: Train+Validation & Test.
    assert args.train ^ args.test  # check XOR logic

    if args.train:
        train(config=config)
    else:
        test(config=config)
