import argparse

import numpy as np
import torch
import yaml
from data_utils import split_dataset
from datasets import BerkeleySegmentation, BrainArman, DiabeticMacularEdema, PolyP, Retina
from model import CUTSEncoder
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import AttributeHashmap, LatentEvaluator, MSELoss, NTXentLoss


def parse_settings(config: AttributeHashmap):
    # fix typing issues
    config.learning_rate = float(config.learning_rate)
    config.weight_decay = float(config.weight_decay)
    # for ablation test
    if config.model_setting == 'no_recon':
        config.lambda_recon_loss = 0
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
    ratios = tuple([c/sum(ratios) for c in ratios])
    train_set, validation_set, test_set = split_dataset(
        dataset=dataset, splits=ratios, random_seed=config.random_seed)

    # Load into DataLoader
    if mode == 'train':
        train_set = DataLoader(
            dataset=train_set, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
        validation_set = DataLoader(
            dataset=validation_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
        return train_set, validation_set
    else:
        test_set = DataLoader(
            dataset=test_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
        return test_set


def train(config: AttributeHashmap):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_set, validation_set = prepare_dataset(
        config=config, mode='train')

    # Build the model
    model = CUTSEncoder().to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    loss_fn_recon = MSELoss()
    loss_fn_contrastive = NTXentLoss()

    for _ in range(config.max_epochs):
        train_loss, validation_loss = 0, 0

        model.train()
        for _, (x_train, _) in tqdm(enumerate(train_set), total=len(train_set)):
            x_train = x_train.type(torch.FloatTensor).to(device)
            optimizer.zero_grad()
            _, x_anchors, x_recon, z_anchors, z_positives = model(x_train)

            loss_recon = loss_fn_recon(x_anchors, x_recon)
            loss_contrastive = loss_fn_contrastive(z_anchors, z_positives)
            loss = config.lambda_contrastive_loss * \
                loss_contrastive + config.lambda_recon_loss * loss_recon

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss / len(train_set)
        print('train loss: {:.3f}'.format(train_loss))

        model.eval()
        with torch.no_grad():
            for _, (x_validation, y_validation) in tqdm(enumerate(validation_set), total=len(validation_set)):
                x_validation = x_validation.type(torch.FloatTensor).to(device)
                z, x_anchors, x_recon, z_anchors, z_positives = model(
                    x_validation)

                loss_recon = loss_fn_recon(x_anchors, x_recon)
                loss_contrastive = loss_fn_contrastive(z_anchors, z_positives)
                loss = config.lambda_contrastive_loss * \
                    loss_contrastive + config.lambda_recon_loss * loss_recon

                validation_loss += loss.item()

        validation_loss = validation_loss / len(validation_set)
        validation_dice_coeff = np.mean(validation_dice_coeff)
        print('validation loss: %.3f' % (validation_loss))

    model.save_weights(config.model_save_path)
    return


def test(config: AttributeHashmap):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_set = prepare_dataset(config=config, mode='test')

    # Build the model
    model = CUTSEncoder().to(device)
    model.load_weights(config.model_save_path)

    loss_fn_recon = MSELoss()
    loss_fn_contrastive = NTXentLoss()
    latent_evaluator = LatentEvaluator(oneshot_prior=config.oneshot_prior)

    test_loss, test_dice_coeffs = 0, []
    model.eval()
    with torch.no_grad():
        for _, (x_test, y_test) in tqdm(enumerate(test_set), total=len(test_set)):
            x_test = x_test.type(torch.FloatTensor).to(device)
            z, x_anchors, x_recon, z_anchors, z_positives = model(
                x_test)

            loss_recon = loss_fn_recon(x_anchors, x_recon)
            loss_contrastive = loss_fn_contrastive(z_anchors, z_positives)
            loss = config.lambda_contrastive_loss * \
                loss_contrastive + config.lambda_recon_loss * loss_recon

            test_loss += loss.item()

            dice_coeffs = latent_evaluator.dice(z, y_test)
            test_dice_coeffs.extend(dice_coeffs)

    test_loss = test_loss / len(test_set)
    test_dice_coeffs = np.mean(test_dice_coeffs)
    print('Test loss: %.3f dice coeff: %.3f' % (test_loss, test_dice_coeffs))
    return


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
        test(config=config)
    else:
        test(config=config)
