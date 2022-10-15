import argparse
import os
import random

import numpy as np
import torch
import yaml
from data_utils import ExtendedDataset, split_dataset
from datasets import BerkeleySegmentation, BrainArman, DiabeticMacularEdema, PolyP, Retina
from model import CUTSEncoder
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import AttributeHashmap, EarlyStopping, LatentEvaluator, MSELoss, NTXentLoss, log


def parse_settings(config: AttributeHashmap, log_settings: bool = True):
    # fix typing issues
    config.learning_rate = float(config.learning_rate)
    config.weight_decay = float(config.weight_decay)

    # for ablation test
    if config.model_setting == 'no_recon':
        config.lambda_recon_loss = 0
    if config.model_setting == 'no_contrastive':
        config.lambda_contrastive_loss = 0

    # Initialize log file.
    config.log_dir = config.log_folder + '/' + \
        os.path.basename(
            config.config_file_name).replace('.yaml', '') + '_log.txt'
    if log_settings:
        log_str = 'Config: \n'
        for key in config.keys():
            log_str += '%s: %s\n' % (key, config[key])
        log_str += '\nTraining History:'
        log(log_str, filepath=config.log_dir, to_console=True)
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
        raise Exception(
            'Dataset not found. Check `dataset_name` in config yaml file.')

    num_image_channel = dataset.num_image_channel()

    # Train/Val/Test Split
    ratios = [float(c) for c in config.train_val_test_ratio.split(':')]
    ratios = tuple([c/sum(ratios) for c in ratios])
    train_set, val_set, test_set = split_dataset(
        dataset=dataset, splits=ratios, random_seed=config.random_seed)

    # Load into DataLoader
    if mode == 'train':
        # Round up the dataset to avoid trailing batch.
        if len(train_set) > config.batch_size:
            desired_len = len(train_set) + config.batch_size % len(train_set)
        else:
            desired_len = config.batch_size
        train_set = ExtendedDataset(
            dataset=train_set, desired_len=desired_len)

        train_set = DataLoader(
            dataset=train_set, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
        val_set = DataLoader(
            dataset=val_set, batch_size=len(val_set), shuffle=False, num_workers=config.num_workers)
        return train_set, val_set, num_image_channel
    else:
        test_set = DataLoader(
            dataset=test_set, batch_size=len(test_set), shuffle=False, num_workers=config.num_workers)
        return test_set, num_image_channel


def train(config: AttributeHashmap):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_set, val_set, num_image_channel = \
        prepare_dataset(config=config, mode='train')

    # Build the model
    model = CUTSEncoder(
        in_channels=num_image_channel,
        num_kernels=config.num_kernels,
        random_seed=config.random_seed,
        sampled_patches_per_image=config.sampled_patches_per_image).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    loss_fn_recon = MSELoss()
    loss_fn_contrastive = NTXentLoss(batch_size=config.batch_size)
    early_stopper = EarlyStopping(
        mode='min', patience=config.patience, percentage=False)

    best_val_loss = np.inf

    for epoch_idx in tqdm(range(config.max_epochs)):
        train_loss_recon, train_loss_contrastive, train_loss = 0, 0, 0

        model.train()
        for _, (x_train, _) in enumerate(train_set):
            x_train = x_train.type(torch.FloatTensor).to(device)
            _, x_anchors, x_recon, z_anchors, z_positives = model(x_train)

            loss_recon = loss_fn_recon(x_anchors, x_recon)
            loss_contrastive = loss_fn_contrastive(z_anchors, z_positives)
            loss = config.lambda_contrastive_loss * \
                loss_contrastive + config.lambda_recon_loss * loss_recon

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_recon += loss_recon.item()
            train_loss_contrastive += loss_contrastive.item()
            train_loss += loss.item()

        train_loss_recon = train_loss_recon / len(train_set)
        train_loss_contrastive = train_loss_contrastive / len(train_set)
        train_loss = train_loss / len(train_set)

        log('Train [%s/%s] recon loss: %.3f, contrastive loss: %.3f, total loss: %.3f' % (
            epoch_idx+1, config.max_epochs, train_loss_recon, train_loss_contrastive, train_loss),
            filepath=config.log_dir, to_console=False)

        val_loss_recon, val_loss_contrastive, val_loss = 0, 0, 0
        model.eval()
        with torch.no_grad():
            for _, (x_val, _) in enumerate(val_set):
                x_val = x_val.type(torch.FloatTensor).to(device)
                _, x_anchors, x_recon, z_anchors, z_positives = model(x_val)

                loss_recon = loss_fn_recon(x_anchors, x_recon)
                loss_contrastive = loss_fn_contrastive(z_anchors, z_positives)
                loss = config.lambda_contrastive_loss * \
                    loss_contrastive + config.lambda_recon_loss * loss_recon

                val_loss_recon += loss_recon.item()
                val_loss_contrastive += loss_contrastive.item()
                val_loss += loss.item()

        val_loss_recon = val_loss_recon / len(val_set)
        val_loss_contrastive = val_loss_contrastive / len(val_set)
        val_loss = val_loss / len(val_set)
        log('Validation [%s/%s] recon loss: %.3f, contrastive loss: %.3f, total loss: %.3f' % (
            epoch_idx+1, config.max_epochs, val_loss_recon, val_loss_contrastive, val_loss),
            filepath=config.log_dir, to_console=False)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_weights(config.model_save_path)
            log('CUTSEncoder: Model weights successfully saved.',
                filepath=config.log_dir, to_console=False)

        if early_stopper.step(val_loss):
            # If the validation loss stop decreasing, stop training.
            log('Early stopping criterion met. Ending training.',
                filepath=config.log_dir, to_console=True)
            break

    return


def test(config: AttributeHashmap):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_set, num_image_channel = prepare_dataset(config=config, mode='test')

    # Build the model
    model = CUTSEncoder(
        in_channels=num_image_channel,
        num_kernels=config.num_kernels,
        random_seed=config.random_seed,
        sampled_patches_per_image=config.sampled_patches_per_image).to(device)
    model.load_weights(config.model_save_path)
    log('CUTSEncoder: Model weights successfully loaded.',
        filepath=config.log_dir, to_console=False)

    loss_fn_recon = MSELoss()
    loss_fn_contrastive = NTXentLoss(batch_size=config.batch_size)
    latent_evaluator = LatentEvaluator(
        oneshot_prior=config.oneshot_prior, random_seed=config.random_seed)

    test_loss, test_dice_coeffs = 0, []
    model.eval()

    # Results to save.
    test_images, test_labels, test_mc_segs, test_bin_segs = None, None, None, None
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

            dice_coeffs, multiclass_segs, binary_segs = latent_evaluator.dice(
                z, y_test)
            test_dice_coeffs.extend(dice_coeffs)

            if test_images is None:
                test_images = x_test.cpu().detach().numpy()
                test_labels = y_test.cpu().detach().numpy()
                test_mc_segs = multiclass_segs
                test_bin_segs = binary_segs
            else:
                test_images = np.concatenate(
                    (test_images, x_test.cpu().detach().numpy()), axis=0)
                test_labels = np.concatenate(
                    (test_labels, y_test.cpu().detach().numpy()), axis=0)
                test_mc_segs = np.concatenate(
                    (test_mc_segs, multiclass_segs.cpu().detach().numpy()), axis=0)
                test_bin_segs = np.concatenate(
                    (test_bin_segs, binary_segs.cpu().detach().numpy()), axis=0)

    test_loss = test_loss / len(test_set)
    test_dice_coeffs = np.mean(test_dice_coeffs)
    log('Test loss: %.3f dice coeff: %.3f' % (test_loss, test_dice_coeffs),
        filepath=config.log_dir, to_console=False)

    log('Saving images, labels, multi-class segmentations and binary segmentations.')
    os.makedirs(config.output_save_path, exist_ok=True)
    np.save(file=config.output_save_path + 'test_images.npy', arr=test_images)
    np.save(file=config.output_save_path + 'test_labels.npy', arr=test_labels)
    np.save(file=config.output_save_path +
            'test_multiclass_segs.npy', arr=test_mc_segs)
    np.save(file=config.output_save_path +
            'test_binary_segs.npy', arr=test_bin_segs)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entry point to run CUTS.')
    parser.add_argument(
        '--mode', help='`train` or `test`?', required=True)
    parser.add_argument(
        '--config', help='Path to config yaml file.', required=True)
    args = vars(parser.parse_args())

    args = AttributeHashmap(args)
    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config.config_file_name = args.config
    config = parse_settings(config, log_settings=args.mode == 'train')

    # Currently supports 2 modes: `train`: Train+Validation+Test & `test`: Test.
    assert args.mode in ['train', 'test']

    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    if args.mode == 'train':
        train(config=config)
        test(config=config)
    elif args.mode == 'test':
        test(config=config)
