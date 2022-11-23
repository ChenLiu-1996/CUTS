import argparse
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
from utils import AttributeHashmap, EarlyStopping, LatentEvaluator, NTXentLoss, log, parse_settings


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
    '''
    Dataset Split.
    Something special here.
    Since the method is unsupervised/self-supervised, we believe it is justifiable to:
        (1) Split the dataset to a train set and a validation set when training the model.
        (2) Use the entire dataset as the test set for evaluating the segmentation performance.
    We believe this is reasonable because ground truth label is NOT used during the training stage.
    '''
    # Load into DataLoader
    if mode == 'train':
        ratios = [float(c) for c in config.train_val_ratio.split(':')]
        ratios = tuple([c / sum(ratios) for c in ratios])
        train_set, val_set = split_dataset(dataset=dataset,
                                           splits=ratios,
                                           random_seed=config.random_seed)

        min_batch_per_epoch = 5
        desired_len = max(len(train_set),
                          config.batch_size * min_batch_per_epoch)
        train_set = ExtendedDataset(dataset=train_set, desired_len=desired_len)

        train_set = DataLoader(dataset=train_set,
                               batch_size=config.batch_size,
                               shuffle=True,
                               num_workers=config.num_workers)
        val_set = DataLoader(dataset=val_set,
                             batch_size=len(val_set),
                             shuffle=False,
                             num_workers=config.num_workers)
        return train_set, val_set, num_image_channel
    else:
        test_set = DataLoader(dataset=dataset,
                              batch_size=len(dataset),
                              shuffle=False,
                              num_workers=config.num_workers)
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
    optimizer = optim.AdamW(model.parameters(),
                            lr=config.learning_rate,
                            weight_decay=config.weight_decay)

    loss_fn_recon = torch.nn.MSELoss()
    loss_fn_contrastive = NTXentLoss()
    early_stopper = EarlyStopping(mode='min',
                                  patience=config.patience,
                                  percentage=False)

    best_val_loss = np.inf

    for epoch_idx in tqdm(range(config.max_epochs)):
        train_loss_recon, train_loss_contrastive, train_loss = 0, 0, 0

        model.train()
        for _, (x_train, _) in enumerate(train_set):
            x_train = x_train.type(torch.FloatTensor).to(device)
            _, patch_real, patch_recon, z_anchors, z_positives = model(x_train)

            loss_recon = loss_fn_recon(patch_real, patch_recon)
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

        log('Train [%s/%s] recon loss: %.3f, contrastive loss: %.3f, total loss: %.3f'
            % (epoch_idx + 1, config.max_epochs, train_loss_recon,
               train_loss_contrastive, train_loss),
            filepath=config.log_dir,
            to_console=False)

        val_loss_recon, val_loss_contrastive, val_loss = 0, 0, 0
        model.eval()
        with torch.no_grad():
            for _, (x_val, _) in enumerate(val_set):
                x_val = x_val.type(torch.FloatTensor).to(device)
                _, patch_real, patch_recon, z_anchors, z_positives = model(
                    x_val)

                loss_recon = loss_fn_recon(patch_real, patch_recon)
                loss_contrastive = loss_fn_contrastive(z_anchors, z_positives)
                loss = config.lambda_contrastive_loss * \
                    loss_contrastive + config.lambda_recon_loss * loss_recon

                val_loss_recon += loss_recon.item()
                val_loss_contrastive += loss_contrastive.item()
                val_loss += loss.item()

        val_loss_recon = val_loss_recon / len(val_set)
        val_loss_contrastive = val_loss_contrastive / len(val_set)
        val_loss = val_loss / len(val_set)
        log('Validation [%s/%s] recon loss: %.3f, contrastive loss: %.3f, total loss: %.3f'
            % (epoch_idx + 1, config.max_epochs, val_loss_recon,
               val_loss_contrastive, val_loss),
            filepath=config.log_dir,
            to_console=False)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_weights(config.model_save_path)
            log('CUTSEncoder: Model weights successfully saved.',
                filepath=config.log_dir,
                to_console=False)

        if early_stopper.step(val_loss):
            # If the validation loss stop decreasing, stop training.
            log('Early stopping criterion met. Ending training.',
                filepath=config.log_dir,
                to_console=True)
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
        filepath=config.log_dir,
        to_console=False)

    loss_fn_recon = torch.nn.MSELoss()
    loss_fn_contrastive = NTXentLoss()
    latent_evaluator = LatentEvaluator(
        segmentation_paradigm=config.segmentation_paradigm,
        pos_enc_gamma=config.pos_enc_gamma,
        save_path=config.output_save_path,
        num_workers=config.num_workers,
        random_seed=config.random_seed)

    test_loss_recon, test_loss_contrastive, test_loss, test_dice_coeffs = 0, 0, 0, []
    model.eval()

    with torch.no_grad():
        for _, (x_test, y_test) in enumerate(test_set):
            x_test = x_test.type(torch.FloatTensor).to(device)
            z, patch_real, patch_recon, z_anchors, z_positives = model(x_test)

            loss_recon = loss_fn_recon(patch_real, patch_recon)
            loss_contrastive = loss_fn_contrastive(z_anchors, z_positives)
            loss = config.lambda_contrastive_loss * \
                loss_contrastive + config.lambda_recon_loss * loss_recon

            test_loss_recon += loss_recon.item()
            test_loss_contrastive += loss_contrastive.item()
            test_loss += loss.item()

            # Each pixel embedding recons to a patch.
            # Here we only take the center pixel of the reconed patch and collect into a reconed image.
            B, L, H, W = z.shape
            z_for_recon = z.permute((0, 2, 3, 1)).reshape(B, H * W, L)
            patch_recon = model.recon(z_for_recon)
            C = patch_recon.shape[2]
            P = patch_recon.shape[-1]
            patch_recon = patch_recon[:, :, :, P // 2, P // 2]
            patch_recon = patch_recon.permute((0, 2, 1)).reshape(B, C, H, W)

            dice_coeffs = latent_evaluator.eval(image_batch=x_test,
                                                recon_batch=patch_recon,
                                                label_true_batch=y_test,
                                                latent_batch=z,
                                                metric=config.test_metric)
            test_dice_coeffs.extend(dice_coeffs)

    test_loss_recon = test_loss_recon / len(test_set)
    test_loss_contrastive = test_loss_contrastive / len(test_set)
    test_loss = test_loss / len(test_set)
    test_dice_coeffs_mean, test_dice_coeffs_std = \
        np.mean(test_dice_coeffs), np.std(test_dice_coeffs)

    log('Test recon loss: %.3f, contrastive loss: %.3f, total loss: %.3f. dice coeff: %.3f \u00B1 %.3f'
        % (test_loss_recon, test_loss_contrastive, test_loss,
           test_dice_coeffs_mean, test_dice_coeffs_std),
        filepath=config.log_dir,
        to_console=True)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entry point to run CUTS.')
    parser.add_argument('--mode', help='`train` or `test`?', required=True)
    parser.add_argument('--config',
                        help='Path to config yaml file.',
                        required=True)
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
