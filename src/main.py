import argparse

import numpy as np
import torch
import yaml
from data_utils.prepare_dataset import prepare_dataset
from model import CUTSEncoder
from tqdm import tqdm
from utils.attribute_hashmap import AttributeHashmap
from utils.early_stop import EarlyStopping
from utils.log_util import log
from utils.losses import NTXentLoss
from utils.output_saver import OutputSaver
from utils.parse import parse_settings
from utils.seed import seed_everything


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
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    early_stopper = EarlyStopping(mode='min',
                                  patience=config.patience,
                                  percentage=False)

    loss_fn_recon = torch.nn.MSELoss()
    loss_fn_contrastive = NTXentLoss()

    best_val_loss = np.inf

    for epoch_idx in tqdm(range(config.max_epochs)):
        train_loss_recon, train_loss_contrastive, train_loss = 0, 0, 0

        model.train()
        for _, (x_train, _) in enumerate(train_set):
            B = x_train.shape[0]

            x_train = x_train.type(torch.FloatTensor).to(device)
            _, patch_real, patch_recon, z_anchors, z_positives = model(x_train)

            loss_recon = loss_fn_recon(patch_real, patch_recon)
            loss_contrastive = loss_fn_contrastive(z_anchors, z_positives)
            loss = config.lambda_contrastive_loss * \
                loss_contrastive + (1 - config.lambda_contrastive_loss) * loss_recon

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_recon += loss_recon.item() * B
            train_loss_contrastive += loss_contrastive.item() * B
            train_loss += loss.item() * B

        train_loss_recon = train_loss_recon / len(train_set.dataset)
        train_loss_contrastive = train_loss_contrastive / len(
            train_set.dataset)
        train_loss = train_loss / len(train_set.dataset)

        log('Train [%s/%s] recon loss: %.3f, contrastive loss: %.3f, total loss: %.3f'
            % (epoch_idx + 1, config.max_epochs, train_loss_recon,
               train_loss_contrastive, train_loss),
            filepath=config.log_dir,
            to_console=False)

        val_loss_recon, val_loss_contrastive, val_loss = 0, 0, 0
        model.eval()
        with torch.no_grad():
            for _, (x_val, _) in enumerate(val_set):
                B = x_val.shape[0]

                x_val = x_val.type(torch.FloatTensor).to(device)
                _, patch_real, patch_recon, z_anchors, z_positives = model(
                    x_val)

                loss_recon = loss_fn_recon(patch_real, patch_recon)
                loss_contrastive = loss_fn_contrastive(z_anchors, z_positives)
                loss = config.lambda_contrastive_loss * \
                    loss_contrastive + (1 - config.lambda_contrastive_loss) * loss_recon

                val_loss_recon += loss_recon.item() * B
                val_loss_contrastive += loss_contrastive.item() * B
                val_loss += loss.item() * B

        val_loss_recon = val_loss_recon / len(val_set.dataset)
        val_loss_contrastive = val_loss_contrastive / len(val_set.dataset)
        val_loss = val_loss / len(val_set.dataset)
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
    device = torch.device('cpu')
    test_set, num_image_channel = prepare_dataset(config=config, mode='test')

    # Build the model
    model = CUTSEncoder(
        in_channels=num_image_channel,
        num_kernels=config.num_kernels,
        random_seed=config.random_seed,
        sampled_patches_per_image=config.sampled_patches_per_image).to(device)
    model.load_weights(config.model_save_path, device=device)
    log('CUTSEncoder: Model weights successfully loaded.', to_console=True)

    loss_fn_recon = torch.nn.MSELoss()
    loss_fn_contrastive = NTXentLoss()
    output_saver = OutputSaver(save_path=config.output_save_path,
                               random_seed=config.random_seed)

    test_loss_recon, test_loss_contrastive, test_loss = 0, 0, 0
    model.eval()

    with torch.no_grad():
        for _, (x_test, y_test) in enumerate(test_set):
            x_test = x_test.type(torch.FloatTensor).to(device)
            z, patch_real, patch_recon, z_anchors, z_positives = model(x_test)

            loss_recon = loss_fn_recon(patch_real, patch_recon)
            loss_contrastive = loss_fn_contrastive(z_anchors, z_positives)
            loss = config.lambda_contrastive_loss * \
                loss_contrastive + (1 - config.lambda_contrastive_loss) * loss_recon

            B = x_test.shape[0]
            test_loss_recon += loss_recon.item() * B
            test_loss_contrastive += loss_contrastive.item() * B
            test_loss += loss.item() * B

            # Each pixel embedding recons to a patch.
            # Here we only take the center pixel of the reconed patch and collect into a reconed image.
            B, L, H, W = z.shape
            z_for_recon = z.permute((0, 2, 3, 1)).reshape(B, H * W, L)
            patch_recon = model.recon(z_for_recon)
            C = patch_recon.shape[2]
            P = patch_recon.shape[-1]
            patch_recon = patch_recon[:, :, :, P // 2, P // 2]
            patch_recon = patch_recon.permute((0, 2, 1)).reshape(B, C, H, W)

            output_saver.save(image_batch=x_test,
                              recon_batch=patch_recon,
                              label_true_batch=y_test if config.no_label is False else None,
                              latent_batch=z)

    test_loss_recon = test_loss_recon / len(test_set.dataset)
    test_loss_contrastive = test_loss_contrastive / len(test_set.dataset)
    test_loss = test_loss / len(test_set.dataset)

    log('Test recon loss: %.3f, contrastive loss: %.3f, total loss: %.3f.' %
        (test_loss_recon, test_loss_contrastive, test_loss),
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

    assert args.mode in ['train', 'test']

    seed_everything(config.random_seed)

    if args.mode == 'train':
        train(config=config)
        test(config=config)
    elif args.mode == 'test':
        test(config=config)
