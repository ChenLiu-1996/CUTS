import argparse
import os

import monai
import numpy as np
import torch
import yaml
from data_utils.prepare_dataset import prepare_dataset, prepare_dataset_supervised
from tqdm import tqdm
from utils.attribute_hashmap import AttributeHashmap
from utils.early_stop import EarlyStopping
from utils.log_util import log
from utils.metrics import dice_coeff, ergas, guided_relabel, hausdorff, per_class_dice_coeff, per_class_hausdorff, \
    range_aware_ssim, rmse
from utils.parse import parse_settings
from utils.seed import seed_everything


def save_weights(model_save_path: str, model):
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    return


def load_weights(model_save_path: str, model, device: torch.device):
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    return


def save_numpy(config: AttributeHashmap, image_batch: torch.Tensor,
               label_true_batch: torch.Tensor, label_pred_batch: torch.Tensor):
    image_batch = image_batch.cpu().detach().numpy()
    label_true_batch = label_true_batch.cpu().detach().numpy()
    label_pred_batch = label_pred_batch.cpu().detach().numpy()
    # channel-first to channel-last
    image_batch = np.moveaxis(image_batch, 1, -1)

    B = image_batch.shape[0]

    # Save the images, labels, and predictions as numpy files for future reference.
    save_path_numpy = '%s/%s/' % (config.output_save_path,
                                  'numpy_files_seg_supervised_%s%s' %
                                  (config.supervised_network,
                                   '_pretrained' if config.pretrained else ''))
    os.makedirs(save_path_numpy, exist_ok=True)
    for image_idx in tqdm(range(B)):
        with open(
                '%s/%s' %
            (save_path_numpy, 'sample_%s.npz' % str(image_idx).zfill(5)),
                'wb+') as f:
            np.savez(f,
                     image=image_batch[image_idx, ...],
                     label_true=label_true_batch[image_idx, ...],
                     label_pred=label_pred_batch[image_idx, ...])
    return


def train(config: AttributeHashmap):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_set, val_set, _, num_image_channel, num_classes = \
        prepare_dataset_supervised(config=config)

    # Build the model
    if config.supervised_network == 'unet':
        model = torch.hub.load(
            'mateuszbuda/brain-segmentation-pytorch',
            'unet',
            in_channels=num_image_channel,
            out_channels=1 if num_classes == 1 else num_classes + 1,
            init_features=config.num_kernels,
            pretrained=config.pretrained).to(device)
    elif config.supervised_network == 'nnunet':
        model = torch.nn.Sequential(
            monai.networks.nets.DynUNet(
                spatial_dims=2,
                in_channels=num_image_channel,
                out_channels=1 if num_classes == 1 else num_classes + 1,
                kernel_size=[5, 5, 5, 5],
                filters=[16, 32, 64, 128],
                strides=[1, 1, 1, 1],
                upsample_kernel_size=[1, 1, 1, 1]),
            torch.nn.Sigmoid()).to(device)
    else:
        raise ValueError(
            '`main_supervised.py: config.supervised_network = %s is not supported.'
            % config.supervised_network)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    early_stopper = EarlyStopping(mode='min',
                                  patience=config.patience,
                                  percentage=False)

    if num_classes == 1:
        loss_fn_supervised = torch.nn.BCELoss()
    else:
        loss_fn_supervised = torch.nn.CrossEntropyLoss()

    best_val_loss = np.inf

    for epoch_idx in tqdm(range(config.max_epochs)):
        train_loss = 0
        train_metrics = {
            'dice': [],
            'hausdorff': [],
            'ssim': [],
            'ergas': [],
            'rmse': [],
        }

        model.train()
        for _, (x_train, seg_true) in enumerate(train_set):
            x_train = x_train.type(torch.FloatTensor).to(device)
            seg_pred = model(x_train)
            if num_classes == 1:
                seg_pred = seg_pred.squeeze(1).type(
                    torch.FloatTensor).to(device)
                seg_pred_metric = (seg_pred > 0.5).type(
                    torch.FloatTensor).to(device)
                seg_true = seg_true.type(torch.FloatTensor).to(device)
            else:
                seg_pred_metric = torch.argmax(seg_pred, dim=1).type(
                    torch.LongTensor).to(device)
                seg_true = seg_true.type(torch.LongTensor).to(device)

            loss = loss_fn_supervised(seg_pred, seg_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            for batch_idx in range(seg_true.shape[0]):
                if num_classes == 1:
                    train_metrics['dice'].append(
                        dice_coeff(
                            label_pred=seg_pred_metric[
                                batch_idx, ...].cpu().detach().numpy(),
                            label_true=seg_true[batch_idx,
                                                ...].cpu().detach().numpy()))
                    train_metrics['hausdorff'].append(
                        hausdorff(
                            label_pred=seg_pred_metric[
                                batch_idx, ...].cpu().detach().numpy(),
                            label_true=seg_true[batch_idx,
                                                ...].cpu().detach().numpy()))
                else:
                    train_metrics['dice'].append(
                        per_class_dice_coeff(
                            label_pred=seg_pred_metric[
                                batch_idx, ...].cpu().detach().numpy(),
                            label_true=seg_true[batch_idx,
                                                ...].cpu().detach().numpy()))
                    train_metrics['hausdorff'].append(
                        per_class_hausdorff(
                            label_pred=seg_pred_metric[
                                batch_idx, ...].cpu().detach().numpy(),
                            label_true=seg_true[batch_idx,
                                                ...].cpu().detach().numpy()))
                train_metrics['ssim'].append(
                    range_aware_ssim(
                        label_pred=seg_pred_metric[batch_idx,
                                                   ...].cpu().detach().numpy(),
                        label_true=seg_true[batch_idx,
                                            ...].cpu().detach().numpy()))
                train_metrics['ergas'].append(
                    ergas(
                        seg_pred_metric[batch_idx, ...].cpu().detach().numpy(),
                        seg_true[batch_idx, ...].cpu().detach().numpy()))
                train_metrics['rmse'].append(
                    rmse(
                        seg_pred_metric[batch_idx, ...].cpu().detach().numpy(),
                        seg_true[batch_idx, ...].cpu().detach().numpy()))

        train_loss = train_loss / len(train_set)

        log('Train [%s/%s] loss: %.3f, dice: %.3f \u00B1 %.3f, hausdorff:  %.3f \u00B1 %.3f, SSIM: %.3f \u00B1 %.3f, ERGAS: %.3f \u00B1 %.3f, RMSE: %.3f \u00B1 %.3f.'
            %
            (epoch_idx, config.max_epochs, train_loss,
             np.mean(train_metrics['dice']), np.std(train_metrics['dice']) /
             np.sqrt(len(train_metrics['dice'])),
             np.mean(train_metrics['hausdorff']),
             np.std(train_metrics['hausdorff']) /
             np.sqrt(len(train_metrics['hausdorff'])),
             np.mean(train_metrics['ssim']), np.std(train_metrics['ssim']) /
             np.sqrt(len(train_metrics['ssim'])),
             np.mean(train_metrics['ergas']), np.std(train_metrics['ergas']) /
             np.sqrt(len(train_metrics['ergas'])),
             np.mean(train_metrics['rmse']), np.std(train_metrics['rmse']) /
             np.sqrt(len(train_metrics['rmse']))),
            filepath=config.log_dir,
            to_console=False)

        val_loss = 0
        model.eval()
        val_metrics = {
            'dice': [],
            'hausdorff': [],
            'ssim': [],
            'ergas': [],
            'rmse': [],
        }
        with torch.no_grad():
            for _, (x_val, seg_true) in enumerate(val_set):
                x_val = x_val.type(torch.FloatTensor).to(device)
                seg_pred = model(x_val)
                if num_classes == 1:
                    seg_pred = seg_pred.squeeze(1).type(
                        torch.FloatTensor).to(device)
                    seg_pred_metric = (seg_pred > 0.5).type(
                        torch.FloatTensor).to(device)
                    seg_true = seg_true.type(torch.FloatTensor).to(device)
                else:
                    seg_pred_metric = torch.argmax(seg_pred, dim=1).type(
                        torch.LongTensor).to(device)
                    seg_true = seg_true.type(torch.LongTensor).to(device)

                loss = loss_fn_supervised(seg_pred, seg_true)

                val_loss += loss.item()

                for batch_idx in range(seg_true.shape[0]):
                    if num_classes == 1:
                        val_metrics['dice'].append(
                            dice_coeff(label_pred=seg_pred_metric[
                                batch_idx, ...].cpu().detach().numpy(),
                                       label_true=seg_true[
                                           batch_idx,
                                           ...].cpu().detach().numpy()))
                        val_metrics['hausdorff'].append(
                            hausdorff(label_pred=seg_pred_metric[
                                batch_idx, ...].cpu().detach().numpy(),
                                      label_true=seg_true[
                                          batch_idx,
                                          ...].cpu().detach().numpy()))
                    else:
                        val_metrics['dice'].append(
                            per_class_dice_coeff(
                                label_pred=seg_pred_metric[
                                    batch_idx, ...].cpu().detach().numpy(),
                                label_true=seg_true[
                                    batch_idx, ...].cpu().detach().numpy()))
                        val_metrics['hausdorff'].append(
                            per_class_hausdorff(
                                label_pred=seg_pred_metric[
                                    batch_idx, ...].cpu().detach().numpy(),
                                label_true=seg_true[
                                    batch_idx, ...].cpu().detach().numpy()))

                    val_metrics['ssim'].append(
                        range_aware_ssim(
                            label_pred=seg_pred_metric[
                                batch_idx, ...].cpu().detach().numpy(),
                            label_true=seg_true[batch_idx,
                                                ...].cpu().detach().numpy()))
                    val_metrics['ergas'].append(
                        ergas(
                            seg_pred_metric[batch_idx,
                                            ...].cpu().detach().numpy(),
                            seg_true[batch_idx, ...].cpu().detach().numpy()))
                    val_metrics['rmse'].append(
                        rmse(
                            seg_pred_metric[batch_idx,
                                            ...].cpu().detach().numpy(),
                            seg_true[batch_idx, ...].cpu().detach().numpy()))

        val_loss = val_loss / len(val_set)

        log('Validation [%s/%s] loss: %.3f, dice: %.3f \u00B1 %.3f, hausdorff:  %.3f \u00B1 %.3f, SSIM: %.3f \u00B1 %.3f, ERGAS: %.3f \u00B1 %.3f, RMSE: %.3f \u00B1 %.3f.'
            %
            (epoch_idx, config.max_epochs, val_loss,
             np.mean(val_metrics['dice']),
             np.std(val_metrics['dice']) / np.sqrt(len(val_metrics['dice'])),
             np.mean(val_metrics['hausdorff']),
             np.std(val_metrics['hausdorff']) / np.sqrt(
                 len(val_metrics['hausdorff'])), np.mean(val_metrics['ssim']),
             np.std(val_metrics['ssim']) / np.sqrt(len(val_metrics['ssim'])),
             np.mean(val_metrics['ergas']), np.std(val_metrics['ergas']) /
             np.sqrt(len(val_metrics['ergas'])), np.mean(val_metrics['rmse']),
             np.std(val_metrics['rmse']) / np.sqrt(len(val_metrics['rmse']))),
            filepath=config.log_dir,
            to_console=False)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_weights(config.model_save_path, model)
            log('%s: Model weights successfully saved.' %
                config.supervised_network,
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
    _, _, test_set, num_image_channel, num_classes = \
        prepare_dataset_supervised(config=config)

    # Build the model
    if config.supervised_network == 'unet':
        model = torch.hub.load(
            'mateuszbuda/brain-segmentation-pytorch',
            'unet',
            in_channels=num_image_channel,
            out_channels=1 if num_classes == 1 else num_classes + 1,
            init_features=config.num_kernels,
            pretrained=config.pretrained).to(device)
    elif config.supervised_network == 'nnunet':
        model = torch.nn.Sequential(
            monai.networks.nets.DynUNet(
                spatial_dims=2,
                in_channels=num_image_channel,
                out_channels=1 if num_classes == 1 else num_classes + 1,
                kernel_size=[5, 5, 5, 5],
                filters=[16, 32, 64, 128],
                strides=[1, 1, 1, 1],
                upsample_kernel_size=[1, 1, 1, 1]),
            torch.nn.Sigmoid()).to(device)
    else:
        raise ValueError(
            '`main_supervised.py: config.supervised_network = %s is not supported.'
            % config.supervised_network)
    load_weights(config.model_save_path, model, device=device)
    log('%s: Model weights successfully loaded.' % config.supervised_network,
        to_console=True)

    if num_classes == 1:
        loss_fn_supervised = torch.nn.BCELoss()
    else:
        loss_fn_supervised = torch.nn.CrossEntropyLoss()

    test_loss = 0
    test_metrics = {
        'dice': [],
        'hausdorff': [],
        'ssim': [],
        'ergas': [],
        'rmse': [],
    }
    model.eval()

    with torch.no_grad():
        for _, (x_test, seg_true) in enumerate(test_set):
            x_test = x_test.type(torch.FloatTensor).to(device)
            seg_pred = model(x_test)
            if num_classes == 1:
                seg_pred = seg_pred.squeeze(1).type(
                    torch.FloatTensor).to(device)
                seg_pred_metric = (seg_pred > 0.5).type(
                    torch.FloatTensor).to(device)
                seg_true = seg_true.type(torch.FloatTensor).to(device)
            else:
                seg_pred_metric = torch.argmax(seg_pred, dim=1).type(
                    torch.LongTensor).to(device)
                seg_true = seg_true.type(torch.LongTensor).to(device)

            loss = loss_fn_supervised(seg_pred, seg_true)

            # NOTE: Stictly speaking, we shall not do the guided relabel here
            # because this is a supervised segmentation task. However, we do
            # it to make it a "fair comparison" with our unsupervised counterparts.
            # This DOES NOT CHANGE the binary segmentation results.
            # This ONLY AFFECTS (improves) the multi-class segmentation results.
            seg_pred_relabeled = np.zeros_like(
                seg_pred_metric.cpu().detach().numpy())
            for batch_idx in range(seg_true.shape[0]):
                seg_pred_relabeled[batch_idx, ...] = guided_relabel(
                    label_pred=seg_pred_metric[batch_idx,
                                               ...].cpu().detach().numpy(),
                    label_true=seg_true[batch_idx, ...].cpu().detach().numpy())

            for batch_idx in range(seg_true.shape[0]):
                if num_classes == 1:
                    test_metrics['dice'].append(
                        dice_coeff(
                            label_pred=seg_pred_relabeled[batch_idx, ...],
                            label_true=seg_true[batch_idx,
                                                ...].cpu().detach().numpy()))
                    test_metrics['hausdorff'].append(
                        hausdorff(
                            label_pred=seg_pred_relabeled[batch_idx, ...],
                            label_true=seg_true[batch_idx,
                                                ...].cpu().detach().numpy()))
                else:
                    test_metrics['dice'].append(
                        per_class_dice_coeff(
                            label_pred=seg_pred_relabeled[batch_idx, ...],
                            label_true=seg_true[batch_idx,
                                                ...].cpu().detach().numpy()))
                    test_metrics['hausdorff'].append(
                        per_class_hausdorff(
                            label_pred=seg_pred_relabeled[batch_idx, ...],
                            label_true=seg_true[batch_idx,
                                                ...].cpu().detach().numpy()))
                test_metrics['ssim'].append(
                    range_aware_ssim(
                        label_pred=seg_pred_relabeled[batch_idx, ...],
                        label_true=seg_true[batch_idx,
                                            ...].cpu().detach().numpy()))
                test_metrics['ergas'].append(
                    ergas(seg_pred_relabeled[batch_idx, ...],
                          seg_true[batch_idx, ...].cpu().detach().numpy()))
                test_metrics['rmse'].append(
                    rmse(seg_pred_relabeled[batch_idx, ...],
                         seg_true[batch_idx, ...].cpu().detach().numpy()))

            test_loss += loss.item()

    test_loss = test_loss / len(test_set)

    log('Test loss: %.3f, dice: %.3f \u00B1 %.3f, hausdorff: %.3f \u00B1 %.3f, SSIM: %.3f \u00B1 %.3f, ERGAS: %.3f \u00B1 %.3f, RMSE: %.3f \u00B1 %.3f.'
        % (test_loss, np.mean(test_metrics['dice']),
           np.std(test_metrics['dice']) / np.sqrt(len(test_metrics['dice'])),
           np.mean(test_metrics['hausdorff']),
           np.std(test_metrics['hausdorff']) / np.sqrt(
               len(test_metrics['hausdorff'])), np.mean(test_metrics['ssim']),
           np.std(test_metrics['ssim']) / np.sqrt(len(test_metrics['ssim'])),
           np.mean(test_metrics['ergas']), np.std(test_metrics['ergas']) /
           np.sqrt(len(test_metrics['ergas'])), np.mean(test_metrics['rmse']),
           np.std(test_metrics['rmse']) / np.sqrt(len(test_metrics['rmse']))),
        filepath=config.log_dir,
        to_console=True)
    return


def infer(config: AttributeHashmap):
    device = torch.device('cpu')
    entire_set, _ = \
        prepare_dataset(config=config, mode='test')
    _, _, _, num_image_channel, num_classes = prepare_dataset_supervised(
        config=config)

    # Build the model
    if config.supervised_network == 'unet':
        model = torch.hub.load(
            'mateuszbuda/brain-segmentation-pytorch',
            'unet',
            in_channels=num_image_channel,
            out_channels=1 if num_classes == 1 else num_classes + 1,
            init_features=config.num_kernels,
            pretrained=config.pretrained).to(device)
    elif config.supervised_network == 'nnunet':
        model = torch.nn.Sequential(
            monai.networks.nets.DynUNet(
                spatial_dims=2,
                in_channels=num_image_channel,
                out_channels=1 if num_classes == 1 else num_classes + 1,
                kernel_size=[5, 5, 5, 5],
                filters=[16, 32, 64, 128],
                strides=[1, 1, 1, 1],
                upsample_kernel_size=[1, 1, 1, 1]),
            torch.nn.Sigmoid()).to(device)
    else:
        raise ValueError(
            '`main_supervised.py: config.supervised_network = %s is not supported.'
            % config.supervised_network)
    load_weights(config.model_save_path, model, device=device)
    log('%s: Model weights successfully loaded.' % config.supervised_network,
        to_console=True)

    model.eval()

    with torch.no_grad():
        for _, (x, seg_true) in enumerate(entire_set):
            x = x.type(torch.FloatTensor).to(device)
            seg_pred = model(x)
            if num_classes == 1:
                seg_pred = seg_pred.squeeze(1).type(
                    torch.FloatTensor).to(device)
                seg_pred = (seg_pred > 0.5).type(torch.FloatTensor).to(device)
                seg_true = seg_true.type(torch.FloatTensor).to(device)
            else:
                seg_pred = torch.argmax(seg_pred, dim=1).type(
                    torch.LongTensor).to(device)
                seg_true = seg_true.type(torch.LongTensor).to(device)

            save_numpy(config=config,
                       image_batch=x,
                       label_true_batch=seg_true,
                       label_pred_batch=seg_pred)

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

    assert args.mode in ['train', 'test', 'infer']

    seed_everything(config.random_seed)

    if args.mode == 'train':
        train(config=config)
        test(config=config)
        # infer(config=config)
    elif args.mode == 'test':
        test(config=config)
    elif args.mode == 'infer':
        infer(config=config)
