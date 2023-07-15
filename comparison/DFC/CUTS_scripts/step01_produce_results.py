import argparse
import os
import sys
from glob import glob
import numpy as np

import torch.multiprocessing
import yaml
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')

# Import from our CUTS codebase.
import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-4])
sys.path.insert(0, import_dir + '/src/utils/')
from attribute_hashmap import AttributeHashmap
from output_saver import squeeze_excessive_dimension

sys.path.insert(0, import_dir + '/src/data_utils/')
from prepare_dataset import prepare_dataset

# Import from DFC.
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import cv2
import sys
import numpy as np
import torch.nn.init

use_cuda = torch.cuda.is_available()


def parse_settings(config: AttributeHashmap) -> AttributeHashmap:
    CUTS_ROOT = '/'.join(os.path.realpath(__file__).split('/')[:-4])
    for key in config.keys():
        if type(config[key]) == str and '$CUTS_ROOT' in config[key]:
            config[key] = config[key].replace('$CUTS_ROOT', CUTS_ROOT)

    for key in config.keys():
        if type(config[key]) == list:
            for i, item in enumerate(config[key]):
                if type(item) == str and '$CUTS_ROOT' in item:
                    config[key][i] = item.replace('$CUTS_ROOT', CUTS_ROOT)

    if 'lr' in config.keys():
        config.lr = float(config.lr)
    return config


def run(test_set, save_path) -> None:

    args = {
        'scribble': False,
        'nChannel': 10,
        'maxIter': 1000,
        'minLabels': 3,
        'lr': 0.1,
        'nConv': 2,
        'visualize': False,
        'stepsize_sim': 1,
        'stepsize_con': 1,
        'stepsize_scr': 0.5,
    }

    args = AttributeHashmap(args)

    # CNN model
    class MyNet(nn.Module):

        def __init__(self, input_dim):
            super(MyNet, self).__init__()
            self.conv1 = nn.Conv2d(input_dim,
                                   args.nChannel,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
            self.bn1 = nn.BatchNorm2d(args.nChannel)
            self.conv2 = nn.ModuleList()
            self.bn2 = nn.ModuleList()
            for i in range(args.nConv - 1):
                self.conv2.append(
                    nn.Conv2d(args.nChannel,
                              args.nChannel,
                              kernel_size=3,
                              stride=1,
                              padding=1))
                self.bn2.append(nn.BatchNorm2d(args.nChannel))
            self.conv3 = nn.Conv2d(args.nChannel,
                                   args.nChannel,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)
            self.bn3 = nn.BatchNorm2d(args.nChannel)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.bn1(x)
            for i in range(args.nConv - 1):
                x = self.conv2[i](x)
                x = F.relu(x)
                x = self.bn2[i](x)
            x = self.conv3(x)
            x = self.bn3(x)
            return x

    for i, (x, y) in enumerate(tqdm(test_set)):
        if use_cuda:
            img = x.cuda().float()
        else:
            img = x.float()
        if img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)

        img = (img + 1) / 2

        # load image
        im = np.uint8(
            img.permute(0, 2, 3, 1).squeeze(0).cpu().detach().numpy() * 255)
        data = Variable(img, requires_grad=True)

        # load scribble
        if args.scribble:
            mask = cv2.imread(
                args.input.replace('.' + args.input.split('.')[-1],
                                   '_scribble.png'), -1)
            mask = mask.reshape(-1)
            mask_inds = np.unique(mask)
            mask_inds = np.delete(mask_inds, np.argwhere(mask_inds == 255))
            inds_sim = torch.from_numpy(np.where(mask == 255)[0])
            inds_scr = torch.from_numpy(np.where(mask != 255)[0])
            target_scr = torch.from_numpy(mask.astype(np.int))
            if use_cuda:
                inds_sim = inds_sim.cuda()
                inds_scr = inds_scr.cuda()
                target_scr = target_scr.cuda()
            target_scr = Variable(target_scr)
            # set minLabels
            args.minLabels = len(mask_inds)

        # train
        model = MyNet(data.size(1))
        if use_cuda:
            model.cuda()
        model.train()

        # similarity loss definition
        loss_fn = torch.nn.CrossEntropyLoss()

        # scribble loss definition
        loss_fn_scr = torch.nn.CrossEntropyLoss()

        # continuity loss definition
        loss_hpy = torch.nn.L1Loss(size_average=True)
        loss_hpz = torch.nn.L1Loss(size_average=True)

        HPy_target = torch.zeros(im.shape[0] - 1, im.shape[1], args.nChannel)
        HPz_target = torch.zeros(im.shape[0], im.shape[1] - 1, args.nChannel)
        if use_cuda:
            HPy_target = HPy_target.cuda()
            HPz_target = HPz_target.cuda()

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        label_colours = np.random.randint(255, size=(100, 3))

        for batch_idx in range(args.maxIter):
            # forwarding
            optimizer.zero_grad()
            output = model(data)[0]
            output = output.permute(1, 2,
                                    0).contiguous().view(-1, args.nChannel)

            outputHP = output.reshape(
                (im.shape[0], im.shape[1], args.nChannel))
            HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
            HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
            lhpy = loss_hpy(HPy, HPy_target)
            lhpz = loss_hpz(HPz, HPz_target)

            ignore, target = torch.max(output, 1)
            im_target = target.data.cpu().numpy()
            nLabels = len(np.unique(im_target))
            if args.visualize:
                im_target_rgb = np.array(
                    [label_colours[c % args.nChannel] for c in im_target])
                im_target_rgb = im_target_rgb.reshape(im.shape).astype(
                    np.uint8)
                cv2.imshow("output", im_target_rgb)
                cv2.waitKey(10)

            # loss
            if args.scribble:
                loss = args.stepsize_sim * loss_fn(
                    output[inds_sim],
                    target[inds_sim]) + args.stepsize_scr * loss_fn_scr(
                        output[inds_scr], target_scr[inds_scr]
                    ) + args.stepsize_con * (lhpy + lhpz)
            else:
                loss = args.stepsize_sim * loss_fn(
                    output, target) + args.stepsize_con * (lhpy + lhpz)

            loss.backward()
            optimizer.step()

            print(batch_idx, '/', args.maxIter, '|', ' label num :', nLabels,
                  ' | loss :', loss.item())

            if nLabels <= args.minLabels:
                print("nLabels", nLabels, "reached minLabels", args.minLabels,
                      ".")
                break

        output = model(data)[0]
        output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)
        ignore, target = torch.max(output, 1)
        im_target = target.data.cpu().numpy()
        label_dfc = im_target.reshape(im.shape[0], im.shape[1])

        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        x = np.moveaxis(x, 1, -1)
        y = np.moveaxis(y, 1, -1)
        x = squeeze_excessive_dimension(x)
        y = squeeze_excessive_dimension(y)

        assert x.shape[0] == 1
        assert y.shape[0] == 1
        x = x.squeeze(0)
        y = y.squeeze(0)

        with open('%s/%s' % (save_path, 'sample_%s.npz' % str(i).zfill(5)),
                  'wb+') as f:
            np.savez(f, image=x, label=y, label_dfc=label_dfc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Path to CUTS config yaml file.',
                        required=True)
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    config_file_name = args.config
    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config.config_file_name = config_file_name
    config = parse_settings(config)

    # Use batch size of 1.
    config.batch_size = 1

    test_set, _ = prepare_dataset(config, mode='test')

    save_path = '%s/%s' % (config.output_save_path, 'numpy_files_seg_DFC')

    os.makedirs(save_path, exist_ok=True)

    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    run(test_set, save_path)
