from glob import glob

import numpy as np
from PIL import Image


def get_data_retina(base_path='/data/lab/datasets/amodio/retina',
                    image_folder='selected_128',
                    label_folder='label_128'):
    data, label = [], []

    fns_img = sorted(glob('%s/%s/*' % (base_path, image_folder)))
    fns_label = sorted(glob('%s/%s/*' % (base_path, label_folder)))

    for fn in fns_img:
        img = np.array(Image.open(fn))
        data.append(img)

    for fn in fns_label:
        # sanity check
        fn_bases = [fn_.split('/')[-1].split('.')[0][5:] for fn_ in fns_img]
        if fn.split('/')[-1].split('.')[0][5:] not in fn_bases:
            continue
        img = np.load(fn)
        label.append(img)

    data = np.array(data)
    label = np.array(label)

    data = (data / 255 * 2) - 1  # standardize to [-1, 1]

    return data, label, fns_img


def get_data_berkeley(base_path='/data/lab/datasets/amodio/berkeley_image_segmentation/',
                      image_folder='128x128_imgs',
                      label_folder='128x128_segmented'):
    fns_img = sorted(glob('%s/%s/*.jpg' % (base_path, image_folder)))
    fns_label = sorted(glob('%s/%s/*.jpg' % (base_path, label_folder)))

    fn_bases = [fn.split('/')[-1].split('_')[1][:-4] for fn in fns_img]
    with open('%s/npz_data.npz' % base_path, 'rb') as f:
        npzfile = np.load(f)
        all_images = npzfile['all_images']
        all_images_segmented = npzfile['all_images_segmented']
    data = all_images
    label = all_images_segmented.squeeze(-1)

    data = data * 2 - 1  # standardize to [-1, 1]

    return data, label, fns_img


def get_data_polyp(base_path='/data/lab/datasets/amodio/polyp/selected_imgs_resized/'):
    with open('%s/selected_imgs_resized.npz' % base_path, 'rb') as f:
        npzfile = np.load(f)
        imgs = npzfile['imgs']
        label = npzfile['labels'][:, :, :, 0]
        fns = npzfile['fns'].tolist()

    imgs = (imgs * 2) - 1  # standardize to [-1, 1]

    label = np.where(label > .5, 1, 0)

    return imgs, label, fns


def get_data_macular_edema(base_path='/home/amodio/contrastive/diabetic_macular_edema/'):
    with open('%s/data.npz' % base_path, 'rb') as f:
        npzfile = np.load(f)
        imgs = npzfile['data']
        label = npzfile['labels'][:, :, :, 0]

    imgs = np.repeat(imgs, 3, axis=-1)

    fns = list(range(imgs.shape[0]))

    imgs = (imgs * 2) - 1  # standardize to [-1, 1]

    return imgs, label, fns


def get_data_brain():
    with open('output_brain_arman/output.npz', 'rb') as f:
        npzfile = np.load(f)
        data = npzfile['data']
        label = npzfile['label'][:, :, :, 0]
        predicted_segmentations = npzfile['predicted_segmentations']
    # imgs = data / data.max()
    imgs = data / np.percentile(data, 99)
    imgs = np.where(imgs > 1., 1., imgs)

    imgs = np.where(imgs == 0, 1., imgs)

    imgs = (imgs * 2) - 1

    return imgs, label, list(range(data.shape[0]))
