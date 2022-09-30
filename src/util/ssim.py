import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def check_ssim(img, r0, c0, r, c):
    img = img.permute(1, 2, 0).detach().numpy()
    img_ = Image.fromarray(np.uint8(img*255))
    im_new = _add_margin(img_, 8, 8, 8, 8, (0, 0, 0))
    x1, x2 = r0, r0+16
    y1, y2 = c0, c0+16

    x1_, x2_ = r, r+16
    y1_, y2_ = c, c+16

    im_new = torch.from_numpy(np.array(im_new))
    return ssim(im_new.detach().numpy()[x1:x2, y1:y2, :], im_new.detach().numpy()[x1_:x2_, y1_:y2_, :], multichannel=True)


def _add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result
