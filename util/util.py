"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import re
import importlib
import torch
from argparse import Namespace
import numpy as np
from PIL import Image
import os
import sys
import argparse
import scipy.io as scio
#from numba import u1, u2, jit

# returns a configuration for creating a generator
# |default_opt| should be the opt of the current experiment
# |**kwargs|: if any configuration should be overriden, it can be specified here

colormap = scio.loadmat('./util/color150.mat')['colors']
def masktorgb(x):
    mask = np.zeros((x.shape[0], 3, x.shape[2], x.shape[2]), dtype=np.uint8)
    for k in range(x.shape[0]):
        for i in range(x.shape[2]):
            for j in range(x.shape[3]):
                mask[k, :, i, j] = colormap[x[k, 0, i, j] - 1]
    return mask

def feature_normalize(feature_in):
    feature_in_norm = torch.norm(feature_in, 2, 1, keepdim=True) + sys.float_info.epsilon
    feature_in_norm = torch.div(feature_in, feature_in_norm)
    return feature_in_norm

def weighted_l1_loss(input, target, weights):
    out = torch.abs(input - target)
    out = out * weights.expand_as(out)
    loss = out.mean()
    return loss

def mse_loss(input, target=0):
    return torch.mean((input - target)**2)

def vgg_preprocess(tensor, vgg_normal_correct=False):
    if vgg_normal_correct:
        tensor = (tensor + 1) / 2
    # input is RGB tensor which ranges in [0,1]
    # output is BGR tensor which ranges in [0,255]
    tensor_bgr = torch.cat((tensor[:, 2:3, :, :], tensor[:, 1:2, :, :], tensor[:, 0:1, :, :]), dim=1)
    # tensor_bgr = tensor[:, [2, 1, 0], ...]
    tensor_bgr_ml = tensor_bgr - torch.Tensor([0.40760392, 0.45795686, 0.48501961]).type_as(tensor_bgr).view(1, 3, 1, 1)
    tensor_rst = tensor_bgr_ml * 255
    return tensor_rst

def copyconf(default_opt, **kwargs):
    conf = argparse.Namespace(**vars(default_opt))
    for key in kwargs:
        print(key, kwargs[key])
        setattr(conf, key, kwargs[key])
    return conf


def tile_images(imgs, picturesPerRow=4):
    """ Code borrowed from
    https://stackoverflow.com/questions/26521365/cleanly-tile-numpy-array-of-images-stored-in-a-flattened-1d-format/26521997
    """

    # Padding
    if imgs.shape[0] % picturesPerRow == 0:
        rowPadding = 0
    else:
        rowPadding = picturesPerRow - imgs.shape[0] % picturesPerRow
    if rowPadding > 0:
        imgs = np.concatenate([imgs, np.zeros((rowPadding, *imgs.shape[1:]), dtype=imgs.dtype)], axis=0)

    # Tiling Loop (The conditionals are not necessary anymore)
    tiled = []
    for i in range(0, imgs.shape[0], picturesPerRow):
        tiled.append(np.concatenate([imgs[j] for j in range(i, i + picturesPerRow)], axis=1))

    tiled = np.concatenate(tiled, axis=0)
    return tiled


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True, tile=False):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if image_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(image_tensor.size(0)):
            one_image = image_tensor[b]
            one_image_np = tensor2im(one_image)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = tile_images(images_np)
            return images_tiled
        else:
            return images_np

    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)
    image_numpy = image_tensor.detach().cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)


# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8, tile=False):
    if label_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(label_tensor.size(0)):
            one_image = label_tensor[b]
            one_image_np = tensor2label(one_image, n_label, imtype)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = tile_images(images_np)
            return images_tiled
        else:
            images_np = images_np[0]
            return images_np

    if label_tensor.dim() == 1:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    result = label_numpy.astype(imtype)
    return result


def save_image(image_numpy, image_path, create_dir=False):
    if create_dir:
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
    if len(image_numpy.shape) == 2:
        image_numpy = np.expand_dims(image_numpy, axis=2)
    if image_numpy.shape[2] == 1:
        image_numpy = np.repeat(image_numpy, 3, 2)
    image_pil = Image.fromarray(image_numpy)

    # save to png
    image_pil.save(image_path.replace('.jpg', '.png'))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]


def natural_sort(items):
    items.sort(key=natural_keys)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def print_network(model):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Network [%s] was created. Total number of parameters: %.1f million. '
            'To see the architecture, do print(network).'
            % (type(model).__name__, num_params / 1000000))

def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    if cls is None:
        print("In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name))
        exit(0)

    return cls


def save_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_path = os.path.join(opt.checkpoints_dir, opt.name, save_filename)
    torch.save(net.cpu().state_dict(), save_path)
    if len(opt.gpu_ids) and torch.cuda.is_available():
        net.cuda()


def load_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    save_path = os.path.join(save_dir, save_filename)
    if not os.path.exists(save_path):
        print('not find model :' + save_path + ', do not load model!')
        return net
    weights = torch.load(save_path)
    try:
        net.load_state_dict(weights)
    except KeyError:
        print('key error, not load!')
    except RuntimeError as err:
        print(err)
        net.load_state_dict(weights, strict=False)
        print('loaded with strict=False')
    return net


###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    if N == 35:  # cityscape
        cmap = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (111, 74, 0), (81, 0, 81),
                         (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70), (102, 102, 156), (190, 153, 153),
                         (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
                         (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                         (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)],
                        dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i + 1  # let's give 0 a color
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b

        if N == 182:  # COCO
            important_colors = {
                'sea': (54, 62, 167),
                'sky-other': (95, 219, 255),
                'tree': (140, 104, 47),
                'clouds': (170, 170, 170),
                'grass': (29, 195, 49)
            }
            for i in range(N):
                name = util.coco.id2label(i)
                if name in important_colors:
                    color = important_colors[name]
                    cmap[i] = np.array(list(color))

    return cmap


class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

def print_current_errors(opt, epoch, i, errors, t):
    message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
    for k, v in errors.items():
        #print(v)
        #if v != 0:
        v = v.mean().float()
        message += '%s: %.3f ' % (k, v)

    print(message)
    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)

# class Distortion_with_flow(object):
#     """Elastic distortion
#     """

#     def __init__(self):
#         return

#     def __call__(self, inputs, dx, dy):
#         inputs = np.array(inputs)
#         shape = inputs.shape[0], inputs.shape[1]
#         inputs = np.array(inputs)
#         # x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
#         # remap_image = cv2.remap(inputs, (dy + y).astype(np.float32), (dx + x).astype(np.float32), interpolation=cv2.INTER_LINEAR)
#         # backward mapping

#         remap_image = forward_mapping(inputs, dy, dx, maxIter=3, precision=1e-3)
#         # remap_image = cv2.remap(inputs, (dy + y).astype(np.float32), (dx + x).astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
#         # remap_image = inputs

#         return Image.fromarray(remap_image)

# def forward_mapping(source_image, u, v, maxIter=5, precision=1e-2):
#     '''
#     warp the image according to the forward flow
#     u: horizontal
#     v: vertical
#     '''
#     H = source_image.shape[0]
#     W = source_image.shape[1]

#     distortImg = np.array(np.zeros((H + 1, W + 1, 3)), dtype=np.uint8)
#     distortImg[0:H, 0:W] = source_image[0:H, 0:W]
#     distortImg[H, 0:W] = source_image[H - 1, 0:W]
#     distortImg[0:H, W] = source_image[0:H, W - 1]
#     distortImg[H, W] = source_image[H - 1, W - 1]

#     padu = np.array(np.zeros((H + 1, W + 1)), dtype=np.float32)
#     padu[0:H, 0:W] = u[0:H, 0:W]
#     padu[H, 0:W] = u[H - 1, 0:W]
#     padu[0:H, W] = u[0:H, W - 1]
#     padu[H, W] = u[H - 1, W - 1]

#     padv = np.array(np.zeros((H + 1, W + 1)), dtype=np.float32)
#     padv[0:H, 0:W] = v[0:H, 0:W]
#     padv[H, 0:W] = v[H - 1, 0:W]
#     padv[0:H, W] = v[0:H, W - 1]
#     padv[H, W] = v[H - 1, W - 1]

#     resultImg = np.array(np.zeros((H, W, 3)), dtype=np.uint8)
#     iterSearch(distortImg, resultImg, padu, padv, W, H, maxIter, precision)
#     return resultImg

# @jit(nopython=True)
# def iterSearchShader(padu, padv, xr, yr, W, H, maxIter, precision):
#     # print('processing location', (xr, yr))
#     #
#     if abs(padu[yr, xr]) < precision and abs(padv[yr, xr]) < precision:
#         return xr, yr

#     else:
#         # Our initialize method in this paper, can see the overleaf for detail
#         if (xr + 1) <= (W - 1):
#             dif = padu[yr, xr + 1] - padu[yr, xr]
#             u_next = padu[yr, xr] / (1 + dif)
#         else:
#             dif = padu[yr, xr] - padu[yr, xr - 1]
#             u_next = padu[yr, xr] / (1 + dif)

#         if (yr + 1) <= (H - 1):
#             dif = padv[yr + 1, xr] - padv[yr, xr]
#             v_next = padv[yr, xr] / (1 + dif)
#         else:
#             dif = padv[yr, xr] - padv[yr - 1, xr]
#             v_next = padv[yr, xr] / (1 + dif)

#         i = xr - u_next
#         j = yr - v_next
#         i_int = int(i)
#         j_int = int(j)

#         # The same as traditional iterative search method
#         for iter in range(maxIter):
#             if 0 <= i <= (W - 1) and 0 <= j <= (H - 1):
#                 u11 = padu[j_int, i_int]
#                 v11 = padv[j_int, i_int]

#                 u12 = padu[j_int, i_int + 1]
#                 v12 = padv[j_int, i_int + 1]

#                 int1 = padu[j_int + 1, i_int]
#                 v21 = padv[j_int + 1, i_int]

#                 int2 = padu[j_int + 1, i_int + 1]
#                 v22 = padv[j_int + 1, i_int + 1]

#                 u = u11 * (i_int + 1 - i) * (j_int + 1 - j) + u12 * (i - i_int) * (j_int + 1 - j) + int1 * (i_int + 1 - i) * (j - j_int) + int2 * (
#                     i - i_int) * (j - j_int)

#                 v = v11 * (i_int + 1 - i) * (j_int + 1 - j) + v12 * (i - i_int) * (j_int + 1 - j) + v21 * (i_int + 1 - i) * (j - j_int) + v22 * (i - i_int) * (
#                     j - j_int)

#                 i_next = xr - u
#                 j_next = yr - v

#                 if abs(i - i_next) < precision and abs(j - j_next) < precision:
#                     return i, j

#                 i = i_next
#                 j = j_next

#             else:
#                 return i, j

#         # if the search doesn't converge within max iter, it will return the last iter result
#         return i_next, j_next

# @jit(nopython=True)
# def biInterpolation(distorted, i, j):
#     i = u2(i)
#     j = u2(j)
#     Q11 = distorted[j, i]
#     Q12 = distorted[j, i + 1]
#     Q21 = distorted[j + 1, i]
#     Q22 = distorted[j + 1, i + 1]

#     pixel = u1(Q11 * (i + 1 - i) * (j + 1 - j) + Q12 * (i - i) * (j + 1 - j) + Q21 * (i + 1 - i) * (j - j) + Q22 * (i - i) * (j - j))
#     return pixel

# @jit(nopython=True)
# def iterSearch(distortImg, resultImg, padu, padv, W, H, maxIter=5, precision=1e-2):
#     for xr in range(W):
#         for yr in range(H):
#             # (xr, yr) is the point in result image, (i, j) is the search result in distorted image
#             i, j = iterSearchShader(padu, padv, xr, yr, W, H, maxIter, precision)

#             # reflect the pixels outside the border
#             if i > W - 1:
#                 i = 2 * W - 1 - i
#             if i < 0:
#                 i = -i
#             if j > H - 1:
#                 j = 2 * H - 1 - j
#             if j < 0:
#                 j = -j

#             # Bilinear interpolation to get the pixel at (i, j) in distorted image
#             resultImg[yr, xr, 0] = biInterpolation(
#                 distortImg[:, :, 0],
#                 i,
#                 j,
#             )
#             resultImg[yr, xr, 1] = biInterpolation(
#                 distortImg[:, :, 1],
#                 i,
#                 j,
#             )
#             resultImg[yr, xr, 2] = biInterpolation(
#                 distortImg[:, :, 2],
#                 i,
#                 j,
#             )
#     return None
