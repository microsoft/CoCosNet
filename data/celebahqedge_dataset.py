# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import cv2
import torch
import numpy as np
from PIL import Image
from skimage import feature
from data.pix2pix_dataset import Pix2pixDataset
from data.base_dataset import get_params, get_transform

class CelebAHQEdgeDataset(Pix2pixDataset):
    #hair, skin, l_brow, r_blow, l_eye, r_eye, l_ear, r_ear, nose, u_lip, mouth, l_lip, neck, 
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        parser.set_defaults(no_pairing_check=True)
        if is_train:
            parser.set_defaults(load_size=286)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=15)
        parser.set_defaults(contain_dontcare_label=False)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)
        return parser

    def get_paths(self, opt):
        if opt.phase == 'train':
            fd = open(os.path.join(opt.dataroot, 'train.txt'))
            lines = fd.readlines()
            fd.close()
        elif opt.phase == 'test':
            fd = open(os.path.join(opt.dataroot, 'val.txt'))
            lines = fd.readlines()
            fd.close()
        
        image_paths = []
        label_paths = []
        for i in range(len(lines)):
            image_paths.append(os.path.join(opt.dataroot, 'CelebA-HQ-img', lines[i].strip() + '.jpg'))
            subfolder = str(int(lines[i].strip()) // 2000)
            label_paths.append(os.path.join(opt.dataroot, 'CelebAMask-HQ-mask-anno', subfolder, lines[i].strip().zfill(5) + '_{}.png'))

        return label_paths, image_paths

    def get_ref(self, opt):
        extra = ''
        if opt.phase == 'test':
            extra = '_test'
        with open('./data/celebahq_ref{}.txt'.format(extra)) as fd:
            lines = fd.readlines()
        ref_dict = {}
        for i in range(len(lines)):
            items = lines[i].strip().split(',')
            key = items[0]
            if opt.phase == 'test':
                val = items[1:]
            else:
                val = [items[1], items[-1]]
            ref_dict[key] = val
        train_test_folder = ('', '')
        return ref_dict, train_test_folder

    def get_edges(self, edge, t):
        edge[:,1:] = edge[:,1:] | (t[:,1:] != t[:,:-1])
        edge[:,:-1] = edge[:,:-1] | (t[:,1:] != t[:,:-1])
        edge[1:,:] = edge[1:,:] | (t[1:,:] != t[:-1,:])
        edge[:-1,:] = edge[:-1,:] | (t[1:,:] != t[:-1,:])
        return edge

    def get_label_tensor(self, path):
        inner_parts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'nose', 'u_lip', 'mouth', 'l_lip', 'eye_g', 'hair']
        img_path = self.labelpath_to_imgpath(path)
        img = Image.open(img_path).resize((self.opt.load_size, self.opt.load_size), resample=Image.BILINEAR)
        params = get_params(self.opt, img.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        transform_img = get_transform(self.opt, params, method=Image.BILINEAR, normalize=False)

        inner_label = np.ones(img.size, dtype=np.uint8)
        edges = np.zeros(img.size, dtype=np.uint8)
        tensors_dist = 0
        e = 1
        for part in inner_parts:
            edge = np.zeros(img.size, dtype=np.uint8)  #this for distance transform map on each facial part
            if os.path.exists(path.format(part)):
                part_label = Image.open(path.format(part)).convert('L').resize((self.opt.load_size, self.opt.load_size), resample=Image.NEAREST)
                part_label = np.array(part_label)
                if part == 'hair':
                    inner_label[part_label == 255] = 1
                else:
                    inner_label[part_label == 255] = 0
                edges = self.get_edges(edges, part_label)
                edge = self.get_edges(edge, part_label)
            im_dist = cv2.distanceTransform(255-edge*255, cv2.DIST_L1, 3)
            im_dist = np.clip((im_dist / 3), 0, 255).astype(np.uint8)
            tensor_dist = transform_img(Image.fromarray(im_dist))
            tensors_dist = tensor_dist if e == 1 else torch.cat([tensors_dist, tensor_dist])
            e += 1

        # canny edge for background
        canny_edges = feature.canny(np.array(img.convert('L')))
        canny_edges = canny_edges * inner_label

        edges_all = edges + canny_edges
        edges_all[edges_all > 1] = 1
        tensor_edges_all = transform_label(Image.fromarray(edges_all * 255))
        edges[edges > 1] = 1
        tensor_edges = transform_label(Image.fromarray(edges * 255))

        label_tensor = torch.cat((tensor_edges_all, tensors_dist, tensor_edges), dim=0)
        return label_tensor, params

    def imgpath_to_labelpath(self, path):
        root, name = path.split('CelebA-HQ-img/')
        subfolder = str(int(name.split('.')[0]) // 2000)
        label_path = os.path.join(root, 'CelebAMask-HQ-mask-anno', subfolder, name.split('.')[0].zfill(5) + '_{}.png')
        return label_path

    def labelpath_to_imgpath(self, path):
        root= path.replace('\\', '/').split('CelebAMask-HQ-mask-anno/')[0]
        name = os.path.basename(path).split('_')[0]
        img_path = os.path.join(root, 'CelebA-HQ-img', str(int(name)) + '.jpg')
        return img_path

    # In ADE20k, 'unknown' label is of value 0.
    # Change the 'unknown' label to the last label to match other datasets.
    # def postprocess(self, input_dict):
    #     label = input_dict['label']
    #     label = label - 1
    #     label[label == -1] = self.opt.label_nc
    #     input_dict['label'] = label
    #     if input_dict['label_ref'] is not None:
    #         label_ref = input_dict['label_ref']
    #         label_ref = label_ref - 1
    #         label_ref[label_ref == -1] = self.opt.label_nc
    #         input_dict['label_ref'] = label_ref
