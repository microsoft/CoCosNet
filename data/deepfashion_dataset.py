# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import cv2
import torch
import numpy as np
import math
import random
from PIL import Image
from skimage import feature
from data.pix2pix_dataset import Pix2pixDataset
from data.base_dataset import get_params, get_transform

class DeepFashionDataset(Pix2pixDataset):
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
        parser.set_defaults(label_nc=20)
        parser.set_defaults(contain_dontcare_label=False)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)
        return parser

    def get_paths(self, opt):
        #root = os.path.dirname(opt.dataroot) if opt.hdfs else opt.dataroot
        root = opt.dataroot
        if opt.phase == 'train':
            fd = open(os.path.join(root, 'train.txt'))
            lines = fd.readlines()
            fd.close()
        elif opt.phase == 'test':
            fd = open(os.path.join(root, 'val.txt'))
            lines = fd.readlines()
            fd.close()
        
        image_paths = []
        label_paths = []
        for i in range(len(lines)):
            image_paths.append(os.path.join(opt.dataroot, lines[i].strip().replace('\\', '/')))
            label_path = lines[i].strip().replace('img', 'pose').replace('.jpg', '_{}.txt').replace('\\', '/')
            label_paths.append(os.path.join(opt.dataroot, label_path))

        return label_paths, image_paths

    def get_ref_video_like(self, opt):
        pair_path = './data/deepfashion_self_pair.txt'
        with open(pair_path) as fd:
            self_pair = fd.readlines()
            self_pair = [it.strip() for it in self_pair]
        key_name = {}
        for it in self_pair:
            items = it.split(',')
            key_name[items[0]] = items[1:]
        ref_name = './data/deepfashion_ref_test.txt' if opt.phase == 'test' else './data/deepfashion_ref.txt'
        with open(ref_name) as fd:
            ref = fd.readlines()
            ref = [it.strip() for it in ref]
        ref_dict = {}
        #split = 'DeepFashion.zip@/' if opt.hdfs else 'DeepFashion/'
        split = 'DeepFashion/'
        for i in range(len(ref)):
            items = ref[i].strip().split(',')
            if items[0] in key_name.keys():
                #ref_dict[items[0].replace('\\', '/')] = [random.choice(key_name[items[0]]).replace('\\', '/'), random.choice(self.image_paths).split(split)[-1]]
                ref_dict[items[0].replace('\\', '/')] = [it.replace('\\', '/') for it in key_name[items[0]]] + [it.split(split)[-1] for it in random.sample(self.image_paths, min(len(self.image_paths), 20))]
            else:
                ref_dict[items[0].replace('\\', '/')] = [items[-1].replace('\\', '/')] + [it.split(split)[-1] for it in random.sample(self.image_paths, min(len(self.image_paths), 20))]
        train_test_folder = ('', '')
        return ref_dict, train_test_folder

    def get_ref_vgg(self, opt):
        extra = ''
        if opt.phase == 'test':
            extra = '_test'
        with open('./data/deepfashion_ref{}.txt'.format(extra)) as fd:
            lines = fd.readlines()
        ref_dict = {}
        for i in range(len(lines)):
            items = lines[i].strip().split(',')
            key = items[0].replace('\\', '/')
            if opt.phase == 'test':
                val = [it.replace('\\', '/') for it in items[1:]]
            else:
                val = [items[-1].replace('\\', '/'), random.choice(self.image_paths).split('DeepFashion/')[-1].replace('\\', '/')]
            ref_dict[key] = val
        train_test_folder = ('', '')
        return ref_dict, train_test_folder

    def get_ref(self, opt):
        if opt.video_like:
            return self.get_ref_video_like(opt)
        else:
            return self.get_ref_vgg(opt)

    def get_edges(self, edge, t):
        edge[:,1:] = edge[:,1:] | (t[:,1:] != t[:,:-1])
        edge[:,:-1] = edge[:,:-1] | (t[:,1:] != t[:,:-1])
        edge[1:,:] = edge[1:,:] | (t[1:,:] != t[:-1,:])
        edge[:-1,:] = edge[:-1,:] | (t[1:,:] != t[:-1,:])
        return edge

    def get_label_tensor(self, path):
        candidate = np.loadtxt(path.format('candidate'))
        subset = np.loadtxt(path.format('subset'))
        stickwidth = 4
        limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                [1, 16], [16, 18], [3, 17], [6, 18]]

        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
        
        img_path = self.labelpath_to_imgpath(path)
        img = cv2.imread(img_path)
        canvas = np.zeros_like(img)
        for i in range(18):
            index = int(subset[i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
        joints = []
        for i in range(17):
            index = subset[np.array(limbSeq[i]) - 1]
            cur_canvas = canvas.copy()
            if -1 in index:
                joints.append(np.zeros_like(cur_canvas[:, :, 0]))
                continue
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

            joint = np.zeros_like(cur_canvas[:, :, 0])
            cv2.fillConvexPoly(joint, polygon, 255)
            joint = cv2.addWeighted(joint, 0.4, joint, 0.6, 0)
            joints.append(joint)

        pose = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).resize((self.opt.load_size, self.opt.load_size), resample=Image.NEAREST)
        params = get_params(self.opt, pose.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        transform_img = get_transform(self.opt, params, method=Image.BILINEAR, normalize=False)
        tensors_dist = 0
        e = 1
        for i in range(len(joints)):
            im_dist = cv2.distanceTransform(255-joints[i], cv2.DIST_L1, 3)
            im_dist = np.clip((im_dist / 3), 0, 255).astype(np.uint8)
            tensor_dist = transform_img(Image.fromarray(im_dist))
            tensors_dist = tensor_dist if e == 1 else torch.cat([tensors_dist, tensor_dist])
            e += 1

        tensor_pose = transform_label(pose)
        label_tensor = torch.cat((tensor_pose, tensors_dist), dim=0)
        return label_tensor, params

    def imgpath_to_labelpath(self, path):
        label_path = path.replace('\\', '/').replace('/img/', '/pose/').replace('.jpg', '_{}.txt')
        return label_path

    def labelpath_to_imgpath(self, path):
        img_path = path.replace('\\', '/').replace('/pose/', '/img/').replace('_{}.txt', '.jpg')
        return img_path
