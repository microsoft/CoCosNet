# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import cv2
from PIL import Image
import numpy as np
from skimage import feature
# parts = ['skin', 'hair', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'nose', 'u_lip', 'mouth', 'l_lip', 'neck', 
#             'cloth', 'hat', 'eye_g', 'ear_r', 'neck_l']
inner_parts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'nose', 'u_lip', 'mouth', 'l_lip', 'eye_g', 'hair']
root = 'C:/Data/CelebAMask-HQ'

def get_edges(edge, t):
    edge[:,1:] = edge[:,1:] | (t[:,1:] != t[:,:-1])
    edge[:,:-1] = edge[:,:-1] | (t[:,1:] != t[:,:-1])
    edge[1:,:] = edge[1:,:] | (t[1:,:] != t[:-1,:])
    edge[:-1,:] = edge[:-1,:] | (t[1:,:] != t[:-1,:])
    return edge

for i in range(30000):
    img = Image.open(os.path.join(root, 'CelebA-HQ-img', str(i) + '.jpg')).resize((512, 512), resample=Image.BILINEAR)
    inner_label = np.ones(img.size, dtype=np.uint8)
    edges = np.zeros(img.size, dtype=np.uint8)
    subfolder = str(i // 2000)
    for part in inner_parts:
        edge = np.zeros(img.size, dtype=np.uint8)  #this for distance transform map on each facial part
        path = os.path.join(root, 'CelebAMask-HQ-mask-anno', subfolder, str(i).zfill(5) + '_' + part + '.png')
        if os.path.exists(path):
            part_label = Image.open(path).convert('L')
            part_label = np.array(part_label)
            if part == 'hair':
                inner_label[part_label == 255] = 1
            else:
                inner_label[part_label == 255] = 0
            edges = get_edges(edges, part_label)
            edge = get_edges(edge, part_label)
        im_dist = cv2.distanceTransform(255-edge*255, cv2.DIST_L1, 3)
        im_dist = np.clip((im_dist / 3), 0, 255).astype(np.uint8)
        #Image.fromarray(im_dist).save(os.path.join(root, 'CelebAMask-HQ-mask-anno', 'parsing_edges', str(i).zfill(5) + '_{}.png'.format(part)))

    # canny edge for background
    canny_edges = feature.canny(np.array(img.convert('L')))
    canny_edges = canny_edges * inner_label

    edges += canny_edges
    Image.fromarray(edges * 255).save(os.path.join(root, 'CelebAMask-HQ-mask-anno', 'parsing_edges', str(i).zfill(5) + '.png'))