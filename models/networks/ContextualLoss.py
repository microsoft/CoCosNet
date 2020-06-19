# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
from collections import OrderedDict, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from util.util import feature_normalize, mse_loss
import matplotlib.pyplot as plt
import torchvision
import numpy as np

postpa = torchvision.transforms.Compose([
    torchvision.transforms.Lambda(lambda x: x.mul_(1. / 255)),
    torchvision.transforms.Normalize(
        mean=[-0.40760392, -0.45795686, -0.48501961],  #add imagenet mean
        std=[1, 1, 1]),
    torchvision.transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  #turn to RGB
])
postpb = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()])


def post_processing(tensor):
    t = postpa(tensor)  # denormalize the image since the optimized tensor is the normalized one
    t[t > 1] = 1
    t[t < 0] = 0
    img = postpb(t)
    img = np.array(img)
    return img


class ContextualLoss(nn.Module):
    '''
        input is Al, Bl, channel = 1, range ~ [0, 255]
    '''

    def __init__(self):
        super(ContextualLoss, self).__init__()
        return None

    def forward(self, X_features, Y_features, h=0.1, feature_centering=True):
        '''
        X_features&Y_features are are feature vectors or feature 2d array
        h: bandwidth
        return the per-sample loss
        '''
        batch_size = X_features.shape[0]
        feature_depth = X_features.shape[1]
        feature_size = X_features.shape[2]

        # center the feature vector???

        # to normalized feature vectors
        if feature_centering:
            X_features = X_features - Y_features.view(batch_size, feature_depth, -1).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
            Y_features = Y_features - Y_features.view(batch_size, feature_depth, -1).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        X_features = feature_normalize(X_features).view(batch_size, feature_depth, -1)  # batch_size * feature_depth * feature_size^2
        Y_features = feature_normalize(Y_features).view(batch_size, feature_depth, -1)  # batch_size * feature_depth * feature_size^2

        # conine distance = 1 - similarity
        X_features_permute = X_features.permute(0, 2, 1)  # batch_size * feature_size^2 * feature_depth
        d = 1 - torch.matmul(X_features_permute, Y_features)  # batch_size * feature_size^2 * feature_size^2

        # normalized distance: dij_bar
        d_norm = d / (torch.min(d, dim=-1, keepdim=True)[0] + 1e-5)  # batch_size * feature_size^2 * feature_size^2

        # pairwise affinity
        w = torch.exp((1 - d_norm) / h)
        A_ij = w / torch.sum(w, dim=-1, keepdim=True)

        # contextual loss per sample
        CX = torch.mean(torch.max(A_ij, dim=1)[0], dim=-1)
        loss = -torch.log(CX)

        # contextual loss per batch
        # loss = torch.mean(loss)
        return loss


class ContextualLoss_forward(nn.Module):
    '''
        input is Al, Bl, channel = 1, range ~ [0, 255]
    '''

    def __init__(self, opt):
        super(ContextualLoss_forward, self).__init__()
        self.opt = opt
        return None

    def forward(self, X_features, Y_features, h=0.1, feature_centering=True):
        '''
        X_features&Y_features are are feature vectors or feature 2d array
        h: bandwidth
        return the per-sample loss
        '''
        batch_size = X_features.shape[0]
        feature_depth = X_features.shape[1]
        feature_size = X_features.shape[2]

        # to normalized feature vectors
        if feature_centering:
            if self.opt.PONO:
                X_features = X_features - Y_features.mean(dim=1).unsqueeze(dim=1)
                Y_features = Y_features - Y_features.mean(dim=1).unsqueeze(dim=1)
            else:
                X_features = X_features - Y_features.view(batch_size, feature_depth, -1).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
                Y_features = Y_features - Y_features.view(batch_size, feature_depth, -1).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        X_features = feature_normalize(X_features).view(batch_size, feature_depth, -1)  # batch_size * feature_depth * feature_size * feature_size
        Y_features = feature_normalize(Y_features).view(batch_size, feature_depth, -1)  # batch_size * feature_depth * feature_size * feature_size

        # X_features = F.unfold(
        #     X_features, kernel_size=self.opt.match_kernel, stride=1, padding=int(self.opt.match_kernel // 2))  # batch_size * feature_depth_new * feature_size^2
        # Y_features = F.unfold(
        #     Y_features, kernel_size=self.opt.match_kernel, stride=1, padding=int(self.opt.match_kernel // 2))  # batch_size * feature_depth_new * feature_size^2

        # conine distance = 1 - similarity
        X_features_permute = X_features.permute(0, 2, 1)  # batch_size * feature_size^2 * feature_depth
        d = 1 - torch.matmul(X_features_permute, Y_features)  # batch_size * feature_size^2 * feature_size^2

        # normalized distance: dij_bar
        # d_norm = d
        d_norm = d / (torch.min(d, dim=-1, keepdim=True)[0] + 1e-3)  # batch_size * feature_size^2 * feature_size^2

        # pairwise affinity
        w = torch.exp((1 - d_norm) / h)
        A_ij = w / torch.sum(w, dim=-1, keepdim=True)

        # contextual loss per sample
        CX = torch.mean(torch.max(A_ij, dim=-1)[0], dim=1)
        loss = -torch.log(CX)

        # contextual loss per batch
        # loss = torch.mean(loss)
        return loss


class ContextualLoss_complex(nn.Module):
    '''
        input is Al, Bl, channel = 1, range ~ [0, 255]
    '''

    def __init__(self):
        super(ContextualLoss_complex, self).__init__()
        return None

    def forward(self, X_features, Y_features, h=0.1, patch_size=1, direction='forward'):
        '''
        X_features&Y_features are are feature vectors or feature 2d array
        h: bandwidth
        return the per-sample loss
        '''
        batch_size = X_features.shape[0]
        feature_depth = X_features.shape[1]
        feature_size = X_features.shape[2]

        # to normalized feature vectors
        # TODO: center by the mean of Y_features
        X_features = X_features - Y_features.view(batch_size, feature_depth, -1).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        Y_features = Y_features - Y_features.view(batch_size, feature_depth, -1).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        X_features = feature_normalize(X_features)  # batch_size * feature_depth * feature_size^2
        Y_features = feature_normalize(Y_features)  # batch_size * feature_depth * feature_size^2

        # to normalized feature vectors
        X_features = F.unfold(
            X_features, kernel_size=(patch_size, patch_size), stride=(1, 1), padding=(patch_size // 2,
                                                                                      patch_size // 2))  # batch_size * feature_depth_new * feature_size^2
        Y_features = F.unfold(
            Y_features, kernel_size=(patch_size, patch_size), stride=(1, 1), padding=(patch_size // 2,
                                                                                      patch_size // 2))  # batch_size * feature_depth_new * feature_size^2

        # conine distance = 1 - similarity
        X_features_permute = X_features.permute(0, 2, 1)  # batch_size * feature_size^2 * feature_depth
        d = 1 - torch.matmul(X_features_permute, Y_features)  # batch_size * feature_size^2 * feature_size^2

        # normalized distance: dij_bar
        d_norm = d / (torch.min(d, dim=-1, keepdim=True)[0] + 1e-5)  # batch_size * feature_size^2 * feature_size^2

        # pairwise affinity
        w = torch.exp((1 - d_norm) / h)
        A_ij = w / torch.sum(w, dim=-1, keepdim=True)

        # contextual loss per sample
        if direction == 'forward':
            CX = torch.mean(torch.max(A_ij, dim=-1)[0], dim=1)
        else:
            CX = torch.mean(torch.max(A_ij, dim=1)[0], dim=-1)

        loss = -torch.log(CX)
        return loss


class ChamferDistance_patch_loss(nn.Module):
    '''
        input is Al, Bl, channel = 1, range ~ [0, 255]
    '''

    def __init__(self):
        super(ChamferDistance_patch_loss, self).__init__()
        return None

    def forward(self, X_features, Y_features, patch_size=3, image_x=None, image_y=None, h=0.1, Y_features_in=None):
        '''
        X_features&Y_features are are feature vectors or feature 2d array
        h: bandwidth
        return the per-sample loss
        '''
        batch_size = X_features.shape[0]
        feature_depth = X_features.shape[1]
        feature_size = X_features.shape[2]

        # to normalized feature vectors
        X_features = F.unfold(
            X_features, kernel_size=(patch_size, patch_size), stride=(1, 1), padding=(patch_size // 2,
                                                                                      patch_size // 2))  # batch_size, feature_depth_new * feature_size^2
        Y_features = F.unfold(
            Y_features, kernel_size=(patch_size, patch_size), stride=(1, 1), padding=(patch_size // 2,
                                                                                      patch_size // 2))  # batch_size, feature_depth_new * feature_size^2

        if image_x is not None and image_y is not None:
            image_x = torch.nn.functional.interpolate(image_x, size=(feature_size, feature_size), mode='bilinear').view(batch_size, 3, -1)
            image_y = torch.nn.functional.interpolate(image_y, size=(feature_size, feature_size), mode='bilinear').view(batch_size, 3, -1)

        X_features_permute = X_features.permute(0, 2, 1)  # batch_size * feature_size^2 * feature_depth
        similarity_matrix = torch.matmul(X_features_permute, Y_features)  # batch_size * feature_size^2 * feature_size^2
        NN_index = similarity_matrix.max(dim=-1, keepdim=True)[1].squeeze()

        if Y_features_in is not None:
            loss = torch.mean((X_features - Y_features_in.detach())**2)
            Y_features_in = Y_features_in.detach()
        else:
            loss = torch.mean((X_features - Y_features[:, :, NN_index].detach())**2)
            Y_features_in = Y_features[:, :, NN_index].detach()

        # re-arrange image
        if image_x is not None and image_y is not None:
            image_y_rearrange = image_y[:, :, NN_index]
            image_y_rearrange = image_y_rearrange.view(batch_size, 3, feature_size, feature_size)
            image_x = image_x.view(batch_size, 3, feature_size, feature_size)
            image_y = image_y.view(batch_size, 3, feature_size, feature_size)
        # plt.figure()
        # plt.imshow((post_processing(image_x[0].detach().cpu())))
        # plt.title('image x')
        # plt.figure()
        # plt.imshow((image_y[0]).permute(1, 2, 0).cpu().numpy())
        # plt.title('image y')
        # plt.figure()
        # plt.imshow((image_y_rearrange[0]).permute(1, 2, 0).cpu().numpy())
        # plt.title('corresponded image y')
        # plt.show()

        return loss


class ChamferDistance_loss(nn.Module):
    '''
        input is Al, Bl, channel = 1, range ~ [0, 255]
    '''

    def __init__(self):
        super(ChamferDistance_loss, self).__init__()
        return None

    def forward(self, X_features, Y_features, image_x, image_y, h=0.1, Y_features_in=None):
        '''
        X_features&Y_features are are feature vectors or feature 2d array
        h: bandwidth
        return the per-sample loss
        '''
        batch_size = X_features.shape[0]
        feature_depth = X_features.shape[1]
        feature_size = X_features.shape[2]

        # to normalized feature vectors
        X_features = feature_normalize(X_features).view(batch_size, feature_depth, -1)  # batch_size * feature_depth * feature_size^2
        Y_features = feature_normalize(Y_features).view(batch_size, feature_depth, -1)  # batch_size * feature_depth * feature_size^2
        image_x = torch.nn.functional.interpolate(image_x, size=(feature_size, feature_size), mode='bilinear').view(batch_size, 3, -1)
        image_y = torch.nn.functional.interpolate(image_y, size=(feature_size, feature_size), mode='bilinear').view(batch_size, 3, -1)

        X_features_permute = X_features.permute(0, 2, 1)  # batch_size * feature_size^2 * feature_depth
        similarity_matrix = torch.matmul(X_features_permute, Y_features)  # batch_size * feature_size^2 * feature_size^2
        NN_index = similarity_matrix.max(dim=-1, keepdim=True)[1].squeeze()
        if Y_features_in is not None:
            loss = torch.mean((X_features - Y_features_in.detach())**2)
            Y_features_in = Y_features_in.detach()
        else:
            loss = torch.mean((X_features - Y_features[:, :, NN_index].detach())**2)
            Y_features_in = Y_features[:, :, NN_index].detach()

        # re-arrange image
        image_y_rearrange = image_y[:, :, NN_index]
        image_y_rearrange = image_y_rearrange.view(batch_size, 3, feature_size, feature_size)
        image_x = image_x.view(batch_size, 3, feature_size, feature_size)
        image_y = image_y.view(batch_size, 3, feature_size, feature_size)

        # plt.figure()
        # plt.imshow((post_processing(image_x[0].detach().cpu())))
        # plt.title('image x')
        # plt.figure()
        # plt.imshow((image_y[0]).permute(1, 2, 0).cpu().numpy())
        # plt.title('image y')
        # plt.figure()
        # plt.imshow((image_y_rearrange[0]).permute(1, 2, 0).cpu().numpy())
        # plt.title('corresponded image y')
        # plt.show()

        return loss, Y_features_in, X_features


# class ChamferDistance_loss(nn.Module):
#     '''
#         input is Al, Bl, channel = 1, range ~ [0, 255]
#     '''

#     def __init__(self):
#         super(ChamferDistance_loss, self).__init__()
#         return None

#     def forward(self, X_features, Y_features, image_x, image_y):
#         '''
#         X_features&Y_features are are feature vectors or feature 2d array
#         h: bandwidth
#         return the per-sample loss
#         '''
#         batch_size = X_features.shape[0]
#         feature_depth = X_features.shape[1]
#         feature_size = X_features.shape[2]

#         # to normalized feature vectors
#         X_features = feature_normalize(X_features).view(batch_size, feature_depth, -1)  # batch_size * feature_depth * feature_size^2
#         Y_features = feature_normalize(Y_features).view(batch_size, feature_depth, -1)  # batch_size * feature_depth * feature_size^2
#         image_x = torch.nn.functional.interpolate(image_x, size=(feature_size, feature_size), mode='bilinear').view(batch_size, 3, -1)
#         image_y = torch.nn.functional.interpolate(image_y, size=(feature_size, feature_size), mode='bilinear').view(batch_size, 3, -1)

#         X_features_permute = X_features.permute(0, 2, 1)  # batch_size * feature_size^2 * feature_depth
#         similarity_matrix = torch.matmul(X_features_permute, Y_features)  # batch_size * feature_size^2 * feature_size^2
#         NN_index = similarity_matrix.max(dim=-1, keepdim=True)[1].squeeze()
#         loss = torch.mean((X_features - Y_features[:, :, NN_index].detach())**2)

#         # re-arrange image
#         image_y_rearrange = image_y[:, :, NN_index]
#         image_y_rearrange = image_y_rearrange.view(batch_size, 3, feature_size, feature_size)
#         image_x = image_x.view(batch_size, 3, feature_size, feature_size)
#         image_y = image_y.view(batch_size, 3, feature_size, feature_size)

#         # plt.figure()
#         # plt.imshow((post_processing(image_x[0].detach().cpu())))
#         # plt.title('image x')
#         # plt.figure()
#         # plt.imshow((image_y[0]).permute(1, 2, 0).cpu().numpy())
#         # plt.title('image y')
#         # plt.figure()
#         # plt.imshow((image_y_rearrange[0]).permute(1, 2, 0).cpu().numpy())
#         # plt.title('corresponded image y')
#         # plt.show()

#         return loss

if __name__ == "__main__":
    contextual_loss = ContextualLoss()
    batch_size = 32
    feature_depth = 8
    feature_size = 16
    X_features = torch.zeros(batch_size, feature_depth, feature_size, feature_size)
    Y_features = torch.zeros(batch_size, feature_depth, feature_size, feature_size)

    cx_loss = contextual_loss(X_features, Y_features, 1)
    print(cx_loss)