# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import copy
import sys
import torch
from models.networks.sync_batchnorm import DataParallelWithCallback
from models.pix2pix_model import Pix2PixModel
from models.networks.generator import EMA
import util.util as util

class Pix2PixTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt, resume_epoch=0):
        self.opt = opt
        self.pix2pix_model = Pix2PixModel(opt)
        if len(opt.gpu_ids) > 1:
            self.pix2pix_model = DataParallelWithCallback(self.pix2pix_model,
                                                          device_ids=opt.gpu_ids)
            self.pix2pix_model_on_one_gpu = self.pix2pix_model.module
        else:
            self.pix2pix_model.to(opt.gpu_ids[0])
            self.pix2pix_model_on_one_gpu = self.pix2pix_model

        if opt.use_ema:
            self.netG_ema = EMA(opt.ema_beta)
            for name, param in self.pix2pix_model_on_one_gpu.net['netG'].named_parameters():
                if param.requires_grad:
                    self.netG_ema.register(name, param.data)
            self.netCorr_ema = EMA(opt.ema_beta)
            for name, param in self.pix2pix_model_on_one_gpu.net['netCorr'].named_parameters():
                if param.requires_grad:
                    self.netCorr_ema.register(name, param.data)

        self.generated = None
        if opt.isTrain:
            self.optimizer_G, self.optimizer_D = \
                self.pix2pix_model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr
            if opt.continue_train and opt.which_epoch == 'latest':
                checkpoint = torch.load(os.path.join(opt.checkpoints_dir, opt.name, 'optimizer.pth'))
                self.optimizer_G.load_state_dict(checkpoint['G'])
                self.optimizer_D.load_state_dict(checkpoint['D'])
        self.last_data, self.last_netCorr, self.last_netG, self.last_optimizer_G = None, None, None, None

    def run_generator_one_step(self, data, alpha=1):
        self.optimizer_G.zero_grad()
        g_losses, out = self.pix2pix_model(data, mode='generator', alpha=alpha)
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()
        self.g_losses = g_losses
        self.out = out
        if self.opt.use_ema:
            self.netG_ema(self.pix2pix_model_on_one_gpu.net['netG'])
            self.netCorr_ema(self.pix2pix_model_on_one_gpu.net['netCorr'])

    def run_discriminator_one_step(self, data):
        self.optimizer_D.zero_grad()
        GforD = {}
        GforD['fake_image'] = self.out['fake_image']
        GforD['adaptive_feature_seg'] = self.out['adaptive_feature_seg']
        GforD['adaptive_feature_img'] = self.out['adaptive_feature_img']
        d_losses = self.pix2pix_model(data, mode='discriminator', GforD=GforD)
        d_loss = sum(d_losses.values()).mean()
        d_loss.backward()
        self.optimizer_D.step()
        self.d_losses = d_losses

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.out['fake_image']

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.pix2pix_model_on_one_gpu.save(epoch)
        if self.opt.use_ema:
            self.netG_ema.assign(self.pix2pix_model_on_one_gpu.net['netG'])
            util.save_network(self.pix2pix_model_on_one_gpu.net['netG'], 'G_ema', epoch, self.opt)
            self.netG_ema.resume(self.pix2pix_model_on_one_gpu.net['netG'])

            self.netCorr_ema.assign(self.pix2pix_model_on_one_gpu.net['netCorr'])
            util.save_network(self.pix2pix_model_on_one_gpu.net['netCorr'], 'netCorr_ema', epoch, self.opt)
            self.netCorr_ema.resume(self.pix2pix_model_on_one_gpu.net['netCorr'])
        if epoch == 'latest':
            torch.save({'G': self.optimizer_G.state_dict(),
                        'D': self.optimizer_D.state_dict(),
                        'lr':  self.old_lr,
                        }, os.path.join(self.opt.checkpoints_dir, self.opt.name, 'optimizer.pth'))

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr

    def update_fixed_params(self):
        for param in self.pix2pix_model_on_one_gpu.net['netCorr'].parameters():
            param.requires_grad = True
        G_params = [{'params': self.pix2pix_model_on_one_gpu.net['netG'].parameters(), 'lr': self.opt.lr*0.5}]
        G_params += [{'params': self.pix2pix_model_on_one_gpu.net['netCorr'].parameters(), 'lr': self.opt.lr*0.5}]
        if self.opt.no_TTUR:
            beta1, beta2 = self.opt.beta1, self.opt.beta2
            G_lr = self.opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr = self.opt.lr / 2

        self.optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2), eps=1e-3)