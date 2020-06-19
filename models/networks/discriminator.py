"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer, equal_lr
from models.networks.architecture import Attention
import util.util as util


class MultiscaleDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--netD_subarch', type=str, default='n_layer',
                            help='architecture of each discriminator')
        parser.add_argument('--num_D', type=int, default=2,
                            help='number of discriminators to be used in multiscale')
        opt, _ = parser.parse_known_args()

        # define properties of each discriminator of the multiscale discriminator
        subnetD = util.find_class_in_module(opt.netD_subarch + 'discriminator',
                                            'models.networks.discriminator')
        subnetD.modify_commandline_options(parser, is_train)

        return parser

    def __init__(self, opt, stage1=False):
        super().__init__()
        self.opt = opt
        self.stage1 = stage1

        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt)
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self, opt):
        subarch = opt.netD_subarch
        if subarch == 'n_layer':
            netD = NLayerDiscriminator(opt, stage1=self.stage1)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        segs = []
        cam_logits = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        for name, D in self.named_children():
            out, cam_logit = D(input)
            cam_logits.append(cam_logit)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result, segs, cam_logits


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--n_layers_D', type=int, default=4,
                            help='# layers in each discriminator')
        return parser

    def __init__(self, opt, stage1=False):
        super().__init__()
        self.opt = opt
        self.stage1 = stage1

        kw = 4
        #padw = int(np.ceil((kw - 1.0) / 2))
        padw = int((kw - 1.0) / 2)
        nf = opt.ndf
        input_nc = self.compute_D_input_nc(opt)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            if (((not stage1) and opt.use_attention) or (stage1 and opt.use_attention_st1)) and n == opt.n_layers_D - 1:
                self.attn = Attention(nf_prev, 'spectral' in opt.norm_D)
            if n == opt.n_layers_D - 1 and (not stage1):
                dec = []
                nc_dec = nf_prev
                for _ in range(opt.n_layers_D - 1):
                    dec += [nn.Upsample(scale_factor=2),
                            norm_layer(nn.Conv2d(nc_dec, int(nc_dec//2), kernel_size=3, stride=1, padding=1)),
                            nn.LeakyReLU(0.2, False)]
                    nc_dec = int(nc_dec // 2)
                dec += [nn.Conv2d(nc_dec, opt.semantic_nc, kernel_size=3, stride=1, padding=1)]
                self.dec = nn.Sequential(*dec)
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if opt.D_cam > 0:
            mult = min(2 ** (opt.n_layers_D - 1), 8)
            if opt.eqlr_sn:
                self.gap_fc = equal_lr(nn.Linear(opt.ndf * mult, 1, bias=False))
                self.gmp_fc = equal_lr(nn.Linear(opt.ndf * mult, 1, bias=False))
            else:
                self.gap_fc = nn.utils.spectral_norm(nn.Linear(opt.ndf * mult, 1, bias=False))
                self.gmp_fc = nn.utils.spectral_norm(nn.Linear(opt.ndf * mult, 1, bias=False))
            self.conv1x1 = nn.Conv2d(opt.ndf * mult * 2, opt.ndf * mult, kernel_size=1, stride=1, bias=True)
            self.leaky_relu = nn.LeakyReLU(0.2, True)

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def compute_D_input_nc(self, opt):
        input_nc = opt.label_nc + opt.output_nc
        if opt.contain_dontcare_label:
            input_nc += 1
        return input_nc

    def forward(self, input):
        results = [input]
        seg = None
        cam_logit = None
        for name, submodel in self.named_children():
            if 'model' not in name:
                continue
            if name == 'model3':
                if ((not self.stage1) and self.opt.use_attention) or (self.stage1 and self.opt.use_attention_st1):
                    x = self.attn(results[-1])
                else:
                    x = results[-1]
            else:
                x = results[-1]
            intermediate_output = submodel(x)
            if self.opt.D_cam > 0 and name == 'model3':
                gap = F.adaptive_avg_pool2d(intermediate_output, 1)
                gap_logit = self.gap_fc(gap.view(intermediate_output.shape[0], -1))
                gap_weight = list(self.gap_fc.parameters())[0]
                gap = intermediate_output * gap_weight.unsqueeze(2).unsqueeze(3)

                gmp = F.adaptive_max_pool2d(intermediate_output, 1)
                gmp_logit = self.gmp_fc(gmp.view(intermediate_output.shape[0], -1))
                gmp_weight = list(self.gmp_fc.parameters())[0]
                gmp = intermediate_output * gmp_weight.unsqueeze(2).unsqueeze(3)

                cam_logit = torch.cat([gap_logit, gmp_logit], 1)
                intermediate_output = torch.cat([gap, gmp], 1)
                intermediate_output = self.leaky_relu(self.conv1x1(intermediate_output))
            results.append(intermediate_output)

        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            retu = results[1:]
        else:
            retu = results[-1]
        if seg is None:
            return retu, cam_logit
        else:
            return retu, seg, cam_logit
