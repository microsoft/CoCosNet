"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # for displays
        parser.add_argument('--display_freq', type=int, default=2000, help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')

        # for training
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
        parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--D_steps_per_G', type=int, default=1, help='number of discriminator iterations per generator iterations.')

        # for discriminators
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
        parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
        parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image)')
        parser.add_argument('--no_TTUR', action='store_true', help='Use TTUR training scheme')

        parser.add_argument('--which_perceptual', type=str, default='5_2', help='relu5_2 or relu4_2')
        parser.add_argument('--weight_perceptual', type=float, default=0.01)
        parser.add_argument('--weight_mask', type=float, default=0.0, help='weight of warped mask loss, used in direct/cycle')
        parser.add_argument('--real_reference_probability', type=float, default=0.7, help='self-supervised training probability')
        parser.add_argument('--hard_reference_probability', type=float, default=0.2, help='hard reference training probability')
        parser.add_argument('--weight_gan', type=float, default=10.0, help='weight of all loss in stage1')
        parser.add_argument('--novgg_featpair', type=float, default=10.0, help='in no vgg setting, use pair feat loss in domain adaptation')
        parser.add_argument('--D_cam', type=float, default=0.0, help='weight of CAM loss in D')
        parser.add_argument('--warp_self_w', type=float, default=0.0, help='push warp self to ref')
        parser.add_argument('--fm_ratio', type=float, default=0.1, help='vgg fm loss weight comp with ctx loss')
        parser.add_argument('--use_22ctx', action='store_true', help='if true, also use 2-2 in ctx loss')
        parser.add_argument('--ctx_w', type=float, default=1.0, help='ctx loss weight')
        parser.add_argument('--mask_epoch', type=int, default=-1, help='useful when noise_for_mask is true, first train mask_epoch with mask, the rest epoch with noise')

        self.isTrain = True
        return parser
