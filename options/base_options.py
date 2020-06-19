"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
import argparse
import os
from util import util
import torch
import models
import data
import pickle


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # experiment specifics
        parser.add_argument('--name', type=str, default='label2coco', help='name of the experiment. It decides where to store samples and models')

        parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--model', type=str, default='pix2pix', help='which model to use')
        parser.add_argument('--norm_G', type=str, default='spectralinstance', help='instance normalization or batch normalization')
        parser.add_argument('--norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization')
        parser.add_argument('--norm_E', type=str, default='spectralinstance', help='instance normalization or batch normalization')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        # input/output sizes
        parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
        parser.add_argument('--preprocess_mode', type=str, default='scale_width_and_crop', help='scaling and cropping of images at load time.', choices=("resize_and_crop", "crop", "scale_width", "scale_width_and_crop", "scale_shortside", "scale_shortside_and_crop", "fixed", "none"))
        parser.add_argument('--load_size', type=int, default=256, help='Scale images to this size. The final image will be cropped to --crop_size.')
        parser.add_argument('--crop_size', type=int, default=256, help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')
        parser.add_argument('--label_nc', type=int, default=182, help='# of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.')
        parser.add_argument('--contain_dontcare_label', action='store_true', help='if the label map contains dontcare label (dontcare=255)')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # for setting inputs
        parser.add_argument('--dataroot', type=str, default='/mnt/blob/Dataset/ADEChallengeData2016/images')
        parser.add_argument('--dataset_mode', type=str, default='ade20k')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
        parser.add_argument('--nThreads', default=16, type=int, help='# threads for loading data')
        parser.add_argument('--max_dataset_size', type=int, default=sys.maxsize, help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--load_from_opt_file', action='store_true', help='load the options from checkpoints and use that as default')
        parser.add_argument('--cache_filelist_write', action='store_true', help='saves the current filelist into a text file, so that it loads faster')
        parser.add_argument('--cache_filelist_read', action='store_true', help='reads from the file list cache')

        # for displays
        parser.add_argument('--display_winsize', type=int, default=400, help='display window size')

        # for generator
        parser.add_argument('--netG', type=str, default='spade', help='selects model to use for netG (pix2pixhd | spade)')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')
        parser.add_argument('--z_dim', type=int, default=256,
                            help="dimension of the latent z vector")

        # for instance-wise features
        parser.add_argument('--CBN_intype', type=str, default='warp_mask', help='type of CBN input for framework, warp/mask/warp_mask')
        parser.add_argument('--maskmix', action='store_true', help='use mask in correspondence net')
        parser.add_argument('--use_attention', action='store_true', help='and nonlocal block in G and D')
        parser.add_argument('--warp_mask_losstype', type=str, default='none', help='type of warped mask loss, none/direct/cycle')
        parser.add_argument('--show_warpmask', action='store_true', help='save warp mask')
        parser.add_argument('--match_kernel', type=int, default=3, help='correspondence matrix match kernel size')
        parser.add_argument('--adaptor_kernel', type=int, default=3, help='kernel size in domain adaptor')
        parser.add_argument('--PONO', action='store_true', help='use positional normalization ')
        parser.add_argument('--PONO_C', action='store_true', help='use C normalization in corr module')
        parser.add_argument('--eqlr_sn', action='store_true', help='if true, use equlr, else use sn')
        parser.add_argument('--vgg_normal_correct', action='store_true', help='if true, correct vgg normalization and replace vgg FM model with ctx model')
        parser.add_argument('--weight_domainC', type=float, default=0.0, help='weight of Domain classification loss for domain adaptation')
        parser.add_argument('--domain_rela', action='store_true', help='if true, use Relativistic loss in domain classifier')
        parser.add_argument('--use_ema', action='store_true', help='if true, use EMA in G')
        parser.add_argument('--ema_beta', type=float, default=0.999, help='beta in ema setting') 
        parser.add_argument('--warp_cycle_w', type=float, default=0.0, help='push warp cycle to ref')
        parser.add_argument('--two_cycle', action='store_true', help='input to ref and back')
        parser.add_argument('--apex', action='store_true', help='if true, use apex')
        parser.add_argument('--warp_bilinear', action='store_true', help='if true, upsample warp by bilinear')
        parser.add_argument('--adaptor_res_deeper', action='store_true', help='if true, use 6 res block in domain adaptor')
        parser.add_argument('--adaptor_nonlocal', action='store_true', help='if true, use nonlocal block in domain adaptor')
        parser.add_argument('--adaptor_se', action='store_true', help='if true, use se layer in domain adaptor')
        parser.add_argument('--dilation_conv', action='store_true', help='if true, use dilation conv in domain adaptor when adaptor_res_deeper is True')
        parser.add_argument('--use_coordconv', action='store_true', help='if true, use coordconv in CorrNet')
        parser.add_argument('--warp_patch', action='store_true', help='use corr matrix to warp 4*4 patch')
        parser.add_argument('--warp_stride', type=int, default=4, help='corr matrix 256 / warp_stride')
        parser.add_argument('--mask_noise', action='store_true', help='use noise with mask')
        parser.add_argument('--noise_for_mask', action='store_true', help='replace mask with noise')
        parser.add_argument('--video_like', action='store_true', help='useful in deepfashion')
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)

        # modify dataset-related parser options
        dataset_mode = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_mode)
        parser = dataset_option_setter(parser, self.isTrain)

        opt, unknown = parser.parse_known_args()

        # if there is opt_file, load it.
        # The previous default options will be overwritten
        if opt.load_from_opt_file:
            parser = self.update_options_from_file(parser, opt)

        opt = parser.parse_args()
        self.parser = parser
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def option_file_path(self, opt, makedir=False):
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if makedir:
            util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        file_name = self.option_file_path(opt, makedir=False)
        new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
        return new_opt

    def parse(self, save=False):

        opt = self.gather_options()  #gather options from base, train, dataset, model
        opt.isTrain = self.isTrain   # train or test

        self.print_options(opt)
        if opt.isTrain:
            self.save_options(opt)

        # Set semantic_nc based on the option.
        # This will be convenient in many places
        opt.semantic_nc = opt.label_nc + \
            (1 if opt.contain_dontcare_label else 0)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        assert len(opt.gpu_ids) == 0 or opt.batchSize % len(opt.gpu_ids) == 0, \
            "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
            % (opt.batchSize, len(opt.gpu_ids))

        self.opt = opt
        return self.opt
