# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from data.pix2pix_dataset import Pix2pixDataset

class FlickrDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        if is_train:
            parser.set_defaults(load_size=286)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=150)
        parser.set_defaults(contain_dontcare_label=True)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)
        return parser

    def get_paths(self, opt):
        root = os.path.join(opt.dataroot, 'test/images') if opt.phase == 'test' else os.path.join(opt.dataroot, 'images')
        root_mask = root.replace('images', 'mask')

        image_paths = sorted(os.listdir(root))
        image_paths = [os.path.join(root, it) for it in image_paths]
        label_paths = sorted(os.listdir(root_mask))
        label_paths = [os.path.join(root_mask, it) for it in label_paths]

        return label_paths, image_paths

    def get_ref(self, opt):
        extra = '_test_from_train' if opt.phase == 'test' else ''
        with open('./data/flickr_ref{}.txt'.format(extra)) as fd:
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
        train_test_folder = ('', 'test')
        return ref_dict, train_test_folder
        
    def imgpath_to_labelpath(self, path):
        path_ref_label = path.replace('images', 'mask')
        return path_ref_label
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
