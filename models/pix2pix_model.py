# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
import models.networks as networks
import util.util as util


class Pix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor
        self.alpha = 1

        self.net = torch.nn.ModuleDict(self.initialize_networks(opt))

        # set loss functions
        if opt.isTrain:
            self.vggnet_fix = networks.correspondence.VGG19_feature_color_torchversion(vgg_normal_correct=opt.vgg_normal_correct)
            self.vggnet_fix.load_state_dict(torch.load('models/vgg19_conv.pth'))
            self.vggnet_fix.eval()
            for param in self.vggnet_fix.parameters():
                param.requires_grad = False

            self.vggnet_fix.to(self.opt.gpu_ids[0])
            self.contextual_forward_loss = networks.ContextualLoss_forward(opt)

            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            self.MSE_loss = torch.nn.MSELoss()
            if opt.which_perceptual == '5_2':
                self.perceptual_layer = -1
            elif opt.which_perceptual == '4_2':
                self.perceptual_layer = -2

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode, GforD=None, alpha=1):
        input_label, input_semantics, real_image, self_ref, ref_image, ref_label, ref_semantics = self.preprocess_input(data, )

        self.alpha = alpha
        generated_out = {}
        if mode == 'generator':
            g_loss, generated_out = self.compute_generator_loss(input_label, 
                input_semantics, real_image, ref_label, ref_semantics, ref_image, self_ref)
            
            out = {}
            out['fake_image'] = generated_out['fake_image']
            out['input_semantics'] = input_semantics
            out['ref_semantics'] = ref_semantics
            out['warp_out'] = None if 'warp_out' not in generated_out else generated_out['warp_out']
            out['warp_mask'] = None if 'warp_mask' not in generated_out else generated_out['warp_mask']
            out['adaptive_feature_seg'] = None if 'adaptive_feature_seg' not in generated_out else generated_out['adaptive_feature_seg']
            out['adaptive_feature_img'] = None if 'adaptive_feature_img' not in generated_out else generated_out['adaptive_feature_img']
            out['warp_cycle'] = None if 'warp_cycle' not in generated_out else generated_out['warp_cycle']
            out['warp_i2r'] = None if 'warp_i2r' not in generated_out else generated_out['warp_i2r']
            out['warp_i2r2i'] = None if 'warp_i2r2i' not in generated_out else generated_out['warp_i2r2i']
            return g_loss, out

        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image, GforD, label=input_label)
            return d_loss
        elif mode == 'inference':
            out = {}
            with torch.no_grad():
                out = self.inference(input_semantics, 
                        ref_semantics=ref_semantics, ref_image=ref_image, self_ref=self_ref)
            out['input_semantics'] = input_semantics
            out['ref_semantics'] = ref_semantics
            return out
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params, D_params = list(), list()
        G_params += [{'params': self.net['netG'].parameters(), 'lr': opt.lr*0.5}]
        G_params += [{'params': self.net['netCorr'].parameters(), 'lr': opt.lr*0.5}]
        if opt.isTrain:
            D_params += list(self.net['netD'].parameters())
            if opt.weight_domainC > 0 and opt.domain_rela:
                D_params += list(self.net['netDomainClassifier'].parameters())

        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            G_lr, D_lr = opt.lr, opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2), eps=1e-3)
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.net['netG'], 'G', epoch, self.opt)
        util.save_network(self.net['netD'], 'D', epoch, self.opt)
        util.save_network(self.net['netCorr'], 'Corr', epoch, self.opt)
        if self.opt.weight_domainC > 0 and self.opt.domain_rela: 
            util.save_network(self.net['netDomainClassifier'], 'DomainClassifier', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        net = {}
        net['netG'] = networks.define_G(opt)
        net['netD'] = networks.define_D(opt) if opt.isTrain else None
        net['netCorr'] = networks.define_Corr(opt)
        net['netDomainClassifier'] = networks.define_DomainClassifier(opt) if opt.weight_domainC > 0 and opt.domain_rela else None

        if not opt.isTrain or opt.continue_train:
            net['netG'] = util.load_network(net['netG'], 'G', opt.which_epoch, opt)
            if opt.isTrain:
                net['netD'] = util.load_network(net['netD'], 'D', opt.which_epoch, opt)
            net['netCorr'] = util.load_network(net['netCorr'], 'Corr', opt.which_epoch, opt)
            if opt.weight_domainC > 0 and opt.domain_rela:
                net['netDomainClassifier'] = util.load_network(net['netDomainClassifier'], 'DomainClassifier', opt.which_epoch, opt)
            if (not opt.isTrain) and opt.use_ema:
                net['netG'] = util.load_network(net['netG'], 'G_ema', opt.which_epoch, opt)
                net['netCorr'] = util.load_network(net['netCorr'], 'netCorr_ema', opt.which_epoch, opt)
        return net
        #return netG_stage1, netD_stage1, netG, netD, netE, netCorr

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        if self.opt.dataset_mode == 'celebahq':
            glasses = data['label'][:,1::2,:,:].long()
            data['label'] = data['label'][:,::2,:,:]
            glasses_ref = data['label_ref'][:,1::2,:,:].long()
            data['label_ref'] = data['label_ref'][:,::2,:,:]
            if self.use_gpu():
                glasses = glasses.cuda()
                glasses_ref = glasses_ref.cuda()
        elif self.opt.dataset_mode == 'celebahqedge':
            input_semantics = data['label'].clone().cuda().float()
            data['label'] = data['label'][:,:1,:,:]
            ref_semantics = data['label_ref'].clone().cuda().float()
            data['label_ref'] = data['label_ref'][:,:1,:,:]
        elif self.opt.dataset_mode == 'deepfashion':
            input_semantics = data['label'].clone().cuda().float()
            data['label'] = data['label'][:,:3,:,:]
            ref_semantics = data['label_ref'].clone().cuda().float()
            data['label_ref'] = data['label_ref'][:,:3,:,:]

        # move to GPU and change data types
        if self.opt.dataset_mode != 'deepfashion':
            data['label'] = data['label'].long()
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['image'] = data['image'].cuda()
            data['ref'] = data['ref'].cuda()
            data['label_ref'] = data['label_ref'].cuda()
            if self.opt.dataset_mode != 'deepfashion':
                data['label_ref'] = data['label_ref'].long()
            data['self_ref'] = data['self_ref'].cuda()

        # create one-hot label map
        if self.opt.dataset_mode != 'celebahqedge' and self.opt.dataset_mode != 'deepfashion':
            label_map = data['label']
            bs, _, h, w = label_map.size()
            nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
                else self.opt.label_nc
            input_label = self.FloatTensor(bs, nc, h, w).zero_()
            input_semantics = input_label.scatter_(1, label_map, 1.0)
        
            label_map = data['label_ref']
            label_ref = self.FloatTensor(bs, nc, h, w).zero_()
            ref_semantics = label_ref.scatter_(1, label_map, 1.0)

        if self.opt.dataset_mode == 'celebahq':
            assert input_semantics[:,-3:-2,:,:].sum().cpu().item() == 0
            input_semantics[:,-3:-2,:,:] = glasses
            assert ref_semantics[:,-3:-2,:,:].sum().cpu().item() == 0
            ref_semantics[:,-3:-2,:,:] = glasses_ref
        return data['label'], input_semantics, data['image'], data['self_ref'], data['ref'], data['label_ref'], ref_semantics

    def get_ctx_loss(self, source, target):
        contextual_style5_1 = torch.mean(self.contextual_forward_loss(source[-1], target[-1].detach())) * 8
        contextual_style4_1 = torch.mean(self.contextual_forward_loss(source[-2], target[-2].detach())) * 4
        contextual_style3_1 = torch.mean(self.contextual_forward_loss(F.avg_pool2d(source[-3], 2), F.avg_pool2d(target[-3].detach(), 2))) * 2
        if self.opt.use_22ctx:
            contextual_style2_1 = torch.mean(self.contextual_forward_loss(F.avg_pool2d(source[-4], 4), F.avg_pool2d(target[-4].detach(), 4))) * 1
            return contextual_style5_1 + contextual_style4_1 + contextual_style3_1 + contextual_style2_1
        return contextual_style5_1 + contextual_style4_1 + contextual_style3_1

    def compute_generator_loss(self, input_label, input_semantics, real_image, ref_label=None, ref_semantics=None, ref_image=None, self_ref=None):
        G_losses = {}
        generate_out = self.generate_fake(
            input_semantics, real_image, ref_semantics=ref_semantics, ref_image=ref_image, self_ref=self_ref)

        if 'loss_novgg_featpair' in generate_out and generate_out['loss_novgg_featpair'] is not None:
            G_losses['no_vgg_feat'] = generate_out['loss_novgg_featpair']
        
        if self.opt.warp_cycle_w > 0:
            if not self.opt.warp_patch:
                ref = F.avg_pool2d(ref_image, self.opt.warp_stride)
            else:
                ref = ref_image

            G_losses['G_warp_cycle'] = F.l1_loss(generate_out['warp_cycle'], ref) * self.opt.warp_cycle_w
            if self.opt.two_cycle:
                real = F.avg_pool2d(real_image, self.opt.warp_stride)
                G_losses['G_warp_cycle'] += F.l1_loss(generate_out['warp_i2r2i'], real) * self.opt.warp_cycle_w
                
        if self.opt.warp_self_w > 0:
            # real = F.avg_pool2d(real_image, self.opt.warp_stride)
            # warp = F.avg_pool2d(generate_out['warp_out'], self.opt.warp_stride)
            sample_weights = (self_ref[:, 0, 0, 0] / (sum(self_ref[:, 0, 0, 0]) + 1e-5)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            G_losses['G_warp_self'] = torch.mean(F.l1_loss(generate_out['warp_out'], real_image, reduce=False) * sample_weights) * self.opt.warp_self_w

        pred_fake, pred_real, seg, fake_cam_logit, real_cam_logit = self.discriminate(
            input_semantics, generate_out['fake_image'], real_image)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                            for_discriminator=False) * self.opt.weight_gan

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        fake_features = self.vggnet_fix(generate_out['fake_image'], ['r12', 'r22', 'r32', 'r42', 'r52'], preprocess=True)
        sample_weights = (self_ref[:, 0, 0, 0] / (sum(self_ref[:, 0, 0, 0]) + 1e-5)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        loss = 0
        for i in range(len(generate_out['real_features'])):
            loss += weights[i] * util.weighted_l1_loss(fake_features[i], generate_out['real_features'][i].detach(), sample_weights)
        G_losses['fm'] = loss * self.opt.lambda_vgg * self.opt.fm_ratio
        
        feat_loss = util.mse_loss(fake_features[self.perceptual_layer], generate_out['real_features'][self.perceptual_layer].detach())
        G_losses['perc'] = feat_loss * self.opt.weight_perceptual

        G_losses['contextual'] = self.get_ctx_loss(fake_features, generate_out['ref_features']) * self.opt.lambda_vgg * self.opt.ctx_w

        if self.opt.warp_mask_losstype != 'none':
            ref_label = F.interpolate(ref_label.float(), scale_factor=0.25, mode='nearest').long().squeeze(1)
            gt_label = F.interpolate(input_label.float(), scale_factor=0.25, mode='nearest').long().squeeze(1)
            weights = []
            for i in range(ref_label.shape[0]):
                ref_label_uniq = torch.unique(ref_label[i])
                gt_label_uniq = torch.unique(gt_label[i])
                zero_label = [it for it in gt_label_uniq if it not in ref_label_uniq]
                weight = torch.ones_like(gt_label[i]).float()
                for j in zero_label:
                    weight[gt_label[i] == j] = 0
                weight[gt_label[i] == 0] = 0 #no loss from unknown class
                weights.append(weight.unsqueeze(0))
            weights = torch.cat(weights, dim=0)
            #G_losses['mask'] = (F.cross_entropy(warp_mask, gt_label, reduce =False) * weights.float()).sum() / (weights.sum() + 1e-5) * self.opt.weight_mask
            G_losses['mask'] = (F.nll_loss(torch.log(generate_out['warp_mask'] + 1e-10), gt_label, reduce =False) * weights).sum() / (weights.sum() + 1e-5) * self.opt.weight_mask
        
        #self.fake_image = fake_image
        return G_losses, generate_out

    def compute_discriminator_loss(self, input_semantics, real_image, GforD, label=None):
        D_losses = {}
        with torch.no_grad():
            #fake_image, _, _, _, _ = self.generate_fake(input_semantics, real_image, VGG_feat=False)
            fake_image = GforD['fake_image'].detach()
            fake_image.requires_grad_()
            
        pred_fake, pred_real, seg, fake_cam_logit, real_cam_logit = self.discriminate(
            input_semantics, fake_image, real_image)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                            for_discriminator=True) * self.opt.weight_gan
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                            for_discriminator=True) * self.opt.weight_gan

        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.net['netE'](real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, real_image, ref_semantics=None, ref_image=None, self_ref=None):
        generate_out = {}
        #print(ref_image.max())
        ref_relu1_1, ref_relu2_1, ref_relu3_1, ref_relu4_1, ref_relu5_1 = self.vggnet_fix(ref_image, ['r12', 'r22', 'r32', 'r42', 'r52'], preprocess=True)
        
        coor_out = self.net['netCorr'](ref_image, real_image, input_semantics, ref_semantics, alpha=self.alpha)
        
        generate_out['ref_features'] = [ref_relu1_1, ref_relu2_1, ref_relu3_1, ref_relu4_1, ref_relu5_1]
        generate_out['real_features'] = self.vggnet_fix(real_image, ['r12', 'r22', 'r32', 'r42', 'r52'], preprocess=True)
        
        if self.opt.CBN_intype == 'mask':
            CBN_in = input_semantics
        elif self.opt.CBN_intype == 'warp':
            CBN_in = coor_out['warp_out']
        elif self.opt.CBN_intype == 'warp_mask':
            CBN_in = torch.cat((coor_out['warp_out'], input_semantics), dim=1)

        generate_out['fake_image'] = self.net['netG'](input_semantics, warp_out=CBN_in)

        generate_out = {**generate_out, **coor_out}
        return generate_out

    def inference(self, input_semantics, ref_semantics=None, ref_image=None, self_ref=None):
        generate_out = {}
        coor_out = self.net['netCorr'](ref_image, None, input_semantics, ref_semantics, alpha=self.alpha)
        if self.opt.CBN_intype == 'mask':
            CBN_in = input_semantics
        elif self.opt.CBN_intype == 'warp':
            CBN_in = coor_out['warp_out']
        elif self.opt.CBN_intype == 'warp_mask':
            CBN_in = torch.cat((coor_out['warp_out'], input_semantics), dim=1)

        generate_out['fake_image'] = self.net['netG'](input_semantics, warp_out=CBN_in)
        generate_out = {**generate_out, **coor_out}
        return generate_out

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
        seg = None
        discriminator_out, seg, cam_logit = self.net['netD'](fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)
        fake_cam_logit, real_cam_logit = None, None
        if self.opt.D_cam > 0:
            fake_cam_logit = torch.cat([it[:it.shape[0]//2] for it in cam_logit], dim=1)
            real_cam_logit = torch.cat([it[it.shape[0]//2:] for it in cam_logit], dim=1)
        #fake_cam_logit, real_cam_logit = self.divide_pred(cam_logit)

        return pred_fake, pred_real, seg, fake_cam_logit, real_cam_logit

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

    def compute_D_seg_loss(self, out, gt):
        fake_seg, real_seg = self.divide_pred([out])
        fake_seg_loss = F.cross_entropy(fake_seg[0][0], gt)
        real_seg_loss = F.cross_entropy(real_seg[0][0], gt)

        down_gt = F.interpolate(gt.unsqueeze(1).float(), scale_factor=0.5, mode='nearest').squeeze().long()
        fake_seg_loss_down = F.cross_entropy(fake_seg[0][1], down_gt)
        real_seg_loss_down = F.cross_entropy(real_seg[0][1], down_gt)

        seg_loss = fake_seg_loss + real_seg_loss + fake_seg_loss_down + real_seg_loss_down
        return seg_loss