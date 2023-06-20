import pdb
import torch
from collections import OrderedDict
import torch.nn.functional as F

from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
from .srgan_model import SRGANModel


@MODEL_REGISTRY.register()
class PatchGANModel(SRGANModel):
    """PatchGAN model for image restoration."""

    def __init__(self, opt):
        super(SRGANModel, self).__init__(opt)

        logger = get_root_logger()
        self.conditional = opt.get('conditional', False)
        if self.conditional:
            logger.info('Use conditional GAN.')
        else:
            logger.info('Use unconditional GAN. Reduce to patched ESRGAN')

        self.regional = opt.get('regional', None)
        self.regional_thres = opt.get('regional_thres', 0)
        self.regional_size = opt.get('regional_size', 0)
        logger.info(f'Regional type [{self.regional}]')
        assert self.regional in [None, 'mean', 'thres']
        assert 0 <= self.regional_thres <= 1, f'Regional threshold out of range.'
        assert self.regional_size in [0, 16, 36], f'Unsupported regional size {self.regional_size}'

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            if self.gt_usm:
                self.gt = self.usm_sharpener(self.gt)
        if 'ref' in data:
            self.light_mask = data['ref'].to(self.device)
        if 'mask' in data:
            self.occlu_mask = data['mask'].to(self.device)
        else:
            self.occlu_mask = None

    def optimize_parameters(self, current_iter):
        if self.regional != None:
            patch_masks = self.unfold_patch(1 - self.light_mask, patch_size=self.regional_size)

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        # if hasattr(self, 'mask'):
        #     weight = self.mask
        # else:
        #     weight = None

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt, weight=self.occlu_mask)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # gan loss (relativistic gan)
            if self.conditional:
                real_pair = torch.cat((self.lq, self.gt), dim=1)
                fake_pair = torch.cat((self.lq, self.output), dim=1)
                real_d_pred = self.net_d(real_pair).detach()
                fake_g_pred = self.net_d(fake_pair)
            else:
                real_d_pred = self.net_d(self.gt).detach()
                fake_g_pred = self.net_d(self.output)
            l_g_real = self.cri_gan(real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
            l_g_fake = self.cri_gan(fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)

            if self.regional != None:
                l_g_real = (l_g_real * patch_masks).mean()
                l_g_fake = (l_g_fake * patch_masks).mean()
            l_g_gan = (l_g_real + l_g_fake) / 2

            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # gan loss (relativistic gan)

        # In order to avoid the error in distributed training:
        # "Error detected in CudnnBatchNormBackward: RuntimeError: one of
        # the variables needed for gradient computation has been modified by
        # an inplace operation",
        # we separate the backwards for real and fake, and also detach the
        # tensor for calculating mean.

        if self.conditional:
            real_pair = torch.cat((self.lq, self.gt), dim=1)
            fake_pair = torch.cat((self.lq, self.output), dim=1)
            # real
            fake_d_pred = self.net_d(fake_pair).detach()
            real_d_pred = self.net_d(real_pair)
            l_d_real = self.cri_gan(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
            if self.regional != None:
                l_d_real = (l_d_real * patch_masks).mean()
            l_d_real.backward()
            # fake
            fake_d_pred = self.net_d(fake_pair.detach())
            l_d_fake = self.cri_gan(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True) * 0.5
            if self.regional != None:
                l_d_fake = (l_d_fake * patch_masks).mean()
            l_d_fake.backward()
        else:
            # real
            fake_d_pred = self.net_d(self.output).detach()
            real_d_pred = self.net_d(self.gt)
            l_d_real = self.cri_gan(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
            if self.regional != None:
                l_d_real = (l_d_real * patch_masks).mean()
            l_d_real.backward()
            # fake
            fake_d_pred = self.net_d(self.output.detach())
            l_d_fake = self.cri_gan(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True) * 0.5
            if self.regional != None:
                l_d_fake = (l_d_fake * patch_masks).mean()
            l_d_fake.backward()

        self.optimizer_d.step()

        loss_dict['l_d_real'] = l_d_real
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def unfold_patch(self, mask, patch_size=16):
        N, C, H, W = mask.shape
        assert patch_size in [16, 36], f"Unsupported patch size {patch_size}"
        if patch_size == 16:
            point_shape = 64
            padding = 6
            stride = 4
        elif patch_size == 36:
            point_shape = 32
            padding = 14
            stride = 8
        patch_masks = F.unfold(mask[:, 0:1, :, :], patch_size, 1, padding=padding, stride=stride)
        # patch_masks = patch_masks.view(1, patch_size, patch_size, -1)
        patch_masks = patch_masks.view(N, -1, point_shape, point_shape)
        assert patch_masks.size(1) == patch_size * patch_size, f"Patch sizes not match."

        patch_masks = patch_masks.mean(dim=1, keepdim=True)
        if self.regional == 'thres':
            patch_masks = (patch_masks > self.regional_thres).float()
            # pdb.set_trace()

        return patch_masks
