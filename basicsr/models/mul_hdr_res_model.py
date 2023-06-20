import importlib
import mmcv
import torch
import math
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
import numpy as np
import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.models.base_model import BaseModel
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils import get_root_logger, tensor2img, tensor2raw, tensor2npy
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class MulHDRResModel(SRGANModel):
    """Base model for Multiple HDR restoration under unsupervised settings."""

    def feed_data(self, data):
        self.img_s = data['img_s'].to(self.device)
        self.img_m = data['img_m'].to(self.device)
        self.img_l = data['img_l'].to(self.device)
        self.img_gt = data['img_gt'].to(self.device)

        if 'img_gt' in data:
            self.img_gt = data['img_gt'].to(self.device)

    def init_training_settings(self):
        train_opt = self.opt['train']

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_model_d', None)
        if load_path is not None:
            self.load_network(self.net_d, load_path,
                              self.opt['path']['strict_load'])

        self.net_g.train()
        self.net_d.train()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_perceptual = build_loss(train_opt['gan_opt']).to(self.device)

        if train_opt.get('spatial_opt'):
            self.cri_spatial = build_loss(train_opt['spatial_opt']).to(self.device)
        else:
            self.cri_spatial = None

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def optimize_parameters(self, current_iter):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.img_s, self.img_m, self.img_l)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0
                and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.img_gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # spatial loss
            if self.cri_spatial:
                l_g_spatial = self.cri_spatial(
                    self.output, self._tonemap(self.img_l))
                l_g_total += l_g_spatial
                loss_dict['l_g_spatial'] = l_g_spatial
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(
                    self.output, self._tonemap(self.img_l))
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # gan loss (relativistic gan)
            real_d_pred = self.net_d(self.img_gt).detach()
            fake_g_pred = self.net_d(self.output)
            l_g_real = self.cri_gan(
                real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
            l_g_fake = self.cri_gan(
                fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)
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

        # real
        fake_d_pred = self.net_d(self.output).detach()
        real_d_pred = self.net_d(self.img_gt)
        l_d_real = self.cri_gan(
            real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(self.output.detach())
        l_d_fake = self.cri_gan(
            fake_d_pred - torch.mean(real_d_pred.detach()),
            False,
            is_disc=True) * 0.5
        l_d_fake.backward()
        self.optimizer_d.step()

        loss_dict['l_d_real'] = l_d_real
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            N, C, H, W = self.img_s.shape
            if H > 2000 or W > 2000:
                self.output = self.test_crop9()
            else:
                self.output = self.net_g(self.img_s, self.img_m, self.img_l)
        self.net_g.train()

    def test_crop9(self):
        N, C, H, W = self.img_s.shape
        h, w = math.ceil(H / 3), math.ceil(W / 3)
        rf = 30
        imTL = self.net_g(
            self.img_s[:, :, 0:h + rf, 0:w + rf],
            self.img_m[:, :, 0:h + rf, 0:w + rf],
            self.img_l[:, :, 0:h + rf, 0:w + rf])[:, :, 0:h, 0:w]
        imML = self.net_g(
            self.img_s[:, :, h - rf:2 * h + rf, 0:w + rf],
            self.img_m[:, :, h - rf:2 * h + rf, 0:w + rf],
            self.img_l[:, :, h - rf:2 * h + rf, 0:w + rf])[:, :, rf:(rf + h), 0:w]
        imBL = self.net_g(
            self.img_s[:, :, 2 * h - rf:, 0:w + rf],
            self.img_m[:, :, 2 * h - rf:, 0:w + rf],
            self.img_l[:, :, 2 * h - rf:, 0:w + rf])[:, :, rf:, 0:w]
        imTM = self.net_g(
            self.img_s[:, :, 0:h + rf, w - rf:2 * w + rf],
            self.img_m[:, :, 0:h + rf, w - rf:2 * w + rf],
            self.img_l[:, :, 0:h + rf, w - rf:2 * w + rf])[:, :, 0:h, rf:(rf + w)]
        imMM = self.net_g(
            self.img_s[:, :, h - rf:2 * h + rf, w - rf:2 * w + rf],
            self.img_m[:, :, h - rf:2 * h + rf, w - rf:2 * w + rf],
            self.img_l[:, :, h - rf:2 * h + rf, w - rf:2 * w + rf])[:, :, rf:(rf + h), rf:(rf + w)]
        imBM = self.net_g(
            self.img_s[:, :, 2 * h - rf:, w - rf:2 * w + rf],
            self.img_m[:, :, 2 * h - rf:, w - rf:2 * w + rf],
            self.img_l[:, :, 2 * h - rf:, w - rf:2 * w + rf])[:, :, rf:, rf:(rf + w)]
        imTR = self.net_g(
            self.img_s[:, :, 0:h + rf, 2 * w - rf:],
            self.img_m[:, :, 0:h + rf, 2 * w - rf:],
            self.img_l[:, :, 0:h + rf, 2 * w - rf:])[:, :, 0:h, rf:]
        imMR = self.net_g(
            self.img_s[:, :, h - rf:2 * h + rf, 2 * w - rf:],
            self.img_m[:, :, h - rf:2 * h + rf, 2 * w - rf:],
            self.img_l[:, :, h - rf:2 * h + rf, 2 * w - rf:],)[:, :, rf:(rf + h), rf:]
        imBR = self.net_g(
            self.img_s[:, :, 2 * h - rf:, 2 * w - rf:],
            self.img_m[:, :, 2 * h - rf:, 2 * w - rf:],
            self.img_l[:, :, 2 * h - rf:, 2 * w - rf:])[:, :, rf:, rf:]

        imT = torch.cat((imTL, imTM, imTR), 3)
        imM = torch.cat((imML, imMM, imMR), 3)
        imB = torch.cat((imBL, imBM, imBR), 3)
        output_cat = torch.cat((imT, imM, imB), 2)
        return output_cat

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            if 'img_gt' in visuals:
                gt_img = tensor2img([visuals['img_gt']])
                del self.img_gt

            # tentative for out of GPU memory
            del self.img_s
            del self.img_m
            del self.img_l
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}.png')
                # np.save(save_img_path.replace('.png', '.npy'), sr_img) # replace for raw data.
                mmcv.imwrite(sr_img, save_img_path)
                # mmcv.imwrite(gt_img, save_img_path.replace('syn_val', 'gt'))

            save_npy = self.opt['val'].get('save_npy', None)
            if save_npy:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.npy')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.npy')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}.npy')

                # saving as .npy format.
                np.save(save_img_path, tensor2npy([visuals['result']]))

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    metric_data = dict(img1=sr_img, img2=gt_img)
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['img_l'] = self.img_l.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'img_gt'):
            out_dict['img_gt'] = self.img_gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

    def _tonemap(self, x, type='simple'):
        if type == 'mu_law':
            norm_x = x / x.max()
            mapped_x = np.log(1 + 10000 * norm_x) / np.log(1 + 10000)
        elif type == 'simple':
            mapped_x = x / (x + 0.25)
        elif type == 'same':
            mapped_x = x
        else:
            raise NotImplementedError(
                'tone mapping type [{:s}] is not recognized.'.format(type))
        return mapped_x
