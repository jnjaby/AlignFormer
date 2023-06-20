import importlib
import torch
import pdb
from collections import OrderedDict
from copy import deepcopy
import os
from os import path as osp
import numpy as np
import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel
from basicsr.utils import tensor2npy


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


@MODEL_REGISTRY.register()
class CYCGANModel(SRModel):
    """CYCGAN model for single image super-resolution."""

    def init_training_settings(self):
        train_opt = self.opt['train']

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # define network net_r
        self.net_r = build_network(self.opt['network_r'])
        self.net_r = self.model_to_device(self.net_r)
        self.print_network(self.net_r)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_model_d', None)
        if load_path is not None:
            self.load_network(self.net_d, load_path,
                              self.opt['path']['strict_load'])

        load_path = self.opt['path'].get('pretrain_model_r', None)
        if load_path is not None:
            self.load_network(self.net_r, load_path,
                              self.opt['path']['strict_load'])

        self.net_g.train()
        self.net_d.train()
        self.net_r.train()

        # define losses
        if train_opt.get('pixel_opt', None):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt', None):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt', None):
            self.cri_perceptual = build_loss(train_opt['gan_opt']).to(self.device)

        self.net_d_iters = train_opt['net_d_iters'] if train_opt[
            'net_d_iters'] else 1
        self.net_d_init_iters = train_opt['net_d_init_iters'] if train_opt[
            'net_d_init_iters'] else 0

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(self.net_g.parameters(),
                                                **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_d = torch.optim.Adam(self.net_d.parameters(),
                                                **train_opt['optim_d'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_d)

        # optimizer r
        optim_type = train_opt['optim_r'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_r = torch.optim.Adam(self.net_r.parameters(),
                                                **train_opt['optim_r'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_r)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.real_sharp = data['real_sharp'].to(self.device)
        if 'real_blur' in data:
            self.real_blur = data['real_blur'].to(self.device)

    def optimize_parameters(self, current_iter):
        # optimize net_g and net_r
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.optimizer_r.zero_grad()
        self.fake_blur = self.net_g(self.lq)
        self.fake_sharp = self.net_r(self.fake_blur)

        l_g_total = 0
        l_r_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0
                and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_r_pix = self.cri_pix(self.fake_sharp, self.real_sharp)
                l_r_total = l_r_total + l_r_pix
                loss_dict['l_r_pix'] = l_r_pix

            # perceptual loss (G network)
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(
                    self.fake_blur, self.lq)
                if l_g_percep is not None:
                    l_g_total = l_g_total + l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total = l_g_total + l_g_style
                    loss_dict['l_g_style'] = l_g_style

            # gan loss (relativistic gan)
            real_d_pred = self.net_d(self.real_blur).detach()
            fake_g_pred = self.net_d(self.fake_blur)
            l_g_real = self.cri_gan(
                real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
            l_g_fake = self.cri_gan(
                fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)
            l_g_gan = (l_g_real + l_g_fake) / 2

            l_g_total = l_g_total + l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            (l_g_total + l_r_total).backward()
            # l_r_total.backward()
            self.optimizer_g.step()
            self.optimizer_r.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # gan loss (relativistic gan)
        # real
        fake_d_pred = self.net_d(self.fake_blur).detach()
        real_d_pred = self.net_d(self.real_blur)
        l_d_real = self.cri_gan(
            real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(self.fake_blur.detach())
        l_d_fake = self.cri_gan(
            fake_d_pred - torch.mean(real_d_pred.detach()),
            False, is_disc=True) * 0.5
        l_d_fake.backward()
        self.optimizer_d.step()

        loss_dict['l_d_real'] = l_d_real
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['fake_blur'] = self.fake_blur.detach().cpu()
        out_dict['fake_sharp'] = self.fake_sharp.detach().cpu()
        out_dict['real_sharp'] = self.real_sharp.detach().cpu()
        if hasattr(self, 'real_blur'):
            out_dict['real_blur'] = self.real_blur.detach().cpu()
        return out_dict

    def test(self):
        self.net_g.eval()
        self.net_r.eval()
        with torch.no_grad():
            self.fake_blur = self.net_g(self.lq)
            self.fake_sharp = self.net_r(self.fake_blur)
        self.net_g.train()
        self.net_r.train()

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_network(self.net_g, 'net_r', current_iter)
        self.save_training_state(epoch, current_iter)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val']['metrics'] is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            val_data = {'lq': val_data['lq'], 'real_sharp': val_data['gt']}
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            lq = tensor2npy([visuals['lq']])
            fake_blur = tensor2npy([visuals['fake_blur']])
            fake_sharp = tensor2npy([visuals['fake_sharp']])
            real_sharp = tensor2npy([visuals['real_sharp']])
            if 'real_blur' in visuals:
                real_blur = tensor2npy([visuals['real_blur']])
                del self.real_blur

            # tentative for out of GPU memory
            del self.lq
            del self.fake_blur
            del self.fake_sharp
            del self.real_sharp
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    mkdir_or_exist(osp.join(self.opt['path']['visualization'], img_name))
                    np.save(osp.join(self.opt['path']['visualization'], img_name,
                                     f'{img_name}_lq_{current_iter}.npy'), lq)
                    np.save(osp.join(self.opt['path']['visualization'], img_name,
                                     f'{img_name}_fakeblur_{current_iter}.npy'), fake_blur)
                    np.save(osp.join(self.opt['path']['visualization'], img_name,
                                     f'{img_name}_fakesharp_{current_iter}.npy'), fake_sharp)
                    np.save(osp.join(self.opt['path']['visualization'], img_name,
                                     f'{img_name}_realsharp_{current_iter}.npy'), real_sharp)
                else:
                    mkdir_or_exist(osp.join(self.opt['path']['visualization'], dataset_name))
                    if self.opt['val']['suffix']:
                        np.save(osp.join(self.opt['path']['visualization'], dataset_name,
                                         f'{img_name}_lq_{self.opt["val"]["suffix"]}.npy'), lq)
                        np.save(osp.join(self.opt['path']['visualization'], dataset_name,
                                         f'{img_name}_fakeblur_{self.opt["val"]["suffix"]}.npy'), fake_blur)
                        np.save(osp.join(self.opt['path']['visualization'], dataset_name,
                                         f'{img_name}_fakesharp_{self.opt["val"]["suffix"]}.npy'), fake_sharp)
                        np.save(osp.join(self.opt['path']['visualization'], dataset_name,
                                         f'{img_name}_realsharp_{self.opt["val"]["suffix"]}.npy'), real_sharp)
                    else:
                        np.save(osp.join(self.opt['path']['visualization'], dataset_name,
                                         f'{img_name}_lq.npy'), lq)
                        np.save(osp.join(self.opt['path']['visualization'], dataset_name,
                                         f'{img_name}_fakeblur.npy'), fake_blur)
                        np.save(osp.join(self.opt['path']['visualization'], dataset_name,
                                         f'{img_name}_fakesharp.npy'), fake_sharp)
                        np.save(osp.join(self.opt['path']['visualization'], dataset_name,
                                         f'{img_name}_realsharp.npy'), real_sharp)
                # np.save(fake_blur_path, sr_img) # replace for raw data.
                # mmcv.imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    fake = np.clip(fake_sharp, 0, 1)
                    real = np.clip(real_sharp, 0, 1)

                    metric_data = dict(img1=fake, img2=real)
                    self.metric_results[name] += calculate_metric(metric_data, opt_)

            pbar.update(f'Test {img_name}')

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
