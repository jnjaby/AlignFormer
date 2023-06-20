import importlib
import mmcv
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
import numpy as np
import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, tensor2img, tensor2raw


@MODEL_REGISTRY.register()
class GRModel(BaseModel):
    """Guided restoration model for single image restoration."""

    def __init__(self, opt):
        super().__init__(opt)

        # define guidance network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # define restoration network
        self.net_r = build_network(opt['network_r'])
        self.net_r = self.model_to_device(self.net_r)
        self.print_network(self.net_r)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_model_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path']['strict_load'])

        load_path = self.opt['path'].get('pretrain_model_r', None)
        if load_path is not None:
            self.load_network(self.net_r, load_path,
                              self.opt['path']['strict_load'])

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        self.net_r.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(self.net_g.parameters(), **train_opt['optim_g'])
            self.optimizers.append(self.optimizer_g)
        elif optim_type == 'Fixed':
            for p in self.net_g.parameters():
                p.requires_grad = False
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')

        # optimizer r
        optim_type = train_opt['optim_r'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_r = torch.optim.Adam(self.net_r.parameters(), **train_opt['optim_r'])
            self.optimizers.append(self.optimizer_r)
        elif optim_type == 'Fixed':
            for p in self.net_r.parameters():
                p.requires_grad = False
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        if hasattr(self, 'optimizer_g'):
            self.optimizer_g.zero_grad()
        if hasattr(self, 'optimizer_r'):
            self.optimizer_r.zero_grad()
        self.guide_map = self.net_g(self.lq)
        self.output = self.net_r(torch.cat((self.lq, self.guide_map), dim=1))

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += 1
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        if hasattr(self, 'optimizer_g'):
            self.optimizer_g.step()
        if hasattr(self, 'optimizer_r'):
            self.optimizer_r.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        self.net_r.eval()
        with torch.no_grad():
            self.guide_map = self.net_g(self.lq)
            self.output = self.net_r(torch.cat((self.lq, self.guide_map), dim=1))
        self.net_g.train()
        self.net_r.train()

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
            gd_img = tensor2img([visuals['guide']])
            if 'gt' in visuals:
                # gt_img = tensor2raw([visuals['gt']]) # replace for raw data.
                gt_img = tensor2img([visuals['gt']])
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.guide_map
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    mmcv.imwrite(gd_img,
                                 osp.join(self.opt['path']['visualization'],
                                          img_name,
                                          f'{img_name}_guide_{current_iter}.png'))
                    mmcv.imwrite(sr_img,
                                 osp.join(self.opt['path']['visualization'],
                                          img_name,
                                          f'{img_name}_{current_iter}.png'))
                else:
                    if self.opt['val']['suffix']:
                        mmcv.imwrite(gd_img,
                                     osp.join(self.opt['path']['visualization'], dataset_name,
                                              f'{img_name}_guide_{self.opt["val"]["suffix"]}.png'))
                        mmcv.imwrite(sr_img,
                                     osp.join(self.opt['path']['visualization'], dataset_name,
                                              f'{img_name}_{self.opt["val"]["suffix"]}.png'))
                    else:
                        mmcv.imwrite(gd_img,
                                     osp.join(self.opt['path']['visualization'], dataset_name,
                                              f'{img_name}_guide.png'))
                        mmcv.imwrite(sr_img,
                                     osp.join(self.opt['path']['visualization'], dataset_name,
                                              f'{img_name}.png'))

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
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['guide'] = self.guide_map.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_r, 'net_r', current_iter)
        self.save_training_state(epoch, current_iter)
