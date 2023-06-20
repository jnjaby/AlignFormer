import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
import mmcv
import pdb
import numpy as np
import tqdm
from os import path as osp

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, tensor2img


@MODEL_REGISTRY.register()
class ClassifierModel(BaseModel):
    """Classifier model for different degradation."""

    def __init__(self, opt):
        super().__init__(opt)

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_model_d', None)
        if load_path is not None:
            self.load_network(self.net_d, load_path,
                              self.opt['path']['strict_load'])

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        train_opt = self.opt['train']
        self.net_d.train()

        # define losses
        if train_opt.get('gan_opt'):
            self.cri_pix = build_loss(train_opt['gan_opt']).to(self.device)

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_d = torch.optim.Adam(self.net_d.parameters(),
                                                **train_opt['optim_d'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        loss_dict = OrderedDict()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # real
        real_d_pred = self.net_d(self.gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(self.lq)
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()

        self.real_d_pred = real_d_pred
        self.fake_d_pred = fake_d_pred
        # pdb.set_trace()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def compare_pred_to_label(self, pred, target_is_real):
        target_val = 1 if target_is_real else 0
        target = torch.ones_like(pred) * target_val
        # target = pred.new_ones(pred.size()) * target_val
        # print(pred, target)
        # pdb.set_trace()
        pred = (pred > 0).float() * 1
        correct_tensor = pred.eq(target)

        return correct_tensor

    def test(self):
        self.net_d.eval()
        with torch.no_grad():
            self.lq_pred = self.net_d(self.lq)
            self.gt_pred = self.net_d(self.gt)
            # pdb.set_trace()
            self.lq_output = self.compare_pred_to_label(self.lq_pred,
                                                        target_is_real=False)
            self.gt_output = self.compare_pred_to_label(self.gt_pred,
                                                        target_is_real=True)
        self.net_d.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img):
        dataset_name = dataloader.dataset.opt['name']
        self.metric_results = {
            metric: 0
            for metric in ['lq_acc', 'gt_acc', 'overall_acc']
        }
        pbar = tqdm(total=len(dataloader), unit='image')

        self.sum_lq_pred, self.sum_gt_pred = 0, 0

        # pdb.set_trace()
        # f = open(osp.join(self.opt['path']
        #          ['results_root'], f'pred_log.txt'), 'w')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            self.metric_results['lq_acc'] += self.lq_output.sum().cpu().numpy()
            self.metric_results['gt_acc'] += self.gt_output.sum().cpu().numpy()
            pbar.update(
                f'Test {img_name} {self.lq_pred.sum().cpu().numpy():.4f} {self.gt_pred.sum().cpu().numpy():.4f}')

            # f.write(
            #     f'{idx:04d} {img_name} {self.lq_pred.sum().cpu().numpy():.4f} {self.gt_pred.sum().cpu().numpy():.4f}\n')
            # pdb.set_trace()
            self.sum_lq_pred += self.lq_pred.sum().cpu().numpy()
            self.sum_gt_pred += self.gt_pred.sum().cpu().numpy()

            # tentative for out of GPU memory
            del self.lq
            del self.gt
            del self.lq_output
            del self.gt_output
            torch.cuda.empty_cache()

        # f.close()

        for metric in self.metric_results.keys():
            self.metric_results[metric] /= (idx + 1)

        self.metric_results['overall_acc'] = (self.metric_results['lq_acc'] +
                                              self.metric_results['gt_acc']) / 2

        self.sum_lq_pred /= (idx + 1)
        self.sum_gt_pred /= (idx + 1)

        self._log_validation_metric_values(current_iter, dataset_name,
                                           tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.2%}\n'
        logger = get_root_logger()
        logger.info(log_str)
        logger.info(f'LQ: {self.sum_lq_pred:.4f}; GT: {self.sum_gt_pred:.4f}')
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def save(self, epoch, current_iter):
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
