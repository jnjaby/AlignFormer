import torch
import torch.nn.functional as F
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import numpy as np

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class ContrastModel(BaseModel):
    """Base Contrast model for image restoration."""

    def __init__(self, opt):
        super().__init__(opt)
        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)
            print(f'Loading model from {load_path}')

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

        self.log_dict = OrderedDict()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        # hyper-parameters for loss
        self.margin = train_opt['margin']
        self.safe_radius = train_opt['safe_radius']
        self.scaling_steps = train_opt['scaling_steps']

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.ref = data['ref'].to(self.device)
        self.transformed_coordinates = data['transformed_coordinate'].to(
            self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq, self.ref)

        loss, pos_dist, neg_dist = self.loss_function()

        self.log_dict['loss'] = loss.item()
        self.log_dict['pos_dist'] = pos_dist.item()
        self.log_dict['neg_dist'] = neg_dist.item()

        loss.backward()
        self.optimizer_g.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq, self.ref)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq, self.ref)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='image')
        loss_val_all = 0.
        pos_dist_val_all = 0.
        neg_dist_val_all = 0.

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            loss_val, pos_dist_val, neg_dist_val = self.loss_function()

            # tentative for out of GPU memory
            del self.lq
            del self.ref
            del self.transformed_coordinates
            del self.output
            torch.cuda.empty_cache()

            loss_val_all += loss_val.item()
            pos_dist_val_all += pos_dist_val.item()
            neg_dist_val_all += neg_dist_val.item()

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        loss_val_all = loss_val_all / (idx + 1)
        pos_dist_val_all = pos_dist_val_all / (idx + 1)
        neg_dist_val_all = neg_dist_val_all / (idx + 1)

        # log
        logger = get_root_logger()
        logger.info(
            f'# Validation {dataset_name} # loss_val: {loss_val_all:.4e} '
            f'# positive_distance: {pos_dist_val:.4e} '
            f'# negative_distance: {neg_dist_val:.4e}.')
        if tb_logger:
            tb_logger.add_scalar('loss_val', loss_val_all, current_iter)

            # self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

    def loss_function(self):
        loss = torch.tensor(
            np.array([0], dtype=np.float32), device=self.device)
        pos_dist = 0.
        neg_dist = 0.

        has_grad = False

        n_valid_samples = 0
        batch_size = self.output['dense_features1'].size(0)
        for idx_in_batch in range(batch_size):

            # Network output
            # shape: [c, h1, w1]
            dense_features1 = self.output['dense_features1'][idx_in_batch]
            c, h1, w1 = dense_features1.size()

            # shape: [c, h2, w2]
            dense_features2 = self.output['dense_features2'][idx_in_batch]
            _, h2, w2 = dense_features2.size()

            # shape: [c, h1 * w1]
            all_descriptors1 = F.normalize(dense_features1.view(c, -1), dim=0)
            descriptors1 = all_descriptors1

            # Warp the positions from image 1 to image 2
            # shape: [2, h1 * w1], coordinate in [h1, w1] dim,
            # dim 0: y, dim 1: x, positions in feature map
            fmap_pos1 = grid_positions(h1, w1, self.device)
            # shape: [2, h1 * w1], coordinate in image level (4 * h1, 4 * w1)
            pos1 = upscale_positions(
                fmap_pos1, scaling_steps=self.scaling_steps)
            pos1, pos2, ids = warp(pos1, 4 * h1, 4 * w1,
                                   self.transformed_coordinates[idx_in_batch])
            # pos1, pos2, ids = warp(pos1, h1, w1,
            #                        self.transformed_coordinates[idx_in_batch])

            # shape: [2, num_ids]
            fmap_pos1 = fmap_pos1[:, ids]
            # shape: [c, num_ids]
            descriptors1 = descriptors1[:, ids]

            # Skip the pair if not enough GT correspondences are available
            if ids.size(0) < 128:
                continue

            # Descriptors at the corresponding positions
            fmap_pos2 = torch.round(
                downscale_positions(pos2,
                                    scaling_steps=self.scaling_steps)).long()
            descriptors2 = F.normalize(
                dense_features2[:, fmap_pos2[0, :], fmap_pos2[1, :]], dim=0)

            positive_distance = 2 - 2 * (descriptors1.t().unsqueeze(
                1) @ descriptors2.t().unsqueeze(2)).squeeze()

            position_distance = torch.max(
                torch.abs(
                    fmap_pos2.unsqueeze(2).float() - fmap_pos2.unsqueeze(1)),
                dim=0)[0]
            is_out_of_safe_radius = position_distance > self.safe_radius
            distance_matrix = 2 - 2 * (descriptors1.t() @ descriptors2)
            negative_distance2 = torch.min(
                distance_matrix + (1 - is_out_of_safe_radius.float()) * 10.,
                dim=1)[0]

            all_fmap_pos1 = grid_positions(h1, w1, self.device)
            position_distance = torch.max(
                torch.abs(
                    fmap_pos1.unsqueeze(2).float() -
                    all_fmap_pos1.unsqueeze(1)),
                dim=0)[0]
            is_out_of_safe_radius = position_distance > self.safe_radius
            distance_matrix = 2 - 2 * (descriptors2.t() @ all_descriptors1)
            negative_distance1 = torch.min(
                distance_matrix + (1 - is_out_of_safe_radius.float()) * 10.,
                dim=1)[0]

            diff = positive_distance - torch.min(negative_distance1,
                                                 negative_distance2)

            loss = loss + torch.mean(F.relu(self.margin + diff))

            pos_dist = pos_dist + torch.mean(positive_distance)
            neg_dist = neg_dist + torch.mean(
                torch.min(negative_distance1, negative_distance2))

            has_grad = True
            n_valid_samples += 1

        if not has_grad:
            raise NotImplementedError

        loss = loss / n_valid_samples
        pos_dist = pos_dist / n_valid_samples
        neg_dist = neg_dist / n_valid_samples

        return loss, pos_dist, neg_dist


def grid_positions(h, w, device, matrix=False):
    lines = torch.arange(0, h, device=device).view(-1, 1).float().repeat(1, w)
    columns = torch.arange(
        0, w, device=device).view(1, -1).float().repeat(h, 1)
    if matrix:
        return torch.stack([lines, columns], dim=0)
    else:
        return torch.cat([lines.view(1, -1), columns.view(1, -1)], dim=0)


def upscale_positions(pos, scaling_steps=0):
    for _ in range(scaling_steps):
        pos = pos * 2
    return pos


def downscale_positions(pos, scaling_steps=0):
    for _ in range(scaling_steps):
        pos = pos / 2
    return pos


def warp(pos1, max_h, max_w, transformed_coordinates):
    device = pos1.device
    ids = torch.arange(0, pos1.size(1), device=device)

    transformed_coordinates = transformed_coordinates[::4, ::4, :2]
    # transformed_coordinates = transformed_coordinates[..., :2]
    # dim 0: x, dim 1: y
    pos2 = transformed_coordinates.permute(2, 0, 1).reshape(2, -1)
    transformed_x = pos2[0, :]
    transformed_y = pos2[1, :]

    # eliminate the outlier pixels
    valid_ids_x = torch.min(transformed_x > 10, transformed_x < (max_w - 10))
    valid_ids_y = torch.min(transformed_y > 10, transformed_y < (max_h - 10))

    valid_ids = torch.min(valid_ids_x, valid_ids_y)

    ids = ids[valid_ids]
    pos1 = pos1[:, valid_ids]
    pos2 = pos2[:, valid_ids]

    pos2 = pos2[[1, 0], :]

    return pos1, pos2, ids
