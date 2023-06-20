import mmcv
import numpy as np
import pdb
from torch.utils import data as data

from basicsr.data.transforms import augment, multiple_random_crop, totensor
from basicsr.data.util import (multiple_paths_from_meta_info_file,
                               paired_paths_from_folder,
                               paired_paths_from_lmdb)
from basicsr.utils import FileClient


class TripleNpyDataset(data.Dataset):
    """Triple image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc)
    and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal and vertical flips.
            use_rot (bool): Use rotation (use transposing h and w for
                implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder, self.lq_folder, self.atn_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_atn']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = multiple_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder, self.atn_folder], ['lq', 'gt', 'atn'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

    def _tonemap(self, x, type='simple'):
        if type == 'mu_law':
            norm_x = x / x.max()
            mapped_x = np.log(1 + 10000 * norm_x) / np.log(1 + 10000)
        elif type == 'simple':
            mapped_x = x / (x + 0.25)
        elif type == 'same':
            mapped_x = x
        else:
            raise NotImplementedError('tone mapping type [{:s}] is not recognized.'.format(type))
        return mapped_x

        
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        map_type = self.opt['map_type']

        # Load gt and lq images. Dimension order: HWC; channel order: RGGB;
        # HDR image range: [0, +inf], float32.
        gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']
        atn_path = self.paths[index]['atn_path']
        img_gt = self.file_client.get(gt_path)
        img_lq = self.file_client.get(lq_path)
        img_atn = self.file_client.get(atn_path)

        # tone mapping
        img_gt = self._tonemap(img_gt, type=map_type)
        img_lq = self._tonemap(img_lq, type=map_type)
        img_atn = img_atn[..., None]

        # augmentation
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq, img_atn = multiple_random_crop(img_gt, img_lq, img_atn, gt_size, scale,
                                                gt_path)
            # flip, rotation
            img_gt, img_lq, img_atn = augment([img_gt, img_lq, img_atn], self.opt['use_flip'],
                                     self.opt['use_rot'])

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        # img_gt, img_lq, img_atn = totensor([img_gt, img_lq, img_atn], bgr2rgb=False, float32=True)

        # return {
        #     'lq': img_lq,
        #     'gt': img_gt,
        #     'atn': img_atn,
        #     'lq_path': lq_path,
        #     'gt_path': gt_path,
        #     'atn_path': atn_path
        # }

        # concat attention map into input.
        img_gt, img_lq = totensor([img_gt, np.concatenate((img_lq, img_atn), axis=2)], bgr2rgb=False, float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path,
        }

    def __len__(self):
        return len(self.paths)
