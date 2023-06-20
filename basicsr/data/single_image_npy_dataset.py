import mmcv
import numpy as np
from os import path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.transforms import augment, random_crop
from basicsr.utils import FileClient, img2tensor


class SingleNpyDataset(data.Dataset):
    """Read only lq images in the test phase.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc).

    There are two modes:
    1. 'meta_info_file': Use meta information file to generate paths.
    2. 'folder': Scan folders to generate paths.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for annotation file.
            io_backend (dict): IO backend type and other kwarg.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.lq_folder = opt['dataroot_lq']
        if 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [
                    osp.join(self.lq_folder,
                             line.split(' ')[0]) for line in fin
                ]
        else:
            self.paths = [
                osp.join(self.lq_folder, v)
                for v in mmcv.scandir(self.lq_folder)
            ]

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

        map_type = self.opt['map_type']

        # load lq image
        # HDR image range: [0, +inf], float32.
        lq_path = self.paths[index]
        img_lq = self.file_client.get(lq_path)

        # tone mapping
        img_lq = self._tonemap(img_lq, type=map_type)

        # augmentation
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_lq = random_crop(img_lq, gt_size)
            # flip, rotation
            img_lq = augment([img_lq], self.opt['use_flip'],
                             self.opt['use_rot'])

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        # print(img_lq.shape)
        img_lq = img2tensor(img_lq, bgr2rgb=False, float32=True)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'lq_path': lq_path}

    def __len__(self):
        return len(self.paths)
