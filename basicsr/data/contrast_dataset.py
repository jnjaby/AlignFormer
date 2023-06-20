from logging import raiseExceptions
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_meta_info_file,
                                    paired_paths_from_folder)
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

import numpy as np
import cv2


def center_crop(data, dim):
    """
    Crops center H, W dimensions to match desired dimensions if input dimension is greater.

    Args:
        data (np.array): single channel or 3-channel image array. 
        dim (int, int): Desired dimensions for H, W. i.e dim = (H, W)

    Returns:
        np.array: Input image matched to new dimensions.
    """
    h_start, w_start = [max(data.shape[0] - dim[0], 0) // 2,
                        max(data.shape[1] - dim[1], 0) // 2]
    h_end, w_end = [h_start + min(dim[0], data.shape[0]),
                    w_start + min(dim[1], data.shape[1])]
    return data[h_start:h_end, w_start:w_end]


def image_pair_generation(img,
                          random_perturb_range=(0, 32),
                          cropping_window_size=160):

    if img is not None:
        shape1 = img.shape
        h = shape1[0]
        w = shape1[1]
    else:
        h = 160
        w = 160

    # ===== in image-1
    cropS = cropping_window_size
    x_topleft = np.random.randint(random_perturb_range[1],
                                  max(w, w - cropS - random_perturb_range[1]))
    y_topleft = np.random.randint(random_perturb_range[1],
                                  max(h, h - cropS - random_perturb_range[1]))

    x_topright = x_topleft + cropS
    y_topright = y_topleft

    x_bottomleft = x_topleft
    y_bottomleft = y_topleft + cropS

    x_bottomright = x_topleft + cropS
    y_bottomright = y_topleft + cropS

    tl = (x_topleft, y_topleft)
    tr = (x_topright, y_topright)
    br = (x_bottomright, y_bottomright)
    bl = (x_bottomleft, y_bottomleft)

    rect1 = np.array([tl, tr, br, bl], dtype=np.float32)

    # ===== in image-2
    x2_topleft = x_topleft + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])
    y2_topleft = y_topleft + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])

    x2_topright = x_topright + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])
    y2_topright = y_topright + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])

    x2_bottomleft = x_bottomleft + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])
    y2_bottomleft = y_bottomleft + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])

    x2_bottomright = x_bottomright + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])
    y2_bottomright = y_bottomright + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])

    tl2 = (x2_topleft, y2_topleft)
    tr2 = (x2_topright, y2_topright)
    br2 = (x2_bottomright, y2_bottomright)
    bl2 = (x2_bottomleft, y2_bottomleft)

    rect2 = np.array([tl2, tr2, br2, bl2], dtype=np.float32)

    # ===== homography
    H = cv2.getPerspectiveTransform(src=rect1, dst=rect2)
    H_inverse = np.linalg.inv(H)

    if img is not None:
        img_warped = cv2.warpPerspective(src=img, M=H_inverse, dsize=(w, h))
        return img_warped, H, H_inverse
    else:
        return H_inverse


@DATASET_REGISTRY.register()
class ContrastDataset(data.Dataset):
    """Triple image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT, Reference image triplet.

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
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            # self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder, self.gt_folder]
            # self.io_backend_opt['client_keys'] = ['lq', 'gt']
            # self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
            raise NotImplementedError(f'backend type lmdb not implemented yet.')
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

        # rect1 = np.array([[50, 50],
        #                 [50, 750],
        #                 [750, 750],
        #                 [750, 50]]).astype(np.float32)

        # rect2 = np.array([[66, 68],
        #                 [74, 730],
        #                 [776, 772],
        #                 [742, 46],]).astype(np.float32)

        # # ===== homography
        # H = cv2.getPerspectiveTransform(src=rect1, dst=rect2)
        # H_inverse = np.linalg.inv(H)

        # grid_x, grid_y = np.meshgrid(np.arange(700) + 50,
        #                             np.arange(700) + 50)
        # grid_z = np.ones(grid_x.shape)

        # coordinate = np.stack((grid_x, grid_y, grid_z), axis=0).reshape((3, -1))

        # transformed_coordinate = np.dot(H_inverse, coordinate)
        # transformed_coordinate /= transformed_coordinate[2, :]
        # # the transformed coordinates of the original image
        # transformed_coordinate = transformed_coordinate.transpose(1, 0)
        # transformed_coordinate = np.rint(transformed_coordinate).astype(np.int)[:, :2] - 50
        # self.trans_coor = transformed_coordinate

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq, = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)

            # flip, rotation
            img_gt, img_lq, = augment([img_gt, img_lq], self.opt['use_flip'], self.opt['use_rot'])
        else:
            img_gt = center_crop(img_gt, (512, 512))
            img_lq = center_crop(img_lq, (512, 512))

        h, w, _ = img_gt.shape
        # image pair generation
        img_transformed, H, H_inverse = image_pair_generation(
            img_gt, (0, 10), int(min(h, w) * 0.75))

        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        grid_z = np.ones(grid_x.shape)

        coordinate = np.stack((grid_x, grid_y, grid_z), axis=0).reshape((3, -1))
        transformed_coordinate = np.dot(H_inverse, coordinate)
        transformed_coordinate /= transformed_coordinate[2, :]
        # the transformed coordinates of the original image
        transformed_coordinate = transformed_coordinate.transpose(
            1, 0).reshape(h, w, 3)

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_transformed, img_lq = img2tensor([img_transformed, img_lq],
                                             bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_transformed, self.mean, self.std, inplace=True)

        return {'lq': img_lq,
                'ref': img_transformed,
                'transformed_coordinate': transformed_coordinate,
                'lq_path': lq_path,
                'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
