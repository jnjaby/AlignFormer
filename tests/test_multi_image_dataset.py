import math
import os
import torchvision.utils

from basicsr.data import build_dataloader, build_dataset


def main(mode='folder'):
    """Test multiple image dataset.

    Args:
        mode: There are three modes: 'lmdb', 'folder', 'meta_info_file'.
    """
    opt = {}
    opt['dist'] = False
    opt['phase'] = 'train'

    opt['name'] = 'DIV2K'
    opt['type'] = 'TripleImageDataset'
    if mode == 'folder':
        opt['dataroot_gt'] = '/mnt/lustre/rcfeng/UDC/dataset/png_crops/gt/train/'
        opt['dataroot_lq'] = '/mnt/lustre/rcfeng/UDC/dataset/png_crops/lq/train/'
        opt['dataroot_ref'] = '/mnt/lustre/rcfeng/UDC/dataset/png_crops/ref/train/'
        opt['filename_tmpl'] = '{}'
        opt['io_backend'] = dict(type='disk')
    elif mode == 'meta_info_file':
        opt['dataroot_gt'] = '/mnt/lustre/rcfeng/UDC/dataset/png_crops/gt/test/'
        opt['dataroot_lq'] = '/mnt/lustre/rcfeng/UDC/dataset/png_crops/lq/test/'
        opt['dataroot_ref'] = '/mnt/lustre/rcfeng/UDC/dataset/png_crops/ref/test/'
        opt['meta_info_file'] = '/mnt/lustre/rcfeng/UDC/dataset/png_crops/meta_info_subset_test.txt'  # noqa:E501
        opt['filename_tmpl'] = '{}'
        opt['io_backend'] = dict(type='disk')

    opt['gt_size'] = 128
    opt['use_flip'] = True
    opt['use_rot'] = True

    opt['use_shuffle'] = True
    opt['num_worker_per_gpu'] = 2
    opt['batch_size_per_gpu'] = 16
    opt['scale'] = 1

    opt['dataset_enlarge_ratio'] = 1

    os.makedirs('tmp', exist_ok=True)

    dataset = build_dataset(opt)
    data_loader = build_dataloader(dataset, opt, num_gpu=0, dist=opt['dist'], sampler=None)

    nrow = int(math.sqrt(opt['batch_size_per_gpu']))
    padding = 2 if opt['phase'] == 'train' else 0

    print('start...')
    for i, data in enumerate(data_loader):
        if i > 5:
            break
        print(i)

        lq = data['lq']
        gt = data['gt']
        ref = data['ref']
        lq_path = data['lq_path']
        gt_path = data['gt_path']
        ref_path = data['ref_path']
        print(lq_path, gt_path, ref_path)
        torchvision.utils.save_image(lq, f'tmp/lq_{i:03d}.png', nrow=nrow, padding=padding, normalize=False)
        torchvision.utils.save_image(gt, f'tmp/gt_{i:03d}.png', nrow=nrow, padding=padding, normalize=False)
        torchvision.utils.save_image(ref, f'tmp/ref_{i:03d}.png', nrow=nrow, padding=padding, normalize=False)


if __name__ == '__main__':
    main()
