import math
import mmcv
import torchvision.utils

from basicsr.data import create_dataloader, create_dataset


def main(mode='meta_info_file'):
    """Test paired image dataset.

    Args:
        mode: There are three modes: 'lmdb', 'folder', 'meta_info_file'.
    """
    opt = {}
    opt['dist'] = False
    opt['phase'] = 'train'

    opt['name'] = 'HDR800'
    opt['type'] = 'PairedImgPSFNpyDataset'
    opt['lq_map_type'] = 'simple'
    opt['gt_map_type'] = 'simple'

    if mode == 'meta_info_file':
        opt['folders'] = dict(rot_5={})
        # opt['folders']['rot_5'] = {}
        opt['folders']['rot_5']['dataroot_gt'] = '/mnt/lustre/rcfeng/UDC/dataset/npy_crops/gt_500/train/'
        opt['folders']['rot_5']['dataroot_lq'] = '/mnt/lustre/rcfeng/UDC/dataset/npy_crops/sim_500/ZTE_new_5/train/'
        opt['folders']['rot_5']['meta_info_file'] = '/mnt/lustre/rcfeng/UDC/dataset/kernel_info_list/ZTE_new/ZTE_new_code_5_train.txt'
        opt['filename_tmpl'] = '{}'
        opt['io_backend'] = dict(type='npy')

    opt['gt_size'] = 512
    opt['use_flip'] = False
    opt['use_rot'] = False

    opt['use_shuffle'] = False
    opt['num_worker_per_gpu'] = 2
    opt['batch_size_per_gpu'] = 16
    opt['scale'] = 1

    opt['dataset_enlarge_ratio'] = 1

    mmcv.mkdir_or_exist('tmp')

    dataset = create_dataset(opt)
    data_loader = create_dataloader(
        dataset, opt, num_gpu=0, dist=opt['dist'], sampler=None)

    nrow = int(math.sqrt(opt['batch_size_per_gpu']))
    padding = 2 if opt['phase'] == 'train' else 0

    print('start...')
    for i, data in enumerate(data_loader):
        if i > 5:
            break
        print(i)

        lq = data['lq']
        gt = data['gt']
        lq_path = data['lq_path']
        gt_path = data['gt_path']
        print(lq_path, gt_path)
        torchvision.utils.save_image(
            lq,
            f'tmp/lq_{i:03d}.png',
            nrow=nrow,
            padding=padding,
            normalize=False)
        torchvision.utils.save_image(
            gt,
            f'tmp/gt_{i:03d}.png',
            nrow=nrow,
            padding=padding,
            normalize=False)


if __name__ == '__main__':
    main()
