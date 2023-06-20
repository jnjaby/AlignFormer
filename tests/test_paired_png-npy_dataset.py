import math
import mmcv
import torchvision.utils

from basicsr.data import build_dataloader, build_dataset


def main(mode='folder'):
    """Test paired image dataset.

    Args:
        mode: There are three modes: 'lmdb', 'folder', 'meta_info_file'.
    """
    opt = {}
    opt['dist'] = False
    opt['phase'] = 'train'

    opt['name'] = 'ip110'
    opt['type'] = 'PairedNpyPngDataset'

    if mode == 'meta_info_file':
        opt['folders'] = dict(rot_5={})
        # opt['folders']['rot_5'] = {}
        opt['folders']['rot_5']['dataroot_gt'] = '../../UDC/dataset/npy_crops/gt_500/train/'
        opt['folders']['rot_5']['dataroot_lq'] = '../../UDC/dataset/npy_crops/sim_500/ZTE_new_5/train/'
        opt['folders']['rot_5']['meta_info_file'] = '../../UDC/dataset/kernel_info_list/ZTE_new/ZTE_new_code_5_train.txt'
        opt['filename_tmpl'] = '{}'
        opt['io_backend'] = dict(type='npy')
    elif mode == 'folder':
        opt['dataroot_gt'] = '../../UDC/dataset/iphone_pair_sub110/output_117/train/'
        opt['dataroot_lq'] = '../../UDC/dataset/iphone_pair_sub110/lq_npy/train/'
        opt['io_backend'] = dict(type='disk')

    opt['gt_size'] = 256
    opt['use_flip'] = True
    opt['use_rot'] = True

    opt['use_shuffle'] = False
    opt['num_worker_per_gpu'] = 0
    opt['batch_size_per_gpu'] = 4
    opt['scale'] = 1

    opt['dataset_enlarge_ratio'] = 1

    mmcv.mkdir_or_exist('tmp')

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
