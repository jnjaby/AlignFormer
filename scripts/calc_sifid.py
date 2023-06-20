import argparse
from collections import OrderedDict, defaultdict
import pyiqa
import torch
import cv2
import mmcv
import os

import os
import pathlib
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.functional import adaptive_avg_pool2d
from torchvision import models
from matplotlib.pyplot import imread
import numpy as np
from tqdm import tqdm
from scipy import linalg

import pdb


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=False,
                 normalize_input=True,
                 requires_grad=False):
        """Build pretrained InceptionV3
        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradient. Possibly useful
            for finetuning the network
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
        ]

        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                nn.MaxPool2d(kernel_size=3, stride=2),
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                nn.MaxPool2d(kernel_size=3, stride=2),
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
            ]
            self.blocks.append(nn.Sequential(*block3))

        if self.last_needed_block >= 4:
            block4 = [
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block4))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.upsample(x,
                           size=(299, 299),
                           mode='bilinear',
                           align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp


def get_imglist(root, subset=None):
    if subset:
        img_list = [os.path.join(root, img) for img in sorted(os.listdir(root)) if img in subset]
    else:
        img_list = [os.path.join(root, img) for img in sorted(os.listdir(root))]
    return img_list


def imread(path):
    img = mmcv.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.
    return img.astype(np.float32)


def img2tensor(img):
    return torch.from_numpy(img.transpose(2, 0, 1))[None, ...]


def tensor2img(tensor):
    if tensor.dim() == 4:
        return tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    elif tensor.dim() == 3:
        return tensor.squeeze().cpu().numpy()
    else:
        raise ValueError('Dimension of tensor should be 3 or 4.')


def get_activations(files, model, batch_size=1, dims=64,
                    cuda=False, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if len(files) % batch_size != 0:
        print(('Warning: number of images is not a multiple of the '
               'batch size. Some samples are going to be ignored.'))
    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    n_batches = len(files) // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, dims))

    for i in range(n_batches):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)
        start = i * batch_size
        end = start + batch_size

        images = np.array([imread(str(f)).astype(np.float32)
                           for f in files[start:end]])

        images = images[:, :, :, 0:3]
        # Reshape to (n_images, 3, height, width)
        images = images.transpose((0, 3, 1, 2))
        #images = images[0,:,:,:]
        images /= 255

        batch = torch.from_numpy(images).type(torch.FloatTensor)
        if cuda:
            batch = batch.cuda()

        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.

        # if pred.shape[2] != 1 or pred.shape[3] != 1:
        #    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr = pred.cpu().data.numpy().transpose(0, 2, 3, 1).reshape(batch_size * pred.shape[2] * pred.shape[3], -1)

    if verbose:
        print(' done')

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=1,
                                    dims=64, cuda=False, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the inception model.
    -- sigma : The covariance matrix of the activations of the inception model.
    """
    act = get_activations(files, model, batch_size, dims, cuda, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def _compute_statistics_of_path(files, model, batch_size, dims, cuda):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        path = pathlib.Path(path)
        files = sorted(list(path.glob('*.jpg')) + list(path.glob('*.png')))
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, cuda)

    return m, s


def calculate_sifid_given_paths(path1, path2, batch_size, dims, cuda=True, suffix='png', subset=None):
    """Calculates the SIFID of two paths"""

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()

    # path1 = pathlib.Path(path1)
    files1 = get_imglist(path1, subset)
    files2 = get_imglist(path2, subset)

    assert len(files1) == len(files2)
    # path2 = pathlib.Path(path2)
    # files2 = sorted(list(path2.glob('*.%s' % suffix)))

    # pdb.set_trace()
    fid_values = []
    # Im_ind = []
    for i in tqdm(range(len(files2))):
        m1, s1 = calculate_activation_statistics([files1[i]], model, batch_size, dims, cuda)
        m2, s2 = calculate_activation_statistics([files2[i]], model, batch_size, dims, cuda)
        fid_values.append(calculate_frechet_distance(m1, s1, m2, s2))
        # file_num1 = files1[i].name
        # file_num2 = files2[i].name
        # Im_ind.append(int(file_num1[:-4]))
        # Im_ind.append(int(file_num2[:-4]))
    return fid_values


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()

    # parser.add_argument('--id', type=str, help='File id')
    # parser.add_argument('--output', type=str, help='Save path')
    # args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sub_set = sorted(os.listdir('/mnt/lustre/rcfeng/UDC/dataset/iphone_pair_sub300/output_165/test_sub'))
    print(f'{len(sub_set)} images in subset.')
    ref_path = '/mnt/lustre/rcfeng/UDC/dataset/iphone_pair_sub300/ref/test_sub/'
    # model_dict = {
    #     '230': '/mnt/lustre/rcfeng/CVPR23/results/Ablation/230',
    #     '232': '/mnt/lustre/rcfeng/CVPR23/results/Ablation/232',
    #     '201': '/mnt/lustre/rcfeng/CVPR23/results/Ablation/201',
    #     '220': '/mnt/lustre/rcfeng/CVPR23/results/Ablation/220',
    #     'PPMUNet_212': '/mnt/lustre/rcfeng/CVPR23/results/Ours/PPM-UNet_212/',
    #     'PPMUNet_214': '/mnt/lustre/rcfeng/CVPR23/results/Ours/PPM-UNet_214/',
    # }

    model_dict = {
        '230': '/mnt/lustre/rcfeng/CVPR23/results/Ablation/230/',
        '232': '/mnt/lustre/rcfeng/CVPR23/results/Ablation/232/',
        '201': '/mnt/lustre/rcfeng/CVPR23/results/Ablation/201/',
        '220': '/mnt/lustre/rcfeng/CVPR23/results/Ablation/220/',
    #     'PPMUNet_212': '/mnt/lustre/rcfeng/CVPR23/results/Ours/PPM-UNet_212/',
        'DISCNet': '/mnt/lustre/rcfeng/CVPR23/results/DISCNet_pretrain/',
        'BNUDC': '/mnt/lustre/rcfeng/CVPR23/results/BNUDC_pretrain/',
        'MUNIT': '/mnt/lustre/rcfeng/CVPR23/results/MUNIT/',
        'TSIT': '/mnt/lustre/rcfeng/CVPR23/results/TSIT/',
        'A_DISCNet': '/mnt/lustre/rcfeng/CVPR23/results/DISCNet_retrain/',
        'A_BNUDC': '/mnt/lustre/rcfeng/CVPR23/results/BNUDC_retrain/',
        'PPMUNet_214': '/mnt/lustre/rcfeng/CVPR23/results/Ours/PPM-UNet_214/',
        '234': '/mnt/lustre/rcfeng/CVPR23/results/Ablation/234/'
    }

    sifid_dict = OrderedDict()
    for model_name, root_path in model_dict.items():
        # imgs = get_imglist(root_path)
        # assert len(imgs) == 336

        sifid_values = calculate_sifid_given_paths(ref_path, root_path, 1, 64, subset=sub_set)

        sifid_values = np.asarray(sifid_values, dtype=np.float32)
        sifid_dict[model_name] = sifid_values
        # numpy.save('SIFID', sifid_values)
        print(f'{model_name} >>> {sifid_values.mean()}')

    # pdb.set_trace()
