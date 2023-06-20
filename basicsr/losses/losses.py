import math
import pdb
import cv2
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.vgg_arch import VGGFeatureExtractor
# from basicsr.archs.pwcnet_arch import FlowGenerator
from basicsr.archs.raft_arch import FlowGenerator
from basicsr.archs.arch_util import flow_warp
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


@weighted_loss
def modified_mse_loss(pred, target, eps=1e-3):
    # weight = target.detach()
    weight = pred.detach()
    return ((pred - target) / (weight + eps))**2


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class ModifiedMSELoss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(ModifiedMSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * modified_mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class GradientLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(GradientLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

        kernel_v = torch.FloatTensor(
            [[0, -1, 0], [0, 0, 0], [0, 1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_h = torch.FloatTensor(
            [[0, 0, 0], [-1, 0, 1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)

        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)

    def forward(self, pred, target, weight=None, **kwargs):
        pred_0_v = F.conv2d(pred[:, 0:1], self.weight_v, padding=1)
        pred_0_h = F.conv2d(pred[:, 0:1], self.weight_h, padding=1)
        pred_1_v = F.conv2d(pred[:, 1:2], self.weight_v, padding=1)
        pred_1_h = F.conv2d(pred[:, 1:2], self.weight_h, padding=1)
        pred_2_v = F.conv2d(pred[:, 2:3], self.weight_v, padding=1)
        pred_2_h = F.conv2d(pred[:, 2:3], self.weight_h, padding=1)

        pred_0 = torch.sqrt(torch.pow(pred_0_v, 2) + torch.pow(pred_0_h, 2) + 1e-6)
        pred_1 = torch.sqrt(torch.pow(pred_1_v, 2) + torch.pow(pred_1_h, 2) + 1e-6)
        pred_2 = torch.sqrt(torch.pow(pred_2_v, 2) + torch.pow(pred_2_h, 2) + 1e-6)
        pred_grad = torch.cat([pred_0, pred_1, pred_2], dim=1)

        target_0_v = F.conv2d(target[:, 0:1], self.weight_v, padding=1)
        target_0_h = F.conv2d(target[:, 0:1], self.weight_h, padding=1)
        target_1_v = F.conv2d(target[:, 1:2], self.weight_v, padding=1)
        target_1_h = F.conv2d(target[:, 1:2], self.weight_h, padding=1)
        target_2_v = F.conv2d(target[:, 2:3], self.weight_v, padding=1)
        target_2_h = F.conv2d(target[:, 2:3], self.weight_h, padding=1)

        target_0 = torch.sqrt(torch.pow(target_0_v, 2) + torch.pow(target_0_h, 2) + 1e-6)
        target_1 = torch.sqrt(torch.pow(target_1_v, 2) + torch.pow(target_1_h, 2) + 1e-6)
        target_2 = torch.sqrt(torch.pow(target_2_v, 2) + torch.pow(target_2_h, 2) + 1e-6)
        target_grad = torch.cat([target_0, target_1, target_2], dim=1)

        return self.loss_weight * l1_loss(pred_grad, target_grad,
                                          weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class SpatialLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(SpatialLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

        kernel_left = torch.FloatTensor(
            [[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor(
            [[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor(
            [[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor(
            [[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(
            data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, pred, target, weight=None, **kwargs):
        pred_mean = torch.mean(pred, 1, keepdim=True)
        target_mean = torch.mean(target, 1, keepdim=True)

        pred_pool = self.pool(pred_mean)
        target_pool = self.pool(target_mean)

        D_pred_letf = F.conv2d(pred_pool, self.weight_left, padding=1)
        D_pred_right = F.conv2d(pred_pool, self.weight_right, padding=1)
        D_pred_up = F.conv2d(pred_pool, self.weight_up, padding=1)
        D_pred_down = F.conv2d(pred_pool, self.weight_down, padding=1)

        D_target_letf = F.conv2d(target_pool, self.weight_left, padding=1)
        D_target_right = F.conv2d(target_pool, self.weight_right, padding=1)
        D_target_up = F.conv2d(target_pool, self.weight_up, padding=1)
        D_target_down = F.conv2d(target_pool, self.weight_down, padding=1)

        D_left = torch.pow(D_pred_letf - D_target_letf, 2)
        D_right = torch.pow(D_pred_right - D_target_right, 2)
        D_up = torch.pow(D_pred_up - D_target_up, 2)
        D_down = torch.pow(D_pred_down - D_target_down, 2)
        E = D_left + D_right + D_up + D_down

        return self.loss_weight * torch.mean(E)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

        Args:
            loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super(WeightedTVLoss, self).forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super(WeightedTVLoss, self).forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss


@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram


@LOSS_REGISTRY.register()
class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0, reduction='mean'):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss(reduction=reduction)
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss(reduction=reduction)
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.

        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight


@LOSS_REGISTRY.register()
class MultiScaleGANLoss(GANLoss):
    """
    MultiScaleGANLoss accepts a list of predictions
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(MultiScaleGANLoss, self).__init__(gan_type, real_label_val, fake_label_val, loss_weight)

    def forward(self, input, target_is_real, is_disc=False):
        """
        The input is a list of tensors, or a list of (a list of tensors)
        """
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    # Only compute GAN loss for the last layer
                    # in case of multiscale feature matching
                    pred_i = pred_i[-1]
                # Safe operaton: 0-dim tensor calling self.mean() does nothing
                loss_tensor = super().forward(pred_i, target_is_real, is_disc).mean()
                loss += loss_tensor
            return loss / len(input)
        else:
            return super().forward(input, target_is_real, is_disc)


def r1_penalty(real_pred, real_img):
    """R1 regularization for discriminator. The core idea is to
        penalize the gradient on real data alone: when the
        generator distribution produces the true data distribution
        and the discriminator is equal to 0 on the data manifold, the
        gradient penalty ensures that the discriminator cannot create
        a non-zero gradient orthogonal to the data manifold without
        suffering a loss in the GAN game.

        Ref:
        Eq. 9 in Which training methods for GANs do actually converge.
        """
    grad_real = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)[0]
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(fake_img.shape[2] * fake_img.shape[3])
    grad = autograd.grad(outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)[0]
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_lengths.detach().mean(), path_mean.detach()


def gradient_penalty_loss(discriminator, real_data, fake_data, weight=None):
    """Calculate gradient penalty for wgan-gp.

    Args:
        discriminator (nn.Module): Network for the discriminator.
        real_data (Tensor): Real input data.
        fake_data (Tensor): Fake input data.
        weight (Tensor): Weight tensor. Default: None.

    Returns:
        Tensor: A tensor for gradient penalty.
    """

    batch_size = real_data.size(0)
    alpha = real_data.new_tensor(torch.rand(batch_size, 1, 1, 1))

    # interpolate between real_data and fake_data
    interpolates = alpha * real_data + (1. - alpha) * fake_data
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    if weight is not None:
        gradients = gradients * weight

    gradients_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
    if weight is not None:
        gradients_penalty /= torch.mean(weight)

    return gradients_penalty


@LOSS_REGISTRY.register()
class GANFeatLoss(nn.Module):
    """Define feature matching loss for gans

    Args:
        criterion (str): Support 'l1', 'l2', 'charbonnier'.
        loss_weight (float): Loss weight. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, criterion='l1', loss_weight=1.0, reduction='mean'):
        super(GANFeatLoss, self).__init__()
        if criterion == 'l1':
            self.loss_op = L1Loss(loss_weight, reduction)
        elif criterion == 'l2':
            self.loss_op = MSELoss(loss_weight, reduction)
        elif criterion == 'charbonnier':
            self.loss_op = CharbonnierLoss(loss_weight, reduction)
        else:
            raise ValueError(f'Unsupported loss mode: {criterion}. ' f'Supported ones are: l1|l2|charbonnier')

        self.loss_weight = loss_weight

    def forward(self, pred_fake, pred_real):
        num_D = len(pred_fake)
        loss = 0
        for i in range(num_D):  # for each discriminator
            # last output is the final prediction, exclude it
            num_intermediate_outputs = len(pred_fake[i]) - 1
            for j in range(num_intermediate_outputs):  # for each layer output
                unweighted_loss = self.loss_op(pred_fake[i][j], pred_real[i][j].detach())
                loss += unweighted_loss / num_D
        return loss * self.loss_weight


def compute_cx(dist_tilde, band_width):
    w = torch.exp((1 - dist_tilde) / band_width)  # Eq(3)
    cx = w / torch.sum(w, dim=2, keepdim=True)  # Eq(4)
    return cx


def compute_relative_distance(dist_raw):
    dist_min, _ = torch.min(dist_raw, dim=2, keepdim=True)
    dist_tilde = dist_raw / (dist_min + 1e-5)
    return dist_tilde


def compute_cosine_distance(x, y):
    # mean shifting by channel-wise mean of `y`.
    y_mu = y.mean(dim=(0, 2, 3), keepdim=True)
    x_centered = x - y_mu
    y_centered = y - y_mu

    # L2 normalization
    x_normalized = F.normalize(x_centered, p=2, dim=1)
    y_normalized = F.normalize(y_centered, p=2, dim=1)

    # channel-wise vectorization
    N, C, *_ = x.size()
    x_normalized = x_normalized.reshape(N, C, -1)  # (N, C, H*W)
    y_normalized = y_normalized.reshape(N, C, -1)  # (N, C, H*W)

    # consine similarity
    cosine_sim = torch.bmm(x_normalized.transpose(1, 2),
                           y_normalized)  # (N, H*W, H*W)

    # convert to distance
    dist = 1 - cosine_sim
    return dist


def compute_l1_distance(x, y):
    N, C, H, W = x.size()
    x_vec = x.view(N, C, -1)
    y_vec = y.view(N, C, -1)

    dist = x_vec.unsqueeze(2) - y_vec.unsqueeze(3)
    dist = dist.sum(dim=1).abs()
    dist = dist.transpose(1, 2).reshape(N, H * W, H * W)
    dist = dist.clamp(min=0.)

    return dist


def compute_l2_distance(x, y):
    N, C, H, W = x.size()
    x_vec = x.view(N, C, -1)
    y_vec = y.view(N, C, -1)
    x_s = torch.sum(x_vec ** 2, dim=1, keepdim=True)
    y_s = torch.sum(y_vec ** 2, dim=1, keepdim=True)

    A = y_vec.transpose(1, 2) @ x_vec
    dist = y_s - 2 * A + x_s.transpose(1, 2)
    dist = dist.transpose(1, 2).reshape(N, H * W, H * W)
    dist = dist.clamp(min=0.)

    return dist


def compute_meshgrid(shape):
    N, C, H, W = shape
    rows = torch.arange(0, H, dtype=torch.float32) / (H + 1)
    cols = torch.arange(0, W, dtype=torch.float32) / (W + 1)

    feature_grid = torch.meshgrid(rows, cols)
    feature_grid = torch.stack(feature_grid).unsqueeze(0)
    feature_grid = torch.cat([feature_grid for _ in range(N)], dim=0)

    return feature_grid


@weighted_loss
def contextual_loss(pred, target, band_width=0.5, loss_type='cosine'):
    """
    Computes contepredtual loss between pred and target.
    Parameters
    ---
    pred : torch.Tensor
        features of shape (N, C, H, W).
    target : torch.Tensor
        features of shape (N, C, H, W).
    band_width : float, optional
        a band-width parameter used to convert distance to similarity.
        in the paper, this is described as :math:`h`.
    loss_type : str, optional
        a loss type to measure the distance between features.
        Note: `l1` and `l2` frequently raises OOM.
    Returns
    ---
    cx_loss : torch.Tensor
        contextual loss between x and y (Eq (1) in the paper)
    """

    assert pred.size() == target.size(), 'input tensor must have the same size.'
    assert loss_type in ['cosine', 'l1', 'l2'], f"select a loss type from \
                                                {['cosine', 'l1', 'l2']}."

    N, C, H, W = pred.size()

    if loss_type == 'cosine':
        dist_raw = compute_cosine_distance(pred, target)
    elif loss_type == 'l1':
        dist_raw = compute_l1_distance(pred, target)
    elif loss_type == 'l2':
        dist_raw = compute_l2_distance(pred, target)

    dist_tilde = compute_relative_distance(dist_raw)
    cx = compute_cx(dist_tilde, band_width)
    cx = torch.mean(torch.max(cx, dim=1)[0], dim=1)  # Eq(1)
    cx_loss = -torch.log(cx + 1e-5)  # Eq(5)
    # pdb.set_trace()
    return cx_loss


@LOSS_REGISTRY.register()
class ContextualLoss(nn.Module):
    """
    Creates a criterion that measures the contextual loss.

    Args
    ---
    band_width : int, optional
        a band_width parameter described as :math:`h` in the paper.
    use_vgg : bool, optional
        if you want to use VGG feature, set this `True`.
    vgg_layer : str, optional
        intermidiate layer name for VGG feature.
        Now we support layer names:
            `['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']`
    """

    def __init__(self, band_width=0.5, loss_type='cosine',
                 use_vgg=True, vgg_layer='conv4_4',
                 loss_weight=1.0, reduction='mean'):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')
        if loss_type not in ['cosine', 'l1', 'l2']:
            raise ValueError(f'Unsupported loss mode: {reduction}.')

        assert band_width > 0, 'band_width parameter must be positive.'

        self.band_width = band_width
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.loss_type = loss_type

        if use_vgg:
            self.vgg_model = VGGFeatureExtractor(
                layer_name_list=[vgg_layer],
                vgg_type='vgg19')

    def forward(self, pred, target, weight=None, **kwargs):
        assert hasattr(self, 'vgg_model'), 'Please specify VGG model.'
        assert pred.shape[1] == 3 and target.shape[1] == 3,\
            'VGG model takes 3 chennel images.'

        # picking up vgg feature maps
        pred_features = self.vgg_model(pred)
        target_features = self.vgg_model(target.detach())

        cx_loss = 0
        for k in pred_features.keys():
            # cx_loss += contextual_loss(pred_features[k], target_features[k],
            #                            band_width=self.band_width, loss_type=self.loss_type,
            #                            weight=weight, reduction=self.reduction)
            if weight != None:
                scaled_weight = F.interpolate(weight, size=target_features[k].shape[2:])
            else:
                scaled_weight = None
            cx_loss += contextual_loss(target_features[k], pred_features[k],
                                       band_width=self.band_width, loss_type=self.loss_type,
                                       weight=None, reduction=self.reduction)

        cx_loss *= self.loss_weight
        return cx_loss


@weighted_loss
def cobi_loss(pred, target, weight_sp=0.1, band_width=0.5, loss_type='cosine'):
    """
    Computes CoBi loss between pred and target.
    Parameters
    ---
    pred : torch.Tensor
        features of shape (N, C, H, W).
    target : torch.Tensor
        features of shape (N, C, H, W).
    weight_sp : float, optional
        a balancing weight between spatial and feature loss.
    band_width : float, optional
        a band-width parameter used to convert distance to similarity.
        in the paper, this is described as :math:`h`.
    loss_type : str, optional
        a loss type to measure the distance between features.
        Note: `l1` and `l2` frequently raises OOM.
    Returns
    ---
    cx_loss : torch.Tensor
        contextual loss between x and y (Eq (1) in the paper)
    """

    assert pred.size() == target.size(), 'input tensor must have the same size.'
    assert loss_type in ['cosine', 'l1', 'l2'], f"select a loss type from \
                                                {['cosine', 'l1', 'l2']}."

    N, C, H, W = pred.size()

    # spatial loss
    grid = compute_meshgrid(pred.shape).to(pred.device)
    dist_raw = compute_l2_distance(grid, grid)
    dist_tilde = compute_relative_distance(dist_raw)
    cx_sp = compute_cx(dist_tilde, band_width)

    # feature loss
    if loss_type == 'cosine':
        dist_raw = compute_cosine_distance(pred, target)
    elif loss_type == 'l1':
        dist_raw = compute_l1_distance(pred, target)
    elif loss_type == 'l2':
        dist_raw = compute_l2_distance(pred, target)
    dist_tilde = compute_relative_distance(dist_raw)
    cx_feat = compute_cx(dist_tilde, band_width)

    # combine loss
    cx_combine = (1 - weight_sp) * cx_feat + weight_sp * cx_sp

    k_max_NC, _ = torch.max(cx_combine, dim=2, keepdim=True)

    cx = k_max_NC.mean(dim=1)
    cx_loss = -torch.log(cx + 1e-5)

    return cx_loss


@LOSS_REGISTRY.register()
class CoBiLoss(nn.Module):
    """
    Creates a criterion that measures the boci loss.
    """

    def __init__(self, band_width=0.5, weight_sp=0.1, loss_type='cosine',
                 use_vgg=True, vgg_layer='conv4_4',
                 loss_weight=1.0, reduction='mean'):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')
        if loss_type not in ['cosine', 'l1', 'l2']:
            raise ValueError(f'Unsupported loss mode: {reduction}.')

        assert band_width > 0, 'band_width parameter must be positive.'
        assert weight_sp >= 0 and weight_sp <= 1, 'weight_sp out of range [0, 1].'

        self.band_width = band_width
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.loss_type = loss_type
        self.weight_sp = weight_sp

        if use_vgg:
            self.vgg_model = VGGFeatureExtractor(
                layer_name_list=[vgg_layer],
                vgg_type='vgg19')

    def forward(self, pred, target, weight=None, **kwargs):
        assert hasattr(self, 'vgg_model'), 'Please specify VGG model.'
        assert pred.shape[1] == 3 and target.shape[1] == 3,\
            'VGG model takes 3 chennel images.'

        # picking up vgg feature maps
        pred_features = self.vgg_model(pred)
        target_features = self.vgg_model(target.detach())

        cx_loss = 0
        for k in pred_features.keys():
            cx_loss += cobi_loss(pred_features[k], target_features[k],
                                 weight_sp=self.weight_sp, band_width=self.band_width,
                                 loss_type=self.loss_type, weight=weight, reduction=self.reduction)

        cx_loss *= self.loss_weight
        return cx_loss


@weighted_loss
def mask_contextual_loss(pred, target, mask, band_width=0.5, loss_type='cosine'):
    """
    Computes contepredtual loss between pred and target.
    Parameters
    ---
    pred : torch.Tensor
        features of shape (N, C, H, W).
    target : torch.Tensor
        features of shape (N, C, H, W).
    band_width : float, optional
        a band-width parameter used to convert distance to similarity.
        in the paper, this is described as :math:`h`.
    loss_type : str, optional
        a loss type to measure the distance between features.
        Note: `l1` and `l2` frequently raises OOM.
    Returns
    ---
    cx_loss : torch.Tensor
        contextual loss between x and y (Eq (1) in the paper)
    """

    assert pred.size() == target.size(), 'input tensor must have the same size.'
    # assert pred.size() == mask.size(), 'input tensor must have the same size.'
    assert loss_type in ['cosine', 'l1', 'l2'], f"select a loss type from \
                                                {['cosine', 'l1', 'l2']}."

    N, C, H, W = pred.size()

    mask = F.interpolate(mask[:, None, ...], size=(H, W), mode='bilinear', align_corners=True)

    if loss_type == 'cosine':
        dist_raw = compute_cosine_distance(pred, target)
    elif loss_type == 'l1':
        dist_raw = compute_l1_distance(pred, target)
    elif loss_type == 'l2':
        dist_raw = compute_l2_distance(pred, target)

    dist_tilde = compute_relative_distance(dist_raw)
    cx = compute_cx(dist_tilde, band_width)
    cx = torch.max(cx, dim=1)[0].reshape(-1, 1, H, W) * mask
    # pdb.set_trace()
    cx = torch.mean(cx, dim=(1, 2, 3))  # Eq(1)
    cx_loss = -torch.log(cx + 1e-5)  # Eq(5)
    return cx_loss


@LOSS_REGISTRY.register()
class MaskContextualLoss(nn.Module):
    """
    Creates a criterion that measures the masked contextual loss.

    Args
    ---
    band_width : int, optional
        a band_width parameter described as :math:`h` in the paper.
    use_vgg : bool, optional
        if you want to use VGG feature, set this `True`.
    vgg_layer : str, optional
        intermidiate layer name for VGG feature.
        Now we support layer names:
            `['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']`
    """

    def __init__(self, band_width=0.5, loss_type='cosine',
                 use_vgg=True, vgg_layer='conv4_4',
                 loss_weight=1.0, reduction='mean',
                 mask_type='flow', alpha=0.01, beta=10):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')
        if loss_type not in ['cosine', 'l1', 'l2']:
            raise ValueError(f'Unsupported loss mode: {loss_type}.')
        if mask_type != 'flow':
            raise ValueError(f'Unsupported mask type: {mask_type}.')

        assert band_width > 0, 'band_width parameter must be positive.'

        self.band_width = band_width
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.loss_type = loss_type
        self.alpha = alpha
        self.beta = beta

        if mask_type == 'flow':
            self.flow_model = FlowGenerator(
                path='experiments/pretrained_models/flownet/pwc_net.pth.tar')

        if use_vgg:
            self.vgg_model = VGGFeatureExtractor(
                layer_name_list=[vgg_layer],
                vgg_type='vgg19')

    def forward(self, pred, target, weight=None, **kwargs):
        assert hasattr(self, 'vgg_model'), 'Please specify VGG model.'
        assert pred.shape[1] == 3 and target.shape[1] == 3,\
            'VGG model takes 3 chennel images.'

        # picking up vgg feature maps
        pred_features = self.vgg_model(pred)
        target_features = self.vgg_model(target.detach())
        occlusion_mask = self.mask_occlusion(pred, target).detach()

        cx_loss = 0
        for k in pred_features.keys():
            cx_loss += mask_contextual_loss(target_features[k], pred_features[k],
                                            mask=occlusion_mask, band_width=self.band_width,
                                            loss_type=self.loss_type, weight=weight,
                                            reduction=self.reduction)

        cx_loss *= self.loss_weight
        return cx_loss

    def mask_occlusion(self, pred, target, forward=True):
        with torch.no_grad():
            w_f = self.flow_model(pred.detach(), target.detach())
            w_b = self.flow_model(target.detach(), pred.detach())

            if forward:
                wb_warpped = flow_warp(w_b, w_f.permute(0, 2, 3, 1))

            left_condition = torch.norm(w_f + wb_warpped, dim=1)
            right_condition = self.alpha * (torch.norm(w_f, dim=1) +
                                            torch.norm(wb_warpped, dim=1)) + self.beta
            mask = (left_condition < right_condition)
        return mask.float()


@LOSS_REGISTRY.register()
class CostVolumeLoss(nn.Module):
    """
    Creates a criterion that measures the cost volume loss.

    Args
    ---
    use_mask : bool, optional
        whether to use occlusion mask. Default: False.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', kernel_size=5,
                 bound='min', use_mask=False, alpha=0.1, beta=0.5,
                 use_vgg=False, vgg_layer='conv4_4'):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')
        if bound not in ['min', 'max']:
            raise ValueError(f'Unsupported bound mode: {bound}.')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.patch_shape = kernel_size ** 2
        self.bound = bound
        self.use_mask = use_mask
        self.alpha = alpha
        self.beta = beta
        self.use_vgg = use_vgg
        self.vgg_layer = vgg_layer

        padding = (kernel_size - 1) // 2
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding,
                                dilation=1, stride=1)

        self.flow_model = FlowGenerator(load_path='experiments/pretrained_models/RAFT/raft-things.pth')

        if use_vgg:
            self.vgg_model = VGGFeatureExtractor(
                layer_name_list=[vgg_layer],
                vgg_type='vgg19')

    def forward(self, pred, target, weight=None, **kwargs):
        assert hasattr(self, 'flow_model'), 'Please specify optical flow model.'
        assert pred.shape[1] == 3 and target.shape[1] == 3,\
            'flow model takes 3 chennel images.'

        N, C, H, W = pred.shape

        # picking up vgg feature maps
        source_img = pred.detach()
        reference_img = target.detach()
        with torch.no_grad():
            w_f = self.flow_model(source_img, reference_img)
            target_warpped = flow_warp(reference_img, w_f.permute(0, 2, 3, 1))

        if self.use_vgg:
            pred_features = self.vgg_model(pred)[self.vgg_layer]
            target_features = self.vgg_model(target_warpped)[self.vgg_layer]
            expand_source, unfold_target = self.unfold_and_expand(pred_features, target_features)
        else:
            expand_source, unfold_target = self.unfold_and_expand(pred, target_warpped)

        diff_map = F.l1_loss(expand_source, unfold_target, reduction='none')
        diff_map = torch.mean(diff_map, dim=1)  # average over channel dimension
        # print(diff_map.shape)

        # bound inside the patch of cost volume.
        if self.bound == 'min':
            diff_map = torch.min(diff_map, dim=1, keepdim=True)[0]
        elif self.bound == 'max':
            diff_map = torch.max(diff_map, dim=1, keepdim=True)[0]
        else:
            raise ValueError(f'Unrecognized bound mode: {self.bound}.')

        if self.use_mask:
            occlusion_mask = self.mask_occlusion(pred, target).unsqueeze(1)
            # pdb.set_trace()
            occlusion_mask = F.interpolate(occlusion_mask, size=diff_map.shape[-2:],
                                           mode='bilinear', align_corners=True).detach()
            diff_map = diff_map * occlusion_mask

        loss = torch.mean(diff_map)
        cv_loss = self.loss_weight * loss
        return cv_loss

    def mask_occlusion(self, pred, target, forward=True):
        with torch.no_grad():
            w_f = self.flow_model(pred.detach(), target.detach())
            w_b = self.flow_model(target.detach(), pred.detach())

            if forward:
                wb_warpped = flow_warp(w_b, w_f.permute(0, 2, 3, 1))

            left_condition = torch.norm(w_f + wb_warpped, dim=1)
            right_condition = self.alpha * (torch.norm(w_f, dim=1) +
                                            torch.norm(wb_warpped, dim=1)) + self.beta
            mask = (left_condition < right_condition)
        return mask.float()

    def unfold_and_expand(self, pred, target):
        assert pred.shape == target.shape, "Pred and Target shape mismatch."

        N, C, H, W = pred.shape
        unfold_target = self.unfold(target).view(N, C, self.patch_shape, H, W).detach()
        expand_source = pred.unsqueeze(2).repeat(1, 1, self.patch_shape, 1, 1)
        return expand_source, unfold_target


@weighted_loss
def mask_cv_loss(pred, target, bound='min'):
    diff_map = F.l1_loss(pred, target, reduction='none')
    diff_map = torch.mean(diff_map, dim=1)  # average over channel dimension
    # print(diff_map.shape)

    # bound inside the patch of cost volume.
    if bound == 'min':
        diff_map = torch.min(diff_map, dim=1, keepdim=True)[0]
    elif bound == 'max':
        diff_map = torch.max(diff_map, dim=1, keepdim=True)[0]
    else:
        raise ValueError(f'Unrecognized bound mode: {bound}.')

    return diff_map


@LOSS_REGISTRY.register()
class MaskCostVolumeLoss(nn.Module):
    """
    Creates a criterion that measures the cost volume loss.

    Args
    ---
    use_mask : bool, optional
        whether to use occlusion mask. Default: False.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', kernel_size=5,
                 bound='min', use_vgg=False, vgg_layer='conv4_4'):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')
        if bound not in ['min', 'max']:
            raise ValueError(f'Unsupported bound mode: {bound}.')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.patch_shape = kernel_size ** 2
        self.bound = bound
        self.use_vgg = use_vgg
        self.vgg_layer = vgg_layer

        self.padding = (kernel_size - 1) // 2
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=self.padding,
                                dilation=1, stride=1)

        if use_vgg:
            self.vgg_model = VGGFeatureExtractor(
                layer_name_list=[vgg_layer],
                vgg_type='vgg19')

    def forward(self, pred, target, weight=None, **kwargs):
        assert pred.shape[1] == 3 and target.shape[1] == 3,\
            'flow model takes 3 chennel images.'

        N, C, H, W = pred.shape

        # picking up vgg feature maps
        if self.use_vgg:
            pred_features = self.vgg_model(pred)[self.vgg_layer]
            target_features = self.vgg_model(target.detach())[self.vgg_layer]
            expand_source, unfold_target = self.unfold_and_expand(pred_features, target_features)
        else:
            expand_source, unfold_target = self.unfold_and_expand(pred, target.detach())

        if weight != None and weight.size(1) == 3:
            weight = weight[:, :1, :, :]
            border = torch.zeros_like(weight)
            border[..., self.padding:-self.padding, self.padding:-self.padding] = 1
            # pdb.set_trace()
            weight *= border

        cv_loss = mask_cv_loss(expand_source, unfold_target, weight=weight, bound=self.bound)
        return self.loss_weight * cv_loss

    def unfold_and_expand(self, pred, target):
        assert pred.shape == target.shape, "Pred and Target shape mismatch."

        N, C, H, W = pred.shape
        unfold_target = self.unfold(target).view(N, C, self.patch_shape, H, W).detach()
        expand_source = pred.unsqueeze(2).repeat(1, 1, self.patch_shape, 1, 1)
        return expand_source, unfold_target


def generate_weight(A, B, epislon=1e-4):
    N, C, H, W = A.shape
    B = F.adaptive_avg_pool2d(B, [1, 1])
    B = B.repeat(1, 1, H, W)

    combination = (A * B).sum(1)
    combination = combination.view(N, -1)
    combination = F.relu(combination) + epislon
    return combination


def normalize_feature(x):
    x = x - x.mean(dim=1, keepdim=True)
    return x


def calc_emd_distance(cost_map, weight_p, weight_t, solver='opencv'):
    N = cost_map.shape[0]
    sim_score = torch.zeros_like(cost_map, device=cost_map.device)

    if solver == 'opencv':  # use openCV solver
        for i in range(N):
            _, flow = emd_inference_opencv(1 - cost_map[i, :, :], weight_p[i, :], weight_t[i, :])

            sim_score[i, :, :] = cost_map[i, :, :] * \
                torch.from_numpy(flow).to(cost_map.device)

        sim_score = sim_score.sum(-1)
        return sim_score
    elif solver == 'qpth':
        raise NotImplementedError('QPTH solver is not implemented yet.')
    else:
        raise ValueError('Unknown Solver')


def emd_inference_opencv(cost_matrix, weight1, weight2):
    # cost matrix is a tensor of shape [N,N]
    cost_matrix = cost_matrix.detach().cpu().numpy()

    weight1 = F.relu(weight1) + 1e-5
    weight2 = F.relu(weight2) + 1e-5

    weight1 = (weight1 * (weight1.shape[0] / weight1.sum().item())).view(-1, 1).detach().cpu().numpy()
    weight2 = (weight2 * (weight2.shape[0] / weight2.sum().item())).view(-1, 1).detach().cpu().numpy()

    cost, _, flow = cv2.EMD(weight1, weight2, cv2.DIST_USER, cost_matrix)
    return cost, flow


def EMD_loss(pred, target, temperature=0.5, loss_type='cosine', solver='opencv'):
    """
    Computes EMD loss between pred and target.
    Parameters
    ---
    pred : torch.Tensor
        features of shape (N, C, H, W).
    target : torch.Tensor
        features of shape (N, C, H, W).
    temperature : float, optional
        a band-width parameter used to convert distance to similarity.
        in the paper, this is described as :math:`h`.
    loss_type : str, optional
        a loss type to measure the distance between features.
        Note: `l1` and `l2` frequently raises OOM.
    solver: str, optional
        solver used to solve LP.
    Returns
    ---
    emd_loss : torch.Tensor
        EMD loss between x and y
    """

    assert pred.size() == target.size(), 'input tensor must have the same size.'
    assert loss_type in ['cosine', 'l1', 'l2'], f"select a loss type from \
                                                {['cosine', 'l1', 'l2']}."
    assert solver in ['opencv', 'QPTH'], f"select a solver from {['opencv', 'QPTH']}."

    N, C, H, W = pred.size()

    weight_p = generate_weight(pred, target)  # return N, HW
    weight_t = generate_weight(target, pred)  # return N, HW

    # # similarity from EMD.
    # pred = normalize_feature(pred)
    # target = normalize_feature(target)

    # pred = pred.view(N, C, -1)
    # target = target.view(N, C, -1)

    # if loss_type == 'cosine':
    #     pred = pred.permute(0, 2, 1).unsqueeze(-2) # N, HW, 1, C
    #     target = target.permute(0, 2, 1).unsqueeze(-3) # N, 1, HW, C
    #     similarity_map = F.cosine_similarity(pred, target, dim=-1) # N, HW, HW
    # elif loss_type == 'l1':
    #     dist_map = pred.unsqueeze(-1) - target.unsqueeze(-2)
    #     dist_map = dist_map.sum(dim=1).abs().clamp(min=0.)
    #     similarity_map = 1 - dist_map
    # elif loss_type == 'l2':
    #     dist_map = (pred.unsqueeze(-1) - target.unsqueeze(-2))**2
    #     dist_map = dist_map.sum(dim=1).sqrt()
    #     similarity_map = 1 - dist_map

    # similarity from CX.
    if loss_type == 'cosine':
        dist_raw = compute_cosine_distance(pred, target)
    elif loss_type == 'l1':
        dist_raw = compute_l1_distance(pred, target)
    elif loss_type == 'l2':
        dist_raw = compute_l2_distance(pred, target)

    dist_tilde = compute_relative_distance(dist_raw)
    similarity_map = compute_cx(dist_tilde, temperature)

    sim_score = calc_emd_distance(similarity_map, weight_p, weight_t, solver='opencv')
    # sim_score = torch.exp()  # Eq(3)

    # pdb.set_trace()

    # emd_loss = -sim_score.sum(dim=1)
    emd_loss = -torch.log(sim_score.sum(dim=1) + 1e-8)  # Eq(5)
    return emd_loss.mean()


@LOSS_REGISTRY.register()
class DeepEMDLoss(nn.Module):
    """
    Creates a criterion that measures the EMD loss.

    Args
    ---
    band_width : int, optional
        a band_width parameter described as :math:`h` in the paper.
    use_vgg : bool, optional
        if you want to use VGG feature, set this `True`.
    vgg_layer : str, optional
        intermidiate layer name for VGG feature.
        Now we support layer names:
            `['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']`
    """

    def __init__(self, temperature=0.5, loss_type='cosine',
                 use_vgg=True, vgg_layers=['conv4_4'], solver='opencv',
                 loss_weight=1.0, reduction='sum'):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')
        if loss_type not in ['cosine', 'l1', 'l2']:
            raise ValueError(f'Unsupported loss mode: {loss_type}.')
        if solver not in ['opencv', 'qpth']:
            raise ValueError(f'Unsupported loss mode: {solver}.')

        assert temperature > 0, 'temperature parameter must be positive.'

        self.temperature = temperature
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.loss_type = loss_type
        self.solver = solver

        if use_vgg:
            self.vgg_model = VGGFeatureExtractor(
                layer_name_list=vgg_layers,
                vgg_type='vgg19')

    def forward(self, pred, target, weight=None, **kwargs):
        assert hasattr(self, 'vgg_model'), 'Please specify VGG model.'
        assert pred.shape[1] == 3 and target.shape[1] == 3,\
            'VGG model takes 3 chennel images.'

        # picking up vgg feature maps
        pred_features = self.vgg_model(pred)
        target_features = self.vgg_model(target.detach())

        logits_loss = 0
        for k in pred_features.keys():
            logits_loss += EMD_loss(pred_features[k], target_features[k],
                                    temperature=self.temperature, loss_type=self.loss_type,
                                    solver=self.solver)
            # logits_loss += -torch.log(emd_loss + 1e-5)

        logits_loss *= self.loss_weight
        return logits_loss


def RelaxEMD_loss(pred, target, temperature=0.5, loss_type='cosine', match_type='dual_softmax'):
    """
    Computes EMD loss between pred and target.
    Parameters
    ---
    pred : torch.Tensor
        features of shape (N, C, H, W).
    target : torch.Tensor
        features of shape (N, C, H, W).
    temperature : float, optional
        a band-width parameter used to convert distance to similarity.
        in the paper, this is described as :math:`h`.
    loss_type : str, optional
        a loss type to measure the distance between features.
        Note: `l1` and `l2` frequently raises OOM.
    match_type: str, optional
        match_type used to solve LP.
    Returns
    ---
    emd_loss : torch.Tensor
        EMD loss between x and y
    """

    assert pred.size() == target.size(), 'input tensor must have the same size.'
    assert loss_type in ['cosine', 'l1', 'l2'], f"select a loss type from \
                                                {['cosine', 'l1', 'l2']}."
    assert match_type in ['dual_softmax', 'sinkhorn'], f"select a match_type from {['dual_softmax', 'sinkhorn']}."

    N, C, H, W = pred.size()

    pred = pred.view(N, C, -1)
    target = target.view(N, C, -1)

    # normalize
    pred, target = map(lambda feat: feat / feat.shape[-2]**0.5, [pred, target])

    if match_type == 'dual_softmax':
        sim_matrix = torch.einsum('ncl,ncs->nls', pred, target) / temperature
        conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

    # if loss_type == 'cosine':
    #     pred = pred.permute(0, 2, 1).unsqueeze(-2) # N, HW, 1, C
    #     target = target.permute(0, 2, 1).unsqueeze(-3) # N, 1, HW, C
    #     similarity_map = F.cosine_similarity(pred, target, dim=-1) # N, HW, HW
    # elif loss_type == 'l1':
    #     dist_map = pred.unsqueeze(-1) - target.unsqueeze(-2)
    #     dist_map = dist_map.sum(dim=1).abs().clamp(min=0.)
    #     similarity_map = 1 - dist_map
    # elif loss_type == 'l2':
    #     dist_map = (pred.unsqueeze(-1) - target.unsqueeze(-2))**2
    #     dist_map = dist_map.sum(dim=1).sqrt()
    #     similarity_map = 1 - dist_map

    # similarity from CX.
    if loss_type == 'cosine':
        dist_raw = compute_cosine_distance(pred, target)
    elif loss_type == 'l1':
        dist_raw = compute_l1_distance(pred, target)
    elif loss_type == 'l2':
        dist_raw = compute_l2_distance(pred, target)

    dist_tilde = compute_relative_distance(dist_raw)
    similarity_map = compute_cx(dist_tilde, temperature)

    sim_score = calc_emd_distance(similarity_map, weight_p, weight_t, solver='opencv')
    # sim_score = torch.exp()  # Eq(3)

    # pdb.set_trace()

    # emd_loss = -sim_score.sum(dim=1)
    emd_loss = -torch.log(sim_score.sum(dim=1) + 1e-8)  # Eq(5)
    return emd_loss.mean()


# @LOSS_REGISTRY.register()
class RelaxEMDLoss(nn.Module):
    """
    Creates a criterion that measures the EMD loss.

    Args
    ---
    band_width : int, optional
        a band_width parameter described as :math:`h` in the paper.
    use_vgg : bool, optional
        if you want to use VGG feature, set this `True`.
    vgg_layer : str, optional
        intermidiate layer name for VGG feature.
        Now we support layer names:
            `['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']`
    """

    def __init__(self, temperature=0.5, loss_type='cosine',
                 use_vgg=True, vgg_layers=['conv4_4'], match_type='dual_softmax',
                 loss_weight=1.0, reduction='sum'):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')
        if loss_type not in ['cosine', 'l1', 'l2']:
            raise ValueError(f'Unsupported loss mode: {loss_type}.')
        if match_type not in ['dual_softmax', 'sinkhorn']:
            raise ValueError(f'Unsupported match type: {match_type}.')

        assert temperature > 0, 'temperature parameter must be positive.'

        self.temperature = temperature
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.loss_type = loss_type
        self.match_type = match_type

        if use_vgg:
            self.vgg_model = VGGFeatureExtractor(
                layer_name_list=vgg_layers,
                vgg_type='vgg19')

    def forward(self, pred, target, weight=None, **kwargs):
        assert hasattr(self, 'vgg_model'), 'Please specify VGG model.'
        assert pred.shape[1] == 3 and target.shape[1] == 3,\
            'VGG model takes 3 chennel images.'

        # picking up vgg feature maps
        pred_features = self.vgg_model(pred)
        target_features = self.vgg_model(target.detach())

        logits_loss = 0
        for k in pred_features.keys():
            logits_loss += RelaxEMD_loss(pred_features[k], target_features[k],
                                         temperature=self.temperature, loss_type=self.loss_type,
                                         match_type=self.match_type)
            # logits_loss += -torch.log(emd_loss + 1e-5)

        logits_loss *= self.loss_weight
        return logits_loss
