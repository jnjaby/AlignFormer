import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops.upfirdn2d import upfirdn2d
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class DAModule(nn.Module):
    def __init__(self, in_ch, feat_ch, out_ch, demodulate=True, load_path=None,
                 requires_grad=False):
        super().__init__()

        self.guide_net = nn.Sequential(
            nn.Conv2d(in_ch, feat_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(1),
        )

        self.conv0 = ModulatedStyleConv(in_ch, feat_ch, feat_ch, kernel_size=3,
                                            activate=True, demodulate=demodulate)
        self.conv11 = ModulatedStyleConv(feat_ch, feat_ch, feat_ch, kernel_size=3,
                                downsample=True, activate=True, demodulate=demodulate)
        self.conv12 = ModulatedStyleConv(feat_ch, feat_ch, feat_ch, kernel_size=3,
                                downsample=False, activate=True, demodulate=demodulate)
        self.conv21 = ModulatedStyleConv(feat_ch, feat_ch, feat_ch, kernel_size=3,
                                downsample=True, activate=True, demodulate=demodulate)
        self.conv22 = ModulatedStyleConv(feat_ch, feat_ch, feat_ch, kernel_size=3,
                                downsample=False, activate=True, demodulate=demodulate)
        self.conv31 = ModulatedStyleConv(feat_ch, feat_ch, feat_ch, kernel_size=3,
                                downsample=False, activate=True, demodulate=demodulate)
        self.conv32 = ModulatedStyleConv(feat_ch, feat_ch, feat_ch, kernel_size=3,
                                downsample=False, activate=True, demodulate=demodulate)
        self.conv41 = ModulatedStyleConv(feat_ch, feat_ch, feat_ch, kernel_size=3,
                                upsample=True, activate=True, demodulate=demodulate)
        self.conv42 = ModulatedStyleConv(feat_ch, feat_ch, feat_ch, kernel_size=3,
                                upsample=False, activate=True, demodulate=demodulate)
        self.conv51 = ModulatedStyleConv(feat_ch, feat_ch, feat_ch, kernel_size=3,
                                upsample=True, activate=True, demodulate=demodulate)
        self.conv52 = ModulatedStyleConv(feat_ch, feat_ch, feat_ch, kernel_size=3,
                                upsample=False, activate=True, demodulate=demodulate)
        self.conv6 = ModulatedStyleConv(feat_ch, feat_ch, out_ch, kernel_size=3,
                                activate=False, demodulate=demodulate)

        if load_path:
            self.load_state_dict(torch.load(
                load_path, map_location=lambda storage, loc: storage)['params_ema'])

        if not requires_grad:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.train()
            for param in self.parameters():
                param.requires_grad = True

    def forward(self, x, ref):
        if not self.training:
            N, C, H, W = x.shape
            mod_size = 4
            H_pad = mod_size - H % mod_size if not H % mod_size == 0 else 0
            W_pad = mod_size - W % mod_size if not W % mod_size == 0 else 0
            x = F.pad(x, (0, W_pad, 0, H_pad), 'replicate')

        style_guidance = self.guide_net(ref)

        feat0 = self.conv0(x, style_guidance)
        feat1 = self.conv11(feat0, style_guidance)
        feat1 = self.conv12(feat1, style_guidance)
        feat2 = self.conv21(feat1, style_guidance)
        feat2 = self.conv22(feat2, style_guidance)
        feat3 = self.conv31(feat2, style_guidance)
        feat3 = self.conv32(feat3, style_guidance)
        feat4 = self.conv41(feat3 + feat2, style_guidance)
        feat4 = self.conv42(feat4, style_guidance)
        feat5 = self.conv51(feat4 + feat1, style_guidance)
        feat5 = self.conv52(feat5, style_guidance)
        feat6 = self.conv6(feat5 + feat0, style_guidance)

        out = feat6
        if not self.training:
            out = out[:, :, :H, :W]

        return out


class ModulatedStyleConv(nn.Module):
    def __init__(self,
                 in_ch,
                 feat_ch,
                 out_ch,
                 kernel_size,
                 upsample=False,
                 downsample=False,
                 activate=False,
                 blur_kernel=[1, 3, 3, 1],
                 demodulate=True,
                 eps=1e-8,):
        super(ModulatedStyleConv, self).__init__()
        self.eps = eps
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.demodulate = demodulate
        self.upsample = upsample
        self.downsample = downsample
        self.activate = activate
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.style_weight = nn.Sequential(
            nn.Conv2d(feat_ch, in_ch, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),)

        self.style_bias = nn.Sequential(
            nn.Conv2d(feat_ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),)

        self.weight = nn.Parameter(
            torch.randn(1, out_ch, in_ch, kernel_size, kernel_size))

        # build blurry layer for upsampling
        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(blur_kernel, (pad0, pad1), upsample_factor=factor)
        # build blurry layer for downsampling
        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        if activate:
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, style):
        n, c, h, w = x.shape
        # process style code
        # pdb.set_trace()

        style_w = self.style_weight(style).view(n, 1, c, 1, 1)
        style_b = self.style_bias(style).view(n, self.out_ch, 1, 1)

        # combine weight and style
        weight = self.weight * style_w
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(n, self.out_ch, 1, 1, 1)

        weight = weight.view(n * self.out_ch, c, self.kernel_size,
                             self.kernel_size)

        if self.upsample:
            x = x.view(1, n * c, h, w)
            weight = weight.view(n, self.out_ch, c, self.kernel_size,
                                 self.kernel_size)
            weight = weight.transpose(1, 2).reshape(n * c, self.out_ch,
                                                    self.kernel_size,
                                                    self.kernel_size)
            x = F.conv_transpose2d(x, weight, padding=0, stride=2, groups=n)
            x = x.reshape(n, self.out_ch, *x.shape[-2:])
            x = self.blur(x)
        elif self.downsample:
            x = self.blur(x)
            x = x.view(1, n * self.in_ch, *x.shape[-2:])
            x = F.conv2d(x, weight, stride=2, padding=0, groups=n)
            x = x.view(n, self.out_ch, *x.shape[-2:])
        else:
            x = x.view(1, n * c, h, w)
            x = F.conv2d(x, weight, stride=1, padding=self.padding, groups=n)
            x = x.view(n, self.out_ch, *x.shape[-2:])

        out = x + style_b

        if self.activate:
            out = self.act(out)

        return out


class Blur(nn.Module):

    def __init__(self, kernel, pad, upsample_factor=1):
        super(Blur, self).__init__()
        kernel = _make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * (upsample_factor**2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, x):
        return upfirdn2d(x, self.kernel, pad=self.pad)


def _make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


if __name__ == '__main__':
    height, width = 256, 256
    model = DAModule(
        in_ch=3, feat_ch=64, out_ch=3, demodulate=True)
    print(model)

    src = torch.randn((2, 3, height, width))
    ref = torch.randn((2, 3, height, width))
    model.eval()
    with torch.no_grad():
        out = model(src, ref)
    model.train()

    print(out.shape)
