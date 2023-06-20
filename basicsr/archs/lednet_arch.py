import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
import numpy as np


class KernelConv2D(nn.Module):
    def __init__(self, ksize=5, act=True):
        super(KernelConv2D, self).__init__()
        self.ksize = ksize
        self.act = act

    def forward(self, feat_in, kernel):
        channels = feat_in.size(1)
        N, kernels, H, W = kernel.size()
        pad = (self.ksize - 1) // 2

        feat_in = F.pad(feat_in, (pad, pad, pad, pad), mode="replicate")
        feat_in = feat_in.unfold(2, self.ksize, 1).unfold(3, self.ksize, 1)
        feat_in = feat_in.permute(0, 2, 3, 1, 4, 5).contiguous()
        feat_in = feat_in.reshape(N, H, W, channels, -1)

        kernel = kernel.permute(0, 2, 3, 1).reshape(N, H, W, channels, -1)
        feat_out = torch.sum(feat_in * kernel, -1)
        feat_out = feat_out.permute(0, 3, 1, 2).contiguous()
        if self.act:
            feat_out = F.leaky_relu(feat_out, negative_slope=0.2, inplace=True)
        return feat_out


class Downsample(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)),
                          int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        # print('Filter size [%i]'%filt_size)
        if(self.filt_size == 1):
            a = np.array([1., ])
        elif(self.filt_size == 2):
            a = np.array([1., 1.])
        elif(self.filt_size == 3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size == 4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size == 5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size == 6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size == 7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


def get_pad_layer(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.PReLU()
            ))
        self.features = nn.ModuleList(self.features)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_dim + reduction_dim * 4, in_dim, kernel_size=3, padding=1, bias=False),
            nn.PReLU())

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        out_feat = self.fuse(torch.cat(out, 1))
        return out_feat


class MyDownSample(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(MyDownSample, self).__init__()

        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias),
                                 nn.PReLU(),
                                 nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=bias),
                                 nn.PReLU(),
                                 Downsample(channels=in_channels, filt_size=3, stride=2),
                                 nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=bias))

        self.bot = nn.Sequential(Downsample(channels=in_channels, filt_size=3, stride=2),
                                 nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top + bot
        return out


class MyUpSample(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(MyUpSample, self).__init__()

        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias),
                                 nn.PReLU(),
                                 nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2,
                                                    padding=1, output_padding=1, bias=bias),
                                 nn.PReLU(),
                                 nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=bias))

        self.bot = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias),
                                 nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top + bot
        return out


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, mode=None, bias=True):
        super(BasicBlock, self).__init__()
        self.mode = mode

        self.body1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=(kernel_size - 1) // 2, bias=bias),
            nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=(kernel_size - 1) // 2, bias=bias),
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=(kernel_size - 1) // 2, bias=bias),
            nn.PReLU()
        )
        if mode == 'down':
            self.reshape_conv = MyDownSample(in_channels, out_channels)
        elif mode == 'up':
            self.reshape_conv = MyUpSample(in_channels, out_channels)

    def forward(self, x):
        res = self.body1(x)
        out = res + x
        out = self.body2(out)
        if self.mode is not None:
            out = self.reshape_conv(out)
        return out


class BasicBlock_D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, mode=None, bias=True):
        super(BasicBlock_D, self).__init__()
        self.mode = mode
        if mode == 'down':
            self.reshape_conv = MyDownSample(in_channels, out_channels)
        elif mode == 'up':
            self.reshape_conv = MyUpSample(in_channels, out_channels)

        self.body1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2, bias=bias),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2, bias=bias)
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2, bias=bias),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2, bias=bias)
        )

    def forward(self, x):
        if self.mode is not None:
            x = self.reshape_conv(x)
        res1 = self.body1(x)
        out1 = res1 + x
        res2 = self.body2(out1)
        out = res2 + out1
        return out


# Channel Attention (CA) Layer
class CurveCALayer(nn.Module):
    def __init__(self, channel, n_curve):
        super(CurveCALayer, self).__init__()
        self.n_curve = n_curve
        self.relu = nn.ReLU(inplace=False)
        self.predict_a = nn.Sequential(
            nn.Conv2d(channel, channel, 5, stride=1, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channel, n_curve, 1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        # clip the input features into range of [0,1]
        a = self.predict_a(x)
        x = self.relu(x) - self.relu(x - 1)
        for i in range(self.n_curve):
            x = x + a[:, i:i + 1] * x * (1 - x)

        return x


@ARCH_REGISTRY.register()
class LEDNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, ppm=True, fac=False,
                 skip=False, curve=False, side_loss=False):
        super(LEDNet, self).__init__()
        self.ppm = ppm
        self.fac = fac
        self.curve = curve
        self.side_loss = side_loss
        self.skip = skip

        # ch1, ch2, ch3, ch4 = 32, 64, 128, 256
        ch1, ch2, ch3, ch4 = 32, 64, 128, 128
        self.E_block1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1, 3, stride=1, padding=1), nn.PReLU(),
            BasicBlock(ch1, ch2, mode='down'))
        self.E_block2 = BasicBlock(ch2, ch3, mode='down')
        self.E_block3 = BasicBlock(ch3, ch4, mode='down')

        if self.ppm:
            self.PPM1 = PPM(ch2, ch2 // 4, bins=(1, 2, 3, 6))
            self.PPM2 = PPM(ch3, ch3 // 4, bins=(1, 2, 3, 6))
            self.PPM3 = PPM(ch4, ch4 // 4, bins=(1, 2, 3, 6))

        # curve CA
        if self.curve:
            self.curve_n = 3
            self.conv_1c = CurveCALayer(ch2, self.curve_n)
            self.conv_2c = CurveCALayer(ch3, self.curve_n)
            self.conv_3c = CurveCALayer(ch4, self.curve_n)

        if self.side_loss:
            self.side_out = nn.Conv2d(ch4, out_channels, 3, stride=1, padding=1)

        self.M_block1 = BasicBlock(ch4, ch4)
        self.M_block2 = BasicBlock(ch4, ch4)

        # dynamic filter
        if self.fac:
            ks_2d = 5
            self.conv_fac_k3 = nn.Sequential(
                nn.Conv2d(ch4, ch4, 3, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(ch4, ch4, 3, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(ch4, ch4, 3, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(ch4, ch4 * ks_2d**2, 1, stride=1))

            self.conv_fac_k2 = nn.Sequential(
                nn.Conv2d(ch3, ch3, 3, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(ch3, ch3, 3, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(ch3, ch3, 3, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(ch3, ch3 * ks_2d**2, 1, stride=1))

            self.conv_fac_k1 = nn.Sequential(
                nn.Conv2d(ch2, ch2, 3, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(ch2, ch2, 3, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(ch2, ch2, 3, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(ch2, ch2 * ks_2d**2, 1, stride=1))

            self.kconv_deblur = KernelConv2D(ksize=ks_2d, act=True)

        self.D_block3 = BasicBlock_D(ch4, ch4)
        self.D_block2 = BasicBlock_D(ch4, ch3, mode='up')
        self.D_block1 = BasicBlock_D(ch3, ch2, mode='up')
        self.D_block0 = nn.Sequential(
            BasicBlock_D(ch2, ch1, mode='up'),
            nn.Conv2d(ch1, out_channels, 3, stride=1, padding=1))

    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """
        if not self.training:
            N, C, H, W = x.shape
            mod_size = 8
            H_pad = mod_size - H % mod_size if not H % mod_size == 0 else 0
            W_pad = mod_size - W % mod_size if not W % mod_size == 0 else 0
            x = F.pad(x, (0, W_pad, 0, H_pad), 'replicate')
        # pdb.set_trace()

        # Dncoder
        e_feat1 = self.E_block1(x)  # 64 1/2
        if self.ppm:
            e_feat1 = self.PPM1(e_feat1)
        if self.curve:
            e_feat1 = self.conv_1c(e_feat1)

        e_feat2 = self.E_block2(e_feat1)  # 128 1/4
        if self.ppm:
            e_feat2 = self.PPM2(e_feat2)
        if self.curve:
            e_feat2 = self.conv_2c(e_feat2)

        e_feat3 = self.E_block3(e_feat2)  # 256 1/8
        if self.ppm:
            e_feat3 = self.PPM3(e_feat3)
        if self.curve:
            e_feat3 = self.conv_3c(e_feat3)

        # pdb.set_trace()
        if self.side_loss:
            out_side = self.side_out(e_feat3)

        # Mid
        m_feat = self.M_block1(e_feat3)
        m_feat = self.M_block2(m_feat)

        # Decoder
        d_feat3 = self.D_block3(m_feat)  # 256 1/8
        if self.fac:
            kernel_3 = self.conv_fac_k3(e_feat3)
            d_feat3 = self.kconv_deblur(d_feat3, kernel_3)
        elif self.skip:
            d_feat3 = d_feat3 + e_feat3

        d_feat2 = self.D_block2(d_feat3)  # 128 1/4
        if self.fac:
            kernel_2 = self.conv_fac_k2(e_feat2)
            d_feat2 = self.kconv_deblur(d_feat2, kernel_2)
        elif self.skip:
            d_feat2 = d_feat2 + e_feat2

        d_feat1 = self.D_block1(d_feat2)  # 64 1/4
        if self.fac:
            kernel_1 = self.conv_fac_k1(e_feat1)
            d_feat1 = self.kconv_deblur(d_feat1, kernel_1)
        elif self.skip:
            d_feat1 = d_feat1 + e_feat1

        out = self.D_block0(d_feat1)

        if not self.training:
            out = out[:, :, :H, :W]

        if self.side_loss:
            return out_side, out
        else:
            return out


if __name__ == '__main__':
    height = 128
    width = 128
    model = LEDNet(
        ppm=True,
        fac=False,
        curve=True)
    print(model)

    src = torch.randn((2, 3, height, width))
    model.eval()
    with torch.no_grad():
        out = model(src)
    model.train()

    print(out.shape)
