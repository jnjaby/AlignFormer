import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


def softmax_attention(q, k, v):
    # b x n x d x h x w
    h, w = q.shape[-2], q.shape[-1]

    q = q.flatten(-2).transpose(-2, -1)  # b x n x hw x d
    k = k.flatten(-2)   # b x n x d x hw
    v = v.flatten(-2).transpose(-2, -1)

    N = k.shape[-1]     # ?????? maybe change to k.shape[-2]????
    attn = torch.matmul(q / N ** 0.5, k)
    attn = F.softmax(attn, dim=-1)
    output = torch.matmul(attn, v)

    output = output.transpose(-2, -1)
    output = output.view(*output.shape[:-1], h, w)

    return output, attn


def dotproduct_attention(q, k, v):
    # b x n x d x h x w
    h, w = q.shape[-2], q.shape[-1]

    q = q.flatten(-2).transpose(-2, -1)  # b x n x hw x d
    k = k.flatten(-2)  # b x n x d x hw
    v = v.flatten(-2).transpose(-2, -1)

    N = k.shape[-1]
    attn = None
    # attn = torch.matmul(q / N ** 0.5, k)
    tmp = torch.matmul(k, v) / N
    output = torch.matmul(q, tmp)

    output = output.transpose(-2, -1)
    output = output.view(*output.shape[:-1], h, w)

    return output, attn


def long_range_attention(q, k, v, P_h, P_w):  # fixed patch size
    B, N, C, qH, qW = q.size()
    _, _, _, kH, kW = k.size()

    qQ_h, qQ_w = qH // P_h, qW // P_w
    kQ_h, kQ_w = kH // P_h, kW // P_w

    q = q.reshape(B, N, C, qQ_h, P_h, qQ_w, P_w)
    k = k.reshape(B, N, C, kQ_h, P_h, kQ_w, P_w)
    v = v.reshape(B, N, -1, kQ_h, P_h, kQ_w, P_w)

    q = q.permute(0, 1, 4, 6, 2, 3, 5)   # [b, n, Ph, Pw, d, Qh, Qw]
    k = k.permute(0, 1, 4, 6, 2, 3, 5)
    v = v.permute(0, 1, 4, 6, 2, 3, 5)

    output, attn = softmax_attention(q, k, v)   # attn: [b, n, Ph, Pw, qQh*qQw, kQ_h*kQ_w]
    output = output.permute(0, 1, 4, 5, 2, 6, 3)
    output = output.reshape(B, N, -1, qH, qW)
    return output, attn


def short_range_attention(q, k, v, Q_h, Q_w):  # fixed patch number
    B, N, C, qH, qW = q.size()
    _, _, _, kH, kW = k.size()

    qP_h, qP_w = qH // Q_h, qW // Q_w
    kP_h, kP_w = kH // Q_h, kW // Q_w

    q = q.reshape(B, N, C, Q_h, qP_h, Q_w, qP_w)
    k = k.reshape(B, N, C, Q_h, kP_h, Q_w, kP_w)
    v = v.reshape(B, N, -1, Q_h, kP_h, Q_w, kP_w)

    q = q.permute(0, 1, 3, 5, 2, 4, 6)   # [b, n, Qh, Qw, d, Ph, Pw]
    k = k.permute(0, 1, 3, 5, 2, 4, 6)
    v = v.permute(0, 1, 3, 5, 2, 4, 6)

    output, attn = softmax_attention(q, k, v)   # attn: [b, n, Qh, Qw, qPh*qPw, kPh*kPw]
    output = output.permute(0, 1, 4, 2, 5, 3, 6)
    output = output.reshape(B, N, -1, qH, qW)
    return output, attn


def patch_attention(q, k, v, P):
    # q: [b, nhead, c, h, w]
    q_patch = space_to_depth(q, P)   # [b, nhead, cP^2, h/P, w/P]
    k_patch = space_to_depth(k, P)
    v_patch = space_to_depth(v, P)

    # output: [b, nhead, cP^2, h/P, w/P]
    # attn: [b, nhead, h/P*w/P, h/P*w/P]
    # output, attn = softmax_attention(q_patch, k_patch, v_patch)
    output, attn = dotproduct_attention(q_patch, k_patch, v_patch)
    output = depth_to_space(output, P)  # output: [b, nhead, c, h, w]
    return output, attn


def space_to_depth(x, block_size):
    x_shape = x.shape
    c, h, w = x_shape[-3:]
    if len(x.shape) >= 5:
        x = x.view(-1, c, h, w)
    # pdb.set_trace()
    unfolded_x = F.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(*x_shape[0:-3], c * block_size ** 2, h // block_size, w // block_size)


def depth_to_space(x, block_size):
    x_shape = x.shape
    c, h, w = x_shape[-3:]
    x = x.view(-1, c, h, w)
    y = F.pixel_shuffle(x, block_size)
    return y.view(*x_shape[0:-3], -1, h * block_size, w * block_size)


class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),)

    def forward(self, x):
        return self.conv(x)


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                  nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, x):
        return self.conv(x)


class double_conv_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                  nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, x):
        return self.conv(x)


class double_conv_up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv_up, self).__init__()
        self.conv = nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                                  nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                  nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, x):
        return self.conv(x)


class PosEnSine(nn.Module):
    """
    Code borrowed from DETR: models/positional_encoding.py
    output size: b*(2.num_pos_feats)*h*w
    """

    def __init__(self, num_pos_feats):
        super(PosEnSine, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.normalize = True
        self.scale = 2 * math.pi
        self.temperature = 10000

    def forward(self, x):
        b, c, h, w = x.shape
        not_mask = torch.ones(1, h, w, device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.repeat(b, 1, 1, 1)
        return pos


class MLP(nn.Module):
    """
    conv-based MLP layers.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.linear1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.linear2 = nn.Conv2d(hidden_features, out_features, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


class TransformerDecoderUnit(nn.Module):
    def __init__(self, feat_dim, n_head=8, pos_en_flag=True,
                 mlp_ratio=2, attn_type='softmax', P=None):
        super().__init__()
        self.feat_dim = feat_dim
        self.attn_type = attn_type
        self.pos_en_flag = pos_en_flag
        self.P = P

        self.pos_en = PosEnSine(self.feat_dim // 2)
        self.attn = MultiheadAttention(feat_dim, n_head)   # cross-attention

        mlp_hidden_dim = int(feat_dim * mlp_ratio)
        self.mlp = MLP(in_features=feat_dim, hidden_features=mlp_hidden_dim)

        self.norm = nn.GroupNorm(1, self.feat_dim)

    def forward(self, q, k, v):
        if self.pos_en_flag:
            q_pos_embed = self.pos_en(q)
            k_pos_embed = self.pos_en(k)
        else:
            q_pos_embed = 0
            k_pos_embed = 0

        # cross-multi-head attention
        out = self.attn(q=q + q_pos_embed, k=k + k_pos_embed, v=v, attn_type=self.attn_type, P=self.P)[0]

        # feed forward
        out2 = self.mlp(out)
        out = out + out2
        out = self.norm(out)

        return out


class Unet(nn.Module):
    def __init__(self, in_ch, feat_ch, out_ch):
        super().__init__()
        self.conv_in = single_conv(in_ch, feat_ch)

        self.conv1 = double_conv_down(feat_ch, feat_ch)
        self.conv2 = double_conv_down(feat_ch, feat_ch)
        self.conv3 = double_conv(feat_ch, feat_ch)
        self.conv4 = double_conv_up(feat_ch, feat_ch)
        self.conv5 = double_conv_up(feat_ch, feat_ch)
        self.conv6 = double_conv(feat_ch, out_ch)

    def forward(self, x):
        feat0 = self.conv_in(x)    # H, W
        feat1 = self.conv1(feat0)   # H/2, W/2
        feat2 = self.conv2(feat1)    # H/4, W/4
        feat3 = self.conv3(feat2)    # H/4, W/4
        feat3 = feat3 + feat2     # H/4
        feat4 = self.conv4(feat3)    # H/2, W/2
        feat4 = feat4 + feat1    # H/2, W/2
        feat5 = self.conv5(feat4)   # H
        feat5 = feat5 + feat0   # H
        feat6 = self.conv6(feat5)

        return feat0, feat1, feat2, feat3, feat4, feat6


class MultiheadAttention(nn.Module):
    def __init__(self, feat_dim, n_head, d_k=None, d_v=None):
        super().__init__()
        if d_k is None:
            d_k = feat_dim // n_head
        if d_v is None:
            d_v = feat_dim // n_head

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # pre-attention projection
        self.w_qs = nn.Conv2d(feat_dim, n_head * d_k, 1, bias=False)
        self.w_ks = nn.Conv2d(feat_dim, n_head * d_k, 1, bias=False)
        self.w_vs = nn.Conv2d(feat_dim, n_head * d_v, 1, bias=False)

        # after-attention combine heads
        self.fc = nn.Conv2d(n_head * d_v, feat_dim, 1, bias=False)

    def forward(self, q, k, v, attn_type='softmax', **kwargs):
        # input: b x d x h x w
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        # Pass through the pre-attention projection: b x (nhead*dk) x h x w
        # Separate different heads: b x nhead x dk x h x w
        q = self.w_qs(q).view(q.shape[0], n_head, d_k, q.shape[2], q.shape[3])
        k = self.w_ks(k).view(k.shape[0], n_head, d_k, k.shape[2], k.shape[3])
        v = self.w_vs(v).view(v.shape[0], n_head, d_v, v.shape[2], v.shape[3])

        # -------------- Attention -----------------
        if attn_type == 'softmax':
            q, attn = softmax_attention(q, k, v)  # b x n x dk x h x w --> b x n x dv x h x w
        elif attn_type == 'dotproduct':
            q, attn = dotproduct_attention(q, k, v)
        elif attn_type == 'patch':
            q, attn = patch_attention(q, k, v, P=kwargs['P'])
        elif attn_type == 'sparse_long':
            q, attn = long_range_attention(q, k, v, P_h=kwargs['ah'], P_w=kwargs['aw'])
        elif attn_type == 'sparse_short':
            q, attn = short_range_attention(q, k, v, Q_h=kwargs['ah'], Q_w=kwargs['aw'])
        else:
            raise NotImplementedError(f'Unknown attention type {attn_type}')
        # ------------ end Attention ---------------

        # Concatenate all the heads together: b x (n*dv) x h x w
        q = q.reshape(q.shape[0], -1, q.shape[3], q.shape[4])
        q = self.fc(q)   # b x d x h x w

        return q, attn


@ARCH_REGISTRY.register()
class Texformer(nn.Module):
    def __init__(self,
                 feat_dim=128,
                 ref_ch=3,
                 src_ch=3,
                 out_ch=3,
                 nhead=8,
                 mlp_ratio=2,
                 pos_en_flag=True,
                 **kwargs):
        super().__init__()

        self.unet_q = Unet(src_ch, feat_dim, feat_dim)
        self.unet_k = Unet(ref_ch, feat_dim, feat_dim)
        self.unet_v = Unet(ref_ch, feat_dim, feat_dim)

        self.trans_dec = nn.ModuleList([
            TransformerDecoderUnit(feat_dim, nhead, pos_en_flag, mlp_ratio,
                                   attn_type=kwargs['attn_type'], P=kwargs['P']),
            TransformerDecoderUnit(feat_dim, nhead, pos_en_flag, mlp_ratio,
                                   attn_type=kwargs['attn_type'], P=kwargs['P']),
            TransformerDecoderUnit(feat_dim, nhead, pos_en_flag, mlp_ratio,
                                   attn_type=kwargs['attn_type'], P=kwargs['P'])
        ])

        self.conv0 = double_conv(feat_dim, feat_dim)
        self.conv1 = double_conv_down(feat_dim, feat_dim)
        self.conv2 = double_conv_down(feat_dim, feat_dim)
        self.conv3 = double_conv(feat_dim, feat_dim)
        self.conv4 = double_conv_up(feat_dim, feat_dim)
        self.conv5 = double_conv_up(feat_dim, feat_dim)

        self.conv6 = nn.Sequential(single_conv(feat_dim, feat_dim),
                                   nn.Conv2d(feat_dim, out_ch, 3, 1, 1))

    def forward(self, src, ref):
        assert src.shape == ref.shape, "Shapes of source and reference images \
                                        mismatch."
        if not self.training:
            N, C, H, W = src.shape
            mod_size = 4
            H_pad = mod_size - H % mod_size if not H % mod_size == 0 else 0
            W_pad = mod_size - W % mod_size if not W % mod_size == 0 else 0
            src = F.pad(src, (0, W_pad, 0, H_pad), 'replicate')
            ref = F.pad(ref, (0, W_pad, 0, H_pad), 'replicate')

        q_feat = self.unet_q(src)
        k_feat = self.unet_k(ref)
        v_feat = self.unet_v(ref)

        outputs = []
        for i in range(3):
            outputs.append(self.trans_dec[i](q_feat[i + 3], k_feat[i + 3], v_feat[i + 3]))

        f0 = self.conv0(outputs[2])  # H, W
        f1 = self.conv1(f0)  # H/2, W/2
        f1 = f1 + outputs[1]
        f2 = self.conv2(f1)  # H/4, W/4
        f2 = f2 + outputs[0]
        f3 = self.conv3(f2)  # H/4, W/4
        f3 = f3 + outputs[0] + f2
        f4 = self.conv4(f3)   # H/2, W/2
        f4 = f4 + outputs[1] + f1
        f5 = self.conv5(f4)   # H, W
        f5 = f5 + outputs[2] + f0

        out = self.conv6(f5)

        if not self.training:
            out = out[:, :, :H, :W]
        return out


@ARCH_REGISTRY.register()
class SingleUNet(Unet):
    def forward(self, x):
        if not self.training:
            N, C, H, W = x.shape
            mod_size = 4
            H_pad = mod_size - H % mod_size if not H % mod_size == 0 else 0
            W_pad = mod_size - W % mod_size if not W % mod_size == 0 else 0
            x = F.pad(x, (0, W_pad, 0, H_pad), 'replicate')

        # pdb.set_trace()
        feat0 = self.conv_in(x)    # H, W
        feat1 = self.conv1(feat0)   # H/2, W/2
        feat2 = self.conv2(feat1)    # H/4, W/4
        feat3 = self.conv3(feat2)    # H/4, W/4
        feat3 = feat3 + feat2     # H/4
        feat4 = self.conv4(feat3)    # H/2, W/2
        feat4 = feat4 + feat1    # H/2, W/2
        feat5 = self.conv5(feat4)   # H
        feat5 = feat5 + feat0   # H
        out = self.conv6(feat5)

        if not self.training:
            out = out[:, :, :H, :W]

        return out


@ARCH_REGISTRY.register()
class PPM_UNet(Unet):
    def __init__(self, in_ch, feat_ch, out_ch, bins):
        super().__init__(in_ch, feat_ch, out_ch)
        self.ppm = PPM(in_dim=feat_ch, bins=bins)

    def forward(self, x):
        if not self.training:
            N, C, H, W = x.shape
            mod_size = 4
            H_pad = mod_size - H % mod_size if not H % mod_size == 0 else 0
            W_pad = mod_size - W % mod_size if not W % mod_size == 0 else 0
            x = F.pad(x, (0, W_pad, 0, H_pad), 'replicate')

        # pdb.set_trace()
        feat0 = self.conv_in(x)    # H, W
        feat1 = self.conv1(feat0)   # H/2, W/2
        feat2 = self.conv2(feat1)    # H/4, W/4
        feat3 = self.conv3(feat2)    # H/4, W/4

        feat3 = self.ppm(feat3)     # H/4, W/4

        feat3 = feat3 + feat2     # H/4, W/4
        feat4 = self.conv4(feat3)    # H/2, W/2
        feat4 = feat4 + feat1    # H/2, W/2
        feat5 = self.conv5(feat4)   # H
        feat5 = feat5 + feat0   # H
        out = self.conv6(feat5)

        if not self.training:
            out = out[:, :, :H, :W]

        return out


class PPM(nn.Module):
    def __init__(self, in_dim, bins):
        super(PPM, self).__init__()
        reduction_dim = in_dim // len(bins)

        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)
        self.conv = single_conv(in_dim * 2, in_dim)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        out = self.conv(torch.cat(out, 1))
        return out


if __name__ == '__main__':
    height = 128
    width = 128
    model = Texformer(
        feat_dim=128,
        nhead=8,
        mlp_ratio=2,)
    print(model)

    src = torch.randn((2, 3, height, width))
    ref = torch.randn((2, 3, height, width))
    model.eval()
    with torch.no_grad():
        out = model(src, ref)
    model.train()

    print(out.shape)
