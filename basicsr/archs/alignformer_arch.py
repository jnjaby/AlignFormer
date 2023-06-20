from builtins import print
import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
# from basicsr.archs.pwcnet_arch import FlowGenerator
from basicsr.archs.arch_util import resize_flow, flow_warp
from mmcv.cnn import constant_init


def softmax_attention(q, k, v):
    # n x 1(k^2) x nhead x d x h x w
    h, w = q.shape[-2], q.shape[-1]

    q = q.permute(0, 2, 4, 5, 1, 3)  # n x nhead x h x w x 1 x d
    k = k.permute(0, 2, 4, 5, 3, 1)  # n x nhead x h x w x d x k^2
    v = v.permute(0, 2, 4, 5, 1, 3)  # n x nhead x h x w x k^2 x d

    N = q.shape[-1]  # scaled attention
    attn = torch.matmul(q / N ** 0.5, k)
    attn = F.softmax(attn, dim=-1)
    output = torch.matmul(attn, v)

    output = output.permute(0, 4, 1, 5, 2, 3).squeeze(1)
    attn = attn.permute(0, 4, 1, 5, 2, 3).squeeze(1)

    return output, attn


# def dotproduct_attention(q, k, v):
#     # n x 1(k^2) x nhead x d x h x w
#     h, w = q.shape[-2], q.shape[-1]

#     q = q.permute(0, 2, 4, 5, 1, 3)  # n x nhead x h x w x 1 x d
#     k = k.permute(0, 2, 4, 5, 3, 1)  # n x nhead x h x w x d x k^2
#     v = v.permute(0, 2, 4, 5, 1, 3)  # n x nhead x h x w x k^2 x d

#     N = q.shape[-1]  # scaled attention
#     attn = torch.matmul(q / N, k)
#     output = torch.matmul(attn, v)

#     output = output.permute(0, 4, 1, 5, 2, 3).squeeze(1)
#     attn = attn.permute(0, 4, 1, 5, 2, 3).squeeze(1)

#     return output, attn


# temporal for global attention.
def dotproduct_attention(q, k, v):
    # b x n x d x h x w
    h, w = q.shape[-2], q.shape[-1]

    q = q.flatten(-2).transpose(-2, -1)  # b x n x hw x d
    k = k.flatten(-2)                    # b x n x d x hw
    v = v.flatten(-2).transpose(-2, -1)  # b x n x hw x d

    N = k.shape[-1]
    attn = None
    tmp = torch.matmul(k, v) / N
    output = torch.matmul(q, tmp)

    output = output.transpose(-2, -1)
    output = output.view(*output.shape[:-1], h, w)

    return output, attn



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


class TransformerUnit(nn.Module):
    def __init__(self, feat_dim, n_head=8, pos_en_flag=True,
                 mlp_ratio=2, k_size=5, attn_type='softmax', fuse_type=None):
        super().__init__()
        self.feat_dim = feat_dim
        self.attn_type = attn_type
        self.fuse_type = fuse_type
        self.pos_en_flag = pos_en_flag

        self.pos_en = PosEnSine(self.feat_dim // 2)
        self.attn = MultiheadAttention(feat_dim, n_head, k_size=k_size)

        mlp_hidden_dim = int(feat_dim * mlp_ratio)
        self.mlp = MLP(in_features=feat_dim, hidden_features=mlp_hidden_dim)
        self.norm = nn.GroupNorm(1, self.feat_dim)

        if fuse_type:
            if fuse_type == 'conv':
                self.fuse_conv = double_conv(in_ch=feat_dim, out_ch=feat_dim)
            elif fuse_type == 'mask':
                self.fuse_conv = double_conv(in_ch=feat_dim, out_ch=feat_dim)

    def forward(self, q, k, v, flow, mask=None):
        if q.shape[-2:] != flow.shape[-2:]:
            # pdb.set_trace()
            flow = resize_flow(flow, 'shape', q.shape[-2:])
        if mask != None and q.shape[-2:] != mask.shape[-2:]:
            # pdb.set_trace()
            mask = F.interpolate(mask, size=q.shape[-2:], mode='nearest')
        if self.pos_en_flag:
            q_pos_embed = self.pos_en(q)
            k_pos_embed = self.pos_en(k)
        else:
            q_pos_embed = 0
            k_pos_embed = 0

        # print(flow)
        # cross-multi-head attention
        out, attn = self.attn(q=q + q_pos_embed, k=k + k_pos_embed, v=v, flow=flow,
                              attn_type=self.attn_type)
        # print(attn.shape)

        if self.fuse_type:
            if self.fuse_type == 'conv':
                out = out + self.fuse_conv(q)
            elif self.fuse_type == 'mask':
                try:
                    assert mask != None, "No mask found."
                except:
                    pdb.set_trace()
                out = (1 - mask) * out + mask * self.fuse_conv(q)

        # feed forward
        out = out + self.mlp(out)
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
    def __init__(self, feat_dim, n_head, k_size=5, d_k=None, d_v=None):
        super().__init__()
        if d_k is None:
            d_k = feat_dim // n_head
        if d_v is None:
            d_v = feat_dim // n_head

        self.n_head = n_head
        self.k_size = k_size
        self.d_k = d_k
        self.d_v = d_v

        # pre-attention projection
        self.w_qs = nn.Conv2d(feat_dim, n_head * d_k, 1, bias=False)
        self.w_ks = nn.Conv2d(feat_dim, n_head * d_k, 1, bias=False)
        self.w_vs = nn.Conv2d(feat_dim, n_head * d_v, 1, bias=False)

        # after-attention combine heads
        self.fc = nn.Conv2d(n_head * d_v, feat_dim, 1, bias=False)

    def forward(self, q, k, v, flow, attn_type='softmax'):
        # input: n x c x h x w
        # flow: n x 2 x h x w
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        # Pass through the pre-attention projection:
        # n x c x h x w   ---->   n x (nhead*dk) x h x w
        q = self.w_qs(q)
        k = self.w_ks(k)
        v = self.w_vs(v)

        n, c, h, w = q.shape

        # ------ Sampling K and V features ---------
        sampling_grid = flow_to_grid(flow, self.k_size)
        # sampled feature
        # n x k^2 x c x h x w
        sample_k_feat = flow_guide_sampler(k, sampling_grid, k_size=self.k_size)
        sample_v_feat = flow_guide_sampler(v, sampling_grid, k_size=self.k_size)

        # Reshape for multi-head attention.
        # n x k^2 x nhead x dk x h x w
        q = q.view(n, 1, n_head, d_k, h, w)
        k = sample_k_feat.view(n, self.k_size**2, n_head, d_k, h, w)
        v = sample_v_feat.view(n, self.k_size**2, n_head, d_v, h, w)

        # -------------- Attention -----------------
        if attn_type == 'softmax':
            # n x 1 x nhead x dk x h x w --> n x nhead x dv x h x w
            q, attn = softmax_attention(q, k, v)
        elif attn_type == 'dot':
            q, attn = dotproduct_attention(q, k, v)
        else:
            raise NotImplementedError(f'Unknown attention type {attn_type}')

        # Concatenate all the heads together
        # n x (nhead*dv) x h x w
        q = q.reshape(n, -1, h, w)
        q = self.fc(q)   # n x c x h x w



        # # --------  Processing in for loop  ---------
        # q_new = torch.zeros((n, n_head, d_v, h, w), device=q.device)
        # attn = torch.zeros((n, n_head, self.k_size**2, h, w))

        # for i in range(n):
        #     # input: 1 x c x h x w
        #     # flow: 1 x 2 x h x w
        #     sampling_grid = flow_to_grid(flow[i:i + 1, ...], self.k_size)
        #     # 1 x k^2 x c x h x w
        #     sample_k_feat = flow_guide_sampler(k[i:i + 1, ...], sampling_grid,
        #                                        k_size=self.k_size)
        #     sample_v_feat = flow_guide_sampler(v[i:i + 1, ...], sampling_grid,
        #                                        k_size=self.k_size)

        #     # 1 x k^2 x nhead x dk x h x w
        #     q_split = q[i:i + 1, ...].view(1, 1, n_head, d_k, h, w)
        #     k_split = sample_k_feat.view(1, self.k_size**2, n_head, d_k, h, w)
        #     v_split = sample_v_feat.view(1, self.k_size**2, n_head, d_v, h, w)

        #     # 1 x 1 x nhead x dk x h x w --> 1 x nhead x dv x h x w
        #     q_split, attn_split = softmax_attention(q_split, k_split, v_split)
        #     q_new[i, ...] = q_split
        #     attn[i, ...] = attn_split

        # # Concatenate all the heads together
        # # n x nhead x dv x h x w   ---->   n x (nhead*dv) x h x w
        # q = q_new.reshape(n, -1, h, w)
        # q = self.fc(q)   # n x c x h x w

        return q, attn


def flow_to_grid(flow, k_size=5):
    # flow (Tensor): Tensor with size (n, 2, h, w), normal value.
    # samples = flow + grid + shift
    # n, h, w, _ = flow.size()
    n, _, h, w = flow.size()
    padding = (k_size - 1) // 2

    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid_y = grid_y[None, ...].expand(k_size**2, -1, -1).type_as(flow)
    grid_x = grid_x[None, ...].expand(k_size**2, -1, -1).type_as(flow)

    shift = torch.arange(0, k_size).type_as(flow) - padding
    shift_y, shift_x = torch.meshgrid(shift, shift)
    shift_y = shift_y.reshape(-1, 1, 1).expand(-1, h, w) # k^2, h, w
    shift_x = shift_x.reshape(-1, 1, 1).expand(-1, h, w) # k^2, h, w

    samples_y = grid_y + shift_y # k^2, h, w
    samples_x = grid_x + shift_x # k^2, h, w
    samples_grid = torch.stack((samples_x, samples_y), 3) # k^2, h, w, 2
    samples_grid = samples_grid[None, ...].expand(n, -1, -1, -1, -1) # n, k^2, h, w, 2

    flow = flow.permute(0, 2, 3, 1)[:, None, ...].expand(-1, k_size**2, -1, -1, -1)

    vgrid = samples_grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[..., 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[..., 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=4).view(-1, h, w, 2)
    # vgrid_scaled.requires_grad = False
    return vgrid_scaled


def flow_guide_sampler(feat, vgrid_scaled, k_size=5, interp_mode='bilinear',
                       padding_mode='zeros', align_corners=True):
    # feat (Tensor): Tensor with size (n, c, h, w).
    # vgrid (Tensor): Tensor with size (nk^2, h, w, 2)
    n, c, h, w = feat.size()
    feat = feat.view(n, 1, c, h, w).expand(-1, k_size**2, -1, -1, -1).reshape(-1, c, h, w)
    sample_feat = F.grid_sample(feat, vgrid_scaled,
                                mode=interp_mode, padding_mode=padding_mode,
                                align_corners=align_corners).view(n, k_size**2, c, h, w)
    return sample_feat


# def zero_flow(src, ref):
#     N, C, H, W = src.shape
#     flow = torch.zeros((N, 2, H, W), dtype=src.dtype, device=src.device)
#     return flow


@ARCH_REGISTRY.register()
class AlignFormer(nn.Module):
    def __init__(self,
                 feat_dim=128,
                 ref_ch=3,
                 src_ch=3,
                 out_ch=3,
                 nhead=8,
                 mlp_ratio=2,
                 pos_en_flag=True,
                 k_size=5,
                 attn_type='softmax',
                 flow_type='pwc',
                 fuse_type=None,
                 dam_flag=False,
                 **kwargs):
        super().__init__()

        self.dam_flag = dam_flag

        if flow_type == 'spynet':
            from basicsr.archs.spynet_arch import FlowGenerator
            self.flow_estimator = FlowGenerator(load_path=kwargs['flow_model_path'])
        elif flow_type == 'pwc':
            from basicsr.archs.pwcnet_arch import FlowGenerator
            self.flow_estimator = FlowGenerator(path=kwargs['flow_model_path'])
        elif flow_type == 'raft':
            from basicsr.archs.raft_arch import FlowGenerator
            self.flow_estimator = FlowGenerator(load_path=kwargs['flow_model_path'],
                                                requires_grad=kwargs['flow_ft'])
        # elif flow_type == 'zero':
        #     self.flow_estimator = zero_flow
        else:
            raise ValueError(f'Unrecognized flow type: {self.flow_type}.')

        if dam_flag:
            if kwargs['dam_ft']:
                assert kwargs['dam_path'] != None
            from basicsr.archs.dam_arch import DAModule
            self.DAM = DAModule(in_ch=src_ch, feat_ch=kwargs['dam_feat'], out_ch=src_ch,
                                demodulate=kwargs['dam_demodulate'],
                                load_path=kwargs['dam_path'], requires_grad=kwargs['dam_ft'])

        self.unet_q = Unet(src_ch, feat_dim, feat_dim)
        self.unet_k = Unet(ref_ch, feat_dim, feat_dim)

        self.trans_unit = nn.ModuleList([
            TransformerUnit(feat_dim, nhead, pos_en_flag, mlp_ratio, k_size, attn_type, fuse_type),
            TransformerUnit(feat_dim, nhead, pos_en_flag, mlp_ratio, k_size, attn_type, fuse_type),
            TransformerUnit(feat_dim, nhead, pos_en_flag, mlp_ratio, k_size, attn_type, fuse_type)])

        self.conv0 = double_conv(feat_dim, feat_dim)
        self.conv1 = double_conv_down(feat_dim, feat_dim)
        self.conv2 = double_conv_down(feat_dim, feat_dim)
        self.conv3 = double_conv(feat_dim, feat_dim)
        self.conv4 = double_conv_up(feat_dim, feat_dim)
        self.conv5 = double_conv_up(feat_dim, feat_dim)

        self.conv6 = nn.Sequential(single_conv(feat_dim, feat_dim),
                                   nn.Conv2d(feat_dim, out_ch, 3, 1, 1))

        if not kwargs['main_ft']:
            self.eval()
            for key, param in self.named_parameters():
                if 'flow_estimator' not in key and 'DAM' not in key:
                    param.requires_grad = False
        else:
            self.train()
            for key, param in self.named_parameters():
                if 'flow_estimator' not in key and 'DAM' not in key:
                    param.requires_grad = True

    def forward(self, src, ref, mask=None):
        assert src.shape == ref.shape, "Shapes of source and reference images \
                                        mismatch."
        if not self.training:
            N, C, H, W = src.shape
            mod_size = 4
            H_pad = mod_size - H % mod_size if not H % mod_size == 0 else 0
            W_pad = mod_size - W % mod_size if not W % mod_size == 0 else 0
            src = F.pad(src, (0, W_pad, 0, H_pad), 'replicate')
            ref = F.pad(ref, (0, W_pad, 0, H_pad), 'replicate')

        if self.dam_flag:
            src = self.DAM(src, ref)

        # with torch.no_grad():
        #     flow = self.flow_estimator(src, ref).detach()
        flow = self.flow_estimator(src, ref)

        q_feat = self.unet_q(src)
        k_feat = self.unet_k(ref)

        outputs = []
        for i in range(3):
            if mask != None:
                mask = mask[:, 0:1, :, :]
                outputs.append(
                    self.trans_unit[i](q_feat[i + 3], k_feat[i + 3], k_feat[i + 3],
                                       flow, mask)
                )
            else:
                outputs.append(
                    self.trans_unit[i](q_feat[i + 3], k_feat[i + 3], k_feat[i + 3], flow)
                )

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


if __name__ == '__main__':
    height = 256
    width = 256
    batch_size = 4
    model = AlignFormer(
        feat_dim=64,
        nhead=4,
        mlp_ratio=2,
        k_size=5,
        attn_type='softmax',
        fuse_type='mask',
        flow_type='raft',
        flow_model_path='../../experiments/pretrained_models/RAFT/raft-things.pth',
        dam_flat=True,
        dam_ft=False,
        dam_feat=32,
        dam_demodulte=True,
        dam_path='../../experiments/FGTransformer/119_DAM_f32_demod_ip110_cx_conv44/models/net_g_latest.pth'
    ).cuda()
    print(model)

    # import torch.autograd.profiler as profiler

    src = torch.randn((batch_size, 3, height, width)).cuda()
    ref = torch.randn((batch_size, 3, height, width)).cuda()
    mask = torch.randn((batch_size, 1, height, width)).cuda()
    # model.eval()

    # warm up
    # for i in range(10):
    #     out = model(src, ref)

    out = model(src, ref, mask)

    # with profiler.profile(with_stack=True, profile_memory=True) as prof:
    #     out = model(src, ref)

    # print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_memory_usage", row_limit=10))

    # with torch.no_grad():
    #     out = model(src, ref)
    # model.train()
    # pdb.set_trace()
    # for k, v in model.named_parameters():
    #     print(k, v.requires_grad)

    print('Max memory usage: {0:.4f} GB'.format(torch.cuda.max_memory_allocated() / 1e9))
    print('Memory usage: {0:.4f} GB'.format(torch.cuda.memory_allocated() / 1e9))
    print(out.shape)
