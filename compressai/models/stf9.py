import torch
import math
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .utils import conv, update_registered_buffers, deconv
from compressai.layers import GDN
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.layers import conv3x3, subpel_conv3x3, Win_noShift_Attention
from compressai.ops import ste_round
from .base import CompressionModel

from .retinanet import model
from collections import OrderedDict

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, inverse=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchSplit(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(dim, dim * 2, bias=False)
        self.norm = norm_layer(dim)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = self.norm(x)
        x = self.reduction(x)           # B, L, C
        x = x.permute(0, 2, 1).contiguous().view(B, 2*C, H, W)
        x = self.shuffle(x)             # B, C//2 ,2H, 2W
        x = x.permute(0, 2, 3, 1).contiguous().view(B, 4 * L, -1)
        return x

class BasicLayer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 inverse=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(

                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                inverse=inverse)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            if isinstance(self.downsample, PatchMerging):
                Wh, Ww = (H + 1) // 2, (W + 1) // 2
            elif isinstance(self.downsample, PatchSplit):
                Wh, Ww = H * 2, W * 2
            return x_down, Wh, Ww
        else:
            return x, H, W


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


class SymmetricalTransFormer6(CompressionModel):
    def __init__(self,
                 pretrain_img_size=256,
                 patch_size=2,
                 in_chans=3,
                 embed_dim=48,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=4,
                 num_slices=6,
                 Mask_win_size=8,
                 num_sliding = 4,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.num_slices = num_slices
        self.max_support_slices = 12
        self.support_num = 24
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                inverse=False)
            self.layers.append(layer)

        depths = depths[::-1]
        num_heads = num_heads[::-1]
        self.syn_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** (3-i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchSplit if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                inverse=True)
            self.syn_layers.append(layer)

        # self.mu_layers = nn.ModuleList()
        # for i_layer in range(self.num_layers):
        #     layer = BasicLayer(
        #         dim=int(32),
        #         depth=depths[i_layer],
        #         num_heads=4,
        #         window_size=window_size,
        #         mlp_ratio=mlp_ratio,
        #         qkv_bias=qkv_bias,
        #         qk_scale=qk_scale,
        #         drop=drop_rate,
        #         attn_drop=attn_drop_rate,
        #         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
        #         norm_layer=norm_layer,
        #         downsample=None,
        #         use_checkpoint=use_checkpoint,
        #         inverse=True)

        # mu_depths = [2,6]
        # self.num_mu = len(mu_depths)
        # self.mu_Swin2 = nn.ModuleList()
        # for i in range(num_slices*4):
        #     mu_layer = nn.ModuleList()
        #     for i_layer in range(self.num_mu):
        #         layer = BasicLayer(
        #             dim=int(384 //self.num_slices),
        #             depth=mu_depths[i_layer],
        #             num_heads=4,
        #             window_size=8,
        #             mlp_ratio=mlp_ratio,
        #             qkv_bias=qkv_bias,
        #             qk_scale=qk_scale,
        #             drop=drop_rate,
        #             attn_drop=attn_drop_rate,
        #             drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
        #             norm_layer=norm_layer,
        #             downsample= None,
        #             use_checkpoint=use_checkpoint,
        #             inverse=True)
        #         mu_layer.append(layer)
        #
        #     self.mu_Swin2.append(mu_layer)
        #
        #
        # sigma_depths = [2,2]
        # self.num_sigma = len(sigma_depths)
        # self.sigma_Swin2 = nn.ModuleList()
        # for i in range(num_slices*4):
        #     sigma_layer = nn.ModuleList()
        #     for i_layer in range(self.num_sigma):
        #         layer = BasicLayer(
        #             dim=int(384 //self.num_slices),
        #             depth=sigma_depths[i_layer],
        #             num_heads=4,
        #             window_size=8,
        #             mlp_ratio=mlp_ratio,
        #             qkv_bias=qkv_bias,
        #             qk_scale=qk_scale,
        #             drop=drop_rate,
        #             attn_drop=attn_drop_rate,
        #             drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
        #             norm_layer=norm_layer,
        #             downsample= None,
        #             use_checkpoint=use_checkpoint,
        #             inverse=True)
        #         sigma_layer.append(layer)
        #
        #     self.sigma_Swin2.append(sigma_layer)
        #
        LRP_depths = [2,6]
        self.num_LRP = len(LRP_depths)
        self.LRP_Swin2 = nn.ModuleList()
        for i in range(num_slices*4):
            LRP_layer = nn.ModuleList()
            for i_layer in range(self.num_LRP):
                layer = BasicLayer(
                    dim=int(384 //self.num_slices),
                    depth=LRP_depths[i_layer],
                    num_heads=4,
                    window_size=8,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    downsample= None,
                    use_checkpoint=use_checkpoint,
                    inverse=True)
                LRP_layer.append(layer)

            self.LRP_Swin2.append(LRP_layer)

        self.end_conv = nn.Sequential(nn.Conv2d(embed_dim, embed_dim * patch_size ** 2, kernel_size=5, stride=1, padding=2),
                                      nn.PixelShuffle(patch_size),
                                      nn.Conv2d(embed_dim, 3, kernel_size=3, stride=1, padding=1),
                                      )

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

        self.num_features = num_features
        N = 192
        M = 384
        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
            Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
        )



        self.g_s1 = nn.Sequential(
            Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 256, kernel_size=5, stride=2),
            GDN(256, inverse=True),
            Win_noShift_Attention(dim=256, num_heads=8, window_size=8, shift_size=4),
            # deconv(N, N, kernel_size=5, stride=2),
            # GDN(N, inverse=True),
            # deconv(N, 3, kernel_size=5, stride=2),
        )

        self.g_s2 = nn.Sequential(
            deconv(256, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        self.human_g_s2 = nn.Sequential(
            Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 256, kernel_size=5, stride=2),
            GDN(256, inverse=True),
            Win_noShift_Attention(dim=256, num_heads=8, window_size=8, shift_size=4),
            deconv(256, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )


        self.h_a = nn.Sequential(
            conv3x3(384, 384),
            nn.GELU(),
            conv3x3(384, 336),
            nn.GELU(),
            conv3x3(336, 288, stride=2),
            nn.GELU(),
            conv3x3(288, 240),
            nn.GELU(),
            conv3x3(240, 192, stride=2),
        )

        self.h_mean_s = nn.Sequential(
            conv3x3(192, 240),
            nn.GELU(),
            subpel_conv3x3(240, 288, 2),
            nn.GELU(),
            conv3x3(288, 336),
            nn.GELU(),
            subpel_conv3x3(336, 384, 2),
            nn.GELU(),
            conv3x3(384, 384),
        )
        self.h_scale_s = nn.Sequential(
            conv3x3(192, 240),
            nn.GELU(),
            subpel_conv3x3(240, 288, 2),
            nn.GELU(),
            conv3x3(288, 336),
            nn.GELU(),
            subpel_conv3x3(336, 384, 2),
            nn.GELU(),
            conv3x3(384, 384),
        )
        number = 2
        self.cc_mean_transforms2 = nn.ModuleList(
            nn.Sequential( # //self.num_slices
                conv(384*self.support_num//self.num_slices  + 384 //self.num_slices * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 384 //self.num_slices, stride=1, kernel_size=3),
            ) for i in range(num_slices*number*number)
        )
        self.cc_scale_transforms2 = nn.ModuleList(
            nn.Sequential( # //self.num_slices
                conv(384*self.support_num//self.num_slices  + 384 //self.num_slices * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 384 //self.num_slices, stride=1, kernel_size=3),
            ) for i in range(num_slices*number*number)
        )
        self.lrp_transforms2 = nn.ModuleList(
            nn.Sequential( # //self.num_slices
                conv(384*self.support_num//self.num_slices  + 384 //self.num_slices * min(i + 1, self.max_support_slices+1), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 384 //self.num_slices, stride=1, kernel_size=3),
            ) for i in range(num_slices*number*number)
        )
        self.entropy_bottleneck = EntropyBottleneck(embed_dim * 4)
        self.entropy_bottleneck_human = EntropyBottleneck(embed_dim * 4)
        self.gaussian_conditional = GaussianConditional(None)
        self.gaussian_conditional_human = GaussianConditional(None)
        self._freeze_stages()

        self.teacherNet = model.resnet50(num_classes=80, pretrained=False)
        self.studentNet = model.studentresnet50(num_classes=80, pretrained=False)

        self.teacherNet.training = True
        self.studentNet.training = True

        # teachercheckpoint = "/home/tianma/Documents/STF-main/coco_resnet_50_map_0_335_state_dict.pt"
        # print("Loading", teachercheckpoint)
        # checkpoint = torch.load(teachercheckpoint, map_location='cuda')
        # self.teacherNet.load_state_dict(checkpoint, strict=True)  #
        # self.studentNet.load_state_dict(checkpoint, strict=True)

        self.promot_g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            nn.GELU(),
            conv(N, N, kernel_size=5, stride=2),
            nn.GELU(),
            conv(N, N, kernel_size=5, stride=2),
            nn.GELU(),
            conv(N, M, kernel_size=5, stride=2),
            nn.GELU(),
            Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
        )

        self.promot_g_s = nn.Sequential(
            Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
            nn.GELU(),
            deconv(M, N, kernel_size=5, stride=2),
            nn.GELU(),
            deconv(N, 256, kernel_size=5, stride=2),
            # nn.GELU(),
            # deconv(N, N, kernel_size=5, stride=2),
            # nn.GELU(),
            # deconv(N, 3, kernel_size=5, stride=2),
        )

        self.promot_h_a = nn.Sequential(
            conv3x3(384, 384),
            nn.GELU(),
            conv3x3(384, 336),
            nn.GELU(),
            conv3x3(336, 288, stride=2),
            nn.GELU(),
            conv3x3(288, 240),
            nn.GELU(),
            conv3x3(240, 192, stride=2),
        )

        self.promot_h_mean_s = nn.Sequential(
            conv3x3(192, 240),
            nn.GELU(),
            subpel_conv3x3(240, 288, 2),
            nn.GELU(),
            conv3x3(288, 336),
            nn.GELU(),
            subpel_conv3x3(336, 384, 2),
            nn.GELU(),
            conv3x3(384, 384),
        )
        self.promot_h_scale_s = nn.Sequential(
            conv3x3(192, 240),
            nn.GELU(),
            subpel_conv3x3(240, 288, 2),
            nn.GELU(),
            conv3x3(288, 336),
            nn.GELU(),
            subpel_conv3x3(336, 384, 2),
            nn.GELU(),
            conv3x3(384, 384),
        )

        self.human_g_a = nn.Sequential(
            conv(6, N, kernel_size=5, stride=2),
            nn.GELU(),
            conv(N, N, kernel_size=5, stride=2),
            nn.GELU(),
            conv(N, N, kernel_size=5, stride=2),
            nn.GELU(),
            conv(N, M, kernel_size=5, stride=2),
            # nn.GELU(),
            # Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
        )

        self.human_g_s = nn.Sequential(
            # Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
            # nn.GELU(),
            deconv(M*2, N, kernel_size=5, stride=2),
            nn.GELU(),
            deconv(N, N, kernel_size=5, stride=2),
            nn.GELU(),
            deconv(N, N, kernel_size=5, stride=2),
            nn.GELU(),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        self.human_h_a = nn.Sequential(
            conv3x3(384, 384),
            nn.GELU(),
            conv3x3(384, 336),
            nn.GELU(),
            conv3x3(336, 288, stride=2),
            nn.GELU(),
            conv3x3(288, 240),
            nn.GELU(),
            conv3x3(240, 192, stride=2),
        )

        self.human_h_mean_s = nn.Sequential(
            conv3x3(192, 240),
            nn.GELU(),
            subpel_conv3x3(240, 288, 2),
            nn.GELU(),
            conv3x3(288, 336),
            nn.GELU(),
            subpel_conv3x3(336, 384, 2),
            nn.GELU(),
            conv3x3(384, 384),
            nn.GELU(),
            conv(384, 384, stride=1, kernel_size=3),
            nn.GELU(),
            conv(384, 384, stride=1, kernel_size=3),
            nn.GELU(),
            conv(384, 384, stride=1, kernel_size=3),
            nn.GELU(),
            conv(384, 384, stride=1, kernel_size=3),
            nn.GELU(),
            conv(384, 384, stride=1, kernel_size=3),

        )
        self.human_h_scale_s = nn.Sequential(
            conv3x3(192, 240),
            nn.GELU(),
            subpel_conv3x3(240, 288, 2),
            nn.GELU(),
            conv3x3(288, 336),
            nn.GELU(),
            subpel_conv3x3(336, 384, 2),
            nn.GELU(),
            conv3x3(384, 384),
            nn.GELU(),
            conv(384, 384, stride=1, kernel_size=3),
            nn.GELU(),
            conv(384, 384, stride=1, kernel_size=3),
            nn.GELU(),
            conv(384, 384, stride=1, kernel_size=3),
            nn.GELU(),
            conv(384, 384, stride=1, kernel_size=3),
            nn.GELU(),
            conv(384, 384, stride=1, kernel_size=3),
        )
        self.human_context_decoder = nn.Sequential(
            conv(384, 384, stride=1, kernel_size=3),
            nn.GELU(),
            conv(384, 384, stride=1, kernel_size=3),
            nn.GELU(),
            conv(384, 384, stride=1, kernel_size=3),
            nn.GELU(),
            conv(384, 384, stride=1, kernel_size=3),
            nn.GELU(),
            conv(384, 384, stride=1, kernel_size=3),
        )

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


    def ZigzagSplits(self,inputs,num_slices):
        B,C,H,W = inputs.shape
        number = 2
        # window_size = 8
        pad_l = pad_t = 0
        pad_r = ((H//number) - H % (H//number)) % (H//number)
        pad_b = ((W//number) - W % (W//number)) % (W//number)
        x = F.pad(inputs, (pad_t, pad_b, pad_l, pad_r))
        _,_,Hpad,Wpad = x.shape
        # print(x.shape)

        x_slices = x.view(B,num_slices,C//num_slices,
                          number,H//number,
                          number,W//number)

        outOrders = []
        embedding_len = (H//number)*(W//number)*C//num_slices

        for i in range(max(number,number)):
            C_index = 0
            H_index = W_index = 0
            for j in range(num_slices*min((i+1),number)*min((i+1),number)):

                if max(H_index,W_index) < i and i > 0:
                    # print('N',C_index,H_index,W_index)
                    if C_index + 2 > num_slices : # or C_index +1> i
                        C_index = 0
                        if H_index + 2 > number or H_index+1 > i:
                            W_index = W_index + 1
                            H_index = 0
                        else:
                            H_index = H_index + 1

                    else:
                        C_index = C_index + 1
                    continue

                # print('Y',C_index,H_index,W_index,'i=',i)
                outOrders.append(x_slices[:,C_index,:,H_index,:,W_index,:].contiguous().unsqueeze(1))
                if C_index + 2 > num_slices : # or C_index +1 > i
                    C_index = 0
                    if H_index + 2 > number or H_index+1 > i:
                        W_index = W_index + 1
                        H_index = 0
                    else:
                        H_index = H_index + 1

                else:
                    C_index = C_index + 1

        zigzag = torch.cat(outOrders, 1).to(inputs.device)

        return zigzag, number, number


    def ZigzagReverse(self,inputs,num_slices,num_H,num_W):
        # inputs : list
        B,N,C,H,W = inputs.shape
        out_C = C*num_slices
        out_H = H*num_H
        out_W = W*num_W

        output = torch.zeros(B,out_C,out_H,out_W, device = inputs.device)
        output = output.view(B,num_slices,C,
                               num_H     ,H,
                               num_W     ,W)
        inputs_index = 0

        for i in range(max(num_H, num_W)):
            C_index = 0
            H_index = W_index = 0
            for j in range(num_slices*min((i+1),num_H)*min((i+1),num_W)):

                if max(H_index,W_index) < i and i > 0:
                    # print('N',C_index,H_index,W_index)
                    if C_index + 2 > num_slices: # or C_index +1> i:
                        C_index = 0
                        if H_index + 2 > num_H or H_index+1 > i:
                            W_index = W_index + 1
                            H_index = 0
                        else:
                            H_index = H_index + 1

                    else:
                        C_index = C_index + 1
                    continue
                # print('Y',C_index,H_index,W_index,'i=',i)
                output[:,C_index,:,H_index,:,W_index,:] = inputs[:,inputs_index]
                inputs_index = inputs_index + 1
                if C_index + 2 > num_slices: # or C_index +1 > i:
                    C_index = 0
                    if H_index + 2 > num_H or H_index+1 > i:
                        W_index = W_index + 1
                        H_index = 0
                    else:
                        H_index = H_index + 1

                else:
                    C_index = C_index + 1

        output = output.view(B,out_C,out_H,out_W).contiguous()
        return output

    def forward(self, x):
        """Forward function."""
        # compressH, Teacher_output_features, Teacher_classification, Teacher_regression, Teacher_anchors = self.teacherNet(x)
        compressH, Teacher_output_features, Teacher_classification, Teacher_regression, Teacher_anchors = None, None, None, None, None
        inputIMGs = x.clone()
        # x = self.patch_embed(x)
        #
        # Wh, Ww = x.size(2), x.size(3)
        # x = x.flatten(2).transpose(1, 2)
        # x = self.pos_drop(x)
        # for i in range(self.num_layers):
        #     layer = self.layers[i]
        #     x, Wh, Ww = layer(x, Wh, Ww)
        #
        # y = x
        y = self.g_a(x)
        # promot_y = self.promot_g_a(x)
        # y = promot_y + y


        C = self.embed_dim * 8
        # y = y.view(-1, Wh, Ww, C).permute(0, 3, 1, 2).contiguous()
        y_shape = y.shape[2:]

        z = self.h_a(y)
        # promot_z = self.promot_h_a(y)
        # z = z + promot_z
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        # promot_latent_scales = self.promot_h_scale_s(z_hat)
        # latent_scales = latent_scales + promot_latent_scales
        latent_means = self.h_mean_s(z_hat)
        # promot_latent_means = self.promot_h_mean_s(z_hat)
        # latent_means = latent_means + promot_latent_means
        B,C,H,W = latent_scales.shape
        number = 2

        # y = y.flatten(2).transpose(1, 2)
        y_zigzag, num_H, num_W = self.ZigzagSplits(y, self.num_slices)
        scales_zigzag, _ , _    = self.ZigzagSplits(latent_scales,self.num_slices)
        means_zigzag, _ , _   = self.ZigzagSplits(latent_means,self.num_slices)

        # y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index in range(self.num_slices* num_H* num_W):
            # support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            support_slices = (y_hat_slices if self.max_support_slices > slice_index else y_hat_slices[slice_index-self.max_support_slices:])

            support_num = self.support_num
            # meanInput = ([means_zigzag[:,slice_index:slice_index+self.max_support_slices,:,:,:]] if slice_index > slice_index else y_hat_slices[slice_index-self.max_support_slices:])
            if slice_index+support_num > self.num_slices* num_H* num_W:
                meanInput = means_zigzag[:,-support_num:,:,:,:].view(-1,384*support_num//self.num_slices,(H//number),(W//number))
            else:
                meanInput = means_zigzag[:,slice_index:slice_index+support_num,:,:,:].view(-1,384*support_num//self.num_slices,(H//number),(W//number))
            mean_support = torch.cat([meanInput] + support_slices, dim=1)
            # print(slice_index,'support_slices',len(support_slices),mean_support.shape)
            mu = self.cc_mean_transforms2[slice_index](mean_support)
            # mu = mu[:, :, :y_shape[0], :y_shape[1]]

            if slice_index+support_num > self.num_slices* num_H* num_W:
                scaleInput = scales_zigzag[:,-support_num:,:,:,:].view(-1,384*support_num//self.num_slices,(H//number),(W//number))
            else:
                scaleInput = scales_zigzag[:,slice_index:slice_index+support_num,:,:,:].view(-1,384*support_num//self.num_slices,(H//number),(W//number))
            # scaleInput = scales_zigzag.view(-1,384*number*number,(H//number),(W//number))
            scale_support = torch.cat([scaleInput] + support_slices, dim=1)
            scale = self.cc_scale_transforms2[slice_index](scale_support)
            # scale = scale[:, :, :y_shape[0], :y_shape[1]]

            # mu1 = mu.clone()
            # mu1 = mu1.permute(0, 2, 3, 1).contiguous().view(-1, (H//number)*(W//number), 384 //self.num_slices)
            # for i in range(self.num_mu):
            #     mu_layer = self.mu_Swin2[slice_index][i]#.to(y.device)
            #     mu1, _, _ = mu_layer(mu1, (H//number), (W//number))
            #
            # mu = mu + mu1.view(-1, (H//number), (W//number), 384 //self.num_slices).permute(0, 3, 1, 2).contiguous()

            # scale1 = scale.clone()
            # scale1 = scale1.permute(0, 2, 3, 1).contiguous().view(-1,  (H//number)*(W//number), 384 //self.num_slices)
            # for i in range(self.num_sigma):
            #     layer = self.sigma_Swin2[slice_index][i]#.to(y.device)
            #     scale1, _, _ = layer(scale1,(H//number), (W//number))
            #
            # scale = scale + scale1.view(-1, (H//number), (W//number), 384 //self.num_slices).permute(0, 3, 1, 2).contiguous()

            _, y_slice_likelihood = self.gaussian_conditional(y_zigzag[:,slice_index,:,:,:], scale, mu)

            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_zigzag[:,slice_index,:,:,:] - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms2[slice_index](lrp_support)

            # lrp1 = lrp.clone()
            # lrp1 = lrp1.permute(0, 2, 3, 1).contiguous().view(-1, (H//number)*(W//number), 384 //self.num_slices)
            # for i in range(self.num_LRP):
            #     layer = self.LRP_Swin2[slice_index][i]#.to(y.device)
            #     lrp1, _, _ = layer(lrp1, (H//number), (W//number))
            #
            # lrp = lrp + lrp1.view(-1, (H//number), (W//number), 384 //self.num_slices).permute(0, 3, 1, 2).contiguous()
            #
            #
            # lrp = 0.5 * torch.tanh(lrp)
            # y_hat_slice += lrp


            y_hat_slices.append(y_hat_slice)

        # y_hat = torch.cat(y_hat_slices, dim=1)
        y_hat = torch.cat(y_hat_slices, dim=1).view(-1,self.num_slices* num_H* num_W,384 //self.num_slices,(H//number),(W//number))
        y_hat = self.ZigzagReverse(y_hat,self.num_slices, number, number)
        y_likelihoods = torch.cat(y_likelihood, dim=1)

        # y_hat = y_hat.permute(0, 2, 3, 1).contiguous().view(-1, Wh*Ww, C)
        # for i in range(self.num_layers):
        #     layer = self.syn_layers[i]
        #     y_hat, Wh, Ww = layer(y_hat, Wh, Ww)

        # x_hat = self.g_s(y_hat)
        h_hat1 = self.g_s1(y_hat)
        # promot_h_hat = self.promot_g_s(y_hat)
        # h_hat = promot_h_hat + h_hat
        decompressImage = self.g_s2(h_hat1)

        # decompressH = self.end_conv(y_hat.view(-1, Wh, Ww, self.embed_dim).permute(0, 3, 1, 2).contiguous())

        # Student_compressH, Student_output_features, Student_classification, Student_regression, Student_anchors, scores, labels, boxes = self.studentNet(h_hat,decompressImage)
        # y_likelihoods, z_likelihoods, Student_classification, Student_regression = None,None,None,None
        Student_compressH, Student_output_features, Student_classification, Student_regression, Student_anchors, scores, labels, boxes = None,None,None,None,None,None,None,None
        # Student_compressH, Student_output_features, Student_classification, Student_regression, Student_anchors, scores, labels, boxes = self.studentNet(x)

        decompressImage2 = self.human_g_s2(y_hat)
        human_support = torch.cat([inputIMGs, decompressImage2], dim=1)
        human_y = self.human_g_a(human_support)

        human_z = self.human_h_a(human_y)
        # promot_z = self.promot_h_a(y)
        # z = z + promot_z
        _, human_z_likelihoods = self.entropy_bottleneck_human(z)
        human_z_offset = self.entropy_bottleneck_human._get_medians()
        human_z_tmp = human_z - human_z_offset
        human_z_hat = ste_round(human_z_tmp) + human_z_offset

        human_latent_scales = self.human_h_scale_s(human_z_hat)
        human_latent_means = self.human_h_mean_s(human_z_hat)

        _, human_y_likelihood = self.gaussian_conditional_human(human_y, human_latent_scales, human_latent_means)

        # human_y_likelihood.append(y_slice_likelihood)
        human_y_hat= ste_round(human_y - human_latent_means) + human_latent_means

        context = self.human_context_decoder(y_hat)
        decoder_support = torch.cat([human_y_hat, context], dim=1)
        human_deimage = self.human_g_s(decoder_support)

        return {
            "decompressedImage":human_deimage,
            "compressH": compressH,
            "decompressH": Student_compressH,
            "likelihoods": {"y": human_y_likelihood, "z": human_z_likelihoods},
            "Student_output_features": Student_output_features,
            "Teacher_output_features": Teacher_output_features,
            "Student_classification": Student_classification,
            "Student_regression": Student_regression,
            "Student_anchors": Student_anchors,
            "scores": scores,
            "labels": labels,
            "boxes": boxes,
        }

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict)
        return net



if __name__ == '__main__':
    # x = [torch.rand(3, 640, 640), torch.rand(3, 640, 640)]
    x = torch.rand(1, 3, 1280, 1280)
    teacher_model = TeacherModel()

    compressH,output_features,classification, regression, anchors = teacher_model(x)
    test_retina = teacher_model.model.eval(x)
