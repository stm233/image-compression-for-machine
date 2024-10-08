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

def mainEncoder(num_layers,embed_dim,depths,num_heads,window_size,mlp_ratio,qkv_bias,qk_scale,drop_rate,attn_drop_rate,dpr,norm_layer,use_checkpoint):
    layers = nn.ModuleList()
    for i_layer in range(num_layers):
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
            downsample=PatchMerging if (i_layer < num_layers - 1) else None,
            use_checkpoint=use_checkpoint,
            inverse=False)
        layers.append(layer)

    return layers


def mainDecoder(num_layers, embed_dim, depths, num_heads, window_size, mlp_ratio, qkv_bias, qk_scale, drop_rate,
                attn_drop_rate, dpr, norm_layer, use_checkpoint):
    syn_layers = nn.ModuleList()
    for i_layer in range(num_layers):
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
            downsample=PatchSplit if (i_layer < num_layers - 1) else None,
            use_checkpoint=use_checkpoint,
            inverse=True)
        syn_layers.append(layer)

    return syn_layers

def mainCNNencoder(N,M):
    return nn.Sequential(
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

def CRC_two_mainCNNencoder(N,M):
    return nn.Sequential(
        conv(6, N, kernel_size=5, stride=2),
        GDN(N),
        conv(N, N, kernel_size=5, stride=2),
        GDN(N),
        Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
        conv(N, N, kernel_size=5, stride=2),
        GDN(N),
        conv(N, M, kernel_size=5, stride=2),
        Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
    )

def mainCNNdecoderPart1(N,M):
    return nn.Sequential(
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

def mainCNNdecoderPart2(N):
    return nn.Sequential(
            deconv(256, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )

def mainCNNdecoder(N,M):
    return nn.Sequential(
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

def CRC_two_mainCNNdecoder(N,M):
    return nn.Sequential(
            Win_noShift_Attention(dim=M*2, num_heads=8, window_size=4, shift_size=2),
            deconv(M*2, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 256, kernel_size=5, stride=2),
            GDN(256, inverse=True),
            Win_noShift_Attention(dim=256, num_heads=8, window_size=8, shift_size=4),
            deconv(256, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )


def mainCNNcontextScale1(N,M):
    return nn.Sequential(
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
def mainCNNcontextScale2(N,M):
    return nn.Sequential(
            Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
            deconv(M, N, kernel_size=3, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=3, stride=2),
            # GDN(N, inverse=True),
            # Win_noShift_Attention(dim=256, num_heads=8, window_size=8, shift_size=4),
            # deconv(256, N, kernel_size=5, stride=2),
            # GDN(N, inverse=True),
            # deconv(N, 3, kernel_size=5, stride=2),
        )

def hyperEncoder():
    return nn.Sequential(
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

def hyperMean():
    return nn.Sequential(
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
def hyperContextMean(support_num,num_slices,max_support_slices,number):
    return nn.ModuleList(
        nn.Sequential(  # //self.num_slices
            conv(384 * support_num // num_slices + 384 // num_slices * min(i, max_support_slices),
                 224, stride=1, kernel_size=3),
            nn.GELU(),
            # conv(224, 176, stride=1, kernel_size=3),
            # nn.GELU(),
            # conv(176, 128, stride=1, kernel_size=3),
            # nn.GELU(),
            conv(224, 64, stride=1, kernel_size=3),
            nn.GELU(),
            conv(64, 384 // num_slices, stride=1, kernel_size=3),
        ) for i in range(num_slices * number * number)
    )

def hyperContextLRP(support_num,num_slices,max_support_slices,number):
    return nn.ModuleList(
            nn.Sequential( # //self.num_slices
                conv(384*support_num//num_slices  + 384 //num_slices * min(i + 1, max_support_slices+1), 224, stride=1, kernel_size=3),
                nn.GELU(),
                # conv(224, 176, stride=1, kernel_size=3),
                # nn.GELU(),
                # conv(176, 128, stride=1, kernel_size=3),
                # nn.GELU(),
                conv(224, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 384 //num_slices, stride=1, kernel_size=3),
            ) for i in range(num_slices*number*number)
        )

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
