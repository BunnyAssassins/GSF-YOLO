# Derived from Ultralytics; AGPL-3.0. See the repository LICENSE.
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.torch_utils import fuse_conv_and_bn
from .conv import Conv

class DFL(nn.Module):

    def __init__(self, c1: int=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (b, _, a) = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

class SPPF(nn.Module):

    def __init__(self, c1: int, c2: int, k: int=5, n: int=3, shortcut: bool=False):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend((self.m(y[-1]) for _ in range(getattr(self, 'n', 3))))
        y = self.cv2(torch.cat(y, 1))
        return y + x if getattr(self, 'add', False) else y

class C2f(nn.Module):

    def __init__(self, c1: int, c2: int, n: int=1, shortcut: bool=False, g: int=1, e: float=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList((Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1]) for m in self.m))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend((m(y[-1]) for m in self.m))
        return self.cv2(torch.cat(y, 1))

class C3(nn.Module):

    def __init__(self, c1: int, c2: int, n: int=1, shortcut: bool=True, g: int=1, e: float=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class Bottleneck(nn.Module):

    def __init__(self, c1: int, c2: int, shortcut: bool=True, g: int=1, k: tuple[int, int]=(3, 3), e: float=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3k2(C2f):

    def __init__(self, c1: int, c2: int, n: int=1, c3k: bool=False, e: float=0.5, attn: bool=False, g: int=1, shortcut: bool=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList((nn.Sequential(Bottleneck(self.c, self.c, shortcut, g), PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))) if attn else C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)))

class C3k(C3):

    def __init__(self, c1: int, c2: int, n: int=1, shortcut: bool=True, g: int=1, e: float=0.5, k: int=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class RepVGGDW(torch.nn.Module):

    def __init__(self, ed: int) -> None:
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        if not hasattr(self, 'conv1'):
            return
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)
        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])
        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b
        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        self.conv = conv
        del self.conv1

class CIB(nn.Module):

    def __init__(self, c1: int, c2: int, shortcut: bool=True, e: float=0.5, lk: bool=False):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Sequential(Conv(c1, c1, 3, g=c1), Conv(c1, 2 * c_, 1), RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_), Conv(2 * c_, c2, 1), Conv(c2, c2, 3, g=c2))
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv1(x) if self.add else self.cv1(x)

class Attention(nn.Module):

    def __init__(self, dim: int, num_heads: int=8, attn_ratio: float=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** (-0.5)
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, C, H, W) = x.shape
        N = H * W
        qkv = self.qkv(x)
        (q, k, v) = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        attn = q.transpose(-2, -1) @ k * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x

class PSABlock(nn.Module):

    def __init__(self, c: int, attn_ratio: float=0.5, num_heads: int=4, shortcut: bool=True) -> None:
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x

class C2PSA(nn.Module):

    def __init__(self, c1: int, c2: int, n: int=1, e: float=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (a, b) = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

class DetailSemanticFusion(nn.Module):

    def __init__(self, ch: tuple=(), c2: int | None=None):
        super().__init__()
        if len(ch) < 2:
            raise ValueError('DetailSemanticFusion requires at least one detail feature and one semantic feature.')
        c2 = c2 or ch[0]
        self.detail_proj = Conv(ch[0], c2, k=3)
        self.semantic_proj = nn.ModuleList((Conv(c, c2, k=1) for c in ch[1:]))
        self.semantic_mix = Conv(c2, c2, k=3)
        self.gate = nn.Sequential(nn.Conv2d(c2 * 2, c2, kernel_size=1, bias=True), nn.Sigmoid())
        self.out = Conv(c2, c2, k=3)

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        detail = self.detail_proj(x[0])
        semantic = 0
        for (proj, feat) in zip(self.semantic_proj, x[1:]):
            semantic = semantic + F.interpolate(proj(feat), size=detail.shape[2:], mode='nearest')
        semantic = self.semantic_mix(semantic)
        gate = self.gate(torch.cat((detail, semantic), dim=1))
        return self.out(detail + gate * semantic)
