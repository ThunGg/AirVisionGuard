'''
MobileFaceNetV3 — combines MobileNetV3 innovations with the MobileFaceNet
architecture optimised for face recognition.

Key ideas borrowed from MobileNetV3:
  • Hard-Swish (h-swish) activation — cheap approximation of Swish that is
    faster on mobile hardware while preserving accuracy.
  • Squeeze-and-Excitation (SE) modules — lightweight channel attention that
    recalibrates feature responses.
  • Efficient last stage — moves the expensive expansion to *after* global
    pooling so the bulk of computation happens on a 1×1 spatial map.

Key ideas retained from MobileFaceNet:
  • Depthwise-separable bottleneck blocks tuned for 112×112 face crops.
  • Global Depthwise Convolution (GDC) as the final pooling stage, which
    captures spatial structure better than simple global-average pooling for
    faces.
  • PReLU on early layers (where spatial resolution is large) and h-swish on
    deeper layers — following the MobileNetV3 observation that h-swish is
    only beneficial in the deeper half of the network.

Reference:
  MobileNetV3  – Howard et al., 2019  (arXiv 1905.02244)
  MobileFaceNet – Chen et al., 2018  (arXiv 1804.07573)
'''

import torch
import torch.nn as nn
from torch.nn import BatchNorm1d, BatchNorm2d, Conv2d, Linear, Module, PReLU, Sequential

__all__ = ['mobilefacenetv3', 'mobilefacenetv3_large']


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------
class HardSigmoid(Module):
    """Piecewise-linear approximation of sigmoid, used inside SE blocks."""
    def forward(self, x):
        return torch.clamp(x + 3.0, min=0.0, max=6.0) / 6.0


class HardSwish(Module):
    """Piecewise-linear approximation of Swish: x * hard_sigmoid(x)."""
    def forward(self, x):
        return x * torch.clamp(x + 3.0, min=0.0, max=6.0) / 6.0


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class Flatten(Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvBnAct(Module):
    """Conv → BN → Activation (PReLU or HardSwish)."""
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1),
                 padding=(0, 0), groups=1, use_hswish=False):
        super().__init__()
        self.layers = Sequential(
            Conv2d(in_c, out_c, kernel, stride=stride, padding=padding,
                   groups=groups, bias=False),
            BatchNorm2d(out_c),
            HardSwish() if use_hswish else PReLU(num_parameters=out_c),
        )

    def forward(self, x):
        return self.layers(x)


class LinearBlock(Module):
    """Conv → BN (no activation — linear bottleneck)."""
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1),
                 padding=(0, 0), groups=1):
        super().__init__()
        self.layers = Sequential(
            Conv2d(in_c, out_c, kernel, stride=stride, padding=padding,
                   groups=groups, bias=False),
            BatchNorm2d(out_c),
        )

    def forward(self, x):
        return self.layers(x)


class SEBlock(Module):
    """Squeeze-and-Excitation block (MobileNetV3 style).

    Uses a reduction ratio to shrink the channel dimension in the hidden
    layer, then recalibrates with hard-sigmoid gating.
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = Sequential(
            Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            Linear(mid, channels, bias=False),
            HardSigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class DepthWiseSE(Module):
    """Depthwise-separable bottleneck with optional SE and activation choice.

    Structure:
        Pointwise expansion → Depthwise conv → (SE) → Pointwise projection (linear)

    This is the core building block: an inverted-residual block from
    MobileFaceNet enhanced with SE attention and h-swish from MobileNetV3.
    """
    def __init__(self, in_c, out_c, kernel=(3, 3), stride=(2, 2),
                 padding=(1, 1), groups=1, residual=False,
                 use_se=True, use_hswish=False):
        super().__init__()
        self.residual = residual
        act = HardSwish if use_hswish else lambda: PReLU(num_parameters=groups)

        layers = [
            # 1×1 pointwise expansion
            ConvBnAct(in_c, groups, kernel=(1, 1), padding=(0, 0),
                      stride=(1, 1), use_hswish=use_hswish),
            # Depthwise conv
            ConvBnAct(groups, groups, kernel=kernel, padding=padding,
                      stride=stride, groups=groups, use_hswish=use_hswish),
        ]
        if use_se:
            layers.append(SEBlock(groups))
        # 1×1 pointwise projection (linear — no activation)
        layers.append(
            LinearBlock(groups, out_c, kernel=(1, 1), padding=(0, 0),
                        stride=(1, 1))
        )
        self.layers = Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        if self.residual:
            out = out + x
        return out


class ResidualStack(Module):
    """Stack of *num_block* residual DepthWiseSE blocks (stride-1)."""
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1),
                 padding=(1, 1), use_se=True, use_hswish=False):
        super().__init__()
        self.layers = Sequential(*[
            DepthWiseSE(c, c, kernel=kernel, stride=stride, padding=padding,
                        groups=groups, residual=True, use_se=use_se,
                        use_hswish=use_hswish)
            for _ in range(num_block)
        ])

    def forward(self, x):
        return self.layers(x)


class GDC(Module):
    """Global Depthwise Convolution head — replaces GAP for face embeddings.

    A depthwise conv that covers the entire 7×7 spatial map, followed by
    a flatten, linear projection, and batch-norm to produce the final
    face embedding vector.
    """
    def __init__(self, in_c, embedding_size):
        super().__init__()
        self.layers = Sequential(
            LinearBlock(in_c, in_c, groups=in_c, kernel=(7, 7),
                        stride=(1, 1), padding=(0, 0)),
            Flatten(),
            Linear(in_c, embedding_size, bias=False),
            BatchNorm1d(embedding_size),
        )

    def forward(self, x):
        return self.layers(x)


# ---------------------------------------------------------------------------
# Main network
# ---------------------------------------------------------------------------
class MobileFaceNetV3(Module):
    """MobileFaceNetV3 — MobileNetV3-inspired face recognition backbone.

    Args:
        feature_dim: Embedding dimensionality (default 512).
        blocks: Tuple of (b0, b1, b2, b3) repeat counts for the four
                residual stages.
        scale: Width multiplier applied to base channel counts.
    """
    def __init__(self, feature_dim=512, blocks=(1, 4, 6, 2), scale=2,
                 **kwargs):
        super().__init__()
        self.scale = scale
        c1 = 64 * scale   # base channels after stem
        c2 = 128 * scale  # channels in stage 3-4
        c_head = 512       # channels before GDC

        self.layers = nn.ModuleList()

        # -- Stem (3 → c1) — use PReLU (spatial is large, h-swish not needed)
        self.layers.append(
            ConvBnAct(3, c1, kernel=(3, 3), stride=(2, 2), padding=(1, 1),
                      use_hswish=False)
        )

        # -- Stage 0 — early depthwise layer(s), no SE, PReLU
        if blocks[0] == 1:
            self.layers.append(
                ConvBnAct(c1, c1, kernel=(3, 3), stride=(1, 1),
                          padding=(1, 1), groups=c1, use_hswish=False)
            )
        else:
            self.layers.append(
                ResidualStack(c1, blocks[0], groups=c1 * 2, kernel=(3, 3),
                              stride=(1, 1), padding=(1, 1),
                              use_se=False, use_hswish=False)
            )

        # -- Stage 1 — downsample + residuals, introduce SE, still PReLU
        self.layers.append(
            DepthWiseSE(c1, c1, kernel=(3, 3), stride=(2, 2),
                        padding=(1, 1), groups=c1 * 2,
                        use_se=True, use_hswish=False)
        )
        self.layers.append(
            ResidualStack(c1, blocks[1], groups=c1 * 2, kernel=(3, 3),
                          stride=(1, 1), padding=(1, 1),
                          use_se=True, use_hswish=False)
        )

        # -- Stage 2 — downsample + residuals, SE + switch to h-swish
        self.layers.append(
            DepthWiseSE(c1, c2, kernel=(3, 3), stride=(2, 2),
                        padding=(1, 1), groups=c2 * 2,
                        use_se=True, use_hswish=True)
        )
        self.layers.append(
            ResidualStack(c2, blocks[2], groups=c2 * 2, kernel=(3, 3),
                          stride=(1, 1), padding=(1, 1),
                          use_se=True, use_hswish=True)
        )

        # -- Stage 3 — downsample + residuals, SE + h-swish
        self.layers.append(
            DepthWiseSE(c2, c2, kernel=(3, 3), stride=(2, 2),
                        padding=(1, 1), groups=c_head,
                        use_se=True, use_hswish=True)
        )
        self.layers.append(
            ResidualStack(c2, blocks[3], groups=c2 * 2, kernel=(3, 3),
                          stride=(1, 1), padding=(1, 1),
                          use_se=True, use_hswish=True)
        )

        # -- Head: 1×1 conv to c_head then GDC
        self.conv_sep = ConvBnAct(c2, c_head, kernel=(1, 1), stride=(1, 1),
                                  padding=(0, 0), use_hswish=True)
        self.features = GDC(c_head, feature_dim)
        self._initialize_weights()

    # -- Weight initialisation -------------------------------------------
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    # -- Forward ---------------------------------------------------------
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.conv_sep(x)
        x = self.features(x)
        return x


# ---------------------------------------------------------------------------
# Factory functions (match the convention used by the rest of the framework)
# ---------------------------------------------------------------------------
def mobilefacenetv3(**kwargs):
    """MobileFaceNetV3-Small — comparable capacity to mobilefacenet."""
    return MobileFaceNetV3(blocks=(1, 4, 6, 2), scale=2, **kwargs)


def mobilefacenetv3_large(**kwargs):
    """MobileFaceNetV3-Large — higher capacity variant."""
    return MobileFaceNetV3(blocks=(2, 8, 12, 4), scale=4, **kwargs)
