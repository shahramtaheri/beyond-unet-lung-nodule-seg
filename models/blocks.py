"""
Building blocks for the proposed Residual-Attention U-Net++.

Implements:
  - ResidualConvBlock : two 3x3 conv-BN-ReLU layers with an identity / 1x1-conv
                        skip connection (Eq. 1-2, Fig. 2.B of the paper).
  - AttentionGate      : additive soft-attention gate applied at every skip
                         connection (Eq. 3-5, Fig. 2.A of the paper).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    """
    Residual convolutional block used to replace the plain conv units of
    U-Net++ in both the encoder and (nested) decoder pathways.

    y = F(x, {W_i}) + H(x)                                            (Eq. 1)

    F(x, {W_i}) = ReLU( BN( W2 * ReLU( BN( W1 * x ) ) ) )              (Eq. 2)

    H(x) is the identity mapping when in_channels == out_channels,
    otherwise a 1x1 convolution (+ BN) that projects x to the output
    dimensionality so the addition is well defined.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # H(x): identity shortcut, or 1x1 projection if channel dims differ.
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual          # y = F(x, W_i) + H(x)
        out = self.relu(out)
        out = self.dropout(out)
        return out


class AttentionGate(nn.Module):
    """
    Additive attention gate applied at every skip connection between the
    encoder (or a nested decoder node) and the decoder, following Oktay et
    al.'s Attention U-Net (Eq. 3-5, Fig. 2.A of the paper).

    Inputs
    ------
    g : gating signal coming from the (coarser-resolution) decoder.
    x : encoder / skip features at the same spatial resolution as the
        decoder branch that will consume the gated output.

    q     = psi^T ( ReLU( Wx * x + Wg * g + b ) ) + b_psi              (Eq. 3)
    alpha = sigmoid(q)                                                (Eq. 4)
    x_att = alpha * x                                                 (Eq. 5)
    """

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int = None):
        super().__init__()
        if inter_channels is None:
            inter_channels = max(skip_channels // 2, 1)

        # Wg * g  (1x1 conv on the gating signal)
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        # Wx * x  (1x1 conv on the encoder/skip features)
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        # psi^T : linear transform -> single-channel attention logits
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
        )
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Align spatial size of the gating signal with the skip features
        # (handles off-by-one mismatches from pooling/upsampling).
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)

        theta_x = self.W_x(x)
        phi_g = self.W_g(g)

        q = self.relu(theta_x + phi_g)
        q = self.psi(q)
        alpha = self.sigmoid(q)                 # attention coefficients, alpha in [0, 1]

        x_att = x * alpha                       # x_att = alpha . x
        return x_att, alpha


class UpSample(nn.Module):
    """Bilinear upsampling by a factor of 2, used between U-Net++ nodes."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, target_size=None) -> torch.Tensor:
        if target_size is not None:
            return F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
