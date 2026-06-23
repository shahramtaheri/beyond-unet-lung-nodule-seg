"""
Beyond U-Net++: Residual and Attention Enhancements for Precise Lung Nodule
Segmentation.

This module implements the proposed architecture described in Section 4.2
of the paper:

  - Backbone: U-Net++ (Zhou et al.) with its nested, densely-connected
    skip-pathway topology, depth = 4 encoder/decoder levels
    (filters: 64, 128, 256, 512) plus a 1024-filter bottleneck.
  - Every convolutional unit (encoder, nested decoder nodes, and final
    decoder) is replaced by a ResidualConvBlock (Sec. 4.2.2).
  - An AttentionGate is inserted on every skip connection feeding a
    decoder node, gated by the coarser-resolution feature map that is
    being upsampled into it (Sec. 4.2.3).
  - Deep supervision at intermediate decoder nodes is omitted, matching
    the paper ("we omit deeply supervised auxiliary outputs ... to keep
    the optimization focused on the final segmentation prediction").
  - Output: single-channel logit map; apply sigmoid externally (the loss
    functions in `losses/` expect probabilities after sigmoid, or you can
    use the provided `predict_mask` helper).

Notation follows the U-Net++ paper: X^{i,j} denotes the node at encoder
depth i (0 = deepest resolution... here we use i = 0 as the finest/input
resolution, growing downward, matching common U-Net++ implementations)
and column j (j = 0 is the pure encoder column; j > 0 are nested decoder
columns).
"""

from typing import Dict

import torch
import torch.nn as nn

from .blocks import ResidualConvBlock, AttentionGate, UpSample


class ResidualAttentionUNetPlusPlus(nn.Module):
    """
    Residual + Attention U-Net++ for 2D lung nodule segmentation.

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for a single CT slice, as in the paper).
    out_channels : int
        Number of output segmentation channels (1 for binary nodule mask).
    base_filters : tuple[int, int, int, int]
        Number of filters at encoder levels 0..3 (default matches the
        paper: 64, 128, 256, 512).
    bottleneck_filters : int
        Number of filters in the bottleneck (default 1024, as in the paper).
    dropout : float
        Dropout probability applied inside each residual block (0.0 to
        disable; not explicitly specified in the paper, default off).
    deep_supervision : bool
        If True, also return intermediate decoder-column outputs
        (X^{0,1}, X^{0,2}, X^{0,3}) in addition to the final mask. The
        paper's main model uses deep_supervision=False.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_filters=(64, 128, 256, 512),
        bottleneck_filters: int = 1024,
        dropout: float = 0.0,
        deep_supervision: bool = False,
    ):
        super().__init__()

        assert len(base_filters) == 4, "Paper uses exactly 4 encoder/decoder levels."
        f0, f1, f2, f3 = base_filters
        fb = bottleneck_filters
        self.deep_supervision = deep_supervision
        self.depth_filters = [f0, f1, f2, f3, fb]  # index 0..3 = encoder levels, 4 = bottleneck

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = UpSample()

        # ----------------------------------------------------------------
        # Encoder column (j = 0): X^{0,0}, X^{1,0}, X^{2,0}, X^{3,0}, X^{4,0}
        # ----------------------------------------------------------------
        self.enc0_0 = ResidualConvBlock(in_channels, f0, dropout)
        self.enc1_0 = ResidualConvBlock(f0, f1, dropout)
        self.enc2_0 = ResidualConvBlock(f1, f2, dropout)
        self.enc3_0 = ResidualConvBlock(f2, f3, dropout)
        self.enc4_0 = ResidualConvBlock(f3, fb, dropout)   # bottleneck, X^{4,0}

        # ----------------------------------------------------------------
        # Nested decoder columns (j = 1, 2, 3, 4) following the U-Net++
        # dense skip-pathway graph:
        #   X^{i,j} = ResBlock( [ X^{i,k} for k < j ] concat Up(X^{i+1,j-1}) )
        # gated through an AttentionGate before concatenation, where the
        # gating signal is the upsampled feature from the row below.
        # ----------------------------------------------------------------

        # --- column j = 1 ---
        self.att0_1 = AttentionGate(gate_channels=f1, skip_channels=f0)
        self.dec0_1 = ResidualConvBlock(f0 + f1, f0, dropout)

        self.att1_1 = AttentionGate(gate_channels=f2, skip_channels=f1)
        self.dec1_1 = ResidualConvBlock(f1 + f2, f1, dropout)

        self.att2_1 = AttentionGate(gate_channels=f3, skip_channels=f2)
        self.dec2_1 = ResidualConvBlock(f2 + f3, f2, dropout)

        self.att3_1 = AttentionGate(gate_channels=fb, skip_channels=f3)
        self.dec3_1 = ResidualConvBlock(f3 + fb, f3, dropout)

        # --- column j = 2 ---
        self.att0_2 = AttentionGate(gate_channels=f1, skip_channels=f0 * 2)
        self.dec0_2 = ResidualConvBlock(f0 * 2 + f1, f0, dropout)

        self.att1_2 = AttentionGate(gate_channels=f2, skip_channels=f1 * 2)
        self.dec1_2 = ResidualConvBlock(f1 * 2 + f2, f1, dropout)

        self.att2_2 = AttentionGate(gate_channels=f3, skip_channels=f2 * 2)
        self.dec2_2 = ResidualConvBlock(f2 * 2 + f3, f2, dropout)

        # --- column j = 3 ---
        self.att0_3 = AttentionGate(gate_channels=f1, skip_channels=f0 * 3)
        self.dec0_3 = ResidualConvBlock(f0 * 3 + f1, f0, dropout)

        self.att1_3 = AttentionGate(gate_channels=f2, skip_channels=f1 * 3)
        self.dec1_3 = ResidualConvBlock(f1 * 3 + f2, f1, dropout)

        # --- column j = 4 (final decoder output row, X^{0,4}) ---
        self.att0_4 = AttentionGate(gate_channels=f1, skip_channels=f0 * 4)
        self.dec0_4 = ResidualConvBlock(f0 * 4 + f1, f0, dropout)

        # ----------------------------------------------------------------
        # Final 1x1 conv + sigmoid (applied outside, see `predict`) head(s)
        # ----------------------------------------------------------------
        self.final_conv = nn.Conv2d(f0, out_channels, kernel_size=1)
        if self.deep_supervision:
            self.final_conv_1 = nn.Conv2d(f0, out_channels, kernel_size=1)
            self.final_conv_2 = nn.Conv2d(f0, out_channels, kernel_size=1)
            self.final_conv_3 = nn.Conv2d(f0, out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _gated_concat(att_gate: AttentionGate, gating_signal, skip_features, target_size):
        """Upsample skip features are NOT touched; the gating signal IS
        upsampled to the skip resolution, gates the skip features, and the
        gated skip features are concatenated with the upsampled gating
        signal for the residual block input."""
        up_gate = nn.functional.interpolate(
            gating_signal, size=target_size, mode="bilinear", align_corners=False
        )
        gated_skip, _ = att_gate(g=up_gate, x=skip_features)
        return torch.cat([gated_skip, up_gate], dim=1)

    def forward(self, x: torch.Tensor):
        # -------------------- Encoder (column j = 0) --------------------
        x0_0 = self.enc0_0(x)                       # f0,  full res
        x1_0 = self.enc1_0(self.pool(x0_0))          # f1,  1/2 res
        x2_0 = self.enc2_0(self.pool(x1_0))          # f2,  1/4 res
        x3_0 = self.enc3_0(self.pool(x2_0))          # f3,  1/8 res
        x4_0 = self.enc4_0(self.pool(x3_0))          # fb,  1/16 res (bottleneck)

        size0 = x0_0.shape[-2:]
        size1 = x1_0.shape[-2:]
        size2 = x2_0.shape[-2:]
        size3 = x3_0.shape[-2:]

        # -------------------- Column j = 1 --------------------
        x0_1 = self.dec0_1(self._gated_concat(self.att0_1, x1_0, x0_0, size0))
        x1_1 = self.dec1_1(self._gated_concat(self.att1_1, x2_0, x1_0, size1))
        x2_1 = self.dec2_1(self._gated_concat(self.att2_1, x3_0, x2_0, size2))
        x3_1 = self.dec3_1(self._gated_concat(self.att3_1, x4_0, x3_0, size3))

        # -------------------- Column j = 2 --------------------
        skip0_2 = torch.cat([x0_0, x0_1], dim=1)
        x0_2 = self.dec0_2(self._gated_concat(self.att0_2, x1_1, skip0_2, size0))

        skip1_2 = torch.cat([x1_0, x1_1], dim=1)
        x1_2 = self.dec1_2(self._gated_concat(self.att1_2, x2_1, skip1_2, size1))

        skip2_2 = torch.cat([x2_0, x2_1], dim=1)
        x2_2 = self.dec2_2(self._gated_concat(self.att2_2, x3_1, skip2_2, size2))

        # -------------------- Column j = 3 --------------------
        skip0_3 = torch.cat([x0_0, x0_1, x0_2], dim=1)
        x0_3 = self.dec0_3(self._gated_concat(self.att0_3, x1_2, skip0_3, size0))

        skip1_3 = torch.cat([x1_0, x1_1, x1_2], dim=1)
        x1_3 = self.dec1_3(self._gated_concat(self.att1_3, x2_2, skip1_3, size1))

        # -------------------- Column j = 4 (final) --------------------
        skip0_4 = torch.cat([x0_0, x0_1, x0_2, x0_3], dim=1)
        x0_4 = self.dec0_4(self._gated_concat(self.att0_4, x1_3, skip0_4, size0))

        logits = self.final_conv(x0_4)

        if self.deep_supervision:
            logits_1 = self.final_conv_1(x0_1)
            logits_2 = self.final_conv_2(x0_2)
            logits_3 = self.final_conv_3(x0_3)
            return {
                "out": logits,
                "ds1": logits_1,
                "ds2": logits_2,
                "ds3": logits_3,
            }

        return logits

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Convenience inference helper: logits -> sigmoid -> binary mask."""
        self.eval()
        logits = self.forward(x)
        if isinstance(logits, dict):
            logits = logits["out"]
        probs = torch.sigmoid(logits)
        return (probs >= threshold).float()


def build_model(config: Dict = None) -> ResidualAttentionUNetPlusPlus:
    """Factory that builds the model from a plain dict config (see configs/default.yaml)."""
    config = config or {}
    model_cfg = config.get("model", {})
    return ResidualAttentionUNetPlusPlus(
        in_channels=model_cfg.get("in_channels", 1),
        out_channels=model_cfg.get("out_channels", 1),
        base_filters=tuple(model_cfg.get("base_filters", [64, 128, 256, 512])),
        bottleneck_filters=model_cfg.get("bottleneck_filters", 1024),
        dropout=model_cfg.get("dropout", 0.0),
        deep_supervision=model_cfg.get("deep_supervision", False),
    )


if __name__ == "__main__":
    # Quick sanity check matching the paper's input size (1 x 512 x 512).
    model = ResidualAttentionUNetPlusPlus()
    dummy = torch.randn(2, 1, 512, 512)
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Output shape: {out.shape}")
    print(f"Total parameters: {n_params / 1e6:.2f} M")
