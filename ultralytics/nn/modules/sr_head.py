# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Super-Resolution auxiliary head for small object detection feature enhancement."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = ["SRAuxHead"]


class ResidualBlock(nn.Module):
    """Simple residual block for SR reconstruction."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.act(x + residual)


class SRAuxHead(nn.Module):
    """Super-Resolution auxiliary head for YOLO detection backbone.

    Takes multi-scale backbone features and reconstructs a super-resolved image.
    The reconstruction loss serves as an auxiliary task to encourage the backbone
    to learn high-frequency detail representations beneficial for small object detection.

    Args:
        in_channels: List of input channels from backbone feature pyramid levels.
        out_channels: Output channels (3 for RGB reconstruction).
        mid_channels: Internal feature channels for SR processing.
        upscale_factor: Upscaling factor (default 2x).
        num_blocks: Number of residual blocks in the reconstruction stage.
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int = 3,
        mid_channels: int = 64,
        upscale_factor: int = 2,
        num_blocks: int = 4,
    ):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.out_channels = out_channels

        # Feature fusion: merge multi-scale backbone features into a single SR stream
        # Each input level gets its own adapter to mid_channels
        self.adapters = nn.ModuleList([
            nn.Sequential(
                Conv(ch, mid_channels, 1, 1),  # 1x1 conv to unify channels
            )
            for ch in in_channels
        ])

        # Main SR reconstruction body
        sr_body = []
        for _ in range(num_blocks):
            sr_body.append(ResidualBlock(mid_channels))
        self.sr_body = nn.Sequential(*sr_body)

        # Upsampling layers (pixel shuffle for efficient upscaling)
        up_layers = []
        curr_channels = mid_channels
        remaining_scale = upscale_factor
        while remaining_scale > 1:
            up_layers.extend([
                nn.Conv2d(curr_channels, curr_channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.SiLU(),
            ])
            remaining_scale //= 2
        self.upsample = nn.Sequential(*up_layers)

        # Final reconstruction layer
        self.reconstruct = nn.Sequential(
            nn.Conv2d(curr_channels, curr_channels // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(curr_channels // 2, out_channels, 3, padding=1),
            nn.Sigmoid(),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for all convolutional layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            features: List of feature maps from backbone [P2_fine, P3, P4, P5_coarse].
                     All will be upsampled to the highest resolution (P2).

        Returns:
            Reconstructed image tensor in [0, 1] range.
        """
        # Get target spatial size (highest resolution feature = P2)
        target_h, target_w = features[0].shape[2:]

        # Adapt and upsample each feature level to target size
        adapted = []
        for i, (feat, adapter) in enumerate(zip(features, self.adapters)):
            x = adapter(feat)
            if x.shape[2:] != (target_h, target_w):
                x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
            adapted.append(x)

        # Fuse multi-scale features (sum fusion for simplicity)
        fused = adapted[0]
        for feat in adapted[1:]:
            fused = fused + feat

        # SR reconstruction
        x = self.sr_body(fused)
        x = self.upsample(x)
        out = self.reconstruct(x)
        return out

    def get_reconstruction_loss(
        self,
        sr_output: torch.Tensor,
        target_img: torch.Tensor,
        loss_type: str = "l1",
    ) -> torch.Tensor:
        """Compute reconstruction loss between SR output and target image.

        Args:
            sr_output: SR reconstructed image (B, 3, H*2, W*2).
            target_img: Ground truth high-resolution image (B, 3, H*2, W*2).
            loss_type: Loss type - "l1", "l2", or "huber".

        Returns:
            Reconstruction loss scalar.
        """
        # Ensure same spatial size
        if sr_output.shape[-2:] != target_img.shape[-2:]:
            target_img = F.interpolate(
                target_img, size=sr_output.shape[-2:], mode='bilinear', align_corners=False
            )

        if loss_type == "l1":
            return F.l1_loss(sr_output, target_img)
        elif loss_type == "l2":
            return F.mse_loss(sr_output, target_img)
        elif loss_type == "huber":
            return F.smooth_l1_loss(sr_output, target_img)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
